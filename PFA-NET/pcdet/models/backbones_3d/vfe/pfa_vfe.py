"""RPFA-Net style Pillar-Feature-Attention encoder.

Reference: Bai et al., "RPFA-Net: a 4D RaDAR Pillar Feature Attention Network
for 3D Object Detection" (ITSC 2021). The key idea of (R)PFA-Net is to replace
PointPillars' simple Pillar Feature Net (per-point Linear + max-pool) with a
self-attention module so the pooled pillar descriptor carries richer, context
aware features -- which notably helps heading/orientation regression on sparse
radar returns.

This module is a drop-in replacement for ``PillarVFE``: it produces the same
``batch_dict['pillar_features']`` tensor of shape ``[num_pillars, C]`` so the
rest of the PointPillars pipeline (scatter -> 2D backbone -> anchor head) is
unchanged. The detector NAME stays ``PointPillar``; only ``MODEL.VFE.NAME`` is
switched to ``RadarPillarFeatureAttention``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PillarPointAttention(nn.Module):
    """Multi-head self-attention over the points inside a single pillar.

    The ``P`` points of each pillar are treated as a length-``P`` sequence; every
    real point attends to every other real point (padded slots are masked out of
    the attention keys). A residual + LayerNorm keeps it stable.
    """

    def __init__(self, dim, num_heads=2, use_norm=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(dim, eps=1e-3)

    def forward(self, x, mask):
        # x: [N, P, C]; mask: [N, P, 1] with 1 for real points, 0 for padding.
        N, P, C = x.shape
        qkv = self.qkv(x).reshape(N, P, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, N, H, P, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [N, H, P, P]
        key_mask = mask.squeeze(-1)[:, None, None, :]  # [N, 1, 1, P]
        attn = attn.masked_fill(key_mask == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        # Every pillar has >=1 real point, so no attention row is fully masked;
        # nan_to_num is a cheap guard against any all-padding edge case.
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (attn @ v).transpose(1, 2).reshape(N, P, C)  # [N, P, C]
        out = self.proj(out)
        out = out + x
        if self.use_norm:
            out = self.norm(out)
        return out


class RadarPillarFeatureAttention(VFETemplate):
    """Pillar feature attention encoder (RPFA-Net), drop-in for ``PillarVFE``."""

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        out_dim = self.num_filters[-1]
        num_heads = getattr(self.model_cfg, "NUM_HEADS", 2)

        self.input_embed = nn.Linear(num_point_features, out_dim, bias=not self.use_norm)
        self.input_norm = nn.BatchNorm1d(out_dim, eps=1e-3, momentum=0.01) if self.use_norm else None
        self.attn = PillarPointAttention(out_dim, num_heads=num_heads, use_norm=self.use_norm)

        self.voxel_x, self.voxel_y, self.voxel_z = voxel_size[0], voxel_size[1], voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        return actual_num.int() > max_num

    def forward(self, batch_dict, **kwargs):
        voxel_features = batch_dict["voxels"]
        voxel_num_points = batch_dict["voxel_num_points"]
        coords = batch_dict["voxel_coords"]

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        if self.with_distance:
            features.append(torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True))
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)  # [N, P, 1]
        features = features * mask

        # Per-point embedding (Linear + BN + ReLU), matching PointPillars' PFN front.
        x = self.input_embed(features)
        if self.input_norm is not None:
            torch.backends.cudnn.enabled = False
            x = self.input_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            torch.backends.cudnn.enabled = True
        x = F.relu(x) * mask

        # RPFA-Net attention over the points of each pillar, then masked max-pool.
        x = self.attn(x, mask) * mask
        x = x.masked_fill(mask == 0, float("-inf"))
        x_max = torch.max(x, dim=1)[0]                     # [N, C]
        x_max = torch.nan_to_num(x_max, neginf=0.0)

        batch_dict["pillar_features"] = x_max
        return batch_dict
