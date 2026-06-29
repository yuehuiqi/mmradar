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
import math

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


def _cfg_get(config, key, default):
    """Read a key from either an EasyDict or a plain dictionary."""
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class RadarAuxiliaryAttention(nn.Module):
    """Identity-initialized motion/intensity/structure attention.

    Motion and intensity are optional because not every radar dataset exposes
    those channels.  The structural branch remains useful for sparse XYZ-only
    data by describing pillar density, geometric spread, and range.  Separate
    branches are combined in logit space, which implements the desired
    "motion OR stable return" behavior without suppressing hovering targets.
    """

    def __init__(
        self,
        dim,
        max_points,
        max_range,
        config,
        raw_num_point_features,
    ):
        super().__init__()
        self.max_points = max(int(max_points), 1)
        self.max_range = max(float(max_range), 1e-3)
        self.motion_index = int(_cfg_get(config, "MOTION_FEATURE_INDEX", -1))
        self.intensity_index = int(_cfg_get(config, "INTENSITY_FEATURE_INDEX", -1))
        self.motion_scale = max(float(_cfg_get(config, "MOTION_SCALE", 1.0)), 1e-6)
        self.intensity_scale = max(float(_cfg_get(config, "INTENSITY_SCALE", 1.0)), 1e-6)
        self.gate_scale = float(_cfg_get(config, "GATE_SCALE", 1.0))
        if not 0.0 < self.gate_scale <= 1.0:
            raise ValueError("AUX_ATTENTION.GATE_SCALE must be in (0, 1]")
        self.use_structure = bool(_cfg_get(config, "USE_STRUCTURE", True))
        self.use_range = bool(_cfg_get(config, "USE_RANGE", True))

        self.has_motion = 0 <= self.motion_index < raw_num_point_features
        self.has_intensity = 0 <= self.intensity_index < raw_num_point_features
        hidden_dim = max(dim // 4, 8)

        self.structure_branch = None
        if self.use_structure:
            structure_dim = 3 if self.use_range else 2
            self.structure_branch = self._make_branch(structure_dim, hidden_dim, dim)
        self.motion_branch = (
            self._make_branch(2, hidden_dim, dim) if self.has_motion else None
        )
        self.intensity_branch = (
            self._make_branch(2, hidden_dim, dim) if self.has_intensity else None
        )

    @staticmethod
    def _make_branch(input_dim, hidden_dim, output_dim):
        branch = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # sigmoid(0) * 2 == 1, so enabling this module starts as an identity
        # mapping and cannot destabilize an existing PFA training recipe.
        nn.init.zeros_(branch[-1].weight)
        nn.init.zeros_(branch[-1].bias)
        return branch

    @staticmethod
    def _masked_mean(values, mask, count):
        return (values * mask).sum(dim=1) / count

    def forward(self, features, raw_points, mask, point_count, pillar_positions):
        logits = features.new_zeros(features.shape)
        count = point_count.type_as(features).clamp_min(1).unsqueeze(-1)

        if self.structure_branch is not None:
            density = torch.log1p(count) / torch.log1p(
                features.new_tensor(float(self.max_points))
            )
            centered = raw_points[:, :, :3] - pillar_positions.unsqueeze(1)
            squared_radius = (centered.square().sum(dim=-1, keepdim=True) * mask)
            spread = torch.sqrt(squared_radius.sum(dim=1) / count + 1e-6)
            structure = [density, torch.log1p(spread)]
            if self.use_range:
                radial_range = torch.norm(pillar_positions, dim=-1, keepdim=True)
                structure.append(radial_range / self.max_range)
            logits = logits + self.structure_branch(torch.cat(structure, dim=-1))

        if self.motion_branch is not None:
            motion = raw_points[:, :, self.motion_index:self.motion_index + 1]
            abs_motion = motion.abs() / self.motion_scale
            mean_motion = self._masked_mean(abs_motion, mask, count)
            max_motion = abs_motion.masked_fill(mask == 0, 0.0).max(dim=1)[0]
            logits = logits + self.motion_branch(
                torch.cat([torch.log1p(mean_motion), torch.log1p(max_motion)], dim=-1)
            )

        if self.intensity_branch is not None:
            intensity = raw_points[
                :, :, self.intensity_index:self.intensity_index + 1
            ] / self.intensity_scale
            mean_intensity = self._masked_mean(intensity, mask, count)
            centered_intensity = intensity - mean_intensity.unsqueeze(1)
            variance = self._masked_mean(centered_intensity.square(), mask, count)
            intensity_stats = torch.cat(
                [
                    torch.sign(mean_intensity) * torch.log1p(mean_intensity.abs()),
                    torch.log1p(torch.sqrt(variance + 1e-6)),
                ],
                dim=-1,
            )
            logits = logits + self.intensity_branch(intensity_stats)

        gate = 1.0 + self.gate_scale * torch.tanh(logits)
        return features * gate


class DynamicPillarGraphBlock(nn.Module):
    """Dynamic EdgeConv over occupied pillars, isolated per batch item."""

    def __init__(self, dim, config):
        super().__init__()
        self.k = max(int(_cfg_get(config, "K", 8)), 1)
        self.neighbor_mode = str(
            _cfg_get(config, "NEIGHBOR_MODE", "hybrid")
        ).lower()
        if self.neighbor_mode not in {"feature", "spatial", "hybrid"}:
            raise ValueError(
                "GRAPH.NEIGHBOR_MODE must be feature, spatial, or hybrid, "
                f"got {self.neighbor_mode!r}"
            )
        self.feature_weight = float(_cfg_get(config, "FEATURE_WEIGHT", 1.0))
        self.position_weight = float(_cfg_get(config, "POSITION_WEIGHT", 0.25))
        self.position_scale = max(
            float(_cfg_get(config, "POSITION_SCALE", 4.0)), 1e-6
        )
        self.query_chunk_size = max(
            int(_cfg_get(config, "QUERY_CHUNK_SIZE", 512)), 1
        )
        self.use_position_delta = bool(
            _cfg_get(config, "USE_POSITION_DELTA", True)
        )
        self.residual_scale = float(_cfg_get(config, "RESIDUAL_SCALE", 1.0))
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("GRAPH.RESIDUAL_SCALE must be in (0, 1]")

        edge_dim = dim * 2 + (3 if self.use_position_delta else 0)
        self.pre_norm = nn.LayerNorm(dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.update_mlp[-1].weight)
        nn.init.zeros_(self.update_mlp[-1].bias)

    def _knn(self, features, positions):
        num_nodes = features.shape[0]
        k = min(self.k, num_nodes - 1)
        if k <= 0:
            return None

        # Neighbor selection itself is discrete. Detaching avoids retaining a
        # large, useless autograd graph while EdgeConv remains fully trainable.
        selection_features = self.pre_norm(features).detach()
        selection_positions = (positions / self.position_scale).detach()
        neighbor_chunks = []
        for start in range(0, num_nodes, self.query_chunk_size):
            end = min(start + self.query_chunk_size, num_nodes)
            distance = None
            if self.neighbor_mode in {"feature", "hybrid"}:
                feature_distance = torch.cdist(
                    selection_features[start:end], selection_features
                ) / (features.shape[-1] ** 0.5)
                distance = self.feature_weight * feature_distance
            if self.neighbor_mode in {"spatial", "hybrid"}:
                position_distance = torch.cdist(
                    selection_positions[start:end], selection_positions
                )
                weighted_position = self.position_weight * position_distance
                distance = (
                    weighted_position
                    if distance is None
                    else distance + weighted_position
                )

            row_index = torch.arange(start, end, device=features.device)
            local_row_index = torch.arange(end - start, device=features.device)
            distance[local_row_index, row_index] = float("inf")
            neighbor_chunks.append(
                torch.topk(distance, k=k, dim=-1, largest=False).indices
            )
        return torch.cat(neighbor_chunks, dim=0)

    def _single_frame_update(self, features, positions):
        neighbor_index = self._knn(features, positions)
        if neighbor_index is None:
            return features.new_zeros(features.shape)

        normalized = self.pre_norm(features)
        center = normalized.unsqueeze(1).expand(-1, neighbor_index.shape[1], -1)
        neighbor = normalized[neighbor_index]
        edge_parts = [center, neighbor - center]
        if self.use_position_delta:
            position_delta = (
                positions[neighbor_index] - positions.unsqueeze(1)
            ) / self.position_scale
            edge_parts.append(position_delta)
        edge_features = self.edge_mlp(torch.cat(edge_parts, dim=-1))
        max_features = edge_features.max(dim=1)[0]
        mean_features = edge_features.mean(dim=1)
        return self.update_mlp(torch.cat([max_features, mean_features], dim=-1))

    def forward(self, features, positions, batch_index):
        updates = torch.zeros_like(features)
        for sample_index in torch.unique(batch_index):
            sample_mask = batch_index == sample_index
            sample_nodes = sample_mask.nonzero(as_tuple=False).squeeze(-1)
            sample_update = self._single_frame_update(
                features[sample_nodes], positions[sample_nodes]
            )
            updates.index_copy_(0, sample_nodes, sample_update)
        return features + self.residual_scale * updates


class FrameContextBlock(nn.Module):
    """Inject inexpensive full-frame context after local graph aggregation."""

    def __init__(self, dim, residual_scale=1.0):
        super().__init__()
        self.residual_scale = residual_scale
        self.pre_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features, batch_index):
        normalized = self.pre_norm(features)
        updates = torch.zeros_like(features)
        for sample_index in torch.unique(batch_index):
            sample_mask = batch_index == sample_index
            sample_nodes = sample_mask.nonzero(as_tuple=False).squeeze(-1)
            sample_features = normalized[sample_nodes]
            global_mean = sample_features.mean(dim=0, keepdim=True)
            global_max = sample_features.max(dim=0, keepdim=True)[0]
            context = torch.cat(
                [
                    sample_features,
                    global_mean.expand_as(sample_features),
                    global_max.expand_as(sample_features),
                ],
                dim=-1,
            )
            updates.index_copy_(0, sample_nodes, self.mlp(context))
        return features + self.residual_scale * updates


class DynamicGraphRadarPillarFeatureAttention(RadarPillarFeatureAttention):
    """DG-PFA: RPFA point attention + dynamic pillar graph + radar priors.

    The original ``RadarPillarFeatureAttention`` remains untouched and
    selectable.  This class has the same input/output contract, so experiments
    can switch between the baseline and the proposed method with only
    ``MODEL.VFE.NAME``.
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        raw_num_point_features = int(num_point_features)
        super().__init__(
            model_cfg=model_cfg,
            num_point_features=num_point_features,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        )
        # Preserve the random stream seen by the unchanged BEV backbone and
        # dense head. Without this isolation, merely constructing extra VFE
        # layers changes all downstream initial weights and invalidates a
        # fixed-seed ablation against the original PFA encoder.
        downstream_rng_state = torch.random.get_rng_state()
        out_dim = self.num_filters[-1]
        graph_config = _cfg_get(self.model_cfg, "GRAPH", {})
        auxiliary_config = _cfg_get(self.model_cfg, "AUX_ATTENTION", {})
        self.mean_pool_scale = float(
            _cfg_get(self.model_cfg, "MEAN_POOL_SCALE", 1.0)
        )
        if not 0.0 < self.mean_pool_scale <= 1.0:
            raise ValueError("VFE.MEAN_POOL_SCALE must be in (0, 1]")

        self.mean_pool_projection = nn.Linear(out_dim, out_dim, bias=False)
        nn.init.zeros_(self.mean_pool_projection.weight)

        fusion_gate_init = float(
            _cfg_get(self.model_cfg, "FINAL_FUSION_GATE_INIT", 0.25)
        )
        if not 0.0 < fusion_gate_init < 1.0:
            raise ValueError("VFE.FINAL_FUSION_GATE_INIT must be in (0, 1)")
        self.final_fusion_norm = nn.LayerNorm(out_dim * 2)
        self.final_fusion_gate = nn.Linear(out_dim * 2, out_dim)
        nn.init.zeros_(self.final_fusion_gate.weight)
        nn.init.constant_(
            self.final_fusion_gate.bias,
            math.log(fusion_gate_init / (1.0 - fusion_gate_init)),
        )
        self.inference_fusion_scale = float(
            _cfg_get(self.model_cfg, "INFERENCE_FUSION_SCALE", 1.0)
        )
        if not 0.0 <= self.inference_fusion_scale <= 1.0:
            raise ValueError("VFE.INFERENCE_FUSION_SCALE must be in [0, 1]")

        max_range = max(
            abs(float(point_cloud_range[0])),
            abs(float(point_cloud_range[1])),
            abs(float(point_cloud_range[3])),
            abs(float(point_cloud_range[4])),
        )
        self.auxiliary_attention = None
        if bool(_cfg_get(auxiliary_config, "ENABLED", True)):
            self.auxiliary_attention = RadarAuxiliaryAttention(
                dim=out_dim,
                max_points=int(_cfg_get(auxiliary_config, "MAX_POINTS", 32)),
                max_range=max_range,
                config=auxiliary_config,
                raw_num_point_features=raw_num_point_features,
            )

        num_graph_layers = max(int(_cfg_get(graph_config, "NUM_LAYERS", 2)), 0)
        self.graph_blocks = nn.ModuleList(
            [
                DynamicPillarGraphBlock(out_dim, graph_config)
                for _ in range(num_graph_layers)
            ]
        )
        self.frame_context = (
            FrameContextBlock(
                out_dim,
                residual_scale=float(
                    _cfg_get(graph_config, "RESIDUAL_SCALE", 1.0)
                ),
            )
            if bool(_cfg_get(graph_config, "USE_GLOBAL_CONTEXT", True))
            else None
        )
        torch.random.set_rng_state(downstream_rng_state)

    def forward(self, batch_dict, **kwargs):
        voxel_features = batch_dict["voxels"]
        voxel_num_points = batch_dict["voxel_num_points"]
        coords = batch_dict["voxel_coords"]

        point_count = voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        points_mean = (
            voxel_features[:, :, :3].sum(dim=1, keepdim=True)
            / point_count.clamp_min(1)
        )
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x
            + self.x_offset
        )
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y
            + self.y_offset
        )
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z
            + self.z_offset
        )

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]
        if self.with_distance:
            features.append(
                torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            )
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(
            voxel_num_points, voxel_count, axis=0
        )
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features = features * mask

        x = self.input_embed(features)
        if self.input_norm is not None:
            torch.backends.cudnn.enabled = False
            x = self.input_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            torch.backends.cudnn.enabled = True
        x = F.relu(x) * mask
        x = self.attn(x, mask) * mask

        x_max = x.masked_fill(mask == 0, float("-inf")).max(dim=1)[0]
        x_max = torch.nan_to_num(x_max, neginf=0.0)
        denominator = voxel_num_points.type_as(x).clamp_min(1).unsqueeze(-1)
        x_mean = (x * mask).sum(dim=1) / denominator
        pillar_features = (
            x_max + self.mean_pool_scale * self.mean_pool_projection(x_mean)
        )

        # Actual mean XYZ is essential here: MMAUD uses one full-height voxel,
        # so voxel-coordinate Z is constant and cannot describe UAV elevation.
        pillar_positions = points_mean.squeeze(1)
        if self.auxiliary_attention is not None:
            pillar_features = self.auxiliary_attention(
                pillar_features,
                voxel_features,
                mask,
                voxel_num_points,
                pillar_positions,
            )

        batch_index = coords[:, 0].long()
        for graph_block in self.graph_blocks:
            pillar_features = graph_block(
                pillar_features, pillar_positions, batch_index
            )
        if self.frame_context is not None:
            pillar_features = self.frame_context(pillar_features, batch_index)

        # Preserve the original RPFA max-pooled descriptor as an explicit
        # bypass. The learned per-pillar, per-channel gate injects graph
        # context only where it is useful, protecting fine localization cues.
        fusion_input = self.final_fusion_norm(
            torch.cat([x_max, pillar_features], dim=-1)
        )
        fusion_gate = (
            self.inference_fusion_scale
            * torch.sigmoid(self.final_fusion_gate(fusion_input))
        )
        pillar_features = x_max + fusion_gate * (pillar_features - x_max)

        batch_dict["pillar_features"] = pillar_features
        return batch_dict
