"""InterFusion (radar-only adaptation) BEV backbone.

The published InterFusion (Wang et al., IROS 2022) fuses 4D-radar and LiDAR via
an interaction (cross-attention) module. Our datasets are radar-only, so the
cross-modal interaction has no second modality to attend to. This module keeps
InterFusion's signature idea -- an attention-based feature interaction on the
BEV plane -- but runs it as a *self*-attention enhancement over the radar BEV
features (every BEV cell aggregates global context from a pooled key/value set).

It wraps the standard SECOND ``BaseBEVBackbone``; the attention is added on top
of ``spatial_features_2d``. ``gamma`` is initialised to 0 so the network starts
identical to plain SECOND and learns how much global context to inject -- this
keeps early training stable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_bev_backbone import BaseBEVBackbone


class InterFusionBEVBackbone(BaseBEVBackbone):
    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg, input_channels)
        c = self.num_bev_features
        reduction = model_cfg.get('ATTN_REDUCTION', 4)
        self.kv_pool = model_cfg.get('ATTN_KV_POOL', 8)
        self.inner = max(c // reduction, 1)
        self.q_conv = nn.Conv2d(c, self.inner, 1, bias=False)
        self.k_conv = nn.Conv2d(c, self.inner, 1, bias=False)
        self.v_conv = nn.Conv2d(c, self.inner, 1, bias=False)
        self.out_conv = nn.Conv2d(self.inner, c, 1, bias=False)
        self.out_norm = nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, data_dict):
        data_dict = super().forward(data_dict)
        x = data_dict['spatial_features_2d']
        B, C, H, W = x.shape

        q = self.q_conv(x).view(B, self.inner, H * W).permute(0, 2, 1)        # [B, HW, inner]
        kv = F.avg_pool2d(x, kernel_size=self.kv_pool, ceil_mode=True)        # [B, C, h, w]
        k = self.k_conv(kv).view(B, self.inner, -1)                          # [B, inner, M]
        v = self.v_conv(kv).view(B, self.inner, -1).permute(0, 2, 1)         # [B, M, inner]

        attn = torch.softmax(torch.bmm(q, k) / (self.inner ** 0.5), dim=-1)  # [B, HW, M]
        ctx = torch.bmm(attn, v).permute(0, 2, 1).contiguous().view(B, self.inner, H, W)
        ctx = self.out_norm(self.out_conv(ctx))

        data_dict['spatial_features_2d'] = x + self.gamma * ctx
        return data_dict
