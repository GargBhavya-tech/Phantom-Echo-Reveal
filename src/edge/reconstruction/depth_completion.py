"""
PHANTOM-ECHO REVEAL — Learned Guided Depth Completion (v23)
depth_completion.py

A REAL neural network (trained from scratch, weights shipped in models/) that
upgrades sparse depth (~500 pts) to an accurate dense depth map using RGB
guidance. It replaces the scipy nearest-neighbour fill whose approximate filled
pixels were the precision bottleneck on real data (held-out precision@5cm 0.63).

Design — guided residual completion (what makes it learnable on little data):
  input  = [ RGB(3), nn_prefilled_depth(1), valid_mask(1) ]   (1,5,H,W)
  output = dense depth (1,1,H,W)
The net is given a nearest-neighbour pre-fill as a strong prior and learns to
*correct* it along RGB edges (depth discontinuities the NN fill smears). This
needs far less data than predicting metric depth from scratch and is guaranteed
to start near the baseline rather than diverge.

The exported TorchScript matches QuantVGGT's existing REAL backend interface
(5-channel in → 1-channel out), so it drops in via `QuantVGGT(checkpoint_path=...)`.
"""
from __future__ import annotations
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

TARGET_H, TARGET_W = 192, 256


def nn_prefill(sparse: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour pre-fill + valid mask (the prior fed to the net)."""
    from scipy.ndimage import distance_transform_edt
    known = sparse > 0.05
    if not known.any():
        return np.full_like(sparse, 2.0), known.astype(np.float32)
    _, idx = distance_transform_edt(~known, return_indices=True)
    filled = sparse[tuple(idx)]
    return filled.astype(np.float32), known.astype(np.float32)


def build_input_tensor(rgb: np.ndarray, sparse: np.ndarray):
    """RGB(uint8 HxWx3) + sparse depth(HxW) → (1,5,H,W) float tensor."""
    import torch
    import torch.nn.functional as F
    filled, mask = nn_prefill(sparse)
    rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    dep_t = torch.from_numpy(filled).float().unsqueeze(0)
    msk_t = torch.from_numpy(mask).float().unsqueeze(0)
    x = torch.cat([rgb_t, dep_t, msk_t], dim=0).unsqueeze(0)  # (1,5,H,W)
    if x.shape[-2:] != (TARGET_H, TARGET_W):
        x = F.interpolate(x, size=(TARGET_H, TARGET_W), mode="bilinear",
                          align_corners=False)
    return x


def make_net():
    """Small RGB-guided U-Net (~120k params) — trainable on CPU in minutes."""
    import torch
    import torch.nn as nn

    class GuidedDepthNet(nn.Module):
        def __init__(self, ch=24):
            super().__init__()
            def cbr(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1),
                                     nn.GroupNorm(max(1, o // 8), o),
                                     nn.ReLU(inplace=True))
            self.e1 = cbr(5, ch)
            self.e2 = cbr(ch, ch * 2)
            self.e3 = cbr(ch * 2, ch * 4)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                                  align_corners=False)
            self.d2 = cbr(ch * 4 + ch * 2, ch * 2)
            self.d1 = cbr(ch * 2 + ch, ch)
            self.out = nn.Conv2d(ch, 1, 1)

        def forward(self, x):
            prior = x[:, 3:4]                      # nn-prefilled depth channel
            s1 = self.e1(x)
            s2 = self.e2(self.pool(s1))
            b = self.e3(self.pool(s2))
            d2 = self.d2(torch.cat([self.up(b), s2], 1))
            d1 = self.d1(torch.cat([self.up(d2), s1], 1))
            residual = self.out(d1)               # learn a correction to the prior
            return (prior + residual).clamp(0.1, 10.0)

    return GuidedDepthNet()


def complete(rgb: np.ndarray, sparse: np.ndarray, model) -> np.ndarray:
    """Run a loaded TorchScript/eager model → dense depth (TARGET_H,TARGET_W)."""
    import torch
    x = build_input_tensor(rgb, sparse)
    with torch.no_grad():
        out = model(x)
    return out.squeeze().cpu().numpy().astype(np.float32)
