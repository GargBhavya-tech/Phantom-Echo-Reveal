"""
PHANTOM-ECHO REVEAL — Real Depth-Completion Network  (v24)
depth_completion_net.py

This is an ACTUAL trained neural network, not the scipy-NN stub. It learns to
complete a dense metric depth map from ~500 sparse depth points + a grayscale
image, which is exactly the job the docs attributed to "QuantVGGT" but that the
shipped code did with nearest-neighbour fill.

It is small on purpose (≈70k params) so it trains on a 1-vCPU box in a couple of
minutes and runs in milliseconds. The architecture is a standard sparse-to-dense
completion encoder-decoder (Ma & Karaman style, scaled down):

    inputs  : [sparse_depth, validity_mask, gray]  (3 × H × W)
    output  : dense_depth                           (1 × H × W)

Honest framing: trained and evaluated on procedurally-generated rooms, so it is a
SYNTHETIC-DOMAIN model. The point it proves is that a learned completion network
beats classical interpolation on held-out room layouts — measured fairly by
masking depth and comparing to ground truth (see train_depth_completion.py).
For real-world RGB, use the Depth-Anything-V2 backend (depth_anything_backend.py).
"""
from __future__ import annotations
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:                       # torch optional at import time
    _TORCH = False


if _TORCH:
    class DepthCompletionNet(nn.Module):
        """Compact sparse-to-dense depth completion U-Net (~70k params)."""

        def __init__(self, base: int = 16):
            super().__init__()

            def cbr(i, o, k=3, s=1):
                return nn.Sequential(
                    nn.Conv2d(i, o, k, s, k // 2), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

            # encoder
            self.e1 = cbr(3, base)            # H
            self.e2 = cbr(base, base * 2, s=2)   # H/2
            self.e3 = cbr(base * 2, base * 4, s=2)   # H/4
            # bottleneck
            self.b = cbr(base * 4, base * 4)
            # decoder (transpose conv up + skip)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
            self.d2 = cbr(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
            self.d1 = cbr(base * 2, base)
            self.out = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(e1)
            e3 = self.e3(e2)
            b = self.b(e3)
            d2 = self.d2(torch.cat([self.up2(b), e2], 1))
            d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
            # predict a positive depth; sparse input is in metres ~[0.3, 8]
            return torch.relu(self.out(d1))


def _to_tensor(sparse: np.ndarray, gray: np.ndarray):
    mask = (sparse > 0.05).astype(np.float32)
    # normalise depth to ~[0,1] by /8m for stable training; undo at output
    x = np.stack([sparse / 8.0, mask, gray / 255.0], 0).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0)


class LearnedDepthCompleter:
    """
    Inference wrapper. Loads a trained checkpoint and completes sparse depth.
    Falls back (returns None) if torch or the checkpoint is unavailable, so the
    caller can drop back to the classical densifier without crashing.
    """
    def __init__(self, checkpoint_path: str, base: int = 16):
        self.ok = False
        if not _TORCH:
            logger.warning("LearnedDepthCompleter: torch unavailable — disabled.")
            return
        import os
        if not os.path.exists(checkpoint_path):
            logger.warning(f"LearnedDepthCompleter: no checkpoint at {checkpoint_path} "
                           f"— train with `python -m src.eval.train_depth_completion`.")
            return
        self.net = DepthCompletionNet(base=base)
        sd = torch.load(checkpoint_path, map_location="cpu")
        self.net.load_state_dict(sd)
        self.net.eval()
        self.ok = True
        logger.info(f"LearnedDepthCompleter: loaded real trained model "
                    f"({sum(p.numel() for p in self.net.parameters())/1e3:.0f}k params).")

    def complete(self, sparse_depth: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
        if not self.ok:
            return None
        with torch.no_grad():
            x = _to_tensor(sparse_depth, gray)
            y = self.net(x)[0, 0].cpu().numpy() * 8.0   # undo normalisation
        # keep observed sparse measurements exact (they are ground truth)
        m = sparse_depth > 0.05
        y[m] = sparse_depth[m]
        return np.clip(y, 0.1, 8.0)
