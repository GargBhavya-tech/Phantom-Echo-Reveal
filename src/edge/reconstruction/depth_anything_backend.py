"""
PHANTOM-ECHO REVEAL — Depth-Anything-V2 backend  (v24)
depth_anything_backend.py

A REAL pretrained monocular metric-depth model for the real-RGB path (photo mode
and real datasets). This is the "real depth model" for real-world input — as
opposed to the synthetic-domain learned completer, which on planar synthetic
rooms loses to interpolation (see output/depth_completion_eval.json).

It is gated behind an explicit enable + an available download, because pretrained
weights live on Hugging Face. In sandboxes / offline demos it degrades gracefully
to the existing path, so nothing crashes if the model isn't present.

ENABLE (on a machine with internet):
    pip install "transformers>=4.45" torch pillow
    export PHANTOM_DEPTH_BACKEND=depth_anything_v2
    # first run downloads ~ depth-anything/Depth-Anything-V2-Small-hf (~100MB)

Then QuantVGGT(mode="depth_anything") (or the photo endpoint) uses it for the
dense depth of real RGB frames; sparse ARKit/acoustic points are then fused on
top to restore metric scale.

Why this is the right tool here: Depth-Anything-V2 is trained on ~62M images and
generalises to unseen real indoor scenes far better than anything trainable in a
hackathon. We use it for *relative* depth and rescale to metric using the sparse
metric points the pipeline already has (least-squares scale+shift alignment) —
the standard monocular-to-metric trick.
"""
from __future__ import annotations
import os
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_ID = os.environ.get("PHANTOM_DEPTH_MODEL",
                           "depth-anything/Depth-Anything-V2-Small-hf")


class DepthAnythingBackend:
    """Lazy-loaded Depth-Anything-V2 wrapper with metric rescaling."""

    def __init__(self, model_id: str = _MODEL_ID, device: Optional[str] = None):
        self.ok = False
        self.model_id = model_id
        self._pipe = None
        try:
            import torch
            from transformers import pipeline
            self._torch = torch
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._pipe = pipeline(task="depth-estimation", model=model_id, device=dev)
            self.ok = True
            logger.info(f"DepthAnythingBackend: loaded {model_id} on {dev}.")
        except Exception as e:
            logger.warning(
                f"DepthAnythingBackend unavailable ({type(e).__name__}: {e}). "
                f"Falling back to the classical densifier. To enable: "
                f"`pip install transformers torch pillow` with internet access.")

    # ── metric rescaling ────────────────────────────────────────────────────
    @staticmethod
    def _fit_scale_shift(rel: np.ndarray, sparse_metric: np.ndarray) -> tuple[float, float]:
        """
        Solve  metric ≈ a*rel + b  on pixels where we have sparse metric depth,
        via least squares. This converts relative monocular depth to metric using
        the pipeline's existing sparse points (ARKit + acoustic).
        """
        m = sparse_metric > 0.05
        if m.sum() < 10:
            return 1.0, 0.0
        r = rel[m].astype(np.float64)
        z = sparse_metric[m].astype(np.float64)
        A = np.stack([r, np.ones_like(r)], 1)
        (a, b), *_ = np.linalg.lstsq(A, z, rcond=None)
        if not np.isfinite(a) or a <= 0:
            return 1.0, 0.0
        return float(a), float(b)

    def estimate_metric_depth(self,
                              rgb: np.ndarray,
                              sparse_metric: Optional[np.ndarray] = None
                              ) -> Optional[np.ndarray]:
        """
        rgb           : (H,W,3) uint8
        sparse_metric : (H,W) metric depth with 0 where unknown (for rescaling)
        returns       : (H,W) metric depth, or None if backend unavailable.
        """
        if not self.ok:
            return None
        from PIL import Image
        img = Image.fromarray(rgb.astype(np.uint8))
        out = self._pipe(img)
        # transformers depth pipeline returns {"depth": PIL, "predicted_depth": tensor}
        rel = np.array(out["depth"], dtype=np.float32)
        if rel.shape[:2] != rgb.shape[:2]:
            # resize relative map to rgb resolution
            rel = np.array(Image.fromarray(rel).resize(
                (rgb.shape[1], rgb.shape[0]), Image.BILINEAR), dtype=np.float32)
        # Depth-Anything outputs inverse-depth-like maps; normalise then rescale
        rel = rel - rel.min()
        if rel.max() > 1e-6:
            rel = rel / rel.max()
        if sparse_metric is not None:
            a, b = self._fit_scale_shift(rel, sparse_metric)
            metric = a * rel + b
        else:
            metric = 0.5 + 4.0 * rel          # fallback heuristic scale
        return np.clip(metric, 0.1, 10.0).astype(np.float32)


_SINGLETON: Optional[DepthAnythingBackend] = None


def get_backend() -> DepthAnythingBackend:
    """Process-wide lazy singleton (model load is expensive)."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = DepthAnythingBackend()
    return _SINGLETON


def is_enabled() -> bool:
    return os.environ.get("PHANTOM_DEPTH_BACKEND", "").lower() == "depth_anything_v2"
