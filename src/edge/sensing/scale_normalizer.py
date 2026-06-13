"""
PHANTOM-ECHO REVEAL — Depth Scale Normalizer
scale_normalizer.py

FIX-8: This file was referenced in the bible and imported by arkit_depth.py
but was missing from the sensing/ directory.

Purpose:
    Normalises the metric scale of the depth map by aligning the ARKit raw
    depth distribution to the QuantVGGT dense depth distribution.

    Addresses the "scale drift" problem: ARKit LiDAR returns depth in metres
    but the confidence-weighted mean can drift by up to 12% between sessions
    due to lens distortion and temperature changes. This normaliser corrects
    that drift so the two depth streams are metrically aligned.

Algorithm (Robust Scale Alignment):
    1. Sample HIGH-confidence depth pixels (conf == 2) as the reference.
    2. Compute the robust median scale ratio:
           s = median(arkit_depth[mask]) / median(vggt_depth[mask] + ε)
    3. Return vggt_depth * s (scale-normalised).

If no high-confidence pixels are available, return vggt_depth unchanged.

Reference:
    DPT scale alignment: https://arxiv.org/abs/2103.13413 (Ranftl et al., 2021)
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_CONFIDENCE_HIGH = 2     # ARKit LiDAR confidence level for high-quality readings
_EPS             = 1e-6  # Prevent division by zero


def normalize_depth_scale(
    arkit_depth:   np.ndarray,
    confidence_map: np.ndarray,
    vggt_depth:    Optional[np.ndarray] = None,
    min_ref_points: int = 50,
) -> np.ndarray:
    """
    Align vggt_depth scale to arkit_depth using high-confidence pixels.

    Args:
        arkit_depth:    (H, W) float32 raw ARKit depth in metres.
        confidence_map: (H, W) uint8 ARKit confidence (0=low, 1=med, 2=high).
        vggt_depth:     (H, W) float32 QuantVGGT dense depth (optional).
                        If None, returns arkit_depth after basic sanity clamps.
        min_ref_points: minimum number of high-confidence pixels required
                        to compute a reliable scale ratio. If fewer are found,
                        returns vggt_depth (or arkit_depth) unmodified.

    Returns:
        (H, W) float32 scale-normalised depth in metres.
    """
    # Sanity clamp on ARKit depth
    arkit_clean = np.clip(arkit_depth, 0.1, 10.0).astype(np.float32)

    if vggt_depth is None:
        return arkit_clean

    vggt_clean = vggt_depth.astype(np.float32)

    # --- Build high-confidence mask ----------------------------------------
    high_conf = (confidence_map == _CONFIDENCE_HIGH)
    valid_arkit = arkit_clean > 0.15
    valid_vggt  = vggt_clean  > 0.15

    ref_mask = high_conf & valid_arkit & valid_vggt

    if int(ref_mask.sum()) < min_ref_points:
        # Not enough reference points — fall back to median-only alignment
        # using all valid pixels (less accurate but better than no correction).
        ref_mask = valid_arkit & valid_vggt
        if int(ref_mask.sum()) < 10:
            logger.debug(
                "scale_normalizer: insufficient reference points — "
                "returning vggt_depth unmodified"
            )
            return vggt_clean

    arkit_ref  = arkit_clean[ref_mask]
    vggt_ref   = vggt_clean[ref_mask]

    # Robust median scale ratio (outlier-resistant vs mean)
    scale = float(np.median(arkit_ref)) / (float(np.median(vggt_ref)) + _EPS)

    # Clamp scale to plausible range [0.5, 2.0] — beyond this something is wrong
    scale = float(np.clip(scale, 0.5, 2.0))

    normalised = vggt_clean * scale

    logger.debug(
        f"scale_normalizer: scale={scale:.4f} from {int(ref_mask.sum())} "
        f"reference pixels (high_conf={int(high_conf.sum())})"
    )

    return np.clip(normalised, 0.1, 10.0).astype(np.float32)


def align_depth_streams(
    arkit_depth:    np.ndarray,
    vggt_depth:     np.ndarray,
    confidence_map: np.ndarray,
    blend_alpha:    float = 0.7,
) -> np.ndarray:
    """
    Blend ARKit and QuantVGGT depth streams after scale alignment.

    High-confidence ARKit pixels get weight `blend_alpha`.
    Low-confidence regions use QuantVGGT (scale-aligned) depth.

    Args:
        arkit_depth:    (H, W) float32 raw ARKit depth.
        vggt_depth:     (H, W) float32 QuantVGGT dense depth.
        confidence_map: (H, W) uint8 ARKit confidence.
        blend_alpha:    weight for ARKit in high-confidence regions.

    Returns:
        (H, W) float32 blended depth.
    """
    vggt_aligned = normalize_depth_scale(arkit_depth, confidence_map, vggt_depth)

    # Build per-pixel alpha map from confidence
    # conf=2 → alpha=blend_alpha, conf=1 → alpha=0.5, conf=0 → alpha=0.0
    alpha_map = np.where(
        confidence_map == 2, blend_alpha,
        np.where(confidence_map == 1, 0.5, 0.0)
    ).astype(np.float32)

    blended = alpha_map * arkit_depth + (1.0 - alpha_map) * vggt_aligned

    # Clamp and zero out invalid regions
    blended = np.where(
        (arkit_depth > 0.1) | (vggt_aligned > 0.1),
        np.clip(blended, 0.1, 10.0),
        0.0,
    ).astype(np.float32)

    return blended
