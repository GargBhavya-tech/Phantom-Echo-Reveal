"""
PHANTOM-ECHO REVEAL — Voxel-Hashed Depth Fusion (TSDF-style denoising)
=====================================================================

Per-frame back-projected depth points scatter around the true surface by the
sensor's noise floor (consumer RGB-D: ~1-3cm + quantisation). Splatting every
frame independently bakes that scatter straight into the reconstruction, which
caps precision at the 5cm scale.

This module applies the core denoising mechanism of a TSDF volume — fuse many
noisy observations of the same surface into one averaged estimate — without the
heavy Open3D dependency (which is not installed here). It is a true voxel-hashed
fusion: points are binned into a fine voxel grid, and each occupied voxel is
collapsed to the confidence-weighted mean of its members. Voxels supported by
too little total weight (isolated single-frame noise) are dropped.

This is not full volumetric ray-cast TSDF (no signed-distance zero-crossing
extraction), but it captures the property that matters for accuracy: averaging
N independent noisy samples of a surface reduces position variance ~1/sqrt(N).
"""

from typing import Optional, Tuple
import os
import numpy as np

_KD_WORKERS = 1 if os.name == "nt" else -1


def knn_smooth(points: np.ndarray,
               k: int = 12,
               max_radius: float = 0.03,
               iters: int = 1) -> np.ndarray:
    """Denoise a point cloud by moving each point to the mean of its nearby
    neighbours, WITHOUT changing the point count (so surface coverage / recall
    is preserved — unlike voxel fusion, which drops sparse voxels).

    This is bilateral-style surface smoothing: averaging local samples of one
    surface cancels independent per-point sensor noise, while the radius cap
    prevents smearing across depth discontinuities / object boundaries.

    Args:
        points:     (N, 3) positions.
        k:          neighbours considered per point (includes self).
        max_radius: neighbours beyond this (metres) are ignored, so corners and
                    edges are not blurred across gaps.
        iters:      smoothing passes (1-2 is plenty; more over-smooths).
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(points, dtype=np.float64).copy()
    n = len(pts)
    if n < k:
        return pts
    for _ in range(max(1, iters)):
        tree = cKDTree(pts)
        dists, idx = tree.query(pts, k=k, workers=_KD_WORKERS)   # (N,k)
        neigh = pts[idx]                                          # (N,k,3)
        w = (dists <= max_radius).astype(np.float64)             # (N,k)
        w[:, 0] = 1.0                                             # always keep self
        wsum = w.sum(axis=1, keepdims=True)                      # (N,1)
        pts = (neigh * w[:, :, None]).sum(axis=1) / wsum         # (N,3)
    return pts


def voxel_fuse(points: np.ndarray,
               weights: Optional[np.ndarray] = None,
               attributes: Optional[np.ndarray] = None,
               voxel_size: float = 0.02,
               min_weight: float = 1.5
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse a point cloud by confidence-weighted averaging within voxels.

    Args:
        points:      (N, 3) positions.
        weights:     (N,) per-point confidence weights (default all 1).
        attributes:  (N, K) optional per-point attributes to average alongside
                     position (e.g. RGB colour). Returned averaged per voxel.
        voxel_size:  edge length of the fusion voxel in metres. Smaller = less
                     quantisation but less averaging.
        min_weight:  minimum summed weight for a voxel to survive. Voxels below
                     this are treated as single-observation noise and dropped.
                     Default 1.5 keeps any voxel seen by ~2+ confident frames.

    Returns:
        (fused_points (M,3), fused_attributes (M,K) or empty, fused_weight (M,))
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        empty_attr = np.zeros((0, attributes.shape[1])) if attributes is not None else np.zeros((0, 0))
        return points.reshape(0, 3), empty_attr, np.zeros(0)

    if weights is None:
        weights = np.ones(len(points), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).clip(1e-3, None)

    keys = np.floor(points / voxel_size).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)

    sum_w = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(sum_w, inv, weights)

    sum_pos = np.zeros((len(uniq), 3), dtype=np.float64)
    np.add.at(sum_pos, inv, points * weights[:, None])
    fused_pos = sum_pos / sum_w[:, None]

    if attributes is not None:
        attributes = np.asarray(attributes, dtype=np.float64)
        sum_attr = np.zeros((len(uniq), attributes.shape[1]), dtype=np.float64)
        np.add.at(sum_attr, inv, attributes * weights[:, None])
        fused_attr = sum_attr / sum_w[:, None]
    else:
        fused_attr = np.zeros((len(uniq), 0))

    keep = sum_w >= min_weight
    return fused_pos[keep], fused_attr[keep], sum_w[keep]
