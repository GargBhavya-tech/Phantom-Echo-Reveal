"""
PHANTOM-ECHO REVEAL — Color Baker (Missing 8 fix)
Bakes Gaussian splat colours onto mesh vertices using batched KD-tree.
Previously missing from the submitted zip entirely.
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def bake_vertex_colors(vertices: np.ndarray,
                        gaussian_positions: np.ndarray,
                        gaussian_colors: np.ndarray,
                        radius: float = 0.05,
                        k: int = 6) -> np.ndarray:
    """
    Assign colours to mesh vertices by inverse-distance-weighted average
    of the k nearest Gaussian splats within `radius` metres.

    Args:
        vertices:           (M,3) mesh vertex positions
        gaussian_positions: (N,3) Gaussian world positions
        gaussian_colors:    (N,3) RGB in [0,1]
        radius:             search radius in metres
        k:                  max neighbours per vertex

    Returns:
        (M,3) float32 vertex colours in [0,1]
    """
    from scipy.spatial import cKDTree

    M = len(vertices)
    N = len(gaussian_positions)
    if N == 0:
        return np.full((M, 3), 0.5, dtype=np.float32)

    logger.info(f"Color baking: {M} vertices ← {N} Gaussians")
    tree = cKDTree(gaussian_positions)
    dist, idx = tree.query(vertices, k=min(k, N), workers=-1)

    if dist.ndim == 1:
        dist = dist.reshape(M, 1)
        idx  = idx.reshape(M, 1)

    weights = 1.0 / (dist + 1e-6)
    weights[dist > radius] = 0.0

    weight_sum = weights.sum(axis=1, keepdims=True)
    no_hit = weight_sum.squeeze() < 1e-9
    if np.any(no_hit):
        weights[no_hit, 0]  = 1.0
        weight_sum[no_hit]  = 1.0

    weights /= weight_sum
    neighbour_colors = gaussian_colors[idx]          # (M, k, 3)
    vertex_colors    = np.einsum("mk,mkc->mc", weights, neighbour_colors)
    return np.clip(vertex_colors, 0, 1).astype(np.float32)
