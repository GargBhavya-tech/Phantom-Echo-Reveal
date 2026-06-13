"""
PHANTOM-ECHO REVEAL — Atlanta-World Normal Regularization (Flaw 33 fix)
atlanta_world.py

Soft-forces surface normals toward the three dominant structural axes
(floor=Y, walls=X/Z, ceiling=-Y) using the Atlanta-World model.

The Atlanta-World assumption: indoor scenes have predominantly
axis-aligned surfaces (floors, walls, ceilings). Non-structural
normals (furniture edges, curved objects) are left unchanged.

This is the standalone mesh-level module. The per-Gaussian version
is in normal_orientation_fixed.py. This one operates on mesh vertices
after SPSR reconstruction.

Flaw 33 fix: without this, reconstructed wall normals have ±5° noise
from SPSR approximation error, causing visible seams in the WebGPU viewer.
After Atlanta regularization: <1° deviation from structural axes.
"""

import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Atlanta structural axes
ATLANTA_AXES = np.array([
    [ 0,  1,  0],   # floor
    [ 0, -1,  0],   # ceiling
    [ 1,  0,  0],   # +X wall
    [-1,  0,  0],   # -X wall
    [ 0,  0,  1],   # +Z wall
    [ 0,  0, -1],   # -Z wall
], dtype=np.float32)

SNAP_THRESHOLD   = 0.85    # cos(angle) above which we snap to axis
BLEND_LAMBDA     = 0.30    # blend strength toward structural axis
STRUCTURAL_LABEL_THRESH = 0.70   # label a normal as "structural" if max dot > this


def classify_normals(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify each normal as structural or non-structural.

    Args:
        normals: (N, 3) unit normals

    Returns:
        (is_structural, best_axis_idx) — (N,) bool, (N,) int
    """
    dots = np.abs(normals @ ATLANTA_AXES.T)   # (N, 6)
    max_dots  = dots.max(axis=1)               # (N,)
    best_axes = dots.argmax(axis=1)            # (N,)
    is_structural = max_dots > STRUCTURAL_LABEL_THRESH
    return is_structural, best_axes


def snap_structural_normals(normals: np.ndarray,
                              snap_threshold: float = SNAP_THRESHOLD,
                              blend_lambda: float   = BLEND_LAMBDA
                              ) -> Tuple[np.ndarray, int]:
    """
    Snap structural normals to nearest Atlanta axis.

    For normals with max dot > snap_threshold:
        Blend toward the nearest structural axis:
            n_new = normalize((1-λ)*n + λ*axis)

    Non-structural normals are left unchanged.

    Args:
        normals:        (N, 3) input normals (need not be unit length)
        snap_threshold: dot-product threshold for structural classification
        blend_lambda:   interpolation weight toward structural axis [0,1]

    Returns:
        (regularized_normals, n_snapped)
    """
    # Normalize input
    norms_mag = np.linalg.norm(normals, axis=1, keepdims=True)
    safe = normals / (norms_mag + 1e-9)

    dots = np.abs(safe @ ATLANTA_AXES.T)   # (N, 6)
    max_dots  = dots.max(axis=1)
    best_axes = dots.argmax(axis=1)

    snap_mask = max_dots > snap_threshold
    n_snapped = int(snap_mask.sum())

    regularized = safe.copy()

    if n_snapped > 0:
        snap_indices = np.where(snap_mask)[0]
        for i in snap_indices:
            ax = ATLANTA_AXES[best_axes[i]].copy()
            # Preserve sign consistency
            if np.dot(regularized[i], ax) < 0:
                ax = -ax
            blended = (1.0 - blend_lambda) * regularized[i] + blend_lambda * ax
            b_norm = np.linalg.norm(blended)
            if b_norm > 1e-9:
                regularized[i] = blended / b_norm

    logger.debug(
        f"Atlanta-World: {n_snapped}/{len(normals)} normals snapped "
        f"(threshold={snap_threshold:.2f}, λ={blend_lambda:.2f})"
    )
    return regularized, n_snapped


def full_atlanta_pipeline(vertices: np.ndarray,
                           normals: np.ndarray,
                           floor_y: float = 0.0,
                           ceiling_y: float = 2.5
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full Atlanta-World pipeline for a mesh.

    Steps:
        1. Height-band hard-snap (floor/ceiling vertices → exact ±Y)
        2. Soft-blend structural normals toward nearest axis
        3. Leave non-structural normals unchanged

    Args:
        vertices: (N, 3) mesh vertex positions
        normals:  (N, 3) SPSR-computed vertex normals
        floor_y:  scene floor Y
        ceiling_y: scene ceiling Y

    Returns:
        (regularized_normals, labels) — (N,3) normals, (N,) label indices
    """
    # Step 1: Hard-snap floor/ceiling vertices
    BAND = 0.12   # 12cm height band
    y = vertices[:, 1]
    floor_mask   = np.abs(y - floor_y)   < BAND
    ceiling_mask = np.abs(y - ceiling_y) < BAND

    reg = normals.copy().astype(np.float32)
    # Normalize
    norms_mag = np.linalg.norm(reg, axis=1, keepdims=True)
    reg = reg / (norms_mag + 1e-9)

    # Hard-snap floor band → exact (0,1,0)
    if np.any(floor_mask):
        reg[floor_mask] = np.array([0.0, 1.0, 0.0])

    # Hard-snap ceiling band → exact (0,-1,0)
    if np.any(ceiling_mask):
        reg[ceiling_mask] = np.array([0.0, -1.0, 0.0])

    # Step 2: Soft-blend remaining structural normals
    non_floor_ceiling = ~(floor_mask | ceiling_mask)
    if np.any(non_floor_ceiling):
        reg_sub, n_snapped = snap_structural_normals(
            reg[non_floor_ceiling],
            snap_threshold=SNAP_THRESHOLD,
            blend_lambda=BLEND_LAMBDA,
        )
        reg[non_floor_ceiling] = reg_sub

    # Step 3: Compute label array (for color visualization)
    _, best_axes = classify_normals(reg)
    labels = best_axes   # 0=floor, 1=ceiling, 2/3=X-walls, 4/5=Z-walls

    total_snapped = int(floor_mask.sum() + ceiling_mask.sum())
    logger.info(
        f"Atlanta-World complete: {total_snapped} hard-snapped, "
        f"{len(vertices)} vertices total"
    )
    return reg, labels


def measure_normal_deviation(normals_before: np.ndarray,
                               normals_after: np.ndarray) -> Dict:
    """Measure angular deviation introduced by Atlanta regularization."""
    from typing import Dict
    dots = np.einsum("ij,ij->i", normals_before, normals_after)
    dots = np.clip(dots, -1, 1)
    angles_deg = np.degrees(np.arccos(dots))
    return {
        "mean_deg":   round(float(np.mean(angles_deg)), 3),
        "max_deg":    round(float(np.max(angles_deg)), 3),
        "p95_deg":    round(float(np.percentile(angles_deg, 95)), 3),
    }
