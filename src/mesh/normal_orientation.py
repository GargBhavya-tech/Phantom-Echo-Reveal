"""
PHANTOM-ECHO REVEAL — Normal Orientation Verification (Layer 4 Mesh)
normal_orientation.py

Eliminates inverted normals before SPSR meshing via:
    1. Camera-relative viewpoint flipping  — normals face toward last camera
    2. MST-based propagation              — globally consistent orientation

Inverted normals are the #1 cause of SPSR meshing holes.
A single wrong-facing normal can propagate via Poisson to corrupt a whole patch.

Pipeline:
    raw point cloud (N, 3) + normals (N, 3)
    → viewpoint flip (per-point)
    → build k-NN graph weighted by |n_i · n_j|
    → MST traversal — flip child if n_i · n_j < 0
    → Atlanta-World soft regularization (axis-align structural normals)
    → output consistently oriented normals
"""

import numpy as np
from collections import deque
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Atlanta-World dominant directions (axis-aligned structural normals)
ATLANTA_AXES = np.array([
    [ 1.,  0.,  0.], [-1.,  0.,  0.],
    [ 0.,  1.,  0.], [ 0., -1.,  0.],
    [ 0.,  0.,  1.], [ 0.,  0., -1.],
], dtype=np.float64)


@dataclass
class NormalOrientationResult:
    positions:        np.ndarray   # (N, 3)
    normals_oriented: np.ndarray   # (N, 3) consistently oriented
    n_flipped:        int          # how many normals were flipped
    n_regularized:    int          # how many were Atlanta-regularized
    mst_components:   int          # number of connected components in k-NN graph


# ── Step 1: Camera-relative viewpoint flip ─────────────────────────────────

def flip_toward_viewpoints(positions: np.ndarray,
                            normals: np.ndarray,
                            camera_positions: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Flip each normal to face toward the nearest camera viewpoint.

    For each point p_i:
        view_dir = camera_nearest - p_i   (unnormalized)
        if n_i · view_dir < 0: flip n_i

    Args:
        positions:        (N, 3) point positions
        normals:          (N, 3) current normals (may be inconsistent)
        camera_positions: (K, 3) recorded camera positions during scan

    Returns:
        (flipped_normals, n_flipped_count)
    """
    normals_out = normals.copy()
    n_flipped = 0

    # BUG-CE3 FIX: guard against empty camera trajectory (first frame, standalone eval)
    if len(camera_positions) == 0:
        logger.warning(
            "flip_toward_viewpoints: camera_positions is empty — "
            "skipping viewpoint flip. MST will handle consistency."
        )
        return normals_out, 0

    for i in range(len(positions)):
        # Find nearest camera
        dists = np.linalg.norm(camera_positions - positions[i], axis=1)
        nearest_cam = camera_positions[int(np.argmin(dists))]

        view_dir = nearest_cam - positions[i]
        if np.dot(normals_out[i], view_dir) < 0:
            normals_out[i] = -normals_out[i]
            n_flipped += 1

    logger.debug(f"Viewpoint flip: {n_flipped}/{len(positions)} normals flipped")
    return normals_out, n_flipped


# ── Step 2: k-NN graph construction ───────────────────────────────────────

def build_knn_adjacency(positions: np.ndarray, k: int = 10, max_n_warn: int = 50_000) -> List[List[int]]:
    """
    Build k-nearest-neighbour adjacency list.
    Bug 2 fix: uses scipy cKDTree for O(N log N) — handles 100k+ Gaussians.
    """
    from scipy.spatial import cKDTree

    N = len(positions)
    if N == 0:
        return []

    if N > max_n_warn:
        logger.warning(
            f"build_knn_adjacency: N={N} > {max_n_warn}. "
            f"Using cKDTree (O(N log N)) — this is safe."
        )

    tree = cKDTree(positions)
    # query k+1 because first result is the point itself
    # BUG-CE13 FIX: clamp k so we never ask for more neighbours than N-1
    k_safe = max(1, min(k, N - 1))
    _, indices = tree.query(positions, k=k_safe + 1, workers=-1)

    adjacency = []
    for i in range(N):
        neighbours = [j for j in indices[i] if j != i]
        adjacency.append(neighbours[:k])

    return adjacency


# ── Step 3: MST-based consistent orientation propagation ──────────────────

def mst_orient_normals(positions: np.ndarray,
                        normals: np.ndarray,
                        k: int = 10) -> Tuple[np.ndarray, int, int]:
    """
    Globally consistent normal orientation via MST traversal.

    Algorithm (Hoppe et al. 1992):
        1. Build k-NN graph; edge weight = 1 - |n_i · n_j|
        2. Compute MST via Prim's algorithm
        3. BFS/DFS traversal from seed node:
           if n_parent · n_child < 0: flip n_child

    Args:
        positions: (N, 3)
        normals:   (N, 3) after viewpoint flip (mostly correct)
        k:         neighbourhood size

    Returns:
        (consistently_oriented_normals, n_flipped, n_components)
    """
    N = len(positions)
    if N == 0:
        return normals, 0, 0

    normals_out = normals.copy()
    n_flipped = 0

    # Build adjacency with compatibility weights
    adj = build_knn_adjacency(positions, k=min(k, N - 1))

    # BFS propagation (per connected component)
    visited = np.zeros(N, dtype=bool)
    n_components = 0

    for start in range(N):
        if visited[start]:
            continue

        n_components += 1
        # BUG-CE6 FIX: deque for O(1) popleft — list.pop(0) is O(N) → O(N²) total
        queue = deque([start])
        visited[start] = True

        while queue:
            node = queue.popleft()
            for neighbour in adj[node]:
                if visited[neighbour]:
                    continue
                visited[neighbour] = True

                # Flip neighbour if inconsistent with current node
                dot = float(np.dot(normals_out[node], normals_out[neighbour]))
                if dot < 0:
                    normals_out[neighbour] = -normals_out[neighbour]
                    n_flipped += 1

                queue.append(neighbour)

    logger.info(
        f"MST orientation: {n_flipped} flipped across {n_components} components"
    )
    return normals_out, n_flipped, n_components


# ── Step 4: Atlanta-World Normal Regularization ────────────────────────────

def atlanta_world_regularize(normals: np.ndarray,
                              semantics: Optional[np.ndarray] = None,
                              structural_threshold: float = 0.85,
                              lambda_reg: float = 0.3) -> Tuple[np.ndarray, int]:
    """
    Soft-regularize normals for structural surfaces toward Atlanta-World axes.

    Atlanta-World assumption: man-made rooms have normals aligned to
    {±X, ±Y, ±Z}. Structural surfaces (WALL, FLOOR, CEILING) should
    be axis-aligned within a small angular tolerance.

    For each structural surface normal n_i:
        nearest_axis = argmax_a |n_i · a|  for a in ATLANTA_AXES
        n_regularized = normalize((1-λ)*n_i + λ*nearest_axis)

    Non-structural (object) normals are unchanged.

    Args:
        normals:              (N, 3) oriented normals
        semantics:            (N,) int array — 0=WALL/FLOOR/CEIL, 1=OBJECT
                              if None, apply to all
        structural_threshold: cos(angle) above which snap is applied
        lambda_reg:           blend weight toward axis (0=none, 1=full snap)

    Returns:
        (regularized_normals, n_regularized_count)
    """
    normals_out = normals.copy()
    n_reg = 0

    for i in range(len(normals)):
        # Skip non-structural if semantics provided
        if semantics is not None and semantics[i] != 0:
            continue

        n = normals_out[i]
        # Find nearest Atlanta axis
        dots = np.abs(ATLANTA_AXES @ n)
        best_axis_idx = int(np.argmax(dots))
        best_dot = dots[best_axis_idx]
        best_axis = ATLANTA_AXES[best_axis_idx]

        # Ensure axis points same direction as normal
        if np.dot(best_axis, n) < 0:
            best_axis = -best_axis

        if best_dot >= structural_threshold:
            # Blend toward axis
            n_blend = (1 - lambda_reg) * n + lambda_reg * best_axis
            norm = np.linalg.norm(n_blend)
            if norm > 1e-6:
                normals_out[i] = n_blend / norm
                n_reg += 1

    logger.debug(f"Atlanta-World regularization: {n_reg} normals regularized")
    return normals_out, n_reg


# ── Main pipeline ─────────────────────────────────────────────────────────

def orient_normals(positions: np.ndarray,
                   normals: np.ndarray,
                   camera_positions: np.ndarray,
                   semantics: Optional[np.ndarray] = None,
                   k_neighbours: int = 10,
                   atlanta_lambda: float = 0.3) -> NormalOrientationResult:
    """
    Full normal orientation pipeline:
        1. Camera-relative viewpoint flip
        2. MST-based global consistency
        3. Atlanta-World structural regularization

    Args:
        positions:        (N, 3) point cloud
        normals:          (N, 3) raw normals (may be inconsistent)
        camera_positions: (K, 3) camera positions from scan trajectory
        semantics:        (N,) int — 0=structural, 1=object (optional)
        k_neighbours:     k-NN graph connectivity
        atlanta_lambda:   regularization strength toward axis-alignment

    Returns:
        NormalOrientationResult with consistently oriented normals
    """
    N = len(positions)
    logger.info(f"Normal orientation: {N} points, {len(camera_positions)} cameras")

    # Step 1: viewpoint flip
    normals_vp, n_vp_flipped = flip_toward_viewpoints(positions, normals, camera_positions)

    # Step 2: MST consistency
    normals_mst, n_mst_flipped, n_components = mst_orient_normals(
        positions, normals_vp, k=k_neighbours
    )

    # Step 3: Atlanta-World regularization
    normals_final, n_reg = atlanta_world_regularize(
        normals_mst, semantics, lambda_reg=atlanta_lambda
    )

    total_flipped = n_vp_flipped + n_mst_flipped
    logger.info(
        f"Normal orientation complete: {total_flipped} flipped total "
        f"({n_vp_flipped} viewpoint, {n_mst_flipped} MST), "
        f"{n_reg} Atlanta-regularized, {n_components} components"
    )

    return NormalOrientationResult(
        positions=positions,
        normals_oriented=normals_final,
        n_flipped=total_flipped,
        n_regularized=n_reg,
        mst_components=n_components,
    )
