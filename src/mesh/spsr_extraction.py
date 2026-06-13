"""
PHANTOM-ECHO REVEAL — SPSR Extraction with Batched Color Baking (Bug 3 fix)
spsr_extraction_fixed.py

Bug 3 fix: bake_vertex_colors used a per-vertex Python loop over Open3D KDTree.
           For 500k vertices this took minutes.
           Replaced with scipy cKDTree batched query — seconds instead.
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# BUG-V18-6 FIX: import semantic labeler so label_vertices() runs after SPSR
try:
    from src.mesh.semantic_labeler import label_vertices, labels_to_colors, LABEL_COLORS
    _SEM_LABELER_OK = True
except ImportError:
    _SEM_LABELER_OK = False
    logger.warning("semantic_labeler not found — semantic vertex labels skipped")


def bake_vertex_colors_fast(vertices: np.ndarray,
                              gaussian_positions: np.ndarray,
                              gaussian_colors: np.ndarray,
                              radius: float = 0.05,
                              k: int = 6) -> np.ndarray:
    """
    Bake Gaussian colors onto mesh vertices using batched KD-tree query.

    Bug 3 fix: original iterated `for i, vert in enumerate(vertices):`
    with a per-vertex kdtree.search_radius_vector_3d() call — O(N) python loop.
    scipy cKDTree.query_ball_point() returns ALL radius neighbours at once.

    Args:
        vertices:           (M, 3) mesh vertex positions
        gaussian_positions: (N, 3) Gaussian world positions
        gaussian_colors:    (N, 3) RGB in [0, 1]
        radius:             search radius in meters
        k:                  max neighbours per vertex (for speed cap)

    Returns:
        (M, 3) float32 vertex colors in [0, 1]
    """
    from scipy.spatial import cKDTree

    M = len(vertices)
    N = len(gaussian_positions)

    if N == 0:
        return np.full((M, 3), 0.5, dtype=np.float32)

    logger.info(f"Color baking: {M} vertices ← {N} Gaussians (batched)")

    tree = cKDTree(gaussian_positions)

    # Batched k-NN query (much faster than per-vertex calls)
    dist, idx = tree.query(vertices, k=min(k, N), workers=-1)

    # Handle scalar dist/idx when k=1
    if k == 1 or (dist.ndim == 1):
        dist = dist.reshape(M, 1)
        idx  = idx.reshape(M, 1)

    # Inverse-distance weighting
    weights = 1.0 / (dist + 1e-6)        # (M, k)
    weights[dist > radius] = 0.0          # zero out neighbours beyond radius

    weight_sum = weights.sum(axis=1, keepdims=True)   # (M, 1)

    # For vertices with no neighbours within radius, use nearest
    no_hit = (weight_sum.squeeze() < 1e-9)
    if np.any(no_hit):
        nearest_idx = idx[:, 0]            # (M,)
        weights[no_hit, 0] = 1.0
        weight_sum[no_hit] = 1.0

    weights /= weight_sum                  # normalize

    # Weighted color interpolation: (M, k) @ colors[idx] → (M, 3)
    # idx shape: (M, k), colors: (N, 3)
    neighbour_colors = gaussian_colors[idx]    # (M, k, 3)
    vertex_colors = np.einsum("mk,mkc->mc", weights, neighbour_colors)
    vertex_colors = np.clip(vertex_colors, 0, 1).astype(np.float32)

    logger.info(f"Color baking complete: {M} vertices colored")
    return vertex_colors


def run_spsr_pipeline(gaussian_positions: np.ndarray,
                       gaussian_normals: np.ndarray,
                       gaussian_colors: np.ndarray,
                       depth: int = 9,
                       density_pct: float = 5.0,
                       output_ply: Optional[str] = None):
    """
    Full SPSR pipeline with batched color baking.

    Returns open3d TriangleMesh or None if open3d unavailable.
    """
    # EDGE-V22 FIX: Poisson needs minimum density for a valid octree; a
    # corner-only scan with <100 points made SPSR fail silently downstream.
    if gaussian_positions is None or len(gaussian_positions) < 100:
        logger.warning(f"SPSR skipped: only "
                       f"{0 if gaussian_positions is None else len(gaussian_positions)} "
                       f"points (<100 minimum for a stable octree)")
        return None
    # EDGE-V22 FIX: Poisson needs minimum density for a valid octree; a
    # corner-only scan with <100 points made SPSR fail silently downstream.
    if gaussian_positions is None or len(gaussian_positions) < 100:
        logger.warning(f"SPSR skipped: only "
                       f"{0 if gaussian_positions is None else len(gaussian_positions)} "
                       f"points (<100 minimum for a stable octree)")
        return None
    try:
        import open3d as o3d
    except ImportError:
        logger.error("open3d not installed. Run: pip install open3d")
        return None

    if len(gaussian_positions) == 0:
        logger.warning("No Gaussians to mesh")
        return None

    logger.info(f"SPSR: {len(gaussian_positions)} input points, depth={depth}")

    # Build oriented point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(gaussian_positions.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(gaussian_normals.astype(np.float64))
    pcd.colors  = o3d.utility.Vector3dVector(gaussian_colors.astype(np.float64))

    # SPSR reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Density-based pruning
    densities_np = np.asarray(densities)
    threshold = np.percentile(densities_np, density_pct)
    keep = densities_np > threshold
    mesh = mesh.select_by_index(np.where(keep)[0])

    # Bug 3 fix: batched vertex color baking
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    vertex_colors = bake_vertex_colors_fast(
        vertices, gaussian_positions, gaussian_colors
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.astype(np.float64))

    logger.info(
        f"SPSR complete: {len(mesh.vertices)} vertices, "
        f"{len(mesh.triangles)} triangles"
    )

    # BUG-V18-6 FIX: Semantic labeling — wire label_vertices() before saving.
    # semantic_labeler.py exists with a correct implementation but was never
    # called from any mesh pipeline. Without this the Semantic Accuracy KPI
    # produces no output and cannot be evaluated.
    sem_labels  = None
    sem_json_path = None
    if _SEM_LABELER_OK:
        try:
            import json as _json
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            vnorms = np.asarray(mesh.vertex_normals, dtype=np.float32)

            sem_labels = label_vertices(verts, vnorms)

            # Blend semantic colours (25%) into texture colours (75%)
            sem_rgb = np.array(
                [LABEL_COLORS.get(l, (0.5, 0.5, 0.5)) for l in sem_labels],
                dtype=np.float64
            )
            if mesh.has_vertex_colors():
                orig = np.asarray(mesh.vertex_colors, dtype=np.float64)
                blended = 0.75 * orig + 0.25 * sem_rgb
            else:
                blended = sem_rgb
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.clip(blended, 0, 1)
            )

            # Export JSON sidecar for eval harness
            if output_ply:
                sem_json_path = output_ply.replace(".ply", "_semantics.json")
                unique, counts = np.unique(sem_labels, return_counts=True)
                sem_data = {
                    "n_vertices":   int(len(verts)),
                    "label_counts": {str(k): int(v)
                                     for k, v in zip(unique, counts)},
                }
                with open(sem_json_path, "w") as _f:
                    _json.dump(sem_data, _f, indent=2)
            logger.info(
                f"Semantic labels applied: "
                f"{dict(zip(*np.unique(sem_labels, return_counts=True)))}"
            )
        except Exception as _se:
            logger.warning(f"Semantic labeling failed (non-fatal): {_se}")

    if output_ply:
        o3d.io.write_triangle_mesh(output_ply, mesh)
        logger.info(f"Mesh saved: {output_ply}")
        if sem_json_path:
            logger.info(f"Semantic sidecar: {sem_json_path}")

    return mesh
