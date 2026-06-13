"""
PHANTOM-ECHO REVEAL — Affordance Router Plane Alignment (Layer 3)
plane_alignment.py

Mathematically slides GREEN Gaussian clusters along their semantic affordance
label using continuous plane-to-point intersections.

Three affordance classes:
    FLOOR-SUPPORTED  — object bottom snaps to nearest floor/support plane
    WALL-MOUNTED     — object back face snaps to nearest wall plane
    CEILING-HUNG     — object top snaps to ceiling plane

Key design: continuous intersection math, NOT discrete grid approximation.
For each GREEN cluster, we solve:
    t* = argmin_t ||C(t) - P_plane||   subject to  n · C(t) = d
where C(t) = cluster_centroid + t * n_plane  (slide along plane normal)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

AFFORDANCE_FLOOR   = "FLOOR_SUPPORTED"
AFFORDANCE_WALL    = "WALL_MOUNTED"
AFFORDANCE_CEILING = "CEILING_HUNG"

SEMANTIC_TO_AFFORDANCE = {
    "CHAIR": AFFORDANCE_FLOOR,   "TABLE": AFFORDANCE_FLOOR,
    "DESK":  AFFORDANCE_FLOOR,   "SOFA":  AFFORDANCE_FLOOR,
    "BED":   AFFORDANCE_FLOOR,   "BOX":   AFFORDANCE_FLOOR,
    "PLANT": AFFORDANCE_FLOOR,   "TV":    AFFORDANCE_WALL,
    "MONITOR": AFFORDANCE_WALL,  "PAINTING": AFFORDANCE_WALL,
    "SHELF": AFFORDANCE_WALL,    "CABINET":  AFFORDANCE_FLOOR,
    "LAMP":  AFFORDANCE_FLOOR,   "LIGHT": AFFORDANCE_CEILING,
    "FAN":   AFFORDANCE_CEILING, "WALL":  AFFORDANCE_WALL,
}


@dataclass
class StructuralPlane:
    """An infinite structural plane: n·x = d."""
    normal: np.ndarray    # (3,) unit normal
    d: float              # offset: n·x = d
    semantic: str         # FLOOR / WALL / CEILING
    confidence: float = 1.0

    def distance_to_point(self, p: np.ndarray) -> float:
        return float(np.dot(self.normal, p) - self.d)

    def project_point(self, p: np.ndarray) -> np.ndarray:
        """Project point p onto the plane."""
        dist = self.distance_to_point(p)
        return p - dist * self.normal


@dataclass
class GreenCluster:
    """A GREEN (IMAGINED) Gaussian cluster to be aligned."""
    region_id: str
    semantic: str
    positions: np.ndarray      # (N, 3) Gaussian positions
    centroid: np.ndarray       # (3,) cluster centroid
    bbox_min: np.ndarray       # (3,)
    bbox_max: np.ndarray       # (3,)

    def translate(self, delta: np.ndarray) -> "GreenCluster":
        return GreenCluster(
            region_id=self.region_id,
            semantic=self.semantic,
            positions=self.positions + delta,
            centroid=self.centroid + delta,
            bbox_min=self.bbox_min + delta,
            bbox_max=self.bbox_max + delta,
        )


@dataclass
class AlignmentResult:
    region_id: str
    affordance: str
    aligned_cluster: GreenCluster
    snap_plane: StructuralPlane
    translation_m: np.ndarray   # (3,) applied translation
    residual_m: float           # distance from support plane after alignment
    success: bool
    reason: str


# ── Continuous plane-to-point intersection ─────────────────────────────────

def slide_to_plane(cluster: GreenCluster,
                   plane: StructuralPlane,
                   affordance: str,
                   tolerance_m: float = 0.02) -> Tuple[np.ndarray, float]:
    """
    Compute translation delta to slide cluster onto the support plane.

    For FLOOR_SUPPORTED: bottom face (bbox_min.y) snaps to plane.
    For WALL_MOUNTED:    back face snaps (face closest to plane).
    For CEILING_HUNG:    top face (bbox_max.y) snaps to plane.

    Continuous intersection:
        Let contact_point = relevant_face_point on cluster
        signed_dist = n · contact_point - d
        delta = -signed_dist * n   (move cluster along normal to reach plane)

    Returns:
        (delta, residual_after_alignment)
    """
    n = plane.normal
    d = plane.d

    if affordance == AFFORDANCE_FLOOR:
        # Bottom face center
        contact = np.array([
            cluster.centroid[0],
            cluster.bbox_min[1],   # lowest Y
            cluster.centroid[2]
        ])
    elif affordance == AFFORDANCE_CEILING:
        # Top face center
        contact = np.array([
            cluster.centroid[0],
            cluster.bbox_max[1],   # highest Y
            cluster.centroid[2]
        ])
    elif affordance == AFFORDANCE_WALL:
        # Face closest to plane (dot product determines which face)
        faces = [cluster.bbox_min, cluster.bbox_max]
        dists = [abs(np.dot(n, f) - d) for f in faces]
        contact = faces[int(np.argmin(dists))]
    else:
        contact = cluster.centroid

    # Signed distance from contact point to plane
    signed_dist = float(np.dot(n, contact) - d)

    # Translation to bring contact point to plane
    delta = -signed_dist * n

    # Residual after translation
    residual = abs(float(np.dot(n, contact + delta) - d))

    return delta, residual


def find_best_support_plane(cluster: GreenCluster,
                             planes: List[StructuralPlane],
                             affordance: str,
                             max_search_radius_m: float = 2.0) -> Optional[StructuralPlane]:
    """
    Find the nearest structural plane matching the affordance type.

    FLOOR_SUPPORTED → look for FLOOR planes below cluster centroid
    WALL_MOUNTED    → look for WALL planes near cluster centroid
    CEILING_HUNG    → look for CEILING planes above cluster centroid
    """
    target_semantic = {
        AFFORDANCE_FLOOR:   "FLOOR",
        AFFORDANCE_WALL:    "WALL",
        AFFORDANCE_CEILING: "CEILING",
    }.get(affordance, "FLOOR")

    candidates = [p for p in planes if p.semantic == target_semantic]
    if not candidates:
        return None

    # Score: proximity to cluster centroid projected onto plane normal
    def score(plane: StructuralPlane) -> float:
        dist = abs(plane.distance_to_point(cluster.centroid))
        if dist > max_search_radius_m:
            return float('inf')
        # Directional check
        if affordance == AFFORDANCE_FLOOR:
            # Plane must be below centroid (negative signed distance from centroid)
            signed = plane.distance_to_point(cluster.centroid)
            if signed < -0.05:   # plane is above centroid — skip
                return float('inf')
        elif affordance == AFFORDANCE_CEILING:
            signed = plane.distance_to_point(cluster.centroid)
            if signed > 0.05:
                return float('inf')
        return dist

    scores = [score(p) for p in candidates]
    best_idx = int(np.argmin(scores))
    if scores[best_idx] == float('inf'):
        return None
    return candidates[best_idx]


# ── Occupancy Oracle Fallback ───────────────────────────────────────────────

def occupancy_oracle_fallback(cluster: GreenCluster,
                               voxel_grid: np.ndarray,
                               voxel_origin: np.ndarray,
                               voxel_size: float,
                               affordance: str) -> Optional[StructuralPlane]:
    """
    Fallback when no structural plane is detected nearby:
    scan the voxel occupancy grid downward (for FLOOR) or inward (for WALL)
    from the cluster centroid and synthesize a virtual plane at the first
    occupied voxel boundary.

    Args:
        cluster:       GREEN cluster to align
        voxel_grid:    (X, Y, Z) bool array — True = occupied
        voxel_origin:  (3,) world position of voxel (0,0,0)
        voxel_size:    meters per voxel
        affordance:    FLOOR_SUPPORTED / WALL_MOUNTED / CEILING_HUNG

    Returns:
        Synthesized StructuralPlane or None if no occupied voxel found
    """
    def world_to_voxel(pt):
        return ((pt - voxel_origin) / voxel_size).astype(int)

    X, Y, Z = voxel_grid.shape
    start_voxel = world_to_voxel(cluster.centroid)

    # Search direction per affordance
    directions = {
        AFFORDANCE_FLOOR:   np.array([ 0, -1,  0]),
        AFFORDANCE_CEILING: np.array([ 0,  1,  0]),
        AFFORDANCE_WALL:    np.array([-1,  0,  0]),  # search toward -X wall first
    }
    normals = {
        AFFORDANCE_FLOOR:   np.array([0.0,  1.0, 0.0]),
        AFFORDANCE_CEILING: np.array([0.0, -1.0, 0.0]),
        AFFORDANCE_WALL:    np.array([1.0,  0.0, 0.0]),
    }
    direction = directions.get(affordance, directions[AFFORDANCE_FLOOR])
    normal    = normals.get(affordance, normals[AFFORDANCE_FLOOR])

    max_steps = int(2.0 / voxel_size)   # search up to 2m
    for step in range(1, max_steps + 1):
        vox = start_voxel + step * direction
        if not (0 <= vox[0] < X and 0 <= vox[1] < Y and 0 <= vox[2] < Z):
            break
        if voxel_grid[vox[0], vox[1], vox[2]]:
            # Found occupied voxel — synthesize plane at its surface
            surface_world = voxel_origin + (vox - direction * 0.5) * voxel_size
            d = float(np.dot(normal, surface_world))
            logger.info(f"Occupancy oracle: found support at voxel {vox}, d={d:.3f}m")
            return StructuralPlane(normal=normal, d=d, semantic="FLOOR", confidence=0.6)

    logger.warning(f"Occupancy oracle: no support found for {cluster.region_id}")
    return None


# ── Main alignment entry point ─────────────────────────────────────────────

def align_cluster(cluster: GreenCluster,
                  structural_planes: List[StructuralPlane],
                  voxel_grid: Optional[np.ndarray] = None,
                  voxel_origin: Optional[np.ndarray] = None,
                  voxel_size: float = 0.05) -> AlignmentResult:
    """
    Align a single GREEN cluster to its semantic affordance plane.

    Tries structural planes first; falls back to occupancy oracle.
    """
    affordance = SEMANTIC_TO_AFFORDANCE.get(cluster.semantic, AFFORDANCE_FLOOR)

    # Step 1: find nearest structural plane
    plane = find_best_support_plane(cluster, structural_planes, affordance)

    # Step 2: fallback to occupancy oracle
    if plane is None and voxel_grid is not None:
        logger.info(f"{cluster.region_id}: no structural plane found — using occupancy oracle")
        plane = occupancy_oracle_fallback(
            cluster, voxel_grid, voxel_origin or np.zeros(3), voxel_size, affordance
        )

    if plane is None:
        logger.warning(f"{cluster.region_id}: alignment failed — no support surface found")
        return AlignmentResult(
            region_id=cluster.region_id,
            affordance=affordance,
            aligned_cluster=cluster,
            snap_plane=StructuralPlane(np.array([0.,1.,0.]), 0., "FLOOR"),
            translation_m=np.zeros(3),
            residual_m=float('inf'),
            success=False,
            reason="No structural plane or occupied voxel found within search radius"
        )

    # Step 3: continuous plane-to-point slide
    delta, residual = slide_to_plane(cluster, plane, affordance)
    aligned = cluster.translate(delta)

    logger.info(
        f"{cluster.region_id} ({cluster.semantic}→{affordance}): "
        f"delta={delta.round(3)}m, residual={residual*100:.1f}cm"
    )

    return AlignmentResult(
        region_id=cluster.region_id,
        affordance=affordance,
        aligned_cluster=aligned,
        snap_plane=plane,
        translation_m=delta,
        residual_m=residual,
        success=residual < 0.05,
        reason=f"Snapped to {plane.semantic} plane (residual={residual*100:.1f}cm)"
    )


def align_all_clusters(clusters: List[GreenCluster],
                        structural_planes: List[StructuralPlane],
                        voxel_grid: Optional[np.ndarray] = None,
                        voxel_origin: Optional[np.ndarray] = None,
                        voxel_size: float = 0.05) -> List[AlignmentResult]:
    """Align all GREEN clusters in one pass."""
    results = []
    for cluster in clusters:
        result = align_cluster(cluster, structural_planes, voxel_grid, voxel_origin, voxel_size)
        results.append(result)

    n_ok  = sum(1 for r in results if r.success)
    n_fail = len(results) - n_ok
    logger.info(f"Plane alignment: {n_ok}/{len(results)} aligned, {n_fail} failed")
    return results
