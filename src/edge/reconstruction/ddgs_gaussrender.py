"""
PHANTOM-ECHO REVEAL — DDGS GaussRender (Layer 1)
Depth-Driven Gaussian Splatting with 2D planar disk constraint.

Key design decision (Flaw 11 fix):
    Standard 3DGS uses 3D ellipsoid Gaussians. When converted to point cloud
    for SPSR meshing, normals are inconsistent → watertight mesh fails.

    DDGS forces each Gaussian to be a 2D planar disk (one axis collapsed):
        scale_z = 0 (or ε = 1e-4)
        → point cloud normals = disk plane normals
        → SPSR produces clean, watertight mesh

Input:  ARKit depth frame + RGB frame + camera pose
Output: List of GaussianDisk objects tagged WHITE/BLUE by confidence
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Confidence tags — aligned with PHANTOM shared tag system
TAG_WHITE = "WHITE"  # ARKit LiDAR confidence ≥ 0.85 (direct measurement) → WHITE
TAG_BLUE  = "BLUE"   # ARKit LiDAR confidence 0.5–0.85 (inferred)         → BLUE
# FIX Bug 4: previous aliases were TAG_WHITE="BLUE" and TAG_BLUE="TEAL"
# This caused ARKit confidence=2 points to emit as BLUE (log-odds 2.0) instead of
# WHITE (log-odds 3.0), permanently disabling the WHITE entry in occupancy_grid.py.
# All WHITE-confidence ARKit geometry was getting 33% less occupancy weight.


@dataclass
class GaussianDisk:
    """
    A 2D planar disk Gaussian splat.

    position:   (3,) world position [meters]
    normal:     (3,) unit surface normal (disk plane normal)
    scale_xy:   (2,) disk radii in local u/v axes [meters]
    color_rgb:  (3,) float32 [0,1]
    opacity:    float 0–1
    confidence: float 0–1 (from ARKit depth confidence)
    tag:        WHITE or BLUE (confidence level)
    semantic:   'WALL', 'FLOOR', 'CEILING', 'OBJECT', 'UNKNOWN'
    plane_d:    float, the D term in Ax+By+Cz+D=0 for the disk plane
    """
    position:   np.ndarray
    normal:     np.ndarray
    scale_xy:   np.ndarray
    color_rgb:  np.ndarray
    opacity:    float = 1.0
    confidence: float = 1.0
    tag:        str = TAG_WHITE
    semantic:   str = "UNKNOWN"
    plane_d:    float = 0.0


def depth_to_world(depth_px: float, u: int, v: int,
                   fx: float, fy: float, cx: float, cy: float,
                   camera_to_world: np.ndarray) -> np.ndarray:
    """
    Back-project a depth pixel to world coordinates.

    Args:
        depth_px:         depth value in meters
        u, v:             pixel column and row
        fx, fy, cx, cy:  camera intrinsics
        camera_to_world:  (4,4) extrinsic matrix

    Returns:
        (3,) world position
    """
    # Camera coordinates
    x_cam = (u - cx) * depth_px / fx
    y_cam = (v - cy) * depth_px / fy
    z_cam = depth_px
    p_cam = np.array([x_cam, y_cam, z_cam, 1.0])

    # World coordinates
    p_world = camera_to_world @ p_cam
    return p_world[:3]


def estimate_normal_from_neighborhood(depth_map: np.ndarray,
                                       u: int, v: int,
                                       fx: float, fy: float,
                                       cx: float, cy: float,
                                       camera_to_world: np.ndarray,
                                       radius: int = 2) -> Optional[np.ndarray]:
    """
    Estimate surface normal at pixel (u,v) using cross-product of
    local depth gradient neighbors.

    Returns unit normal in world space, or None if invalid.
    """
    H, W = depth_map.shape

    # Sample 4 neighbors
    offsets = [(-radius, 0), (radius, 0), (0, -radius), (0, radius)]
    pts = []
    for du, dv in offsets:
        nu, nv = u + du, v + dv
        if 0 <= nu < W and 0 <= nv < H:
            d = depth_map[nv, nu]
            if d > 0.1:  # valid depth threshold
                pts.append(depth_to_world(d, nu, nv, fx, fy, cx, cy, camera_to_world))

    if len(pts) < 3:
        return None

    # Cross-product of two edge vectors
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0] if len(pts) > 2 else pts[3] - pts[0]

    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return None

    n = n / norm
    # Ensure normal faces camera (toward origin in camera space)
    cam_pos = camera_to_world[:3, 3]
    p_world = depth_to_world(depth_map[v, u], u, v, fx, fy, cx, cy, camera_to_world)
    if np.dot(n, cam_pos - p_world) < 0:
        n = -n

    return n


def classify_semantic(normal: np.ndarray,
                       position: np.ndarray,
                       floor_y_threshold: float = 0.3,
                       ceiling_y_threshold: float = 2.0) -> str:
    """
    Classify surface semantic class from normal direction and height.

    Conventions:
        Floor:   |normal · (0,1,0)| > 0.85  AND  y < floor_y_threshold
        Ceiling: |normal · (0,1,0)| > 0.85  AND  y > ceiling_y_threshold
        Wall:    |normal · (0,1,0)| < 0.3
        Object:  everything else
    """
    up = np.array([0.0, 1.0, 0.0])
    dot = abs(np.dot(normal, up))

    if dot > 0.85:
        if position[1] < floor_y_threshold:
            return "FLOOR"
        elif position[1] > ceiling_y_threshold:
            return "CEILING"

    if dot < 0.30:
        return "WALL"

    return "OBJECT"


def arkit_confidence_to_tag(confidence_value: int) -> Tuple[str, float]:
    """
    Map ARKit LiDAR confidence integer (0, 1, 2) to PHANTOM tag.

    ARKit values:
        0 = low confidence
        1 = medium confidence
        2 = high confidence

    Maps to shared tag system: WHITE (high-confidence sensor) / BLUE (medium) / BLUE (low).
    TEAL is reserved for acoustic SAS measurements only and is never assigned here.
    """
    if confidence_value == 2:
        return TAG_WHITE, 0.95   # TAG_WHITE = "WHITE" — high-confidence ARKit
    elif confidence_value == 1:
        return TAG_BLUE,  0.70   # TAG_BLUE  = "BLUE"  — medium-confidence ARKit
    else:
        return TAG_BLUE,  0.40   # TAG_BLUE  = "BLUE"  — low-confidence ARKit


def build_gaussian_scene(depth_map: np.ndarray,
                          confidence_map: np.ndarray,
                          rgb_image: np.ndarray,
                          camera_intrinsics: dict,
                          camera_to_world: np.ndarray,
                          stride: int = 4,
                          max_depth_m: float = 8.0) -> List[GaussianDisk]:
    """
    Convert ARKit depth frame → list of GaussianDisk objects.

    Args:
        depth_map:          (H, W) float32 depth in meters
        confidence_map:     (H, W) uint8, values 0/1/2 (ARKit LiDAR confidence)
        rgb_image:          (H, W, 3) uint8 RGB
        camera_intrinsics:  dict with fx, fy, cx, cy
        camera_to_world:    (4, 4) float64 extrinsic
        stride:             sample every N pixels (4 = 75% downsample)
        max_depth_m:        ignore depth readings beyond this distance

    Returns:
        List of GaussianDisk objects (WHITE + BLUE tags)
    """
    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    H, W = depth_map.shape
    gaussians: List[GaussianDisk] = []

    # BUG-V22-9 FIX: the synthetic densifier (nearest-neighbour fill + boxcar
    # smoothing) smears depth across object boundaries, creating "halo"
    # points floating between surfaces — the main reason BLUE median error
    # was 4cm. Reject pixels at depth discontinuities (standard RGB-D
    # practice): a real surface is locally smooth, an edge pixel is not.
    gy, gx = np.gradient(depth_map)
    edge_mask = (np.abs(gy) + np.abs(gx)) > 0.08   # 8cm/pixel jump = edge

    for v in range(0, H, stride):
        for u in range(0, W, stride):
            d = depth_map[v, u]
            if d < 0.1 or d > max_depth_m:
                continue
            if edge_mask[v, u]:
                continue

            conf_val = int(confidence_map[v, u])
            tag, confidence = arkit_confidence_to_tag(conf_val)

            # World position
            position = depth_to_world(d, u, v, fx, fy, cx, cy, camera_to_world)

            # Surface normal
            normal = estimate_normal_from_neighborhood(
                depth_map, u, v, fx, fy, cx, cy, camera_to_world
            )
            if normal is None:
                # Fallback: use view direction as normal
                cam_pos = camera_to_world[:3, 3]
                view_dir = cam_pos - position
                norm = np.linalg.norm(view_dir)
                if norm < 1e-6:
                    continue
                normal = view_dir / norm

            # Disk scale: proportional to depth (farther → larger splat)
            base_scale = d * 0.01  # 1cm per meter of depth
            scale_xy = np.array([base_scale, base_scale])

            # Color from RGB (bilinear sample)
            color = rgb_image[v, u].astype(np.float32) / 255.0

            # Semantic classification
            semantic = classify_semantic(normal, position)

            # Plane D term: D = -(A*x + B*y + C*z)
            plane_d = -float(np.dot(normal, position))

            gaussians.append(GaussianDisk(
                position=position,
                normal=normal,
                scale_xy=scale_xy,
                color_rgb=color,
                opacity=confidence,
                confidence=confidence,
                tag=tag,
                semantic=semantic,
                plane_d=plane_d
            ))

    logger.info(
        f"DDGS: {len(gaussians)} disk Gaussians from {H}x{W} frame "
        f"(stride={stride}): "
        f"{sum(g.tag==TAG_WHITE for g in gaussians)} WHITE, "
        f"{sum(g.tag==TAG_BLUE for g in gaussians)} BLUE"
    )
    return gaussians


def export_to_ply(gaussians: List[GaussianDisk], path: str) -> None:
    """
    Export Gaussian positions + normals as a PLY point cloud for SPSR meshing.

    The 2D disk constraint guarantees normal consistency → clean SPSR output.
    """
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(gaussians)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    rows = []
    for g in gaussians:
        r, gr, b = (np.clip(g.color_rgb * 255, 0, 255)).astype(int)
        rows.append(
            f"{g.position[0]:.6f} {g.position[1]:.6f} {g.position[2]:.6f} "
            f"{g.normal[0]:.6f} {g.normal[1]:.6f} {g.normal[2]:.6f} "
            f"{r} {gr} {b}"
        )

    with open(path, "w") as f:
        f.write(header)
        f.write("\n".join(rows))

    logger.info(f"Exported {len(gaussians)} Gaussians to {path}")


# ── DDGSGaussRender class wrapper ─────────────────────────────────────────
# FIX: server.py imports DDGSGaussRender but only the function build_gaussian_scene
# existed. Adding this thin OO wrapper so the import resolves.

class DDGSGaussRender:
    """
    Object-oriented wrapper around build_gaussian_scene() + export_to_ply().
    Provides the interface expected by cloud/api/server.py.

    Usage:
        renderer = DDGSGaussRender(stride=4, max_depth_m=8.0)
        gaussians = renderer.process_depth_frame(depth, conf, rgb, c2w, intrinsics)
        renderer.export_ply(gaussians, "scene.ply")
    """

    def __init__(self, stride: int = 4, max_depth_m: float = 8.0):
        self._stride      = stride
        self._max_depth_m = max_depth_m

    def process_depth_frame(
        self,
        depth_map: np.ndarray,
        confidence_map: np.ndarray,
        rgb_image: np.ndarray,
        camera_to_world: np.ndarray,
        camera_intrinsics: dict,
    ) -> List[dict]:
        """
        Run DDGS on one depth frame and return a list of Gaussian dicts
        (compatible with the server.py Gaussian dict format).
        """
        disk_gaussians = build_gaussian_scene(
            depth_map=depth_map,
            confidence_map=confidence_map,
            rgb_image=rgb_image,
            camera_intrinsics=camera_intrinsics,
            camera_to_world=camera_to_world,
            stride=self._stride,
            max_depth_m=self._max_depth_m,
        )
        # Convert GaussianDisk objects → plain dicts for server compatibility
        result = []
        for g in disk_gaussians:
            result.append({
                "position":   g.position.tolist(),
                "normal":     g.normal.tolist(),
                "color":      g.color_rgb.tolist(),
                "scale":      float(g.scale_xy[0]),
                "opacity":    float(g.opacity),
                "confidence": float(g.confidence),
                "tag":        g.tag,
                "semantic":   g.semantic,
                "plane_d":    float(g.plane_d),
                # Add fields expected by ism_filter.extract_walls_from_scene
                "semantic_tag":     g.semantic,
                "confidence_tag":   g.tag,
                "plane_normal":     g.normal.tolist(),
            })
        return result

    def export_ply(self, gaussians_or_disks, path: str) -> None:
        """Export Gaussians to PLY. Accepts both GaussianDisk objects and dicts."""
        if gaussians_or_disks and isinstance(gaussians_or_disks[0], dict):
            # Reconstruct minimal GaussianDisk objects from dicts for export_to_ply
            disks = []
            for g in gaussians_or_disks:
                disk = GaussianDisk(
                    position=np.array(g["position"]),
                    normal=np.array(g.get("normal", [0, 1, 0])),
                    scale_xy=np.array([g.get("scale", 0.05)] * 2),
                    color_rgb=np.array(g.get("color", [0.5, 0.5, 0.5])),
                    opacity=float(g.get("opacity", 1.0)),
                )
                disks.append(disk)
            export_to_ply(disks, path)
        else:
            export_to_ply(gaussians_or_disks, path)


__all__ = [
    "GaussianDisk",
    "DDGSGaussRender",
    "build_gaussian_scene",
    "export_to_ply",
    "depth_to_world",
    "estimate_normal_from_neighborhood",
    "classify_semantic",
    "arkit_confidence_to_tag",
]
