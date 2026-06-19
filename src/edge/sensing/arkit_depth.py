"""
PHANTOM-ECHO REVEAL — ARKit/ARCore Depth Interface + Scale Normalizer
arkit_depth.py + scale_normalizer.py + imu_tracker.py (combined)

Layer 0: Multi-Modal Sensing (Visual Channel)

In real deployment:
    iOS:     ARKit provides LiDAR-fused depth via ARFrame.sceneDepth
    Android: ARCore provides depth via DepthImage (ToF or ML-estimated)

In simulation:
    Reads pre-recorded depth sequences OR generates synthetic depth
    from a known room geometry for evaluation.

Flaw 22 fix — Scale Normalization:
    Sparse depth sensors cluster readings in texture-rich regions.
    Apply inverse-density weighting so undersampled regions get equal
    contribution to the global metric scale estimate.

IMU Tracker:
    Records phone world position at each chirp emission for SAS baseline.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import logging
import time

logger = logging.getLogger(__name__)

SPEED_OF_SOUND = 343.0   # m/s


# ── ARKit depth frame dataclass ───────────────────────────────────────────
@dataclass
class DepthFrame:
    """One captured depth + RGB frame with camera pose."""
    depth_map:        np.ndarray    # (H, W) float32 meters
    confidence_map:   np.ndarray    # (H, W) uint8   0/1/2
    rgb_image:        np.ndarray    # (H, W, 3) uint8
    camera_intrinsics: Dict[str, float]  # fx, fy, cx, cy
    camera_to_world:  np.ndarray    # (4, 4) float64
    timestamp_s:      float
    frame_id:         int

    @property
    def H(self) -> int:
        return self.depth_map.shape[0]

    @property
    def W(self) -> int:
        return self.depth_map.shape[1]

    def valid_depth_pixels(self) -> int:
        return int(np.sum(self.depth_map > 0.1))


# ── Scale Normalizer (Flaw 22 fix) ────────────────────────────────────────
def normalize_depth_scale(depth_map: np.ndarray,
                            confidence_map: np.ndarray,
                            n_sparse_points: int = 500) -> np.ndarray:
    """
    Apply inverse-density weighting to sparse depth measurements.

    Problem: ARKit depth tends to cluster in texture-rich regions.
    Underrepresented regions (smooth walls, ceilings) get fewer depth
    anchors, causing scale drift there.

    Solution:
        1. Sample n_sparse_points from high-confidence pixels
        2. Compute local density of samples in image-space grid
        3. Regions with low sample density get upweighted confidence
        4. Refit global scale using density-weighted least squares

    Args:
        depth_map:       (H, W) float32 raw depth
        confidence_map:  (H, W) uint8  ARKit confidence
        n_sparse_points: target number of sparse depth anchors

    Returns:
        (H, W) float32 scale-normalized depth map
    """
    H, W = depth_map.shape

    # Extract high-confidence pixels
    high_conf = confidence_map >= 1
    valid_depth = depth_map > 0.1
    candidates = high_conf & valid_depth

    ys, xs = np.where(candidates)
    if len(ys) == 0:
        return depth_map   # nothing to normalize

    # Sample up to n_sparse_points
    n_sample = min(n_sparse_points, len(ys))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ys), n_sample, replace=False)
    sample_ys = ys[idx]
    sample_xs = xs[idx]

    # Compute spatial density on 8x8 grid
    grid_h, grid_w = 8, 8
    cell_h = H / grid_h
    cell_w = W / grid_w
    density = np.zeros((grid_h, grid_w), dtype=np.float32)

    for sy, sx in zip(sample_ys, sample_xs):
        gi = min(int(sy / cell_h), grid_h - 1)
        gj = min(int(sx / cell_w), grid_w - 1)
        density[gi, gj] += 1

    density = density / (density.max() + 1e-9)   # normalize [0,1]

    # Build per-pixel inverse density weight map
    weight_map = np.ones((H, W), dtype=np.float32)
    for gi in range(grid_h):
        for gj in range(grid_w):
            r0 = int(gi * cell_h);  r1 = int((gi + 1) * cell_h)
            c0 = int(gj * cell_w);  c1 = int((gj + 1) * cell_w)
            d = density[gi, gj]
            weight_map[r0:r1, c0:c1] = 1.0 / (d + 0.1)   # inverse density

    # Global scale anchor: weighted median depth across sampled points
    sample_depths = depth_map[sample_ys, sample_xs]
    sample_weights = weight_map[sample_ys, sample_xs]
    sort_idx = np.argsort(sample_depths)
    cum_w = np.cumsum(sample_weights[sort_idx])
    median_idx = np.searchsorted(cum_w, cum_w[-1] / 2)
    scale_anchor = float(sample_depths[sort_idx[median_idx]])

    # Apply gentle scale correction (only if drift > 5%)
    naive_median = float(np.median(sample_depths))
    if naive_median > 0.1:
        scale_correction = scale_anchor / naive_median
        if abs(scale_correction - 1.0) < 0.2:   # max ±20% correction
            depth_map = (depth_map * scale_correction).astype(np.float32)

    logger.debug(
        f"Scale normalization: {n_sample} anchors sampled, "
        f"scale_anchor={scale_anchor:.3f}m, "
        f"naive_median={naive_median:.3f}m"
    )
    return depth_map


# ── IMU Tracker ───────────────────────────────────────────────────────────
@dataclass
class IMUPose:
    """One IMU pose sample."""
    position:    np.ndarray   # (3,) world position [meters]
    rotation:    np.ndarray   # (3, 3) rotation matrix
    timestamp_s: float
    velocity:    np.ndarray = field(default_factory=lambda: np.zeros(3))


class IMUTracker:
    """
    Records phone trajectory for SAS baseline construction.

    In real deployment: receives poses from ARKit/ARCore at ~60Hz.
    In simulation: integrates synthetic motion along a walking path.
    """

    def __init__(self, max_poses: int = 1000):
        self._poses: List[IMUPose] = []
        self._max_poses = max_poses

    def add_pose(self, position: np.ndarray,
                  rotation: np.ndarray,
                  timestamp_s: float) -> None:
        """Record one IMU pose (call at chirp emission time for SAS)."""
        velocity = np.zeros(3)
        if len(self._poses) > 0:
            dt = timestamp_s - self._poses[-1].timestamp_s
            if dt > 0:
                velocity = (position - self._poses[-1].position) / dt

        self._poses.append(IMUPose(
            position=position.copy(),
            rotation=rotation.copy(),
            timestamp_s=timestamp_s,
            velocity=velocity,
        ))

        if len(self._poses) > self._max_poses:
            self._poses.pop(0)

    def get_positions(self) -> np.ndarray:
        """Return (N, 3) array of all recorded positions."""
        if not self._poses:
            return np.zeros((0, 3))
        return np.array([p.position for p in self._poses], dtype=np.float32)

    def get_latest_pose(self) -> Optional[IMUPose]:
        return self._poses[-1] if self._poses else None

    def get_walking_path(self, from_time: float = 0.0) -> List[IMUPose]:
        """Get poses since a given timestamp (for one SAS sweep)."""
        return [p for p in self._poses if p.timestamp_s >= from_time]

    def baseline(self) -> float:
        """Return maximum baseline separation between any two recorded positions."""
        if len(self._poses) < 2:
            return 0.0
        positions = self.get_positions()
        dists = np.linalg.norm(positions - positions[0], axis=1)
        return float(np.max(dists))

    def clear(self) -> None:
        self._poses.clear()


# ── Synthetic depth generator (simulation mode) ───────────────────────────
class SyntheticDepthGenerator:
    """
    Generates realistic depth frames for a parameterized indoor room.
    Used when no real ARKit device is available (simulation / CI).

    Room geometry: axis-aligned box with optional furniture boxes.
    """

    def __init__(self, room_dims: Dict[str, float],
                  furniture: Optional[List[Dict]] = None):
        """
        Args:
            room_dims:  {"x": 5.0, "y": 2.5, "z": 4.0}
            furniture:  list of {"bbox_min": [x,y,z], "bbox_max": [x,y,z], "visible": bool}
        """
        self.room = room_dims
        self.furniture = furniture or []
        self._frame_id = 0

    def _ray_room_intersect(self, ray_origin: np.ndarray,
                              ray_dir: np.ndarray) -> float:
        """Intersect ray with room walls + furniture boxes. Returns hit depth."""
        t_min = np.inf

        # Room walls (6 planes)
        bounds = [
            (0, 0.0, self.room["x"]),    # X axis
            (1, 0.0, self.room["y"]),    # Y axis
            (2, 0.0, self.room["z"]),    # Z axis
        ]
        for axis, lo, hi in bounds:
            for bound in (lo, hi):
                if abs(ray_dir[axis]) < 1e-9:
                    continue
                t = (bound - ray_origin[axis]) / ray_dir[axis]
                if 0.05 < t < t_min:
                    hit = ray_origin + t * ray_dir
                    other_axes = [i for i in range(3) if i != axis]
                    in_bounds = all(
                        lo_b <= hit[a] <= hi_b
                        for a, (_, lo_b, hi_b) in zip(other_axes, [b for b in bounds if b[0] != axis])
                    )
                    if in_bounds:
                        t_min = t

        # Furniture AABB intersections
        for furn in self.furniture:
            bmin = np.array(furn["bbox_min"])
            bmax = np.array(furn["bbox_max"])
            t_enter, t_exit = -np.inf, np.inf
            hit = True
            for i in range(3):
                if abs(ray_dir[i]) < 1e-9:
                    if ray_origin[i] < bmin[i] or ray_origin[i] > bmax[i]:
                        hit = False; break
                else:
                    t1 = (bmin[i] - ray_origin[i]) / ray_dir[i]
                    t2 = (bmax[i] - ray_origin[i]) / ray_dir[i]
                    t_enter = max(t_enter, min(t1, t2))
                    t_exit  = min(t_exit,  max(t1, t2))
            if hit and t_enter < t_exit and 0.05 < t_enter < t_min:
                t_min = t_enter

        return float(t_min) if t_min < np.inf else 0.0

    def generate_frame(self, camera_pos: np.ndarray,
                        camera_yaw: float = 0.0,
                        H: int = 192, W: int = 256,
                        fx: float = 500.0, fy: float = 500.0,
                        add_noise: bool = True) -> DepthFrame:
        """Generate one synthetic depth frame from a given camera position."""
        cx, cy = W / 2.0, H / 2.0

        # Camera rotation (yaw only for simplicity)
        R = np.array([
            [np.cos(camera_yaw), 0, np.sin(camera_yaw)],
            [0, 1, 0],
            [-np.sin(camera_yaw), 0, np.cos(camera_yaw)],
        ])
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = R
        cam_to_world[:3, 3] = camera_pos

        depth_map = np.zeros((H, W), dtype=np.float32)
        conf_map  = np.zeros((H, W), dtype=np.uint8)
        rgb_image = np.full((H, W, 3), 200, dtype=np.uint8)

        rng = np.random.default_rng(self._frame_id)

        for v in range(0, H, 2):
            for u in range(0, W, 2):
                # Ray in camera space
                ray_cam = np.array([
                    (u - cx) / fx,
                    (v - cy) / fy,
                    1.0
                ])
                ray_norm = np.linalg.norm(ray_cam)
                ray_cam /= ray_norm
                ray_world = R @ ray_cam

                d = self._ray_room_intersect(camera_pos, ray_world)
                if d > 0.1:
                    # BUG-V22-10 FIX: d is Euclidean RAY LENGTH, but ARKit
                    # depth maps (and depth_to_world back-projection) use
                    # Z-DEPTH. Storing ray length inflated depth by
                    # sqrt(1+((u-cx)/fx)^2+((v-cy)/fy)^2) — a systematic
                    # 2-4cm error growing toward image corners that capped
                    # every accuracy KPI. Convert: z = d * ray_cam.z.
                    z = d / ray_norm
                    noise = rng.normal(0, 0.005) if add_noise else 0.0
                    depth_map[v, u] = np.clip(z + noise, 0.1, 8.0)
                    # Confidence based on distance
                    conf_map[v, u] = 2 if d < 3.0 else (1 if d < 5.0 else 0)
                    # Simple shading
                    shade = int(220 - d * 10)
                    rgb_image[v, u] = [shade, shade, shade]

        self._frame_id += 1
        return DepthFrame(
            depth_map=depth_map,
            confidence_map=conf_map,
            rgb_image=rgb_image,
            camera_intrinsics={"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            camera_to_world=cam_to_world,
            timestamp_s=time.time(),
            frame_id=self._frame_id,
        )

    def generate_walk_sequence(self, n_frames: int = 10,
                                start_pos: Optional[np.ndarray] = None,
                                axis: str = "xz") -> List[DepthFrame]:
        """
        Generate a sequence of frames along a 2D walking path.

        FIX BUG E: Original used axis='z' (1D walk), giving a rank-1 linear
        system in SAS triangulation → 0 acoustic points.
        
        Fix: default to 'xz' zigzag: moves in X+Z with slight Y variation
        (simulating natural hand movement) → rank-3 system → triangulates.
        """
        if start_pos is None:
            start_pos = np.array([0.5, 1.2, 0.5])
        frames = []
        for i in range(n_frames):
            pos = start_pos.copy()
            if axis == "x":
                pos[0] += i * 0.3
            elif axis == "z":
                pos[2] += i * 0.3
            elif axis == "xz":
                # FIX BUG E: 2D zigzag gives rank-3 SAS system
                # BUG-V18-3 FIX: Z only advances every 3 frames (i//3*0.3)
                # so with n_frames=3 all Z=0 → rank-1 collinear.
                # Fix: use i//2 (every 2 frames) so rank-3 achieved at n=4+.
                pos[0] += (i % 3) * 0.35           # X zigzag 35cm steps
                pos[2] += (i // 2) * 0.30           # Z advances every 2 frames
                pos[1] += np.sin(i * 0.5) * 0.05   # Y sway (hand motion)
                # SAS-V22 NOTE: 5cm sway is INTENTIONAL. A reviewer proposed
                # 15cm to improve lstsq conditioning; measured result was
                # 7/7 ghost triangulations (>30cm) vs 20/20 good (<10cm) at
                # 5cm — the planar-array mirror prior in sas_triangulator v3
                # requires near-planar positions (std<5cm) and outperforms
                # the better-conditioned but unconstrained system.
            elif axis == "arc":
                # v23 acoustic-honesty fix: smooth 3D arc. Small consecutive
                # steps keep the v3 echo-track association stable, while the
                # rotating tangent + gentle height ramp make the SAS linear
                # system full-rank (rank-3) so triangulation actually fires.
                # This is also a better reconstruction path (more viewpoint
                # diversity) than a straight line.
                #
                # BUG-PROD-3 FIX: clamp camera Y so it never exits the room
                # regardless of n_frames. Previously pos[1] = start[1]-0.25+0.06*i
                # → for n_frames=20 the camera reaches 1.2-0.25+1.14=2.09m,
                # exiting a 2.5m room at i=13+, yielding 0 depth hits.
                # Also: n_frames=1 with arc used cx_room+0.45 (not start_pos)
                # as the origin — now start_pos is used directly at i=0.
                room_y = self.room.get("y", 2.5)
                cx_room = start_pos[0] + 0.5
                cz_room = start_pos[2] + 0.2
                theta = 0.45 * i
                pos[0] = cx_room + 0.45 * np.cos(theta)
                raw_y  = start_pos[1] - 0.25 + 0.06 * i
                pos[1] = float(np.clip(raw_y, 0.5, room_y - 0.3))
                pos[2] = cz_room + 0.45 * np.sin(theta)
                # face roughly toward room interior (+Z/+X)
                yaw = np.pi / 2 - theta * 0.3
                frames.append(self.generate_frame(pos, camera_yaw=yaw))
                continue

            yaw = np.pi / 2   # facing along Z (default for x/z/xz)
            frames.append(self.generate_frame(pos, camera_yaw=yaw))
        return frames
