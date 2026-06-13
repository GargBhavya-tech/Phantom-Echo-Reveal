"""
PHANTOM-ECHO REVEAL — Spatial Anchor Ray-Space Optimization (Edge)
ray_registration.py

Tracks camera motion during the cloud round-trip (scan → upload → generation → download)
via 1-DoF ray-space locking. Compensates for phone movement so generated
geometry is correctly anchored in world space on return.

Problem:
    Cloud round-trip latency: 300ms–2s (WiFi variable)
    At typical phone motion 0.1–0.5 m/s: 5cm–100cm displacement
    Without compensation: generated geometry appears in wrong world position

Solution (1-DoF ray-space registration):
    Lock the PRIMARY viewing ray (center of frame → scene) as an anchor.
    All subsequent camera poses are expressed as rotations around this ray.
    The ray direction in world space is preserved even as phone moves.
    On geometry return: apply inverse ray rotation to anchor geometry.

Math:
    anchor_ray_world = R_anchor @ [0, 0, 1]^T  (optical axis at scan time)
    For each subsequent frame:
        current_ray = R_current @ [0, 0, 1]^T
        rotation_delta = R s.t. R @ anchor_ray = current_ray
        Solved via Rodrigues: axis = cross(a, b), angle = acos(dot(a, b))
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraAnchor:
    """Spatial anchor captured at scan submission time."""
    world_position:     np.ndarray    # (3,) camera position
    world_rotation:     np.ndarray    # (3, 3) rotation matrix (cols = camera axes)
    anchor_ray_world:   np.ndarray    # (3,) primary viewing ray in world space
    timestamp_s:        float
    frame_id:           int


@dataclass
class PoseUpdate:
    """One IMU/ARKit pose sample during cloud round-trip."""
    world_position:     np.ndarray    # (3,)
    world_rotation:     np.ndarray    # (3, 3)
    timestamp_s:        float


@dataclass
class AnchorCompensation:
    """Compensation transform to apply to returned geometry."""
    translation:        np.ndarray    # (3,) world translation
    rotation:           np.ndarray    # (3, 3) rotation matrix
    max_position_drift_m: float       # distance phone moved during round-trip
    max_angle_drift_deg:  float       # maximum angular drift
    confidence:         float         # 1.0 = perfect anchor, 0.0 = too much drift


# ── Ray-space math ─────────────────────────────────────────────────────────

def rotation_from_optical_axis(camera_rotation: np.ndarray) -> np.ndarray:
    """
    Extract primary viewing ray (optical axis = camera Z) in world space.

    Args:
        camera_rotation: (3, 3) camera-to-world rotation

    Returns:
        (3,) unit vector — optical axis in world coordinates
    """
    # Camera looks along +Z axis in camera frame
    optical_z = np.array([0.0, 0.0, 1.0])
    ray_world = camera_rotation @ optical_z
    norm = np.linalg.norm(ray_world)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return ray_world / norm


def rodrigues_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix R such that R @ a = b.
    Uses Rodrigues formula.

    Args:
        a, b: (3,) unit vectors

    Returns:
        (3, 3) rotation matrix
    """
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)

    cross = np.cross(a, b)
    dot   = np.clip(np.dot(a, b), -1.0, 1.0)
    angle = float(np.arccos(dot))

    if abs(angle) < 1e-7:
        return np.eye(3)
    if abs(angle - np.pi) < 1e-7:
        # 180-degree rotation — find perpendicular axis
        perp = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis) + 1e-9
    else:
        axis = cross / (np.linalg.norm(cross) + 1e-9)

    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    K = np.array([
        [ 0,      -axis[2], axis[1]],
        [ axis[2], 0,      -axis[0]],
        [-axis[1], axis[0], 0      ]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


# ── Stereo Anchor Frame Buffer ─────────────────────────────────────────────

class StereoAnchorFrameBuffer:
    """
    Selects optimal frame pairs with maximum baseline separation and
    depth confidence to condition VideoScene generation.

    Two frames with wide baseline → better 3D conditioning for VideoScene.
    Selection criteria:
        1. Baseline ≥ 20cm (sufficient stereo separation)
        2. Depth confidence: max mean ARKit confidence
        3. Temporal spread: frames not too close in time
    """

    def __init__(self, max_frames: int = 50,
                  min_baseline_m: float = 0.20):
        self._frames: List[dict] = []
        self._max_frames = max_frames
        self._min_baseline = min_baseline_m

    def add_frame(self, position: np.ndarray,
                   rotation: np.ndarray,
                   depth_confidence_mean: float,
                   timestamp_s: float,
                   frame_id: int) -> None:
        self._frames.append({
            "position":    position.copy(),
            "rotation":    rotation.copy(),
            "confidence":  depth_confidence_mean,
            "timestamp_s": timestamp_s,
            "frame_id":    frame_id,
        })
        # Evict oldest
        if len(self._frames) > self._max_frames:
            self._frames.pop(0)

    def select_best_pair(self) -> Optional[Tuple[dict, dict]]:
        """
        Select the frame pair with maximum baseline × mean_confidence product.

        Returns:
            (frame_a, frame_b) pair or None if insufficient frames
        """
        if len(self._frames) < 2:
            return None

        best_score = -1.0
        best_pair  = None

        for i in range(len(self._frames)):
            for j in range(i + 1, len(self._frames)):
                fa, fb = self._frames[i], self._frames[j]
                baseline = float(np.linalg.norm(
                    fa["position"] - fb["position"]
                ))
                if baseline < self._min_baseline:
                    continue

                score = baseline * (fa["confidence"] + fb["confidence"]) / 2.0
                if score > best_score:
                    best_score = score
                    best_pair  = (fa, fb)

        if best_pair is None:
            logger.warning(
                f"No frame pair with baseline ≥ {self._min_baseline}m "
                f"({len(self._frames)} frames in buffer)"
            )
        else:
            fa, fb = best_pair
            baseline = float(np.linalg.norm(fa["position"] - fb["position"]))
            logger.info(
                f"Best stereo pair: frames {fa['frame_id']}+{fb['frame_id']}, "
                f"baseline={baseline:.2f}m, score={best_score:.3f}"
            )

        return best_pair


# ── Main anchor registration ───────────────────────────────────────────────

class RaySpaceRegistrar:
    """
    Tracks camera motion during cloud round-trips and computes
    compensation transforms for returned geometry.
    """

    def __init__(self, max_drift_m: float = 0.5, max_drift_deg: float = 30.0):
        self._anchor: Optional[CameraAnchor] = None
        self._pose_buffer: List[PoseUpdate]  = []
        self._max_drift_m   = max_drift_m
        self._max_drift_deg = max_drift_deg

    def set_anchor(self, cam_position: np.ndarray,
                    cam_rotation: np.ndarray,
                    timestamp_s: float,
                    frame_id: int) -> CameraAnchor:
        """
        Set the anchor frame at cloud submission time.
        Call this immediately BEFORE sending scan to cloud.
        """
        anchor_ray = rotation_from_optical_axis(cam_rotation)
        self._anchor = CameraAnchor(
            world_position=cam_position.copy(),
            world_rotation=cam_rotation.copy(),
            anchor_ray_world=anchor_ray,
            timestamp_s=timestamp_s,
            frame_id=frame_id,
        )
        self._pose_buffer.clear()
        logger.info(
            f"Anchor set: frame={frame_id}, "
            f"pos={cam_position.round(3)}, "
            f"ray={anchor_ray.round(3)}"
        )
        return self._anchor

    def update_pose(self, cam_position: np.ndarray,
                     cam_rotation: np.ndarray,
                     timestamp_s: float) -> None:
        """Call for each ARKit pose update during cloud round-trip."""
        self._pose_buffer.append(PoseUpdate(
            world_position=cam_position.copy(),
            world_rotation=cam_rotation.copy(),
            timestamp_s=timestamp_s,
        ))

    def compute_compensation(self) -> Optional[AnchorCompensation]:
        """
        Compute the compensation transform to apply to returned geometry.
        Call this when geometry returns from the cloud.

        Returns:
            AnchorCompensation or None if anchor not set
        """
        if self._anchor is None:
            logger.warning("No anchor set — cannot compute compensation")
            return None

        if not self._pose_buffer:
            # No movement during round-trip — identity transform
            return AnchorCompensation(
                translation=np.zeros(3),
                rotation=np.eye(3),
                max_position_drift_m=0.0,
                max_angle_drift_deg=0.0,
                confidence=1.0
            )

        # Latest pose
        latest = self._pose_buffer[-1]

        # Position drift
        pos_drift = float(np.linalg.norm(
            latest.world_position - self._anchor.world_position
        ))

        # Angular drift (1-DoF ray-space)
        anchor_ray  = self._anchor.anchor_ray_world
        current_ray = rotation_from_optical_axis(latest.world_rotation)
        dot = float(np.clip(np.dot(anchor_ray, current_ray), -1.0, 1.0))
        angle_drift_deg = float(np.degrees(np.arccos(dot)))

        # Compute compensation rotation (anchor_ray → current_ray)
        R_comp = rodrigues_rotation(current_ray, anchor_ray)

        # Translation: move generated geometry from current phone pos to anchor pos
        t_comp = self._anchor.world_position - latest.world_position

        # Confidence: degrades with drift
        pos_conf   = max(0.0, 1.0 - pos_drift / self._max_drift_m)
        angle_conf = max(0.0, 1.0 - angle_drift_deg / self._max_drift_deg)
        confidence = pos_conf * angle_conf

        logger.info(
            f"Compensation: pos_drift={pos_drift*100:.1f}cm, "
            f"angle_drift={angle_drift_deg:.1f}°, "
            f"confidence={confidence:.2f}"
        )

        if pos_drift > self._max_drift_m or angle_drift_deg > self._max_drift_deg:
            logger.warning(
                f"Excessive drift (pos={pos_drift:.2f}m, angle={angle_drift_deg:.1f}°) "
                f"— generated geometry may be misaligned"
            )

        return AnchorCompensation(
            translation=t_comp,
            rotation=R_comp,
            max_position_drift_m=pos_drift,
            max_angle_drift_deg=angle_drift_deg,
            confidence=confidence,
        )

    def apply_compensation(self, positions: np.ndarray,
                             compensation: AnchorCompensation) -> np.ndarray:
        """
        Apply compensation transform to generated geometry positions.

        Args:
            positions:    (N, 3) generated Gaussian positions
            compensation: computed AnchorCompensation

        Returns:
            (N, 3) compensated positions in correct world frame
        """
        if compensation is None:
            return positions

        # Apply rotation around anchor position
        anchor_pos = self._anchor.world_position if self._anchor else np.zeros(3)
        centered   = positions - anchor_pos
        rotated    = (compensation.rotation @ centered.T).T
        compensated = rotated + anchor_pos + compensation.translation

        logger.debug(
            f"Compensation applied to {len(positions)} points, "
            f"confidence={compensation.confidence:.2f}"
        )
        return compensated
