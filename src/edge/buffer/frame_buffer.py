"""
PHANTOM-ECHO REVEAL — Frame Buffer + Stereo Anchor Selector
frame_buffer.py + crop_extractor.py (combined)

Maintains a rolling buffer of recent camera frames for:
    1. Stereo-Anchor selection (VideoScene conditioning)
    2. Temporal consistency (consecutive frame pairs for training)
    3. Camera trajectory for SAS triangulation

Flaw 20 fix — Stereo-Anchor Selection:
    VideoScene generates flat billboard-like objects when conditioned on
    a single image. Two frames with wide baseline give genuine stereo depth.
    We select the frame pair maximizing:
        score(f) = w1 * baseline(f, f_current)
                 + w2 * depth_confidence(f, bbox)
                 + w3 * bbox_overlap(f, bbox)
    Minimum baseline threshold: 5cm.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

MIN_BASELINE_M   = 0.05    # 5cm minimum stereo baseline
MAX_FRAMES       = 100     # rolling buffer capacity
SCORE_W_BASELINE = 0.5
SCORE_W_CONF     = 0.3
SCORE_W_OVERLAP  = 0.2


@dataclass
class BufferedFrame:
    """One frame entry in the rolling buffer."""
    frame_id:           int
    position:           np.ndarray    # (3,) camera world position
    rotation:           np.ndarray    # (3, 3) camera rotation matrix
    depth_map:          np.ndarray    # (H, W) float32
    confidence_map:     np.ndarray    # (H, W) uint8
    rgb_image:          np.ndarray    # (H, W, 3) uint8
    camera_intrinsics:  Dict[str, float]
    timestamp_s:        float
    mean_confidence:    float = 0.0   # mean ARKit confidence in [0,2]

    def __post_init__(self):
        if self.mean_confidence == 0.0:
            valid = self.depth_map > 0.1
            if np.any(valid):
                self.mean_confidence = float(
                    self.confidence_map[valid].mean()
                )


class FrameBuffer:
    """
    Rolling buffer of recent camera frames with stereo-anchor selection.
    """

    def __init__(self, max_frames: int = MAX_FRAMES):
        self._frames: List[BufferedFrame] = []
        self._max_frames = max_frames

    def add_frame(self, frame: BufferedFrame) -> None:
        """Add one frame to the buffer, evicting oldest if full."""
        self._frames.append(frame)
        if len(self._frames) > self._max_frames:
            self._frames.pop(0)

    def __len__(self) -> int:
        return len(self._frames)

    def latest(self) -> Optional[BufferedFrame]:
        return self._frames[-1] if self._frames else None

    def get_positions(self) -> np.ndarray:
        """Return (N, 3) camera positions array."""
        if not self._frames:
            return np.zeros((0, 3))
        return np.array([f.position for f in self._frames], dtype=np.float32)

    def get_rotations(self) -> np.ndarray:
        """Return (N, 3, 3) rotation matrices."""
        if not self._frames:
            return np.zeros((0, 3, 3))
        return np.array([f.rotation for f in self._frames], dtype=np.float32)

    # ── Stereo-Anchor selection (Flaw 20 fix) ─────────────────────────────
    def select_stereo_anchor(self,
                              current_frame: BufferedFrame,
                              target_bbox_min: Optional[np.ndarray] = None,
                              target_bbox_max: Optional[np.ndarray] = None
                              ) -> Optional[BufferedFrame]:
        """
        Select the best stereo anchor frame from the buffer.

        Maximizes:
            score(f) = w_baseline * baseline(f, current)
                     + w_conf     * mean_confidence(f)
                     + w_overlap  * bbox_overlap(f, bbox)

        Args:
            current_frame:    current camera frame (the query)
            target_bbox_min:  (3,) target occluded region bbox
            target_bbox_max:  (3,) target occluded region bbox

        Returns:
            Best BufferedFrame or None if no valid anchor exists
        """
        if len(self._frames) < 2:
            logger.warning("Frame buffer has <2 frames — no stereo anchor")
            return None

        best_frame = None
        best_score = -np.inf

        for frame in self._frames:
            if frame.frame_id == current_frame.frame_id:
                continue

            # Baseline
            baseline = float(np.linalg.norm(frame.position - current_frame.position))
            if baseline < MIN_BASELINE_M:
                continue

            # Normalize baseline (expect 0–2m range)
            baseline_score = min(1.0, baseline / 2.0)

            # Depth confidence
            conf_score = frame.mean_confidence / 2.0   # ARKit conf in [0,2]

            # BBox overlap (fraction of bbox visible in this frame)
            overlap_score = 0.5   # default if no bbox provided
            if target_bbox_min is not None and target_bbox_max is not None:
                overlap_score = self._bbox_visibility_score(
                    frame, target_bbox_min, target_bbox_max
                )

            score = (SCORE_W_BASELINE * baseline_score
                     + SCORE_W_CONF    * conf_score
                     + SCORE_W_OVERLAP * overlap_score)

            if score > best_score:
                best_score = score
                best_frame = frame

        if best_frame is None:
            logger.warning(
                f"No stereo anchor with baseline ≥ {MIN_BASELINE_M*100:.0f}cm "
                f"({len(self._frames)} frames in buffer)"
            )
        else:
            baseline = float(np.linalg.norm(
                best_frame.position - current_frame.position
            ))
            logger.info(
                f"Stereo anchor: frame {best_frame.frame_id}, "
                f"baseline={baseline:.2f}m, score={best_score:.3f}"
            )

        return best_frame

    def _bbox_visibility_score(self,
                                 frame: BufferedFrame,
                                 bbox_min: np.ndarray,
                                 bbox_max: np.ndarray) -> float:
        """
        Estimate what fraction of a 3D bbox is visible in this frame.
        Projects bbox corners and checks how many land in image bounds.
        """
        intrinsics = frame.camera_intrinsics
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        H, W = frame.depth_map.shape

        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = frame.rotation
        cam_to_world[:3, 3] = frame.position
        world_to_cam = np.linalg.inv(cam_to_world)

        # 8 corners of bbox
        corners = np.array([
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
        ])

        visible = 0
        for corner in corners:
            corner_h = np.append(corner, 1.0)
            p_cam = (world_to_cam @ corner_h)[:3]
            if p_cam[2] <= 0.1:
                continue
            u = fx * p_cam[0] / p_cam[2] + cx
            v = fy * p_cam[1] / p_cam[2] + cy
            if 0 <= u < W and 0 <= v < H:
                visible += 1

        return visible / 8.0

    # ── Crop extractor ─────────────────────────────────────────────────────
    def extract_stereo_crops(self,
                               current_frame: BufferedFrame,
                               bbox_min: np.ndarray,
                               bbox_max: np.ndarray,
                               crop_size: Tuple[int, int] = (224, 224)
                               ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract two RGB crops for VideoScene stereo conditioning.
        Crop 1: from current frame around the occluded region.
        Crop 2: from best stereo anchor frame.

        Returns:
            (crop_current, crop_anchor) — (H, W, 3) uint8 each, or None
        """
        anchor = self.select_stereo_anchor(current_frame, bbox_min, bbox_max)

        crop_current = self._project_crop(current_frame, bbox_min, bbox_max, crop_size)
        crop_anchor  = self._project_crop(anchor, bbox_min, bbox_max, crop_size) if anchor else None

        return crop_current, crop_anchor

    def _project_crop(self,
                        frame: Optional[BufferedFrame],
                        bbox_min: np.ndarray,
                        bbox_max: np.ndarray,
                        crop_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Project bbox into frame image space and extract a crop."""
        if frame is None:
            return None
        try:
            import cv2
        except ImportError:
            return None

        intrinsics = frame.camera_intrinsics
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        H, W = frame.depth_map.shape

        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = frame.rotation
        cam_to_world[:3, 3] = frame.position
        world_to_cam = np.linalg.inv(cam_to_world)

        # Project bbox center
        center = (bbox_min + bbox_max) / 2
        center_h = np.append(center, 1.0)
        p_cam = (world_to_cam @ center_h)[:3]

        if p_cam[2] <= 0.1:
            return None

        u_c = int(fx * p_cam[0] / p_cam[2] + cx)
        v_c = int(fy * p_cam[1] / p_cam[2] + cy)

        # Crop radius (proportional to bbox size in image space)
        bbox_size_m = np.linalg.norm(bbox_max - bbox_min)
        crop_r = int(fx * bbox_size_m / (2 * max(p_cam[2], 0.1)))
        crop_r = max(30, min(crop_r, min(H, W) // 2))

        u0 = max(0, u_c - crop_r); u1 = min(W, u_c + crop_r)
        v0 = max(0, v_c - crop_r); v1 = min(H, v_c + crop_r)

        if u1 <= u0 or v1 <= v0:
            return None

        crop = frame.rgb_image[v0:v1, u0:u1]
        if crop.size == 0:
            return None

        crop_resized = cv2.resize(crop, (crop_size[1], crop_size[0]))
        return crop_resized

    def get_consecutive_pairs(self,
                               max_pairs: int = 100) -> List[Tuple[BufferedFrame, BufferedFrame]]:
        """
        Return consecutive (frame_t, frame_t+1) pairs for temporal consistency.
        """
        pairs = []
        for i in range(min(len(self._frames) - 1, max_pairs)):
            pairs.append((self._frames[i], self._frames[i + 1]))
        return pairs
