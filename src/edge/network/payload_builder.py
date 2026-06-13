"""
PHANTOM-ECHO REVEAL — Payload Builder
payload_builder.py

Wraps raw sensor data into typed ScanPayload / RevealPayload objects
ready for transmission to the cloud API.

Responsibilities:
    - Convert numpy arrays to JSON-serializable lists
    - Attach session_id and frame metadata
    - Cap payload size (subsample depth if > MAX_DEPTH_PIXELS)
    - Build RevealPayload from tap coordinates + anchor bbox
"""

import numpy as np
import time
import logging
from typing import Dict, Optional, List, Any

from src.shared.gaussian_format import ScanPayload, RevealPayload

logger = logging.getLogger(__name__)

MAX_DEPTH_PIXELS = 50_000   # cap before serialization (~200KB)


def build_scan_payload(session_id: str,
                        frame_id: int,
                        depth_map: np.ndarray,
                        confidence_map: np.ndarray,
                        rgb_image: np.ndarray,
                        cam_to_world: np.ndarray,
                        camera_intrinsics: Dict[str, float],
                        audio_signal: Optional[np.ndarray] = None,
                        phone_position: Optional[np.ndarray] = None,
                        subsample_stride: int = 1) -> ScanPayload:
    """
    Build a ScanPayload from raw sensor arrays.

    Args:
        subsample_stride: stride for depth subsampling (1 = full res)
    """
    H, W = depth_map.shape

    if subsample_stride > 1:
        depth_sub = depth_map[::subsample_stride, ::subsample_stride]
        conf_sub  = confidence_map[::subsample_stride, ::subsample_stride]
        rgb_sub   = rgb_image[::subsample_stride, ::subsample_stride]
        H_s, W_s = depth_sub.shape
    else:
        depth_sub = depth_map
        conf_sub  = confidence_map
        rgb_sub   = rgb_image
        H_s, W_s = H, W

    # Auto-subsample if still too large
    total_px = H_s * W_s
    if total_px > MAX_DEPTH_PIXELS:
        extra_stride = int(np.ceil(np.sqrt(total_px / MAX_DEPTH_PIXELS)))
        depth_sub = depth_sub[::extra_stride, ::extra_stride]
        conf_sub  = conf_sub[::extra_stride,  ::extra_stride]
        rgb_sub   = rgb_sub[::extra_stride,   ::extra_stride]
        H_s, W_s  = depth_sub.shape
        logger.debug(f"Auto-subsampled depth to {H_s}×{W_s} ({H_s*W_s} px)")

    return ScanPayload(
        session_id=session_id,
        frame_id=frame_id,
        timestamp_s=time.time(),
        depth_h=H_s,
        depth_w=W_s,
        depth_flat=depth_sub.flatten().tolist(),
        confidence_flat=conf_sub.flatten().tolist(),
        rgb_flat=rgb_sub.flatten().tolist(),
        cam_to_world=cam_to_world.tolist(),
        camera_intrinsics=camera_intrinsics,
        audio_flat=audio_signal.tolist() if audio_signal is not None else None,
        phone_position=phone_position.tolist() if phone_position is not None else None,
    )


def build_reveal_payload(session_id: str,
                          region_id: str,
                          semantic: str,
                          confidence_tag: str,
                          bbox_min: np.ndarray,
                          bbox_max: np.ndarray,
                          floor_y: float = 0.0,
                          ceiling_y: float = 2.5,
                          stereo_crop_a: Optional[np.ndarray] = None,
                          stereo_crop_b: Optional[np.ndarray] = None,
                          acoustic_point: Optional[np.ndarray] = None,
                          acoustic_distance_m: Optional[float] = None) -> RevealPayload:
    """
    Build a RevealPayload from a tap/trigger event.
    """
    return RevealPayload(
        session_id=session_id,
        region_id=region_id,
        semantic=semantic,
        confidence_tag=confidence_tag,
        bbox_min=bbox_min.tolist(),
        bbox_max=bbox_max.tolist(),
        floor_y=floor_y,
        ceiling_y=ceiling_y,
        stereo_crop_a=stereo_crop_a.flatten().tolist() if stereo_crop_a is not None else None,
        stereo_crop_b=stereo_crop_b.flatten().tolist() if stereo_crop_b is not None else None,
        acoustic_point=acoustic_point.tolist() if acoustic_point is not None else None,
    )


def estimate_payload_size_bytes(payload: ScanPayload) -> int:
    """Quick estimate of JSON payload size in bytes."""
    n_depth = payload.depth_h * payload.depth_w
    # float32 → 8 chars avg JSON, uint8 → 3 chars, RGB → 3 bytes × 3 chars
    return (n_depth * 8 +    # depth
            n_depth * 3 +    # confidence
            n_depth * 9 +    # rgb
            512)             # headers/metadata
