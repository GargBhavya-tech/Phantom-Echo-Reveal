"""
PHANTOM-ECHO REVEAL — Static/Dynamic Gaussian Separation
static_dynamic_sep.py

Layer 1: Separates Gaussian cloud into:
    STATIC  — permanent geometry (walls, floors, furniture)
    DYNAMIC — moving objects (tagged ORANGE)

Uses SlotLSTM tracker output to mask out dynamic Gaussians from the
static reconstruction, preventing ghost geometry contamination.

Flaw 35 fix: dynamic objects no longer corrupt the static map.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

from src.edge.tracking.slot_lstm import SlotLSTMTracker, DynamicTrack

logger = logging.getLogger(__name__)

ORANGE_TAG = "ORANGE"


def separate_static_dynamic(
    gaussians: List[Dict[str, Any]],
    tracker: SlotLSTMTracker,
    depth_map: np.ndarray,
    rgb_image: np.ndarray,
    intrinsics: Dict[str, float],
    cam_to_world: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split Gaussian list into static and dynamic subsets.

    Steps:
        1. Run SlotLSTM tracker update on current frame
        2. Get dynamic object bboxes from confirmed tracks
        3. Mask Gaussians inside any dynamic bbox → ORANGE
        4. Return (static_gaussians, dynamic_gaussians)

    Args:
        gaussians:    full Gaussian list from DDGS
        tracker:      SlotLSTMTracker instance (stateful across frames)
        depth_map:    (H, W) float32 current depth
        rgb_image:    (H, W, 3) uint8 current RGB
        intrinsics:   camera intrinsics dict {fx, fy, cx, cy}
        cam_to_world: (4, 4) camera pose matrix

    Returns:
        (static_gaussians, dynamic_gaussians)
    """
    if not gaussians:
        return [], []

    # Update tracker
    confirmed_tracks = tracker.update(
        depth_map, rgb_image, intrinsics, cam_to_world
    )

    if not confirmed_tracks:
        logger.debug("No dynamic tracks — all Gaussians are static")
        return gaussians, []

    # Build position array
    positions = np.array(
        [g.get("position", [0, 0, 0]) for g in gaussians],
        dtype=np.float32
    )

    # Get dynamic mask
    dynamic_mask = tracker.get_dynamic_gaussian_mask(positions, expand_m=0.1)
    n_dynamic = int(dynamic_mask.sum())

    static_gaussians = []
    dynamic_gaussians = []

    for i, g in enumerate(gaussians):
        if dynamic_mask[i]:
            g_copy = dict(g)
            g_copy["tag"] = ORANGE_TAG
            dynamic_gaussians.append(g_copy)
        else:
            static_gaussians.append(g)

    logger.info(
        f"Static/Dynamic separation: "
        f"{len(static_gaussians)} static, {n_dynamic} ORANGE "
        f"({len(confirmed_tracks)} tracks)"
    )

    return static_gaussians, dynamic_gaussians


def merge_for_viewer(static_gaussians: List[Dict[str, Any]],
                      dynamic_gaussians: List[Dict[str, Any]]
                      ) -> List[Dict[str, Any]]:
    """Merge static + dynamic for the WebGPU viewer (shows both layers)."""
    return static_gaussians + dynamic_gaussians
