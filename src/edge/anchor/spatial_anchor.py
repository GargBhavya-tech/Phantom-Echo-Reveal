"""
PHANTOM-ECHO REVEAL — Spatial Anchor Manager
spatial_anchor.py

Manages ARKit/ARCore world anchors for persistent Gaussian placement.
Ensures generated Gaussians stay locked to real-world geometry across
phone movement and session resumption.

In real deployment:
    iOS:     uses ARAnchor + ARAnchorManager
    Android: uses ARCore Anchor + Earth anchors (if available)

In simulation:
    Maintains a dict of anchor_id → world_transform
    Transforms are updated by simulated IMU drift.

Flaw 28 fix: without anchors, generated Gaussians drift when the phone
moves. Each reveal call gets an anchor; generated Gaussians are stored
in anchor-local coordinates and transformed back to world on query.
"""

import numpy as np
import uuid
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SpatialAnchor:
    """One world-locked anchor point."""
    anchor_id:       str
    world_transform: np.ndarray    # (4, 4) anchor-to-world transform
    created_at:      float
    semantic:        str
    bbox_min:        np.ndarray    # (3,) in world space
    bbox_max:        np.ndarray    # (3,) in world space
    confidence:      float = 1.0
    is_tracked:      bool  = True
    gaussian_ids:    List[int] = field(default_factory=list)

    @property
    def position(self) -> np.ndarray:
        return self.world_transform[:3, 3]

    def world_to_anchor(self, world_pt: np.ndarray) -> np.ndarray:
        """Transform world point to anchor-local coordinates."""
        inv = np.linalg.inv(self.world_transform)
        pt_h = np.append(world_pt, 1.0)
        return (inv @ pt_h)[:3]

    def anchor_to_world(self, local_pt: np.ndarray) -> np.ndarray:
        """Transform anchor-local point to world coordinates."""
        pt_h = np.append(local_pt, 1.0)
        return (self.world_transform @ pt_h)[:3]


class SpatialAnchorManager:
    """
    Manages the full set of spatial anchors for a session.

    Anchors are created at tap time (Mode A) or at autonomous trigger
    points (Mode B). Each anchor holds the Gaussian IDs generated for
    that region so they can be rigidly updated if the anchor pose changes.
    """

    def __init__(self, max_anchors: int = 100,
                  drift_tolerance_m: float = 0.02):
        self._anchors: Dict[str, SpatialAnchor] = {}
        self._max_anchors = max_anchors
        self._drift_tol = drift_tolerance_m

    def create_anchor(self,
                       world_position: np.ndarray,
                       semantic: str,
                       bbox_min: np.ndarray,
                       bbox_max: np.ndarray,
                       confidence: float = 1.0) -> SpatialAnchor:
        """
        Create a new spatial anchor at the given world position.

        Args:
            world_position: (3,) center of the revealed region
            semantic:       object class label
            bbox_min/max:  region bounds in world space

        Returns:
            New SpatialAnchor
        """
        anchor_id = str(uuid.uuid4())[:8]

        # Build identity-rotation anchor transform
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = world_position

        anchor = SpatialAnchor(
            anchor_id=anchor_id,
            world_transform=transform,
            created_at=time.time(),
            semantic=semantic,
            bbox_min=bbox_min.copy(),
            bbox_max=bbox_max.copy(),
            confidence=confidence,
        )

        if len(self._anchors) >= self._max_anchors:
            # Evict oldest anchor
            oldest = min(self._anchors.values(), key=lambda a: a.created_at)
            del self._anchors[oldest.anchor_id]
            logger.debug(f"Evicted oldest anchor: {oldest.anchor_id}")

        self._anchors[anchor_id] = anchor
        logger.info(f"Created anchor {anchor_id} for {semantic} at {world_position.round(3)}")
        return anchor

    def update_anchor_pose(self,
                            anchor_id: str,
                            new_world_transform: np.ndarray) -> Optional[np.ndarray]:
        """
        Update anchor world transform (e.g. after ARKit relocalization).

        Returns delta transform so caller can update associated Gaussian positions.
        """
        if anchor_id not in self._anchors:
            return None

        anchor = self._anchors[anchor_id]
        old_transform = anchor.world_transform.copy()
        anchor.world_transform = new_world_transform.copy()

        # Compute delta
        delta = new_world_transform @ np.linalg.inv(old_transform)

        # Check drift magnitude
        drift = float(np.linalg.norm(new_world_transform[:3, 3] - old_transform[:3, 3]))
        if drift > self._drift_tol:
            logger.warning(
                f"Anchor {anchor_id}: drift={drift:.3f}m > tolerance={self._drift_tol}m"
            )

        return delta

    def get_anchor(self, anchor_id: str) -> Optional[SpatialAnchor]:
        return self._anchors.get(anchor_id)

    def get_anchors_in_radius(self,
                               center: np.ndarray,
                               radius_m: float) -> List[SpatialAnchor]:
        """Return all anchors within radius of a point."""
        result = []
        for anchor in self._anchors.values():
            dist = float(np.linalg.norm(anchor.position - center))
            if dist <= radius_m:
                result.append(anchor)
        return result

    def attach_gaussians(self,
                          anchor_id: str,
                          gaussian_indices: List[int]) -> None:
        """Record which Gaussian indices belong to this anchor."""
        if anchor_id in self._anchors:
            self._anchors[anchor_id].gaussian_ids.extend(gaussian_indices)

    def transform_gaussians_on_update(self,
                                       all_positions: np.ndarray,
                                       anchor_id: str,
                                       new_transform: np.ndarray) -> np.ndarray:
        """
        Rigidly move all Gaussians attached to an anchor when its pose changes.

        Args:
            all_positions: (N, 3) full Gaussian position array
            anchor_id:     which anchor's Gaussians to update
            new_transform: new anchor-to-world transform (4, 4)

        Returns:
            (N, 3) updated positions
        """
        delta = self.update_anchor_pose(anchor_id, new_transform)
        if delta is None:
            return all_positions

        anchor = self._anchors[anchor_id]
        updated = all_positions.copy()

        for idx in anchor.gaussian_ids:
            if 0 <= idx < len(updated):
                pt_h = np.append(updated[idx], 1.0)
                updated[idx] = (delta @ pt_h)[:3]

        return updated

    def mark_lost(self, anchor_id: str) -> None:
        """Mark an anchor as not tracked (ARKit lost tracking)."""
        if anchor_id in self._anchors:
            self._anchors[anchor_id].is_tracked = False
            logger.warning(f"Anchor {anchor_id} tracking LOST")

    def mark_recovered(self, anchor_id: str) -> None:
        if anchor_id in self._anchors:
            self._anchors[anchor_id].is_tracked = True
            logger.info(f"Anchor {anchor_id} tracking RECOVERED")

    def all_anchors(self) -> List[SpatialAnchor]:
        return list(self._anchors.values())

    def __len__(self) -> int:
        return len(self._anchors)
