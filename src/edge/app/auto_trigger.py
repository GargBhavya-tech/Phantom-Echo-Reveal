"""
PHANTOM-ECHO REVEAL — Mode B: Autonomous Nav2 Trigger
auto_trigger.py

Mode B runs autonomously during robot navigation.
Nav2 triggers a reveal whenever:
    1. A RED voxel region appears in the local costmap ahead of the robot
    2. The region is large enough (> MIN_VOXEL_COUNT voxels)
    3. No reveal is currently in flight for that region
    4. The 10s watchdog has not fired

Full pipeline:
    Nav2 goal callback → detect frontier RED region
    → auto_trigger fires → same reveal pipeline as Mode A
    → Gaussians inserted into map for Nav2 costmap update

Flaw 41 fix: Mode B was calling reveal synchronously inside the Nav2
callback, blocking the navigation stack. Now async with queue.
"""

import asyncio
import numpy as np
import time
import logging
import threading
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field

from src.edge.app.tap_handler import TapHandler
from src.navigation.occupancy_grid import OccupancyGrid
from src.shared.gaussian_format import TAG_RED, Constants

logger = logging.getLogger(__name__)

MIN_VOXEL_COUNT   = 50       # min RED voxels to trigger a reveal
COOLDOWN_S        = 5.0      # minimum seconds between auto reveals
WATCHDOG_TIMEOUT  = Constants.WATCHDOG_TIMEOUT_S   # 10s


@dataclass
class AutoTriggerEvent:
    """One autonomous trigger event."""
    region_center: np.ndarray
    bbox_min:      np.ndarray
    bbox_max:      np.ndarray
    voxel_count:   int
    semantic:      str = "UNKNOWN"
    timestamp:     float = field(default_factory=time.time)
    processed:     bool = False


class AutoTrigger:
    """
    Mode B autonomous reveal trigger.

    Monitors the occupancy grid for large RED regions ahead of the robot
    and fires reveal requests via the TapHandler pipeline.

    Usage:
        trigger = AutoTrigger(tap_handler, occ_grid, on_gaussians_callback)
        trigger.start()
        # Nav2 calls trigger.notify_goal_reached(position) as robot moves
        trigger.stop()
    """

    def __init__(self,
                 tap_handler: TapHandler,
                 occ_grid: OccupancyGrid,
                 on_gaussians: Optional[Callable[[List[Dict]], None]] = None,
                 min_voxels: int = MIN_VOXEL_COUNT,
                 cooldown_s: float = COOLDOWN_S):
        self._tap      = tap_handler
        self._grid     = occ_grid
        self._callback = on_gaussians
        self._min_voxels = min_voxels
        self._cooldown   = cooldown_s

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_trigger: Dict[str, float] = {}   # region_key → last trigger time
        self._watchdog_last_progress = time.time()
        self._all_gaussians: List[Dict] = []

    def start(self) -> None:
        """Start the autonomous trigger background thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="AutoTrigger"
        )
        self._thread.start()
        logger.info("AutoTrigger started (Mode B)")

    def stop(self) -> None:
        """Gracefully stop the trigger."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("AutoTrigger stopped")

    def notify_nav2_position(self,
                              robot_position: np.ndarray,
                              look_ahead_m: float = 1.5) -> None:
        """
        Called by Nav2 bridge each time the robot moves.
        Scans ahead for large RED regions.

        Flaw 41 fix: puts event on queue instead of blocking call.
        """
        if not self._running:
            return

        # Update watchdog
        self._watchdog_last_progress = time.time()

        # Scan ahead for RED regions
        events = self._detect_red_regions(robot_position, look_ahead_m)

        for event in events:
            region_key = self._region_key(event.bbox_min, event.bbox_max)
            last = self._last_trigger.get(region_key, 0.0)

            if time.time() - last < self._cooldown:
                continue   # cooldown not expired

            if not self._queue.full():
                try:
                    self._queue.put_nowait(event)
                    self._last_trigger[region_key] = time.time()
                    logger.info(
                        f"AutoTrigger queued: {event.voxel_count} RED voxels "
                        f"at {event.region_center.round(2)}"
                    )
                except Exception:
                    pass

    def notify_goal_reached(self, goal_position: np.ndarray) -> None:
        """Called when Nav2 reaches a navigation goal."""
        self.notify_nav2_position(goal_position, look_ahead_m=2.0)

    def set_gaussians(self, gaussians: List[Dict]) -> None:
        """Update reference to current scene Gaussians."""
        self._all_gaussians = gaussians

    # ── Background processing loop ─────────────────────────────────────────
    def _run_loop(self) -> None:
        """Background thread: processes trigger events asynchronously."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._process_queue())

    async def _process_queue(self) -> None:
        while self._running:
            # Watchdog check
            if time.time() - self._watchdog_last_progress > WATCHDOG_TIMEOUT:
                logger.warning(
                    f"AutoTrigger watchdog: no Nav2 progress in "
                    f"{WATCHDOG_TIMEOUT}s — pausing reveals"
                )
                await asyncio.sleep(2.0)
                continue

            try:
                event = self._queue.get_nowait()
            except Exception:
                await asyncio.sleep(0.1)
                continue

            await self._process_event(event)

    async def _process_event(self, event: AutoTriggerEvent) -> None:
        """Run the reveal pipeline for one auto-trigger event."""
        logger.info(f"AutoTrigger processing: {event.semantic} @ {event.region_center.round(2)}")

        try:
            # Synthesize a "virtual tap" at the region center
            # We don't have a real depth frame, so use region center directly
            new_gaussians, anchor = await self._reveal_from_bbox(
                event.region_center,
                event.bbox_min,
                event.bbox_max,
                event.semantic,
            )

            if new_gaussians:
                self._all_gaussians.extend(new_gaussians)
                if self._callback:
                    self._callback(new_gaussians)
                logger.info(
                    f"AutoTrigger inserted {len(new_gaussians)} Gaussians "
                    f"for {event.semantic}"
                )

        except Exception as e:
            logger.error(f"AutoTrigger process_event failed: {e}")

    async def _reveal_from_bbox(self,
                                  center: np.ndarray,
                                  bbox_min: np.ndarray,
                                  bbox_max: np.ndarray,
                                  semantic: str,
                                  ) -> Tuple[List[Dict], Any]:
        """Fire the reveal pipeline for a known bbox (Mode B variant)."""
        from src.edge.network.payload_builder import build_reveal_payload
        from src.edge.network.gaussian_decoder import decode_reveal_response, filter_duplicates
        from src.shared.gaussian_format import TAG_RED

        payload = build_reveal_payload(
            session_id=self._tap._session,
            region_id=f"auto_{int(time.time())}",
            semantic=semantic,
            confidence_tag=TAG_RED,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            floor_y=self._tap._floor_y,
            ceiling_y=self._tap._ceiling_y,
        )

        payload_dict = {
            "session_id":    payload.session_id,
            "region_id":     payload.region_id,
            "semantic":      payload.semantic,
            "confidence_tag": payload.confidence_tag,
            "bbox_min":      payload.bbox_min,
            "bbox_max":      payload.bbox_max,
            "floor_y":       payload.floor_y,
            "ceiling_y":     payload.ceiling_y,
        }

        compressed = await self._tap._client.reveal(payload_dict)
        if compressed is None:
            compressed = self._tap._synthetic_fallback(semantic, bbox_min, bbox_max)

        new_gaussians = decode_reveal_response(compressed)
        new_gaussians = filter_duplicates(self._all_gaussians, new_gaussians)

        anchor = self._tap._anchors.create_anchor(center, semantic, bbox_min, bbox_max)
        return new_gaussians, anchor

    # ── RED region detection ───────────────────────────────────────────────
    def _detect_red_regions(self,
                              robot_pos: np.ndarray,
                              look_ahead_m: float
                              ) -> List[AutoTriggerEvent]:
        """
        Scan the occupancy grid for large RED regions ahead of the robot.
        Returns list of trigger events sorted by size (largest first).
        """
        try:
            # Get unknown cells within look-ahead radius
            unknown_cells = self._grid.get_unknown_cells_in_radius(
                robot_pos, look_ahead_m
            )
            if not unknown_cells or len(unknown_cells) < self._min_voxels:
                return []

            # Cluster unknown cells into connected regions
            events = []
            positions = np.array(unknown_cells, dtype=np.float32)

            if len(positions) >= self._min_voxels:
                center = positions.mean(axis=0)
                bbox_min = positions.min(axis=0)
                bbox_max = positions.max(axis=0)
                events.append(AutoTriggerEvent(
                    region_center=center,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    voxel_count=len(positions),
                    semantic=self._guess_semantic(center),
                ))

            return sorted(events, key=lambda e: e.voxel_count, reverse=True)[:3]

        except Exception as e:
            logger.debug(f"RED region detection failed: {e}")
            return []

    def _guess_semantic(self, position: np.ndarray) -> str:
        """
        Guess semantic label from position heuristics.
        Wall-height → PAINTING/SHELF; floor-height → SOFA/CHAIR.
        """
        y = float(position[1])
        if y < 0.5:
            return "PLANT"
        elif y < 1.0:
            return "SOFA"
        elif y < 1.5:
            return "CHAIR"
        elif y < 2.0:
            return "SHELF"
        else:
            return "PAINTING"

    @staticmethod
    def _region_key(bbox_min: np.ndarray, bbox_max: np.ndarray) -> str:
        """Stable string key for a region (for cooldown tracking)."""
        center = ((bbox_min + bbox_max) / 2).round(1)
        return f"{center[0]},{center[1]},{center[2]}"
