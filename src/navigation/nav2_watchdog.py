"""
PHANTOM-ECHO REVEAL — Nav2 Watchdog
nav2_watchdog.py

Monitors robot navigation progress. If no forward progress is made
in WATCHDOG_TIMEOUT_S (10s), transitions to fallback mode:
    1. Cancel current Nav2 goal
    2. Select new frontier via active_perception.py
    3. Send new goal
    4. Emit warning to presenter HUD

Flaw 38 fix: without watchdog, robot gets stuck on obstacles and
the demo silently stalls.
"""

import threading
import time
import logging
import numpy as np
from typing import Optional, Callable

from src.shared.gaussian_format import Constants

logger = logging.getLogger(__name__)

WATCHDOG_TIMEOUT  = Constants.WATCHDOG_TIMEOUT_S        # 10s
MIN_PROGRESS_M    = Constants.WATCHDOG_MIN_PROGRESS     # 0.15m


class Nav2Watchdog:
    """
    Monitors Nav2 navigation progress.

    Usage:
        watchdog = Nav2Watchdog(nav2_bridge, active_planner)
        watchdog.start()
        watchdog.update_position(robot_pos)   # call from Nav2 callback
        watchdog.stop()
    """

    def __init__(self,
                 nav2_bridge,
                 active_planner,
                 on_stall: Optional[Callable] = None,
                 timeout_s: float = WATCHDOG_TIMEOUT,
                 min_progress_m: float = MIN_PROGRESS_M):
        self._nav2      = nav2_bridge
        self._planner   = active_planner
        self._on_stall  = on_stall
        self._timeout   = timeout_s
        self._min_prog  = min_progress_m

        self._last_pos:   Optional[np.ndarray] = None
        self._last_prog:  float = time.time()
        self._goal_start: float = time.time()
        self._running:    bool  = False
        self._thread:     Optional[threading.Thread] = None
        self._stall_count: int = 0

    def start(self) -> None:
        self._running = True
        self._last_prog = time.time()
        self._thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="Nav2Watchdog"
        )
        self._thread.start()
        logger.info(f"Nav2Watchdog started (timeout={self._timeout}s)")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def update_position(self, position: np.ndarray) -> None:
        """Call this every time the robot's position is updated."""
        if self._last_pos is not None:
            dist = float(np.linalg.norm(position - self._last_pos))
            if dist >= self._min_prog:
                self._last_prog = time.time()
                self._stall_count = 0
        self._last_pos = position.copy()

    def reset_goal(self) -> None:
        """Reset watchdog when a new Nav2 goal is sent."""
        self._last_prog = time.time()
        self._goal_start = time.time()
        self._stall_count = 0
        logger.debug("Watchdog reset (new goal)")

    def time_since_progress(self) -> float:
        return time.time() - self._last_prog

    def is_stalled(self) -> bool:
        return self.time_since_progress() > self._timeout

    def _watchdog_loop(self) -> None:
        while self._running:
            time.sleep(1.0)
            if self.is_stalled():
                self._handle_stall()

    def _handle_stall(self) -> None:
        self._stall_count += 1
        logger.warning(
            f"Nav2Watchdog: STALL detected (no progress in {self._timeout}s, "
            f"count={self._stall_count})"
        )

        # Cancel current goal
        try:
            self._nav2.cancel_goal()
        except Exception as e:
            logger.debug(f"cancel_goal failed: {e}")

        # Select new frontier
        try:
            frontier = self._planner.compute_frontier()
            if frontier is not None:
                self._nav2.send_goal(frontier.tolist())
                logger.info(f"Watchdog recovery: new goal {frontier.round(2)}")
            else:
                logger.warning("Watchdog: no frontier available, robot stopped")
        except Exception as e:
            logger.error(f"Watchdog recovery failed: {e}")

        # Notify presenter
        if self._on_stall:
            try:
                self._on_stall(self._stall_count)
            except Exception:
                pass

        # Reset timer
        self._last_prog = time.time()
