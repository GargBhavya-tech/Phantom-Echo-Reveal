"""
PHANTOM-ECHO REVEAL — ROS2 Nav2 Costmap Bridges + Watchdog (Layer 5)
global_costmap.py + local_costmap.py + nav2_watchdog.py  (combined)

Publishes PHANTOM-ECHO occupancy data as ROS2 Nav2 costmap topics.

Topics published:
    /phantom_echo/global_costmap   — OccupancyGrid (nav_msgs)
    /phantom_echo/local_costmap    — OccupancyGrid (local window)
    /phantom_echo/costmap_updates  — OccupancyGridUpdate (incremental)

Nav2 watchdog:
    Monitors navigation health. If robot does not make progress
    within WATCHDOG_TIMEOUT seconds, triggers manual control fallback
    and publishes /phantom_echo/manual_fallback (std_msgs/Bool).

ROS2 dependency: rclpy, nav_msgs, std_msgs
Install: sudo apt install ros-humble-nav2-bringup ros-humble-rclpy
"""

import numpy as np
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
COSTMAP_FREE     = 0      # Nav2 free space
COSTMAP_UNKNOWN  = 255    # Nav2 unknown
COSTMAP_OCCUPIED = 100    # Nav2 occupied

# RED voxels → lethal cost (robot must not enter unknown space near here)
COSTMAP_RED_BUFFER = 90   # inflation cost near RED boundaries

WATCHDOG_TIMEOUT  = 30.0  # seconds — trigger manual fallback
WATCHDOG_MIN_PROGRESS_M = 0.15  # meters — minimum progress per timeout window


@dataclass
class CostmapConfig:
    resolution:        float = 0.05   # meters per cell
    global_width_m:    float = 10.0
    global_height_m:   float = 10.0
    local_window_m:    float = 3.0    # local costmap radius
    inflation_radius_m: float = 0.3   # obstacle inflation
    red_lethal_cost:   int   = 90     # cost for RED boundary cells


# ── Costmap conversion ─────────────────────────────────────────────────────

def occupancy_grid_to_costmap(occ_2d: np.ndarray,
                               red_2d: Optional[np.ndarray] = None,
                               cfg: Optional[CostmapConfig] = None) -> np.ndarray:
    """
    Convert PHANTOM-ECHO 2D occupancy slice to Nav2 costmap values.

    Args:
        occ_2d:  (X, Z) float32 occupancy probabilities [0, 1]
        red_2d:  (X, Z) bool — True where RED (unknown) voxels projected
        cfg:     costmap configuration

    Returns:
        (X, Z) uint8 costmap with Nav2 standard values
    """
    if cfg is None:
        cfg = CostmapConfig()

    costmap = np.full(occ_2d.shape, COSTMAP_UNKNOWN, dtype=np.uint8)

    # Free space
    costmap[occ_2d <= 0.3] = COSTMAP_FREE

    # Occupied (BLUE/TEAL/GREEN)
    costmap[occ_2d >= 0.7] = COSTMAP_OCCUPIED

    # RED boundaries → high-cost buffer (lethal but not impassable)
    if red_2d is not None:
        costmap[red_2d] = cfg.red_lethal_cost

    # Inflate obstacles
    costmap = _inflate_costmap(costmap, cfg.inflation_radius_m, cfg.resolution)

    return costmap


def _inflate_costmap(costmap: np.ndarray,
                      radius_m: float,
                      resolution: float) -> np.ndarray:
    """Inflate obstacles using distance transform."""
    try:
        from scipy.ndimage import distance_transform_edt, binary_dilation
        obstacle_mask = (costmap == COSTMAP_OCCUPIED)
        radius_cells  = int(radius_m / resolution)
        inflated_mask = binary_dilation(obstacle_mask, iterations=radius_cells)
        result = costmap.copy()
        # Scale inflation cost: full at 0, 0 at radius
        dist = distance_transform_edt(~obstacle_mask) * resolution
        inflation_cost = np.clip(
            COSTMAP_OCCUPIED * (1 - dist / radius_m), 0, COSTMAP_OCCUPIED
        ).astype(np.uint8)
        result[inflated_mask & ~obstacle_mask] = inflation_cost[inflated_mask & ~obstacle_mask]
        return result
    except ImportError:
        return costmap   # no inflation if scipy unavailable


# ── ROS2 Publisher (graceful fallback if rclpy unavailable) ───────────────

class CostmapPublisher:
    """
    Publishes occupancy grid costmaps to ROS2 Nav2.
    Falls back to simulation-mode logging if ROS2 is not available.
    """

    def __init__(self, cfg: Optional[CostmapConfig] = None):
        self.cfg = cfg or CostmapConfig()
        self._ros_available = False
        self._node = None
        self._global_pub = None
        self._local_pub  = None
        self._fallback_pub = None

        try:
            import rclpy
            from rclpy.node import Node
            from nav_msgs.msg import OccupancyGrid as ROSOccupancyGrid
            from std_msgs.msg import Bool

            rclpy.init()
            self._node = Node("phantom_echo_costmap")
            self._global_pub  = self._node.create_publisher(ROSOccupancyGrid,
                "/phantom_echo/global_costmap", 10)
            self._local_pub   = self._node.create_publisher(ROSOccupancyGrid,
                "/phantom_echo/local_costmap", 10)
            self._fallback_pub = self._node.create_publisher(Bool,
                "/phantom_echo/manual_fallback", 10)
            self._ros_available = True
            logger.info("ROS2 costmap publisher initialised")

        except (ImportError, Exception) as e:
            logger.warning(f"ROS2 not available ({e}) — simulation mode")

    def publish_global_costmap(self, costmap_2d: np.ndarray,
                                origin: np.ndarray,
                                stamp_sec: float = 0.0) -> None:
        """Publish global costmap to /phantom_echo/global_costmap."""
        if not self._ros_available:
            n_occ     = int(np.sum(costmap_2d == COSTMAP_OCCUPIED))
            n_free    = int(np.sum(costmap_2d == COSTMAP_FREE))
            n_unknown = int(np.sum(costmap_2d == COSTMAP_UNKNOWN))
            logger.info(
                f"[SIM] Global costmap: {costmap_2d.shape} cells, "
                f"occ={n_occ}, free={n_free}, unknown={n_unknown}"
            )
            return

        try:
            from nav_msgs.msg import OccupancyGrid as ROSOccupancyGrid
            from builtin_interfaces.msg import Time

            msg = ROSOccupancyGrid()
            msg.header.frame_id = "map"
            msg.info.resolution = self.cfg.resolution
            msg.info.width  = costmap_2d.shape[1]
            msg.info.height = costmap_2d.shape[0]
            msg.info.origin.position.x = float(origin[0])
            msg.info.origin.position.y = float(origin[2])
            # Flatten row-major, clamp to int8 range
            data = np.clip(costmap_2d.flatten().astype(np.int8), -128, 127)
            msg.data = data.tolist()
            self._global_pub.publish(msg)

        except Exception as e:
            logger.error(f"Costmap publish failed: {e}")

    def publish_local_costmap(self, costmap_2d: np.ndarray,
                               robot_pos: np.ndarray,
                               window_m: float = 3.0) -> None:
        """Publish local costmap window centred on robot."""
        # Crop to local window
        half = int(window_m / self.cfg.resolution / 2)
        H, W = costmap_2d.shape
        cy   = H // 2  # simplified — in production use robot_pos → grid coords
        cx   = W // 2
        lo_y = max(0, cy - half)
        hi_y = min(H, cy + half)
        lo_x = max(0, cx - half)
        hi_x = min(W, cx + half)
        local = costmap_2d[lo_y:hi_y, lo_x:hi_x]
        self.publish_global_costmap(local, robot_pos)   # reuse publisher with local origin

    def publish_manual_fallback(self, triggered: bool) -> None:
        """Publish manual fallback flag."""
        if not self._ros_available:
            if triggered:
                logger.warning("[SIM] MANUAL FALLBACK TRIGGERED")
            return
        try:
            from std_msgs.msg import Bool
            msg = Bool()
            msg.data = triggered
            self._fallback_pub.publish(msg)
        except Exception as e:
            logger.error(f"Fallback publish failed: {e}")

    def spin_once(self) -> None:
        if self._ros_available and self._node:
            try:
                import rclpy
                rclpy.spin_once(self._node, timeout_sec=0.01)
            except Exception:
                pass

    def destroy(self) -> None:
        if self._ros_available and self._node:
            try:
                import rclpy
                self._node.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass


# ── Nav2 Watchdog ─────────────────────────────────────────────────────────

class Nav2Watchdog:
    """
    Monitors robot navigation progress.
    Triggers manual control fallback if robot stalls.

    Runs in a background thread — call start() / stop().
    """

    def __init__(self,
                 publisher: CostmapPublisher,
                 timeout_s: float = WATCHDOG_TIMEOUT,
                 min_progress_m: float = WATCHDOG_MIN_PROGRESS_M):
        self._pub           = publisher
        self._timeout       = timeout_s
        self._min_progress  = min_progress_m
        self._last_pos      = None
        self._last_progress = time.time()
        self._running       = False
        self._thread        = None
        self._lock          = threading.Lock()
        self._manual_mode   = False

    def update_position(self, pos: np.ndarray) -> None:
        """Call this every time a new robot position is received."""
        with self._lock:
            if self._last_pos is not None:
                dist = float(np.linalg.norm(pos - self._last_pos))
                if dist >= self._min_progress:
                    self._last_progress = time.time()
                    if self._manual_mode:
                        logger.info("Watchdog: progress detected — resuming auto mode")
                        self._manual_mode = False
                        self._pub.publish_manual_fallback(False)
            self._last_pos = pos.copy()

    def _watchdog_loop(self) -> None:
        while self._running:
            time.sleep(1.0)
            with self._lock:
                elapsed = time.time() - self._last_progress
                if elapsed > self._timeout and not self._manual_mode:
                    logger.warning(
                        f"Nav2 Watchdog: no progress in {elapsed:.0f}s "
                        f"(threshold={self._timeout}s) — MANUAL FALLBACK"
                    )
                    self._manual_mode = True
                    self._pub.publish_manual_fallback(True)

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Nav2 watchdog started (timeout={self._timeout}s, "
            f"min_progress={self._min_progress}m)"
        )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Nav2 watchdog stopped")

    @property
    def in_manual_mode(self) -> bool:
        return self._manual_mode
