"""
PHANTOM-ECHO REVEAL — Nav2 Costmap Publisher
nav2_publisher.py

MISS-3 FIX: global_costmap.py + nav2_bridge.py existed but were never called
from any pipeline. This module is the single entry point that main.py and
main_v2.py call after every Layer 4 update to publish the costmap.

ROS2 available  → publishes /phantom_echo/global_costmap (nav_msgs/OccupancyGrid)
ROS2 not available → writes costmap.npy to output_dir (Gazebo / offline use)
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Any

logger = logging.getLogger(__name__)

# ROS2 state (lazy init)
_ros2_ok  = False
_node     = None
_pub_glob = None

# Log-odds sensor model — matches occupancy_grid.py LOG_ODDS_SENSOR
_LOG_ODDS = {
    "WHITE": 3.0, "BLUE": 2.0, "TEAL": 1.8,
    "GREEN": 0.7, "YELLOW": 1.2, "ORANGE": 0.5, "RED": 0.0,
}


def _try_init_ros2(node_name: str = "phantom_echo_nav") -> bool:
    global _ros2_ok, _node, _pub_glob
    if _ros2_ok:
        return True
    try:
        import rclpy
        from rclpy.node import Node
        from nav_msgs.msg import OccupancyGrid
        if not rclpy.ok():
            rclpy.init()
        _node     = Node(node_name)
        _pub_glob = _node.create_publisher(OccupancyGrid, "/phantom_echo/global_costmap", 10)
        _ros2_ok  = True
        logger.info("ROS2 init OK — publishing /phantom_echo/global_costmap")
    except Exception as e:
        logger.info(f"ROS2 not available ({e}) — file fallback mode")
    return _ros2_ok


def _build_2d_slice(gaussians: List[Any],
                    resolution: float = 0.05,
                    width_m: float = 10.0,
                    height_m: float = 10.0,
                    slice_y: float = 0.5,
                    band_m: float = 0.4) -> np.ndarray:
    """Project Gaussians to 2D occupancy grid at height slice_y ± band_m."""
    W = int(width_m  / resolution)
    H = int(height_m / resolution)
    lo = np.zeros((H, W), dtype=np.float32)

    for g in gaussians:
        pos = g.get("position") if isinstance(g, dict) else getattr(g, "position", None)
        tag = g.get("tag")      if isinstance(g, dict) else getattr(g, "tag", "BLUE")
        if pos is None:
            continue
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        if abs(py - slice_y) > band_m:
            continue
        cx = int(px / resolution)
        cz = int(pz / resolution)
        if 0 <= cz < H and 0 <= cx < W:
            lo[cz, cx] += _LOG_ODDS.get(str(tag).upper(), 0.5)

    # NAV-SAFETY FIX: the naive sigmoid e^x/(1+e^x) overflows float32 once a
    # cell accumulates enough log-odds (lo > ~88) — exactly the densest, most
    # CERTAIN obstacle cells (thick walls, dense floor infill). exp(lo)->inf,
    # then inf/(1+inf)=NaN, and every NaN threshold compare below is False, so
    # the cell silently stays UNKNOWN (-1) instead of LETHAL (100). A Nav2
    # planner could then route the robot straight through a confirmed wall.
    # Use the numerically-stable logistic form (clip the exponent) so the most
    # certain cells correctly saturate to prob≈1.0 → LETHAL.
    prob = 1.0 / (1.0 + np.exp(-np.clip(lo, -60.0, 60.0)))
    assert np.isfinite(prob).all(), "costmap probability contains NaN/inf"
    grid = np.full((H, W), -1, dtype=np.int8)
    grid[prob > 0.70] = 100
    grid[(prob >= 0.30) & (prob <= 0.70)] = 50
    grid[prob < 0.30] = 0
    return grid


def _publish_ros2(grid: np.ndarray, resolution: float,
                   width_m: float, height_m: float) -> None:
    try:
        from nav_msgs.msg import OccupancyGrid as OG
        msg = OG()
        msg.header.frame_id      = "map"
        msg.header.stamp         = _node.get_clock().now().to_msg()
        msg.info.resolution      = float(resolution)
        msg.info.width           = int(width_m  / resolution)
        msg.info.height          = int(height_m / resolution)
        msg.info.origin.orientation.w = 1.0
        msg.data                 = grid.flatten().tolist()
        _pub_glob.publish(msg)
        logger.info(f"ROS2 costmap published: {msg.info.width}×{msg.info.height}")
    except Exception as e:
        logger.warning(f"ROS2 publish failed: {e}")


def _save_to_file(grid: np.ndarray, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(output_dir) / "costmap.npy")
    np.save(path, grid)
    pct = float(np.sum(grid == 100)) / max(grid.size, 1) * 100
    logger.info(f"Costmap saved → {path} ({pct:.1f}% occupied)")


def publish_or_save_costmap(gaussians:  List[Any],
                             floor_y:    float = 0.0,
                             output_dir: str   = "output",
                             resolution: float = 0.05,
                             width_m:    float = 10.0,
                             height_m:   float = 10.0) -> np.ndarray:
    """
    MISS-3 FIX: publish occupancy costmap after every Layer 4 update.
    Called from run_layer_4_output() in main.py.

    1. Build 2D occupancy slice at waist height (floor_y + 0.5m).
    2. Publish to ROS2 if available; save to costmap.npy otherwise.
    3. Return (H, W) int8 grid for inspection / testing.
    """
    g_list = []
    for g in gaussians:
        if isinstance(g, dict):
            g_list.append(g)
        else:
            g_list.append({
                "position": g.position.tolist() if hasattr(g.position, "tolist") else list(g.position),
                "tag":      getattr(g, "tag", "BLUE"),
            })

    slice_y = floor_y + 0.5
    grid    = _build_2d_slice(g_list, resolution=resolution,
                               width_m=width_m, height_m=height_m, slice_y=slice_y)

    if _try_init_ros2():
        _publish_ros2(grid, resolution, width_m, height_m)
    else:
        _save_to_file(grid, output_dir)

    return grid


# ── BUG-EC-COSTMAP FIX: auto-size costmap from actual Gaussian extents ─────

def publish_or_save_costmap_autosized(gaussians: List[Any],
                                       floor_y: float = 0.0,
                                       output_dir: str = "output",
                                       resolution: float = 0.05,
                                       margin_m: float = 0.5) -> np.ndarray:
    """
    BUG-EC-COSTMAP FIX: original hardcoded width_m=10, height_m=10.
    Gaussians outside 10m boundary were silently dropped.
    This version auto-computes map extent from actual Gaussian positions
    + margin_m padding, then delegates to publish_or_save_costmap().

    Use this function as the default call from main_v2.py Layer 4.
    """
    positions = []
    for g in gaussians:
        pos = g.get("position") if isinstance(g, dict) else getattr(g, "position", None)
        if pos is not None:
            positions.append(pos)

    if not positions:
        logger.warning("No Gaussians — using default 10x10m costmap")
        return publish_or_save_costmap(gaussians, floor_y, output_dir, resolution)

    pts = np.array(positions, dtype=np.float32)
    x_range = float(pts[:, 0].max() - pts[:, 0].min()) + 2 * margin_m
    z_range = float(pts[:, 2].max() - pts[:, 2].min()) + 2 * margin_m

    # Round up to nearest 0.5m
    width_m  = max(5.0, round(x_range * 2) / 2)
    height_m = max(5.0, round(z_range * 2) / 2)

    logger.info(f"Auto-sized costmap: {width_m:.1f}m × {height_m:.1f}m "
                f"(from {len(positions)} Gaussians)")

    return publish_or_save_costmap(gaussians, floor_y, output_dir,
                                    resolution, width_m, height_m)
