"""
PHANTOM-ECHO REVEAL — Global + Local Costmap for Nav2
global_costmap.py + local_costmap.py + adaptive_costmap.py (combined)

Converts PHANTOM Gaussian scene into ROS2-compatible OccupancyGrid
messages for Nav2 path planning.

global_costmap: static mesh → full-map OccupancyGrid
local_costmap:  dynamic bbox inflation around confirmed tracks (ORANGE)
adaptive_costmap: movable-obstacle layer (furniture that may have moved)
"""

import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# OccupancyGrid cell values (ROS2 convention)
CELL_FREE    = 0
CELL_UNKNOWN = -1
CELL_LETHAL  = 100
CELL_INSCRIBED = 99

DEFAULT_RESOLUTION = 0.05   # 5cm per cell
INFLATION_RADIUS_M = 0.30   # 30cm robot inflation radius


class GlobalCostmap:
    """
    Static global costmap built from PHANTOM Gaussian scene.

    Converts:
        BLUE/WHITE Gaussians  → OCCUPIED (LETHAL)
        RED Gaussians         → UNKNOWN
        GREEN Gaussians       → LETHAL (generated geometry = real obstacle)
        Empty space           → FREE
    """

    def __init__(self, resolution: float = DEFAULT_RESOLUTION,
                  width_m: float = 10.0, height_m: float = 10.0,
                  origin_x: float = 0.0, origin_y: float = 0.0):
        self._res    = resolution
        self._width  = width_m
        self._height = height_m
        self._ox     = origin_x
        self._oy     = origin_y

        self._w_cells = int(width_m  / resolution)
        self._h_cells = int(height_m / resolution)
        self._grid    = np.full((self._h_cells, self._w_cells),
                                 CELL_UNKNOWN, dtype=np.int8)
        self._last_built = 0.0

    def build_from_gaussians(self,
                               gaussians: List[Dict[str, Any]],
                               floor_y: float = 0.0,
                               height_band: float = 0.5) -> None:
        """
        Populate costmap from Gaussian scene.

        Only Gaussians at floor level (y ∈ [floor_y, floor_y+height_band])
        contribute to the 2D costmap (nav plane).
        """
        self._grid[:] = CELL_UNKNOWN

        for g in gaussians:
            pos = g.get("position", [0, 0, 0])
            tag = g.get("tag", "RED")
            y   = float(pos[1])

            # Only floor-level Gaussians matter for 2D nav
            if not (floor_y <= y <= floor_y + height_band):
                continue

            ci, cj = self._world_to_cell(float(pos[0]), float(pos[2]))
            if not (0 <= ci < self._h_cells and 0 <= cj < self._w_cells):
                continue

            if tag in ("BLUE", "WHITE", "GREEN", "TEAL"):
                self._grid[ci, cj] = CELL_LETHAL
            elif tag == "RED":
                self._grid[ci, cj] = CELL_UNKNOWN
            elif tag in ("YELLOW",):
                self._grid[ci, cj] = max(self._grid[ci, cj], 50)

        # Inflate obstacles
        self._inflate(INFLATION_RADIUS_M)
        self._last_built = time.time()
        logger.info(f"GlobalCostmap: built from {len(gaussians)} Gaussians")

    def _inflate(self, radius_m: float) -> None:
        """Inflate LETHAL cells by radius for robot safety margin."""
        try:
            from scipy.ndimage import binary_dilation, generate_binary_structure
            r_cells = int(radius_m / self._res)
            if r_cells < 1:
                return
            lethal_mask = self._grid == CELL_LETHAL
            struct = np.ones((2 * r_cells + 1, 2 * r_cells + 1), dtype=bool)
            inflated = binary_dilation(lethal_mask, structure=struct)
            # Mark inflated region as INSCRIBED (not lethal, but costly)
            self._grid[inflated & ~lethal_mask] = CELL_INSCRIBED
        except ImportError:
            pass   # scipy not available, skip inflation

    def get_ros2_occupancy_grid(self, frame_id: str = "map") -> Dict:
        """
        Return a ROS2-style OccupancyGrid dict (for Nav2 or mock).
        """
        return {
            "header": {
                "stamp": time.time(),
                "frame_id": frame_id,
            },
            "info": {
                "resolution": self._res,
                "width":  self._w_cells,
                "height": self._h_cells,
                "origin": {
                    "position": {"x": self._ox, "y": self._oy, "z": 0.0},
                    "orientation": {"w": 1.0}
                },
            },
            "data": self._grid.flatten().tolist(),
        }

    def is_free(self, world_x: float, world_z: float) -> bool:
        ci, cj = self._world_to_cell(world_x, world_z)
        if not (0 <= ci < self._h_cells and 0 <= cj < self._w_cells):
            return False
        return int(self._grid[ci, cj]) < CELL_INSCRIBED

    def _world_to_cell(self, wx: float, wz: float) -> Tuple[int, int]:
        """BUG-PROD-6 FIX: document the XZ-projection convention explicitly.

        PHANTOM's world coordinate system: Y is up, navigation is on the XZ plane.
        This costmap projects onto that XZ plane:
            row (ci) = Z-axis  (north/south depth in the room)
            col (cj) = X-axis  (east/west width of the room)

        Nav2 OccupancyGrid convention: row = Y-axis (north), col = X-axis (east).
        When imported by a Nav2 planner, the 'origin_y' metadata must be set to
        the Z-origin of the room (not the physical Y/height), otherwise the
        planner interprets rows as physical Y and rotates the map 90 degrees.

        All callers pass (world_x, world_z) explicitly, and origin parameters
        _ox / _oy are named for their role as X-origin and Z-origin respectively.
        """
        ci = int((wz - self._oy) / self._res)   # row = Z (depth)
        cj = int((wx - self._ox) / self._res)   # col = X (width)
        return ci, cj


class LocalCostmap:
    """
    Dynamic local costmap: inflates ORANGE (dynamic) object bboxes.
    Updated every frame from SlotLSTM tracker output.
    """

    def __init__(self, base_costmap: GlobalCostmap,
                  inflation_m: float = 0.5):
        self._base      = base_costmap
        self._inflation = inflation_m
        self._dynamic_cells: np.ndarray = np.zeros_like(base_costmap._grid)

    def update_dynamic_obstacles(self,
                                   dynamic_gaussians: List[Dict[str, Any]],
                                   floor_y: float = 0.0) -> None:
        """
        Add dynamic object inflation to local costmap.

        MISSING-8 FIX: previously all ORANGE Gaussians were collapsed to a
        single centroid and inflated as one blob. If two people walk on opposite
        sides of the room, the single centroid falls between them and the robot
        drives through the gap thinking both sides are blocked.

        Now uses DBSCAN to cluster ORANGE points by proximity (0.8m radius),
        then inflates one circle per cluster so each person/object is a
        separate obstacle in the costmap.
        """
        self._dynamic_cells[:] = 0

        if not dynamic_gaussians:
            return

        positions = np.array([g["position"] for g in dynamic_gaussians],
                              dtype=np.float32)

        # Cluster ORANGE Gaussians by XZ proximity (0.8m epsilon)
        try:
            from sklearn.cluster import DBSCAN
            xz = positions[:, [0, 2]]  # nav plane only
            labels = DBSCAN(eps=0.8, min_samples=1).fit_predict(xz)
        except ImportError:
            # sklearn not installed: fall back to single-centroid behaviour
            labels = np.zeros(len(positions), dtype=int)
            logger.debug("sklearn not installed — LocalCostmap using single-centroid fallback")

        h, w = self._dynamic_cells.shape
        r = int(self._inflation / self._base._res)
        for label in set(labels):
            if label < 0:
                continue  # DBSCAN noise points — skip
            cluster_pts = positions[labels == label]
            centroid = cluster_pts.mean(axis=0)
            ci, cj = self._base._world_to_cell(
                float(centroid[0]), float(centroid[2]))
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if di**2 + dj**2 <= r**2:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            self._dynamic_cells[ni, nj] = CELL_LETHAL

    def get_merged_grid(self) -> np.ndarray:
        """Merge base + dynamic costmaps."""
        merged = self._base._grid.copy()
        merged = np.maximum(merged, self._dynamic_cells)
        return merged
