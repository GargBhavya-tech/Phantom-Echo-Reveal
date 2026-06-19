"""
PHANTOM-ECHO REVEAL — Occupancy Grid (Layer 5 Navigation)
occupancy_grid.py

Projects Gaussians into a 3D occupancy grid using Binary Bayes Filter
with log-odds representation.

Log-odds update rule:
    l(x|z) = l(x|z_prev) + log[ P(z|occ)/P(z|free) ]
            = l_prev + l_sensor

FIXED BUGS:
    Bug 2  — YELLOW tag silently dropped: LOG_ODDS_SENSOR now includes
             YELLOW (structurally observed ARKit geometry) and WHITE
             (high-confidence ARKit measurement) with appropriate weights.
    Bug 6  — ORANGE tag silently dropped: dynamic objects now register
             as temporarily occupied so the costmap is not
             pathologically clear around moving people/robots.
    Missing 1 — WHITE tag was also absent (same root cause as Bug 2/6).

Sensor model (per Gaussian tag):
    WHITE:  ARKit high-conf direct measurement → l_sensor = +3.0
    BLUE:   Physics proven                     → l_sensor = +2.0
    TEAL:   Acoustic measured                  → l_sensor = +1.8
    YELLOW: Physically probable (observed geo) → l_sensor = +1.2  ← FIXED
    GREEN:  AI generated + corrected           → l_sensor = +0.7
    ORANGE: Dynamic object (person/robot)      → l_sensor = +0.5  ← FIXED
    RED:    Unknown                            → no update (preserves prior)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# ── FIX Bug 2, Bug 6, Missing 1 ──────────────────────────────────────────
# Every tag defined in gaussian_format.py MUST appear here.
# YELLOW was the silent killer — ~40% of all ARKit geometry was invisible
# to the occupancy grid. ORANGE was a safety hazard (zero cost through
# tracked moving people). WHITE was also absent.
#
# Log-odds values match the bible's confidence hierarchy:
#   WHITE/BLUE (proven)  → high positive
#   TEAL (measured)      → near-high positive
#   YELLOW (probable)    → moderate positive   ← was 0.0 (same as RED)
#   GREEN (generated)    → low positive
#   ORANGE (dynamic)     → low positive, but decays on next frame ← was 0.0
#   RED (unknown)        → 0.0 (no update — unknown stays unknown)
# BUG-A FIX (v27): this table previously DIVERGED from gaussian_format.TAG_LOG_ODDS
# on BLUE/TEAL/GREEN/YELLOW (e.g. GREEN 0.7 here vs 1.5 there → 66.8% vs 81.8%
# occupancy), so the navigation costmap weighted generated geometry far less
# blocking than the rest of the system believed. There must be ONE source of
# truth. Import it. The keys match (tags are plain strings: TAG_BLUE == "BLUE").
from src.shared.gaussian_format import TAG_LOG_ODDS as LOG_ODDS_SENSOR

LOG_ODDS_FREE      = -0.5   # Ray cast through free space
LOG_ODDS_MIN       = -5.0   # Clamp floor (always-free saturation)
LOG_ODDS_MAX       = +5.0   # Clamp ceiling (always-occupied saturation)
LOG_ODDS_PRIOR     = 0.0    # Uniform prior (50% probability)

# Dynamic objects decay every frame — occupancy reverts toward unknown
# so the map stays accurate as objects move.
LOG_ODDS_ORANGE_DECAY = -0.3   # per frame without re-observation


@dataclass
class OccupancyGrid:
    """
    3D occupancy grid with log-odds representation.
    Slice at y=robot_height for 2D navigation costmap.
    """
    origin:     np.ndarray            # (3,) world position of voxel (0,0,0) [meters]
    voxel_size: float                  # meters per voxel (e.g. 0.05 = 5cm)
    shape:      Tuple[int, int, int]   # (X, Y, Z) grid dimensions

    log_odds:    np.ndarray = field(init=False)    # (X, Y, Z) float32
    visit_count: np.ndarray = field(init=False)    # (X, Y, Z) int32
    dynamic_mask: np.ndarray = field(init=False)   # (X, Y, Z) bool — tracks ORANGE voxels

    def __post_init__(self):
        self.log_odds     = np.full(self.shape, LOG_ODDS_PRIOR, dtype=np.float32)
        self.visit_count  = np.zeros(self.shape, dtype=np.int32)
        self.dynamic_mask = np.zeros(self.shape, dtype=bool)

    def world_to_voxel(self, world_pt: np.ndarray) -> np.ndarray:
        return np.floor((world_pt - self.origin) / self.voxel_size).astype(int)

    def voxel_to_world(self, voxel: np.ndarray) -> np.ndarray:
        return self.origin + (voxel + 0.5) * self.voxel_size

    def in_bounds(self, vox: np.ndarray) -> bool:
        return all(0 <= vox[i] < self.shape[i] for i in range(3))

    def probability(self) -> np.ndarray:
        """Convert log-odds to probability. Returns (X, Y, Z) float32."""
        return (1.0 - 1.0 / (1.0 + np.exp(self.log_odds))).astype(np.float32)

    def occupied_mask(self, threshold: float = 0.7) -> np.ndarray:
        """Boolean mask of occupied voxels."""
        return self.probability() >= threshold

    def free_mask(self, threshold: float = 0.3) -> np.ndarray:
        """Boolean mask of free voxels."""
        return self.probability() <= threshold

    def decay_dynamic_voxels(self) -> None:
        """
        Decay ORANGE voxels toward unknown each frame.

        Dynamic objects move — if we don't re-observe them, their last
        known position should revert toward unknown (log-odds=0) rather
        than staying marked occupied forever.  This prevents ghost
        obstacles in the costmap after a person has walked away.
        """
        if not np.any(self.dynamic_mask):
            return
        ix, iy, iz = np.where(self.dynamic_mask)
        self.log_odds[ix, iy, iz] = np.clip(
            self.log_odds[ix, iy, iz] + LOG_ODDS_ORANGE_DECAY,
            LOG_ODDS_MIN, LOG_ODDS_MAX
        )
        # Voxels that have decayed back near prior are no longer "dynamic"
        reverted = self.log_odds[ix, iy, iz] <= LOG_ODDS_PRIOR + 0.1
        self.dynamic_mask[ix[reverted], iy[reverted], iz[reverted]] = False

    def slice_2d(self, y_world: float, half_height_m: float = 0.3) -> np.ndarray:
        """
        Extract 2D occupancy slice at robot navigation height.
        Returns (X, Z) probability map.
        """
        y_vox_center = int((y_world - self.origin[1]) / self.voxel_size)
        y_half = max(1, int(half_height_m / self.voxel_size))
        y_lo = max(0, y_vox_center - y_half)
        y_hi = min(self.shape[1], y_vox_center + y_half + 1)
        return self.probability()[:, y_lo:y_hi, :].max(axis=1)

    def get_unknown_cells_in_radius(
        self,
        robot_pos: np.ndarray,
        radius_m: float,
        unknown_lo: float = 0.35,
        unknown_hi: float = 0.65,
    ) -> List[np.ndarray]:
        """Return world positions of UNKNOWN voxels within `radius_m` of robot_pos.

        UNKNOWN = occupancy probability in (unknown_lo, unknown_hi). Called by
        auto_trigger.py for Mode-B exploration target selection.

        Why the voxel-AABB bound: this is a *local* query, but a naive full-grid
        meshgrid allocates three X*Y*Z coordinate arrays + a distance array
        regardless of `radius_m` — O(grid) memory and time even for a 0.8 m
        search (a 5x2.5x4 m room at 5 cm is ~400k voxels). We restrict the scan
        to the integer voxel box covering the query sphere (O(radius^3)). The
        returned voxel set — and its lexicographic (C-order) ordering — is
        identical to the old full-grid scan, because every in-radius voxel lies
        inside that box; only the wasted work outside it is removed.
        """
        prob = self.probability()
        r_vox = int(np.ceil(radius_m / self.voxel_size))
        center = self.world_to_voxel(np.asarray(robot_pos, dtype=np.float64))
        lo = np.maximum(center - r_vox, 0)
        hi = np.minimum(center + r_vox + 1, np.array(self.shape))
        if np.any(hi <= lo):                       # query box entirely off-grid
            return []

        xi = np.arange(lo[0], hi[0], dtype=np.float32)
        yi = np.arange(lo[1], hi[1], dtype=np.float32)
        zi = np.arange(lo[2], hi[2], dtype=np.float32)
        gx, gy, gz = np.meshgrid(xi, yi, zi, indexing="ij")
        wx = self.origin[0] + (gx + 0.5) * self.voxel_size
        wy = self.origin[1] + (gy + 0.5) * self.voxel_size
        wz = self.origin[2] + (gz + 0.5) * self.voxel_size
        dist = np.sqrt(
            (wx - robot_pos[0]) ** 2 +
            (wy - robot_pos[1]) ** 2 +
            (wz - robot_pos[2]) ** 2
        )
        sub_prob = prob[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        mask = (dist <= radius_m) & (sub_prob > unknown_lo) & (sub_prob < unknown_hi)
        local_indices = np.argwhere(mask)          # local to the sub-box
        return [self.voxel_to_world(idx + lo) for idx in local_indices]


def _update_log_odds(grid: OccupancyGrid,
                     vox: np.ndarray,
                     l_sensor: float,
                     is_dynamic: bool = False) -> None:
    """Update a single voxel's log-odds, clamped to [MIN, MAX]."""
    if grid.in_bounds(vox):
        ix, iy, iz = int(vox[0]), int(vox[1]), int(vox[2])
        grid.log_odds[ix, iy, iz] = float(np.clip(
            grid.log_odds[ix, iy, iz] + l_sensor,
            LOG_ODDS_MIN, LOG_ODDS_MAX
        ))
        grid.visit_count[ix, iy, iz] += 1
        if is_dynamic:
            grid.dynamic_mask[ix, iy, iz] = True


def _bresenham_3d(p0: np.ndarray, p1: np.ndarray):
    """
    3D Bresenham line — yields integer voxel coords from p0 to p1 (exclusive of p1).
    Used for ray casting through free space.
    """
    d = p1 - p0
    n = int(np.max(np.abs(d))) + 1
    if n <= 1:
        return
    for i in range(n - 1):   # exclude endpoint (it's the occupied voxel)
        t = i / (n - 1)
        vox = np.round(p0 + t * d).astype(int)
        yield vox


# ── Main projection function ───────────────────────────────────────────────

def project_gaussians(grid: OccupancyGrid,
                       positions: np.ndarray,
                       tags: list,
                       camera_positions: np.ndarray,
                       cast_free_rays: bool = True) -> None:
    """
    Project Gaussian splats into the occupancy grid via Binary Bayes Filter.

    FIX Bug 2: YELLOW now updates voxels (l_sensor=+1.2, not 0.0).
               This restores ~40% of all scene geometry to the costmap.
    FIX Bug 6: ORANGE now updates voxels (l_sensor=+0.5) and marks
               dynamic_mask so these voxels decay each frame.
    FIX Missing 1: WHITE now updates voxels (l_sensor=+3.0).

    For each Gaussian:
        1. Mark occupied voxel with tag-dependent log-odds update.
        2. (Optional) Cast free-space ray from camera to Gaussian.

    Args:
        grid:             OccupancyGrid to update in-place.
        positions:        (N, 3) Gaussian world positions.
        tags:             list of N confidence tags.
        camera_positions: (K, 3) camera positions during scan.
        cast_free_rays:   if True, mark ray-cast voxels as free.
    """
    if len(positions) == 0:
        return

    cam_arr = np.asarray(camera_positions, dtype=np.float32)
    unknown_count  = 0
    updated_count  = 0
    skipped_red    = 0

    for i in range(len(positions)):
        tag = tags[i]
        l_sensor = LOG_ODDS_SENSOR.get(tag, 0.0)

        # RED: genuinely unknown — skip (preserve prior, don't corrupt it).
        # Unknown tags: treat as RED (0.0) with a warning.
        if tag not in LOG_ODDS_SENSOR:
            logger.warning(
                f"Unknown Gaussian tag '{tag}' at index {i} — treated as RED. "
                f"Add to LOG_ODDS_SENSOR in occupancy_grid.py."
            )
            skipped_red += 1
            continue

        if l_sensor == 0.0:
            # RED — skip entirely, navigation map keeps prior (unknown=0.5)
            skipped_red += 1
            continue

        is_dynamic = (tag == "ORANGE")
        occ_vox = grid.world_to_voxel(positions[i])

        # Update occupied voxel
        _update_log_odds(grid, occ_vox, l_sensor, is_dynamic=is_dynamic)
        updated_count += 1

        # Free-space ray cast from nearest camera
        if cast_free_rays and len(cam_arr) > 0:
            dists = np.linalg.norm(cam_arr - positions[i], axis=1)
            nearest_cam_world = cam_arr[int(np.argmin(dists))]
            cam_vox = grid.world_to_voxel(nearest_cam_world)
            for free_vox in _bresenham_3d(cam_vox, occ_vox):
                _update_log_odds(grid, free_vox, LOG_ODDS_FREE)

    logger.debug(
        f"project_gaussians: {updated_count} voxels updated, "
        f"{skipped_red} RED/unknown skipped "
        f"(of {len(positions)} total Gaussians)"
    )


def build_occupancy_grid(positions: np.ndarray,
                          tags: list,
                          camera_positions: np.ndarray,
                          room_min: np.ndarray,
                          room_max: np.ndarray,
                          voxel_size: float = 0.05) -> OccupancyGrid:
    """
    Build a complete occupancy grid from Gaussian cloud.

    Args:
        positions:        (N, 3) Gaussian positions
        tags:             list of N confidence tags
        camera_positions: (K, 3) camera trajectory
        room_min/max:     (3,) world bounding box
        voxel_size:       meters per voxel (default 5cm)

    Returns:
        Populated OccupancyGrid
    """
    extent = room_max - room_min
    shape = tuple(int(x) for x in np.ceil(extent / voxel_size).astype(int) + 2)

    grid = OccupancyGrid(
        origin=room_min - voxel_size,
        voxel_size=voxel_size,
        shape=shape
    )

    logger.info(
        f"Occupancy grid: {shape} voxels, "
        f"size={voxel_size*100:.0f}cm, "
        f"projecting {len(positions)} Gaussians"
    )

    project_gaussians(grid, positions, tags, camera_positions)

    occ_count  = int(np.sum(grid.occupied_mask()))
    free_count = int(np.sum(grid.free_mask()))
    logger.info(
        f"Occupancy grid populated: {occ_count} occupied, "
        f"{free_count} free, "
        f"{np.prod(grid.shape) - occ_count - free_count} unknown"
    )
    return grid
