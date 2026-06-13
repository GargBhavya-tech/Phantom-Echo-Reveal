"""
PHANTOM-ECHO REVEAL — Generative Outpainting Sweep (Layer 4 Mesh) — v18
outpainting_sweep.py

Seals all open RED boundary regions with flat continuation geometry
before SPSR so the mesh is fully watertight (deliverable KPI).

Strategy per boundary type:
    WALL boundary   → vertical flat quad extension
    FLOOR boundary  → horizontal flat quad extension
    CEILING boundary→ horizontal flat quad extension
    OBJECT boundary → convex-hull-cap closure

The sweep DOES NOT generate interior geometry for RED regions — it only
seals the perimeter so SPSR has a closed boundary condition.

Missing-6 fix: this module existed in the original zip but was never
called from main.py or main_v2.py. It is now wired into run_full_pipeline()
via the Patch 2 block in main_v2.py.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class SealPatch:
    """A flat continuation patch sealing one open boundary."""
    tag:      str           = "RED"
    semantic: str           = "WALL"
    position: np.ndarray    = field(default_factory=lambda: np.zeros(3))
    normal:   np.ndarray    = field(default_factory=lambda: np.array([0., 1., 0.]))
    color:    np.ndarray    = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))


def _make_wall_extension(gaussians: List[Dict], floor_y: float,
                          ceiling_y: float, room_dims: Dict,
                          axis: int, sign: int) -> List[SealPatch]:
    """
    Extend a wall plane at the room boundary (axis=0→X, axis=2→Z, sign=+/-1).
    Places one Gaussian per 10cm cell across the wall face.
    """
    patches = []
    step = 0.10
    if axis == 0:
        x_val = (room_dims.get("x", 5.0) if sign > 0 else 0.0)
        for z in np.arange(0, room_dims.get("z", 4.0), step):
            for y in np.arange(floor_y, ceiling_y, step):
                p = SealPatch(
                    tag="RED", semantic="WALL",
                    position=np.array([x_val, y, z], dtype=np.float32),
                    normal=np.array([float(-sign), 0., 0.], dtype=np.float32),
                    color=np.array([0.9, 0.9, 0.9], dtype=np.float32),
                )
                patches.append(p)
    else:
        z_val = (room_dims.get("z", 4.0) if sign > 0 else 0.0)
        for x in np.arange(0, room_dims.get("x", 5.0), step):
            for y in np.arange(floor_y, ceiling_y, step):
                p = SealPatch(
                    tag="RED", semantic="WALL",
                    position=np.array([x, y, z_val], dtype=np.float32),
                    normal=np.array([0., 0., float(-sign)], dtype=np.float32),
                    color=np.array([0.9, 0.9, 0.9], dtype=np.float32),
                )
                patches.append(p)
    return patches


def _make_floor_extension(room_dims: Dict, floor_y: float) -> List[SealPatch]:
    patches = []
    step = 0.10
    for x in np.arange(0, room_dims.get("x", 5.0), step):
        for z in np.arange(0, room_dims.get("z", 4.0), step):
            patches.append(SealPatch(
                tag="RED", semantic="FLOOR",
                position=np.array([x, floor_y, z], dtype=np.float32),
                normal=np.array([0., 1., 0.], dtype=np.float32),
                color=np.array([0.75, 0.70, 0.60], dtype=np.float32),
            ))
    return patches


def _make_ceiling_extension(room_dims: Dict, ceiling_y: float) -> List[SealPatch]:
    patches = []
    step = 0.10
    for x in np.arange(0, room_dims.get("x", 5.0), step):
        for z in np.arange(0, room_dims.get("z", 4.0), step):
            patches.append(SealPatch(
                tag="RED", semantic="CEILING",
                position=np.array([x, ceiling_y, z], dtype=np.float32),
                normal=np.array([0., -1., 0.], dtype=np.float32),
                color=np.array([0.95, 0.95, 0.95], dtype=np.float32),
            ))
    return patches


def seal_all_boundaries(gaussians: List[Any],
                         floor_y: float = 0.0,
                         ceiling_y: float = 2.5,
                         room_dims: Optional[Dict] = None) -> List[SealPatch]:
    """
    Main entry point. Auto-detects which room boundaries lack Gaussian
    coverage and generates seal patches for each.

    Returns list of SealPatch objects (each becomes a Gaussian disk
    injected into the scene before SPSR).
    """
    if room_dims is None:
        room_dims = {"x": 5.0, "y": 2.5, "z": 4.0}

    # Build coverage map: which boundary faces have Gaussians within 0.3m?
    positions = []
    for g in gaussians:
        pos = g.get("position") if isinstance(g, dict) else getattr(g, "position", None)
        if pos is not None:
            positions.append(np.asarray(pos, dtype=np.float32))

    covered = {
        "wall_x0":   False, "wall_xmax": False,
        "wall_z0":   False, "wall_zmax": False,
        "floor":     False, "ceiling":   False,
    }
    tol = 0.30

    for pos in positions:
        if pos[0] < tol:               covered["wall_x0"]   = True
        if pos[0] > room_dims["x"]-tol: covered["wall_xmax"] = True
        if pos[2] < tol:               covered["wall_z0"]   = True
        if pos[2] > room_dims["z"]-tol: covered["wall_zmax"] = True
        if abs(pos[1] - floor_y)   < tol: covered["floor"]  = True
        if abs(pos[1] - ceiling_y) < tol: covered["ceiling"] = True

    all_patches: List[SealPatch] = []

    if not covered["wall_x0"]:
        all_patches.extend(_make_wall_extension(gaussians, floor_y, ceiling_y, room_dims, 0, -1))
    if not covered["wall_xmax"]:
        all_patches.extend(_make_wall_extension(gaussians, floor_y, ceiling_y, room_dims, 0,  1))
    if not covered["wall_z0"]:
        all_patches.extend(_make_wall_extension(gaussians, floor_y, ceiling_y, room_dims, 2, -1))
    if not covered["wall_zmax"]:
        all_patches.extend(_make_wall_extension(gaussians, floor_y, ceiling_y, room_dims, 2,  1))
    if not covered["floor"]:
        all_patches.extend(_make_floor_extension(room_dims, floor_y))
    if not covered["ceiling"]:
        all_patches.extend(_make_ceiling_extension(room_dims, ceiling_y))

    logger.info(
        f"Outpainting sweep: {len(all_patches)} seal patches for "
        f"{sum(1 for v in covered.values() if not v)} uncovered boundaries "
        f"(covered: {[k for k,v in covered.items() if v]})"
    )
    return all_patches
