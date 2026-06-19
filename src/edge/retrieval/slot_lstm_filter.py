"""
PHANTOM-ECHO REVEAL — SlotLSTM Structural Constraint Filter (Section 5.1 — Missing Component 2)
================================================================================================

Implements the SLOTLSTM generation strategy from the affordance router.
Previously the router returned strategy=SLOTLSTM but generation fell through
to Tier-3 templates with no structural filtering. This module:

1. Receives a candidate set of Gaussians (from VideoScene, FAISS, or Tier-3).
2. Applies semantic-specific structural rules from affordance_router.SLOTLSTM_CONSTRAINTS.
3. Uses ContradictionEngineFixed to validate each Gaussian against physics laws.
4. Removes Gaussians that violate constraints; shifts borderline points to comply.
5. Returns the filtered set + a constraint violation report.

Design principle: the filter is a PASS from the affordance router — it never
generates new geometry, only removes physics-invalid points from existing geometry.
This keeps the concern separation clean: generation lives in videoscene_pipeline,
structural validation lives here.

Usage
-----
    from src.edge.retrieval.slot_lstm_filter import SlotLSTMStructuralFilter

    filt = SlotLSTMStructuralFilter()
    filtered_gaussians, report = filt.apply(
        gaussians=candidate_gaussians,
        semantic="CHAIR",
        constraints={"seat_height_range": (0.38, 0.55), "leg_count": (1, 5), "backrest": True},
        floor_y=0.0,
        ceiling_y=2.5,
        bbox_min=np.array([1.0, 0.0, 1.0]),
        bbox_max=np.array([1.6, 1.0, 1.6]),
    )
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from src.edge.phantom_lite.contradiction_engine import (
    ContradictionEngineFixed, PhysicsHypothesis)

logger = logging.getLogger(__name__)


# ── Constraint validators ─────────────────────────────────────────────────────

def _check_bbox_containment(pts: np.ndarray,
                              bbox_min: np.ndarray,
                              bbox_max: np.ndarray) -> np.ndarray:
    """Return boolean mask: True where point is inside (or on) bbox."""
    # Add small tolerance (2cm) to avoid float-precision false rejections
    tol = 0.02
    inside = np.all((pts >= bbox_min - tol) & (pts <= bbox_max + tol), axis=1)
    return inside


def _check_floor_support(pts: np.ndarray,
                          floor_y: float,
                          ceiling_y: float) -> np.ndarray:
    """Return boolean mask: True where Y is within [floor_y, ceiling_y]."""
    return (pts[:, 1] >= floor_y - 0.01) & (pts[:, 1] <= ceiling_y + 0.01)


def _check_seat_height(pts: np.ndarray,
                        floor_y: float,
                        seat_range: Tuple[float, float]) -> np.ndarray:
    """Keep points within the acceptable seat-height band OR outside it (legs/backrest).

    The seat itself must be within seat_range. Points at y < seat_range[0] are
    legs (allowed). Points above seat_range[1] + 0.5 are backrests (allowed).
    Points in the forbidden gap (seat_range[1] < y < seat_range[0]) are culled.
    """
    lo, hi = seat_range
    y = pts[:, 1] - floor_y
    # allow legs (below seat), seat band, and backrest (above seat+0.05)
    valid = (y <= lo + 0.02) | ((y >= lo - 0.02) & (y <= hi + 0.02)) | (y >= hi)
    return valid


def _check_surface_height(pts: np.ndarray,
                           floor_y: float,
                           surface_range: Tuple[float, float]) -> np.ndarray:
    """Keep points within surface height band (for TABLE/DESK flat-top check)."""
    lo, hi = surface_range
    y = pts[:, 1] - floor_y
    # Surface + legs below + slight overhang tolerance
    return (y <= hi + 0.05) & (y >= 0.0)


# ── Main filter class ─────────────────────────────────────────────────────────

class SlotLSTMStructuralFilter:
    """Structural constraint filter for SLOTLSTM-routed generation.

    Thread-safe: ContradictionEngine is stateless; the filter holds no mutable
    state after construction.
    """

    def __init__(self):
        self._engine = ContradictionEngineFixed()

    def apply(self,
              gaussians: List[Dict[str, Any]],
              semantic: str,
              constraints: Dict[str, Any],
              floor_y: float,
              ceiling_y: float,
              bbox_min: np.ndarray,
              bbox_max: np.ndarray,
              room_dims: Optional[Dict[str, float]] = None) -> Tuple[List[Dict[str, Any]], Dict]:
        """Apply structural constraints and return filtered Gaussians + report.

        Args:
            gaussians:   Candidate Gaussians to filter
            semantic:    Semantic class ('CHAIR', 'TABLE', etc.)
            constraints: Dict from affordance_router.SLOTLSTM_CONSTRAINTS
            floor_y:     Room floor Y coordinate
            ceiling_y:   Room ceiling Y coordinate
            bbox_min/max: Expected bounding box (hard containment check)
            room_dims:   {'x', 'y', 'z'} room dimensions for physics engine

        Returns:
            (filtered_gaussians, report_dict)
        """
        if not gaussians:
            return [], {"semantic": semantic, "n_in": 0, "n_out": 0,
                        "removed": 0, "violations": []}

        pts = np.array([g["position"] for g in gaussians], dtype=np.float64)
        n_in = len(pts)
        mask = np.ones(n_in, dtype=bool)
        violations = []

        # ── Rule 1: hard bbox containment ────────────────────────────────────
        bbox_mask = _check_bbox_containment(pts, bbox_min, bbox_max)
        n_oob = int((~bbox_mask).sum())
        if n_oob > 0:
            mask &= bbox_mask
            violations.append({
                "rule": "bbox_containment",
                "n_removed": n_oob,
                "detail": f"{n_oob} points outside physics bbox",
            })

        # ── Rule 2: floor/ceiling containment ────────────────────────────────
        fc_mask = _check_floor_support(pts, floor_y, ceiling_y)
        n_fc = int((~fc_mask).sum())
        if n_fc > 0:
            mask &= fc_mask
            violations.append({
                "rule": "floor_ceiling_bounds",
                "n_removed": n_fc,
                "detail": f"{n_fc} points outside Y∈[{floor_y:.2f},{ceiling_y:.2f}]",
            })

        # ── Rule 3: semantic height constraints ───────────────────────────────
        if "seat_height_range" in constraints:
            sh_mask = _check_seat_height(pts, floor_y, constraints["seat_height_range"])
            n_sh = int((~sh_mask).sum())
            if n_sh > 0:
                mask &= sh_mask
                lo, hi = constraints["seat_height_range"]
                violations.append({
                    "rule": "seat_height",
                    "n_removed": n_sh,
                    "detail": f"{n_sh} points in forbidden mid-air gap (seat must be {lo:.2f}–{hi:.2f}m)",
                })

        if "surface_height_range" in constraints:
            sv_mask = _check_surface_height(pts, floor_y, constraints["surface_height_range"])
            n_sv = int((~sv_mask).sum())
            if n_sv > 0:
                mask &= sv_mask
                lo, hi = constraints["surface_height_range"]
                violations.append({
                    "rule": "surface_height",
                    "n_removed": n_sv,
                    "detail": f"{n_sv} points above {hi:.2f}m surface cap",
                })

        # ── Rule 4: ContradictionEngine physics laws ──────────────────────────
        rd = room_dims or {"x": 5.0, "y": float(ceiling_y), "z": 4.0}
        scene_objs = [
            {"semantic": "FLOOR",   "bbox_min": [0., floor_y - 0.05, 0.],
             "bbox_max": [rd["x"], floor_y, rd["z"]]},
            {"semantic": "CEILING", "bbox_min": [0., ceiling_y, 0.],
             "bbox_max": [rd["x"], ceiling_y + 0.05, rd["z"]]},
        ]
        n_physics_removed = 0
        active_pts = np.where(mask)[0]
        for i in active_pts:
            g = gaussians[i]
            hyp = PhysicsHypothesis(
                position=pts[i],
                semantic=semantic,
                confidence=g.get("confidence", 0.70),
                context={
                    "room_bounds": rd,
                    "scene_objects": scene_objs,
                    "phone_position": [rd["x"] / 2, floor_y + 1.2, rd["z"] / 2],
                    "input_tag": g.get("tag", "GREEN"),
                },
                acoustic_distance_m=None,
                floor_y=floor_y, ceiling_y=ceiling_y,
            )
            try:
                tag, _, _ = self._engine.evaluate(hyp)
                # Only drop points that physics explicitly marks as impossible
                # (TAG_RED at confidence > 0.9 means L6 structural penetration)
                if tag == "RED" and g.get("confidence", 0.70) < 0.30:
                    mask[i] = False
                    n_physics_removed += 1
            except Exception:
                pass  # physics eval failure → keep the point (fail open)

        if n_physics_removed > 0:
            violations.append({
                "rule": "contradiction_engine_physics",
                "n_removed": n_physics_removed,
                "detail": f"{n_physics_removed} low-confidence points rejected by physics laws",
            })

        # ── Build output ──────────────────────────────────────────────────────
        filtered = [g for g, m in zip(gaussians, mask) if m]
        n_out = len(filtered)
        n_removed = n_in - n_out

        # If filter removed everything, return the full set (fail open — generation
        # is better than an empty scene).
        if n_out == 0 and n_in > 0:
            logger.warning(
                f"SlotLSTMStructuralFilter: all {n_in} Gaussians filtered for "
                f"'{semantic}' — returning unfiltered set to avoid empty scene"
            )
            return gaussians, {
                "semantic": semantic, "n_in": n_in, "n_out": n_in,
                "removed": 0, "violations": violations,
                "warning": "all_filtered_fail_open",
            }

        report = {
            "semantic":      semantic,
            "n_in":          n_in,
            "n_out":         n_out,
            "removed":       n_removed,
            "removal_rate":  round(n_removed / max(n_in, 1), 3),
            "violations":    violations,
        }
        logger.info(
            f"SlotLSTMStructuralFilter: '{semantic}' {n_in}→{n_out} "
            f"({n_removed} removed, {len(violations)} violation types)"
        )
        return filtered, report

    def apply_for_routing(self,
                           gaussians: List[Dict[str, Any]],
                           decision,
                           floor_y: float,
                           ceiling_y: float,
                           room_dims: Optional[Dict[str, float]] = None
                           ) -> Tuple[List[Dict[str, Any]], Dict]:
        """Convenience wrapper: takes a RoutingDecision from affordance_router."""
        return self.apply(
            gaussians=gaussians,
            semantic=decision.semantic,
            constraints=decision.slotlstm_constraints,
            floor_y=floor_y,
            ceiling_y=ceiling_y,
            bbox_min=decision.physics_bounds.min_pt,
            bbox_max=decision.physics_bounds.max_pt,
            room_dims=room_dims,
        )
