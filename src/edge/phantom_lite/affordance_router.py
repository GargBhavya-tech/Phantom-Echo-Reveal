"""
PHANTOM-ECHO REVEAL — Smart Generation + Semantic Affordance Router (Layer 3)
Routes each occluded RED region to the correct geometry source:

Priority order (Flaw 18 fix — never generate what physics proves):
    1. BLUE/TEAL  → PROVEN/MEASURED — no generation needed
    2. GREEN      → POSSIBLE — pass to VideoScene with tight constraints
    3. RED        → UNKNOWN — attempt generation, mark as speculative

Semantic Affordance Routing:
    WALL    → FAISS floor-plan retrieval (structural, must be flat+vertical)
    FLOOR   → simple planar extrusion (floors are always flat)
    CEILING → simple planar extrusion (ceilings are always flat)
    CHAIR   → SlotLSTM-filtered VideoScene (seat height, leg count constraints)
    TABLE   → SlotLSTM-filtered VideoScene (surface height, leg support)
    BOX     → primitive geometry (box = box)
    UNKNOWN → full VideoScene generation with PHANTOM-LITE bounding box
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GenerationStrategy(Enum):
    SKIP            = "SKIP"            # Already proven — no generation
    PRIMITIVE       = "PRIMITIVE"       # Trivial geometry (floor, ceiling, box)
    FAISS_RETRIEVAL = "FAISS_RETRIEVAL" # Retrieve from floor-plan DB via FAISS
    SLOTLSTM        = "SLOTLSTM"        # SlotLSTM structural constraint filter
    VIDEOSCENE      = "VIDEOSCENE"      # Full VideoScene generation


@dataclass
class PhysicsBounds:
    """
    Tight physics-derived bounding box for constrained generation.
    VideoScene must stay within these bounds.
    """
    min_pt: np.ndarray    # (3,) absolute minimum corner [meters]
    max_pt: np.ndarray    # (3,) absolute maximum corner [meters]
    floor_y: float
    ceiling_y: float

    # Acoustic constraints (if available)
    acoustic_point: Optional[np.ndarray] = None   # (3,) exact surface from SAS
    acoustic_radius: float = 0.05                  # ±5cm acoustic tolerance

    # Shadow constraints
    max_height: Optional[float] = None             # from shadow geometry L3

    def volume_m3(self) -> float:
        d = np.clip(self.max_pt - self.min_pt, 0, None)
        return float(np.prod(d))

    def as_prompt_string(self) -> str:
        """Format for VideoScene system prompt."""
        lines = [
            f"Generate geometry STRICTLY within bounds:",
            f"  X: [{self.min_pt[0]:.2f}, {self.max_pt[0]:.2f}]m",
            f"  Y: [{self.min_pt[1]:.2f}, {self.max_pt[1]:.2f}]m (floor={self.floor_y:.2f}m, ceiling={self.ceiling_y:.2f}m)",
            f"  Z: [{self.min_pt[2]:.2f}, {self.max_pt[2]:.2f}]m",
        ]
        if self.acoustic_point is not None:
            lines.append(
                f"  MUST include surface at acoustic point "
                f"({self.acoustic_point[0]:.2f}, {self.acoustic_point[1]:.2f}, {self.acoustic_point[2]:.2f})m "
                f"±{self.acoustic_radius*100:.0f}cm"
            )
        if self.max_height is not None:
            lines.append(f"  MAX HEIGHT from floor: {self.max_height:.2f}m (shadow constraint)")
        return "\n".join(lines)


@dataclass
class RoutingDecision:
    """Decision for a single occluded region."""
    region_id:      str
    semantic:       str
    confidence_tag: str                  # BLUE/TEAL/GREEN/RED from contradiction engine
    strategy:       GenerationStrategy
    physics_bounds: PhysicsBounds
    prompt_hint:    str                  # semantic hint for VideoScene
    faiss_query:    Optional[str] = None # FAISS retrieval query string
    slotlstm_constraints: dict = field(default_factory=dict)


# ── Semantic affordance tables ────────────────────────────────────────────────

# Typical height ranges [min, max] in meters for each semantic class
SEMANTIC_HEIGHT_RANGE = {
    "CHAIR":    (0.40, 1.20),
    "TABLE":    (0.60, 1.00),
    "DESK":     (0.65, 0.85),
    "SOFA":     (0.35, 0.90),
    "BED":      (0.30, 0.80),
    "BOX":      (0.10, 1.50),
    "CABINET":  (0.50, 2.20),
    # NEW-BUG-5 FIX: was (0.10, 0.05) — min > max is a bug. That was the shelf
    # THICKNESS, not the height range. A shelf unit ranges from 30cm to 250cm
    # tall. The clamp in route_region sets max_pt[1] = floor_y + h_max, so
    # h_max=0.05 was clamping every SHELF to 5cm above the floor — buried.
    "SHELF":    (0.30, 2.50),
    "DOOR":     (1.90, 2.20),
    "WALL":     (2.00, 4.00),
    "PLANT":    (0.20, 2.00),
    "TV":       (0.40, 1.20),
    "MONITOR":  (0.30, 0.80),
    "UNKNOWN":  (0.10, 3.00),
}

# Which semantics are structural (must be flat/planar)
STRUCTURAL_SEMANTICS = {"WALL", "FLOOR", "CEILING", "COLUMN", "BEAM"}

# Which semantics get SlotLSTM structural filtering
SLOTLSTM_SEMANTICS = {"CHAIR", "TABLE", "DESK", "SOFA", "BED", "SHELF", "CABINET"}

# Standard affordance constraints for SlotLSTM
SLOTLSTM_CONSTRAINTS = {
    "CHAIR": {
        "seat_height_range": (0.38, 0.55),   # meters from floor
        "leg_count": (1, 5),                  # BUG-4 FIX: was (4,4) min==max → rejected stools(1)/pedestals(1)/office chairs(5). Now 1–5.
        "backrest": True,
        "armrest": "optional"
    },
    "TABLE": {
        "surface_height_range": (0.68, 0.80),
        "leg_count": (1, 5),   # BUG-4 FIX: was (4,4) — allow stools/pedestals/5-star bases
        "surface_must_be_flat": True,
        "min_surface_area_m2": 0.25
    },
    "DESK": {
        "surface_height_range": (0.70, 0.80),
        "leg_count": (1, 5),   # BUG-4 FIX: was (4,4) — allow stools/pedestals/5-star bases
        "surface_must_be_flat": True
    },
    "SOFA": {
        "seat_height_range": (0.35, 0.50),
        "backrest_height_range": (0.60, 0.90),
        "min_seat_depth_m": 0.45
    },
}


# ── Router logic ──────────────────────────────────────────────────────────────

def compute_physics_bounds(region_bbox: dict,
                            contradiction_result: dict,
                            floor_y: float,
                            ceiling_y: float) -> PhysicsBounds:
    """
    Compute tight physics bounds from contradiction engine output.

    Args:
        region_bbox:          dict with 'min_pt', 'max_pt' arrays
        contradiction_result: dict with 'law_results', 'acoustic_point', etc.
        floor_y, ceiling_y:   room dimensions

    Returns:
        PhysicsBounds with tightest constraints from all laws
    """
    min_pt = np.array(region_bbox["min_pt"], dtype=np.float64)
    max_pt = np.array(region_bbox["max_pt"], dtype=np.float64)

    # Clamp Y to room bounds
    min_pt[1] = max(min_pt[1], floor_y)
    max_pt[1] = min(max_pt[1], ceiling_y)

    # Extract acoustic point if TEAL
    acoustic_point = None
    if "acoustic_point" in contradiction_result:
        acoustic_point = np.array(contradiction_result["acoustic_point"])

    # Extract max height from shadow constraint
    max_height = None
    for law_result in contradiction_result.get("law_results", []):
        if law_result.get("law_id") == "L3_SHADOW":
            if "max_height" in law_result.get("constraints", {}):
                max_height = law_result["constraints"]["max_height"]

    return PhysicsBounds(
        min_pt=min_pt,
        max_pt=max_pt,
        floor_y=floor_y,
        ceiling_y=ceiling_y,
        acoustic_point=acoustic_point,
        max_height=max_height
    )


def route_region(region_id: str,
                 semantic: str,
                 confidence_tag: str,
                 region_bbox: dict,
                 contradiction_result: dict,
                 floor_y: float = 0.0,
                 ceiling_y: float = 2.5) -> RoutingDecision:
    """
    Route a single occluded region to the correct generation strategy.

    Args:
        region_id:            unique ID for the region
        semantic:             semantic class ('WALL', 'CHAIR', etc.)
        confidence_tag:       'BLUE', 'TEAL', 'GREEN', 'RED'
        region_bbox:          {'min_pt': (3,), 'max_pt': (3,)}
        contradiction_result: output dict from contradiction_engine
        floor_y, ceiling_y:   room dimensions

    Returns:
        RoutingDecision
    """
    physics_bounds = compute_physics_bounds(
        region_bbox, contradiction_result, floor_y, ceiling_y
    )

    # BLUE/TEAL → already proven/measured — skip generation
    if confidence_tag in ("BLUE", "TEAL"):
        logger.info(f"Region {region_id} ({semantic}): SKIP (already {confidence_tag})")
        return RoutingDecision(
            region_id=region_id,
            semantic=semantic,
            confidence_tag=confidence_tag,
            strategy=GenerationStrategy.SKIP,
            physics_bounds=physics_bounds,
            prompt_hint=f"{semantic} already measured — no generation needed"
        )

    # Structural primitives → PRIMITIVE (flat extrusion)
    if semantic in ("FLOOR", "CEILING"):
        logger.info(f"Region {region_id} ({semantic}): PRIMITIVE")
        return RoutingDecision(
            region_id=region_id,
            semantic=semantic,
            confidence_tag=confidence_tag,
            strategy=GenerationStrategy.PRIMITIVE,
            physics_bounds=physics_bounds,
            prompt_hint=f"Extend {semantic.lower()} plane within bounds"
        )

    # WALL → FAISS floor-plan retrieval
    if semantic == "WALL":
        query = _build_faiss_query(semantic, physics_bounds)
        logger.info(f"Region {region_id} (WALL): FAISS retrieval query='{query}'")
        return RoutingDecision(
            region_id=region_id,
            semantic=semantic,
            confidence_tag=confidence_tag,
            strategy=GenerationStrategy.FAISS_RETRIEVAL,
            physics_bounds=physics_bounds,
            prompt_hint="Retrieve matching wall geometry from floor-plan DB",
            faiss_query=query
        )

    # Furniture with known affordances → SlotLSTM
    if semantic in SLOTLSTM_SEMANTICS:
        constraints = SLOTLSTM_CONSTRAINTS.get(semantic, {})
        # Apply semantic height range clamp
        h_min, h_max = SEMANTIC_HEIGHT_RANGE.get(semantic, (0.1, 3.0))
        physics_bounds.max_pt[1] = min(physics_bounds.max_pt[1], floor_y + h_max)
        physics_bounds.min_pt[1] = max(physics_bounds.min_pt[1], floor_y + h_min * 0.5)

        logger.info(f"Region {region_id} ({semantic}): SLOTLSTM with {len(constraints)} constraints")
        return RoutingDecision(
            region_id=region_id,
            semantic=semantic,
            confidence_tag=confidence_tag,
            strategy=GenerationStrategy.SLOTLSTM,
            physics_bounds=physics_bounds,
            prompt_hint=f"Generate physically plausible {semantic.lower()} "
                        f"with height {h_min:.1f}–{h_max:.1f}m",
            slotlstm_constraints=constraints
        )

    # Box-like objects → PRIMITIVE
    if semantic == "BOX":
        return RoutingDecision(
            region_id=region_id, semantic=semantic,
            confidence_tag=confidence_tag,
            strategy=GenerationStrategy.PRIMITIVE,
            physics_bounds=physics_bounds,
            prompt_hint="Generate box primitive within physics bounds"
        )

    # Everything else → full VideoScene
    h_min, h_max = SEMANTIC_HEIGHT_RANGE.get(semantic, (0.1, 3.0))
    prompt = (
        f"Generate a {semantic.lower()} that fits the following constraints:\n"
        + physics_bounds.as_prompt_string()
        + f"\nSemantic: {semantic}, height range: {h_min:.1f}–{h_max:.1f}m"
    )

    logger.info(f"Region {region_id} ({semantic}): VIDEOSCENE full generation")
    return RoutingDecision(
        region_id=region_id,
        semantic=semantic,
        confidence_tag=confidence_tag,
        strategy=GenerationStrategy.VIDEOSCENE,
        physics_bounds=physics_bounds,
        prompt_hint=prompt
    )


def _build_faiss_query(semantic: str, bounds: PhysicsBounds) -> str:
    """Build a FAISS semantic query string for floor-plan retrieval."""
    width  = bounds.max_pt[0] - bounds.min_pt[0]
    height = bounds.max_pt[1] - bounds.min_pt[1]
    depth  = bounds.max_pt[2] - bounds.min_pt[2]
    return (
        f"{semantic.lower()} "
        f"width={width:.1f}m height={height:.1f}m depth={depth:.1f}m "
        f"vertical flat rectangular"
    )


def batch_route(regions: List[dict],
                floor_y: float = 0.0,
                ceiling_y: float = 2.5) -> List[RoutingDecision]:
    """
    Route all occluded regions in one pass.

    Args:
        regions: list of dicts with keys:
                 region_id, semantic, confidence_tag,
                 region_bbox, contradiction_result
        floor_y, ceiling_y: room dimensions

    Returns:
        list of RoutingDecision
    """
    decisions = []
    skip_count = 0
    gen_count  = 0

    for r in regions:
        decision = route_region(
            region_id=r["region_id"],
            semantic=r.get("semantic", "UNKNOWN"),
            confidence_tag=r.get("confidence_tag", "RED"),
            region_bbox=r["region_bbox"],
            contradiction_result=r.get("contradiction_result", {}),
            floor_y=floor_y,
            ceiling_y=ceiling_y
        )
        decisions.append(decision)

        if decision.strategy == GenerationStrategy.SKIP:
            skip_count += 1
        else:
            gen_count += 1

    logger.info(
        f"Routing complete: {skip_count} skipped (proven), "
        f"{gen_count} sent for generation"
    )
    return decisions
