"""
PHANTOM-ECHO REVEAL — PHANTOM-LITE Contradiction Engine (Layer 2)
Canonical single file — legacy contradiction_engine_fixed.py removed.

Fixes applied vs original:
  BUG-CE2  : L6 penetration skips same-semantic structural types
  BUG-CE2b : colour tag is modality-based (TEAL only for pure L5_ACOUSTIC)
  BUG-CE5  : acoustic_distance_m is pure phone→surface distance (no residual offset)
  BUG-CE9  : all 6 room surfaces passed as SceneObjects so L8 fires correctly
  BUG-11   : L6 floor-overlap tolerance matches L1 ±5 cm (prevents false IMPOSSIBLE)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

GRAVITY = np.array([0.0, -9.81, 0.0])
FLOOR_OVERLAP_TOLERANCE = 0.06   # BUG-11 fix: L6 uses same ±6cm as L1


class PhysicsVerdict(Enum):
    PROVEN     = "PROVEN"
    POSSIBLE   = "POSSIBLE"
    IMPOSSIBLE = "IMPOSSIBLE"


@dataclass
class BoundingBox:
    min_pt: np.ndarray
    max_pt: np.ndarray

    def volume(self) -> float:
        d = self.max_pt - self.min_pt
        return float(np.prod(np.clip(d, 0, None)))

    def overlaps(self, other: "BoundingBox") -> bool:
        return all(self.min_pt[i] < other.max_pt[i] and
                   self.max_pt[i] > other.min_pt[i] for i in range(3))

    def center(self) -> np.ndarray:
        return (self.min_pt + self.max_pt) / 2.0


@dataclass
class SceneObject:
    object_id:   str
    semantic:    str
    bbox:        BoundingBox
    has_support: bool = True
    is_visible:  bool = True
    mass_kg:     float = 1.0


@dataclass
class Hypothesis:
    region_id: str
    geometry:  BoundingBox
    semantic:  str
    source:    str


@dataclass
class LawResult:
    law_id:     str
    verdict:    PhysicsVerdict
    confidence: float
    reason:     str
    constraints: Dict = field(default_factory=dict)


# ── L1: Gravity ──────────────────────────────────────────────────────────────

def law_gravity(hypothesis: Hypothesis, floor_y: float,
                scene_objects: List[SceneObject]) -> LawResult:
    # BUG-V18 FIX: structural elements are not subject to gravity.
    # A WALL hypothesis at y=1.25m is not "floating" — it IS the wall structure.
    # Without this exemption every wall/ceiling hypothesis gets IMPOSSIBLE
    # and L8 symmetry PROVEN is always overridden by L1 IMPOSSIBLE.
    if hypothesis.semantic in ("WALL", "CEILING", "FLOOR"):
        return LawResult(
            "L1_GRAVITY", PhysicsVerdict.POSSIBLE, 0.95,
            f"Structural element ({hypothesis.semantic}) — exempt from gravity law"
        )

    hyp_bottom = float(hypothesis.geometry.min_pt[1])

    if hyp_bottom < floor_y - 0.05:
        return LawResult("L1_GRAVITY", PhysicsVerdict.IMPOSSIBLE, 0.99,
                         f"Bottom y={hyp_bottom:.2f}m below floor y={floor_y:.2f}m")

    supported = False
    for obj in scene_objects:
        # FIX-11: Added DESK, CABINET to support surfaces.
        # Previously DESK was missing, causing DESK-supported items to be IMPOSSIBLE.
        if obj.semantic in ("FLOOR", "TABLE", "SHELF", "DESK", "CABINET",
                            "PLATFORM", "COUNTER"):
            support_top = float(obj.bbox.max_pt[1])
            if abs(hyp_bottom - support_top) < 0.06:
                x_ok = (hypothesis.geometry.min_pt[0] < obj.bbox.max_pt[0] and
                        hypothesis.geometry.max_pt[0] > obj.bbox.min_pt[0])
                z_ok = (hypothesis.geometry.min_pt[2] < obj.bbox.max_pt[2] and
                        hypothesis.geometry.max_pt[2] > obj.bbox.min_pt[2])
                if x_ok and z_ok:
                    supported = True
                    break

    if not supported and hyp_bottom > floor_y + 0.35:
        return LawResult("L1_GRAVITY", PhysicsVerdict.IMPOSSIBLE, 0.85,
                         f"Floats at y={hyp_bottom:.2f}m with no support")

    return LawResult("L1_GRAVITY", PhysicsVerdict.POSSIBLE, 0.90,
                     "Gravity constraint satisfied")


# ── L2: Occlusion Geometry ───────────────────────────────────────────────────

def law_occlusion_geometry(hypothesis: Hypothesis, visible_gap_width_m: float,
                            camera_pos: np.ndarray) -> LawResult:
    hyp_center = hypothesis.geometry.center()
    dist = np.linalg.norm(hyp_center - camera_pos)
    if dist < 0.1:
        return LawResult("L2_OCCLUSION", PhysicsVerdict.POSSIBLE, 0.5, "Too close")

    hyp_extent = np.array([
        hypothesis.geometry.max_pt[0] - hypothesis.geometry.min_pt[0],
        hypothesis.geometry.max_pt[2] - hypothesis.geometry.min_pt[2]
    ])
    min_extent = float(np.min(hyp_extent))

    if min_extent < visible_gap_width_m * 0.1:
        return LawResult("L2_OCCLUSION", PhysicsVerdict.IMPOSSIBLE, 0.80,
                         f"Object too narrow ({min_extent:.2f}m) for gap ({visible_gap_width_m:.2f}m)")

    return LawResult("L2_OCCLUSION", PhysicsVerdict.POSSIBLE, 0.85,
                     "Occlusion geometry satisfied")


# ── L3: Shadow Geometry ──────────────────────────────────────────────────────

def law_shadow_geometry(hypothesis: Hypothesis, shadow_endpoint: np.ndarray,
                         light_direction: np.ndarray, floor_y: float) -> LawResult:
    if abs(light_direction[1]) < 0.01:
        return LawResult("L3_SHADOW", PhysicsVerdict.POSSIBLE, 0.5,
                         "Light nearly horizontal")

    hyp_top_y = float(hypothesis.geometry.max_pt[1])
    hyp_center_xz = hypothesis.geometry.center()[[0, 2]]

    t = (floor_y - hyp_top_y) / light_direction[1]
    predicted_shadow_xz = hyp_center_xz + t * light_direction[[0, 2]]
    error = np.linalg.norm(predicted_shadow_xz - shadow_endpoint[[0, 2]])

    if error > 0.5:
        return LawResult("L3_SHADOW", PhysicsVerdict.IMPOSSIBLE, 0.75,
                         f"Shadow offset {error:.2f}m from observed")

    return LawResult("L3_SHADOW", PhysicsVerdict.PROVEN, 0.80,
                     f"Shadow confirms object height (error={error:.2f}m)",
                     constraints={"max_height": hyp_top_y + 0.2})


# ── L4: Light Propagation ────────────────────────────────────────────────────

def law_light_propagation(hypothesis: Hypothesis, lit_surface_point: np.ndarray,
                           light_source: np.ndarray,
                           scene_objects: List[SceneObject]) -> LawResult:
    ray_dir = lit_surface_point - light_source
    ray_len = np.linalg.norm(ray_dir)
    if ray_len < 1e-6:
        return LawResult("L4_LIGHT", PhysicsVerdict.POSSIBLE, 0.5, "Degenerate ray")
    ray_dir_n = ray_dir / ray_len

    t_min, t_max = _ray_aabb_intersect(light_source, ray_dir_n, hypothesis.geometry)
    if t_min is not None and 0 < t_min < ray_len:
        return LawResult("L4_LIGHT", PhysicsVerdict.IMPOSSIBLE, 0.85,
                         "Hypothesis blocks observed lit surface")

    return LawResult("L4_LIGHT", PhysicsVerdict.POSSIBLE, 0.88,
                     "Light propagation satisfied")


def _ray_aabb_intersect(origin, direction, bbox) -> Tuple[Optional[float], Optional[float]]:
    t_min, t_max = -np.inf, np.inf
    for i in range(3):
        if abs(direction[i]) < 1e-9:
            if origin[i] < bbox.min_pt[i] or origin[i] > bbox.max_pt[i]:
                return None, None
        else:
            t1 = (bbox.min_pt[i] - origin[i]) / direction[i]
            t2 = (bbox.max_pt[i] - origin[i]) / direction[i]
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
    if t_min > t_max:
        return None, None
    return t_min, t_max


# ── L5: Acoustic Mirror ──────────────────────────────────────────────────────

def law_acoustic_mirror(hypothesis: Hypothesis, acoustic_distance_m: float,
                         phone_position: np.ndarray,
                         tolerance_m: float = 0.05) -> LawResult:
    # BUG-CE5 fix: use nearest-surface distance, not bbox center distance,
    # so large objects don't get falsely marked IMPOSSIBLE.
    hyp_center  = hypothesis.geometry.center()
    center_dist = float(np.linalg.norm(hyp_center - phone_position))

    # Nearest surface: clamp phone projection to bbox faces
    clamped = np.clip(phone_position,
                      hypothesis.geometry.min_pt,
                      hypothesis.geometry.max_pt)
    nearest_dist = float(np.linalg.norm(clamped - phone_position))

    # Use nearest-surface distance for the comparison
    predicted_dist = nearest_dist
    error = abs(predicted_dist - acoustic_distance_m)

    if error < tolerance_m:
        conf = min(0.99, 0.95 + (tolerance_m - error) / tolerance_m * 0.04)
        return LawResult("L5_ACOUSTIC", PhysicsVerdict.PROVEN, conf,
                         f"Acoustic {acoustic_distance_m:.2f}m matches surface "
                         f"(error={error*100:.1f}cm)")
    elif error < tolerance_m * 4:
        return LawResult("L5_ACOUSTIC", PhysicsVerdict.POSSIBLE, 0.60,
                         f"Acoustic mismatch {error*100:.1f}cm (within 4× tol)")
    else:
        return LawResult("L5_ACOUSTIC", PhysicsVerdict.IMPOSSIBLE, 0.90,
                         f"Acoustic {acoustic_distance_m:.2f}m contradicts "
                         f"nearest surface {predicted_dist:.2f}m (error={error:.2f}m)")


# ── L6: Penetration ──────────────────────────────────────────────────────────

def law_no_penetration(hypothesis: Hypothesis,
                        scene_objects: List[SceneObject]) -> LawResult:
    for obj in scene_objects:
        # BUG-CE2 fix: skip same-semantic structural self-checks
        if obj.semantic == hypothesis.semantic and \
                hypothesis.semantic in ("WALL", "FLOOR", "CEILING"):
            continue

        if obj.semantic in ("WALL", "FLOOR", "CEILING", "SOLID"):
            if hypothesis.geometry.overlaps(obj.bbox):
                overlap = _compute_overlap_volume(hypothesis.geometry, obj.bbox)

                # BUG-11 fix: exclude floor-overlap within ±6 cm (resting objects)
                if obj.semantic == "FLOOR":
                    floor_top = float(obj.bbox.max_pt[1])
                    hyp_bottom = float(hypothesis.geometry.min_pt[1])
                    if abs(hyp_bottom - floor_top) <= FLOOR_OVERLAP_TOLERANCE:
                        continue

                if overlap > 0.001:
                    return LawResult("L6_PENETRATION", PhysicsVerdict.IMPOSSIBLE, 0.99,
                                     f"Penetrates {obj.semantic} (overlap={overlap:.3f}m³)")

    return LawResult("L6_PENETRATION", PhysicsVerdict.POSSIBLE, 0.95,
                     "No penetration detected")


def _compute_overlap_volume(a: BoundingBox, b: BoundingBox) -> float:
    overlap = np.maximum(0, np.minimum(a.max_pt, b.max_pt) - np.maximum(a.min_pt, b.min_pt))
    return float(np.prod(overlap))


# ── L7: Support ──────────────────────────────────────────────────────────────

def law_support(hypothesis: Hypothesis, scene_objects: List[SceneObject],
                floor_y: float) -> LawResult:
    if hypothesis.semantic in ("WALL", "FLOOR", "CEILING"):
        return LawResult("L7_SUPPORT", PhysicsVerdict.POSSIBLE, 0.9,
                         "Structural element")

    hyp_bottom = float(hypothesis.geometry.min_pt[1])

    if abs(hyp_bottom - floor_y) < 0.08:
        return LawResult("L7_SUPPORT", PhysicsVerdict.PROVEN, 0.95, "Resting on floor")

    for obj in scene_objects:
        if obj.semantic in ("TABLE", "SHELF", "DESK", "FLOOR", "SURFACE",
                             "PLATFORM", "COUNTER"):
            if abs(hyp_bottom - float(obj.bbox.max_pt[1])) < 0.08:
                return LawResult("L7_SUPPORT", PhysicsVerdict.POSSIBLE, 0.85,
                                 f"Supported by {obj.semantic}")

    if hyp_bottom > floor_y + 0.15:
        return LawResult("L7_SUPPORT", PhysicsVerdict.IMPOSSIBLE, 0.80,
                         f"No support at y={hyp_bottom:.2f}m")

    return LawResult("L7_SUPPORT", PhysicsVerdict.POSSIBLE, 0.70, "Assumed floor support")


# ── L8: Symmetry Prior ───────────────────────────────────────────────────────

def law_symmetry_prior(hypothesis: Hypothesis, room_bbox: BoundingBox,
                        visible_walls: List[SceneObject]) -> LawResult:
    room_center = room_bbox.center()

    for wall in visible_walls:
        if wall.semantic != "WALL":
            continue
        wall_center = wall.bbox.center()

        sym_x = 2 * room_center[0] - wall_center[0]
        if abs(hypothesis.geometry.center()[0] - sym_x) < 0.3:
            return LawResult("L8_SYMMETRY", PhysicsVerdict.PROVEN, 0.72,
                             f"X-symmetric wall at x={sym_x:.2f}m")

        sym_z = 2 * room_center[2] - wall_center[2]
        if abs(hypothesis.geometry.center()[2] - sym_z) < 0.3:
            return LawResult("L8_SYMMETRY", PhysicsVerdict.PROVEN, 0.72,
                             f"Z-symmetric wall at z={sym_z:.2f}m")

    return LawResult("L8_SYMMETRY", PhysicsVerdict.POSSIBLE, 0.50,
                     "No symmetry match")


# ── Aggregation ──────────────────────────────────────────────────────────────

@dataclass
class ContradictionResult:
    hypothesis:       Hypothesis
    final_verdict:    PhysicsVerdict
    final_confidence: float
    final_tag:        str
    law_results:      List[LawResult]
    reason:           str


def run_contradiction_engine(hypothesis: Hypothesis,
                              scene_context: dict) -> ContradictionResult:
    """
    Run all applicable physical laws and return final verdict + colour tag.

    scene_context keys:
        floor_y, ceiling_y, scene_objects, camera_pos, room_bbox
        [optional] visible_gap_width_m, shadow_endpoint, light_direction,
                   lit_surface_point, light_source,
                   acoustic_distance_m, phone_position
    """
    results: List[LawResult] = []

    floor_y       = scene_context.get("floor_y", 0.0)
    scene_objects = scene_context.get("scene_objects", [])
    camera_pos    = scene_context.get("camera_pos", np.zeros(3))
    room_bbox     = scene_context.get("room_bbox",
                                       BoundingBox(np.zeros(3), np.ones(3) * 5))
    visible_walls = [o for o in scene_objects if o.semantic == "WALL"]

    # Always-run laws
    results.append(law_gravity(hypothesis, floor_y, scene_objects))
    results.append(law_no_penetration(hypothesis, scene_objects))
    results.append(law_support(hypothesis, scene_objects, floor_y))
    results.append(law_symmetry_prior(hypothesis, room_bbox, visible_walls))

    # Optional laws (fired only when data is available)
    if "visible_gap_width_m" in scene_context:
        results.append(law_occlusion_geometry(
            hypothesis, scene_context["visible_gap_width_m"], camera_pos))

    if all(k in scene_context for k in ("shadow_endpoint", "light_direction")):
        results.append(law_shadow_geometry(
            hypothesis, scene_context["shadow_endpoint"],
            scene_context["light_direction"], floor_y))

    if all(k in scene_context for k in ("lit_surface_point", "light_source")):
        results.append(law_light_propagation(
            hypothesis, scene_context["lit_surface_point"],
            scene_context["light_source"], scene_objects))

    if all(k in scene_context for k in ("acoustic_distance_m", "phone_position")):
        results.append(law_acoustic_mirror(
            hypothesis, scene_context["acoustic_distance_m"],
            scene_context["phone_position"]))

    # Aggregate
    impossible = [r for r in results if r.verdict == PhysicsVerdict.IMPOSSIBLE]
    if impossible:
        return ContradictionResult(
            hypothesis=hypothesis,
            final_verdict=PhysicsVerdict.IMPOSSIBLE,
            final_confidence=max(r.confidence for r in impossible),
            final_tag="RED",
            law_results=results,
            reason=" | ".join(r.reason for r in impossible))

    proven = [r for r in results if r.verdict == PhysicsVerdict.PROVEN]
    if proven:
        avg_conf = float(np.mean([r.confidence for r in proven]))
        # BUG-CE2b fix: TEAL only when ALL proven laws are acoustic
        all_acoustic = all(r.law_id == "L5_ACOUSTIC" for r in proven)
        tag = "TEAL" if all_acoustic else "BLUE"
        return ContradictionResult(
            hypothesis=hypothesis,
            final_verdict=PhysicsVerdict.PROVEN,
            final_confidence=avg_conf,
            final_tag=tag,
            law_results=results,
            reason=" | ".join(r.reason for r in proven))

    avg_conf = float(np.mean([r.confidence for r in results]))
    return ContradictionResult(
        hypothesis=hypothesis,
        final_verdict=PhysicsVerdict.POSSIBLE,
        final_confidence=avg_conf,
        final_tag="GREEN",
        law_results=results,
        reason="All laws satisfied — awaiting generation")


# ── Thin class wrapper for main_v2.py compatibility ───────────────────────────

class ContradictionEngineFixed:
    """Thin wrapper so main_v2.py can do engine = ContradictionEngineFixed()."""

    def run(self, hypothesis: Hypothesis, scene_context: dict) -> ContradictionResult:
        return run_contradiction_engine(hypothesis, scene_context)

    def evaluate(self, hyp: "PhysicsHypothesis"):
        """
        BUG-V18-1 FIX: main_v2 calls engine.evaluate(hyp) but only engine.run()
        existed. Added this method so the call resolves without AttributeError.

        BUG-V18-2 FIX: main_v2 builds PhysicsHypothesis with fields:
            position, semantic, confidence, context,
            acoustic_distance_m, floor_y, ceiling_y
        The old Hypothesis dataclass had: region_id, geometry, semantic, source
        — completely different. evaluate() translates the PhysicsHypothesis
        fields into the scene_context dict that run_contradiction_engine expects,
        then wraps position+semantic into a proper Hypothesis.

        Returns: (tag: str, verdict: str, confidence: float)
        """
        pos       = np.array(hyp.position)
        semantic  = str(hyp.semantic)
        floor_y   = float(hyp.floor_y)
        ceiling_y = float(hyp.ceiling_y)
        ctx       = hyp.context or {}

        # BUG-4 FIX: Build a BoundingBox centred at position with a 2cm half-extent
        # (was 5cm). At 5cm, a Gaussian at x=4.92m in a 5.0m room had a box
        # extending to x=4.97m — within the wall plane tolerance — so L6
        # Penetration fired IMPOSSIBLE on perfectly valid wall-adjacent points,
        # demoting them to RED. At 2cm the false-positive rate drops to near zero
        # while still catching genuine solid-object penetration (>2cm overlap).
        half = np.array([0.02, 0.02, 0.02])
        bbox = BoundingBox(min_pt=pos - half, max_pt=pos + half)
        h    = Hypothesis(
            region_id=f"g_{id(hyp)}",
            geometry=bbox,
            semantic=semantic,
            source="GAUSSIAN",
        )

        # Build scene_context from PhysicsHypothesis fields
        room = ctx.get("room_bounds", {"x": 5.0, "y": 2.5, "z": 4.0})
        rd   = room if isinstance(room, dict) else {"x": 5.0, "y": 2.5, "z": 4.0}

        # CE9 FIX: read scene_objects from context if present (main_v2 now passes them).
        # Fall back to building from room_bounds when called without scene_objects
        # (e.g. from main.py or unit tests that pass only room dimensions).
        if "scene_objects" in ctx and ctx["scene_objects"]:
            scene_objects = []
            for obj_dict in ctx["scene_objects"]:
                scene_objects.append(SceneObject(
                    object_id=obj_dict.get("id", obj_dict["semantic"]),
                    semantic=obj_dict["semantic"],
                    bbox=BoundingBox(
                        min_pt=np.array(obj_dict["bbox_min"], dtype=np.float64),
                        max_pt=np.array(obj_dict["bbox_max"], dtype=np.float64),
                    )
                ))
        else:
            # Fallback: build all 6 room surfaces from room_bounds
            scene_objects = [
                SceneObject("floor",      "FLOOR",
                    BoundingBox(np.array([0., floor_y - 0.1, 0.]),
                                np.array([rd["x"], floor_y, rd["z"]]))),
                SceneObject("ceiling",    "CEILING",
                    BoundingBox(np.array([0., ceiling_y, 0.]),
                                np.array([rd["x"], ceiling_y + 0.1, rd["z"]]))),
                SceneObject("wall_front", "WALL",
                    BoundingBox(np.array([0., floor_y, -0.1]),
                                np.array([rd["x"], ceiling_y, 0.0]))),
                SceneObject("wall_back",  "WALL",
                    BoundingBox(np.array([0., floor_y, rd["z"]]),
                                np.array([rd["x"], ceiling_y, rd["z"] + 0.1]))),
                SceneObject("wall_left",  "WALL",
                    BoundingBox(np.array([-0.1, floor_y, 0.]),
                                np.array([0.0, ceiling_y, rd["z"]]))),
                SceneObject("wall_right", "WALL",
                    BoundingBox(np.array([rd["x"], floor_y, 0.]),
                                np.array([rd["x"] + 0.1, ceiling_y, rd["z"]]))),
            ]

        scene_ctx = {
            "floor_y":       floor_y,
            "ceiling_y":     ceiling_y,
            "scene_objects": scene_objects,
            "camera_pos":    np.array(ctx.get("phone_position", [2.5, 1.5, 2.0])),
            "room_bbox":     BoundingBox(
                np.array([0., floor_y, 0.]),
                np.array([rd["x"], ceiling_y, rd["z"]])
            ),
        }

        # Optional law contexts
        if ctx.get("visible_gap_width_m") is not None:
            scene_ctx["visible_gap_width_m"] = float(ctx["visible_gap_width_m"])
        if ctx.get("shadow_endpoint") is not None:
            scene_ctx["shadow_endpoint"]  = np.array(ctx["shadow_endpoint"])
            scene_ctx["light_direction"]  = np.array(
                ctx.get("light_source", [2.5, 2.4, 2.0])
            ) - pos
            nrm = np.linalg.norm(scene_ctx["light_direction"])
            if nrm > 1e-6:
                scene_ctx["light_direction"] /= nrm
        if ctx.get("light_source") is not None and ctx.get("lit_surface_point") is not None:
            scene_ctx["lit_surface_point"] = np.array(ctx["lit_surface_point"])
            scene_ctx["light_source"]      = np.array(ctx["light_source"])
        if hyp.acoustic_distance_m is not None:
            scene_ctx["acoustic_distance_m"] = float(hyp.acoustic_distance_m)
            scene_ctx["phone_position"]       = np.array(
                ctx.get("phone_position", [2.5, 1.5, 2.0])
            )

        # FIX-7 (corrected): Nuanced physics evaluation for all Gaussians.
        #
        # Philosophy from the bible:
        #   WHITE  = sensor-observed, physics POSSIBLE (uncertain)
        #   BLUE   = sensor-observed, physics PROVEN (high-confidence)
        #   RED    = physics IMPOSSIBLE (contradiction found)
        #   GREEN  = generated + physics POSSIBLE
        #   TEAL   = acoustically PROVEN (L5 fired)
        #
        # Correct behaviour:
        #   1. BLUE geometry is already proven — skip re-verification (perf)
        #   2. WHITE/YELLOW/GREEN: run full physics
        #      → if IMPOSSIBLE found: demote to RED  ← contradiction
        #      → if L5_ACOUSTIC PROVEN: promote to TEAL
        #      → if all POSSIBLE: KEEP original sensor tag (don't change WHITE→GREEN)
        #   3. RED/UNKNOWN: run full physics and use engine's tag decision
        #
        # Previous wrong behaviours:
        #   Original: skip all laws for non-RED → BLUE = "ARKit confidence=2", not physics
        #   FIX-7 attempt 1: remove bypass entirely → 5000/5000 Gaussians became RED
        #     because run_contradiction_engine returns tag="GREEN" for POSSIBLE results
        #     (overwriting WHITE/BLUE with GREEN), and L4 room_bounds fires IMPOSSIBLE
        #     on Gaussians at x=5.0m for a 5.0m room (boundary precision issue).

        input_tag = str(ctx.get("input_tag", "")).upper()

        # BLUE = already physics-proven. Skip re-verification (correctness + speed).
        if input_tag == "BLUE":
            return "BLUE", "PROVEN", float(hyp.confidence)

        # Run physics for all other tags
        result = run_contradiction_engine(h, scene_ctx)

        # For sensor-confirmed geometry (WHITE, YELLOW, TEAL, GREEN, ORANGE):
        # Only override the sensor tag if physics found an explicit contradiction.
        # If all laws say POSSIBLE/PROVEN, the sensor tag is authoritative.
        if input_tag and input_tag not in ("", "RED", "UNKNOWN"):
            if result.final_verdict == PhysicsVerdict.IMPOSSIBLE:
                # Physics found a real contradiction — trust it (demote to RED)
                return "RED", result.final_verdict.value, result.final_confidence
            elif (result.final_verdict == PhysicsVerdict.PROVEN and
                  result.final_tag == "TEAL"):
                # L5 Acoustic fired PROVEN → upgrade to TEAL regardless of sensor tag
                return "TEAL", result.final_verdict.value, result.final_confidence
            else:
                # Physics agrees (POSSIBLE/PROVEN without acoustic) → keep sensor tag
                return input_tag, result.final_verdict.value, result.final_confidence

        # RED/UNKNOWN: use the engine's full tag decision
        return result.final_tag, result.final_verdict.value, result.final_confidence


@dataclass
class PhysicsHypothesis:
    """
    BUG-V18-2 FIX: Real dataclass for the fields main_v2 uses.
    Previously was just an alias for Hypothesis (wrong fields).
    """
    position:           np.ndarray
    semantic:           str
    confidence:         float
    context:            dict
    acoustic_distance_m: object   # float or None
    floor_y:            float
    ceiling_y:          float
