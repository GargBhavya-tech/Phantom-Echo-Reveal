"""
PHANTOM-ECHO REVEAL — Agent Tool Registry
==========================================

Each tool here is a thin, typed wrapper around a REAL pipeline module — the same
code the batch and realtime pipelines use. The agent (see planner.py) calls these
to resolve a genuinely-unknown region:

    inspect_region    — geometry/occlusion features (pure, no side effects)
    apply_physics     — PROVE: PHANTOM-LITE 8-law contradiction engine
    acoustic_measure  — MEASURE: forward+inverse acoustic DSP + SAS triangulation
    generate_geometry — IMAGINE: VideoScene generation within occlusion bounds
    plan_viewpoint    — EXPLORE: next-best-view waypoint for the robot (Mode B)

Every tool returns a JSON-serialisable observation dict. Tools report their own
applicability honestly (e.g. acoustics decline an occluded *volume* — sound
reaches surfaces, not interiors), so the planner's job is genuine sequencing and
interpretation, not reading a pre-baked answer.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

import numpy as np

logger = logging.getLogger("phantom.agent.tools")

SIMULATE = os.environ.get("PHANTOM_SIMULATE", "true").lower() == "true"


# ── Scene / region model ─────────────────────────────────────────────────────
@dataclass
class Region:
    """A genuinely-unknown region the agent must resolve."""
    region_id: str
    center: np.ndarray            # (3,) world position [m]
    size: np.ndarray              # (3,) extent [m]
    semantic: str                 # best-guess label (OBJECT/PLATFORM/SOFA/...)
    kind: str                     # "surface_on_floor" | "surface" | "volume"
    note: str = ""                # human description for the trace

    @property
    def bbox_min(self) -> np.ndarray:
        return self.center - self.size / 2.0

    @property
    def bbox_max(self) -> np.ndarray:
        return self.center + self.size / 2.0

    @property
    def volume_m3(self) -> float:
        return float(np.prod(np.maximum(self.size, 1e-6)))


@dataclass
class Scene:
    room_dims: Dict[str, float]
    floor_y: float
    ceiling_y: float
    regions: List[Region]
    # a single occluded acoustic target (the hidden surface behind the sofa)
    acoustic_target: np.ndarray = field(
        default_factory=lambda: np.array([1.9, 0.4, 1.0]))


def default_scene() -> Scene:
    """The canonical demo scene: four unknown regions, one per resolution path.

    1. floor_contact  → physics PROVES it (BLUE)
    2. hidden_surface → acoustics MEASURE it (TEAL)   [matches the bat-sonar demo]
    3. sofa_interior  → generation IMAGINES it (GREEN)
    4. far_void       → too large/unreachable → robot EXPLORES (RED + waypoint)
    """
    return Scene(
        room_dims={"x": 5.0, "y": 2.5, "z": 4.0},
        floor_y=0.0,
        ceiling_y=2.5,
        acoustic_target=np.array([1.9, 0.4, 1.0]),
        regions=[
            Region("floor_contact", np.array([2.5, 0.02, 2.0]),
                   np.array([0.40, 0.04, 0.40]), "PLATFORM", "surface_on_floor",
                   "a flat patch flush with the floor under the table"),
            Region("hidden_surface", np.array([1.9, 0.40, 1.0]),
                   np.array([0.30, 0.30, 0.30]), "OBJECT", "surface",
                   "an occluded surface behind the sofa (no line of sight)"),
            Region("sofa_interior", np.array([2.0, 0.45, 1.4]),
                   np.array([0.80, 0.60, 0.50]), "SOFA", "volume",
                   "the unseen interior volume of an occluded sofa"),
            Region("far_void", np.array([4.2, 1.2, 3.4]),
                   np.array([1.40, 1.60, 1.00]), "OBJECT", "volume",
                   "a large never-scanned corner of the room"),
        ],
    )


# ── shared scene-objects builder (mirrors main_v2 / engine) ──────────────────
def _scene_objects(scene: Scene):
    from src.edge.phantom_lite.contradiction_engine import SceneObject, BoundingBox
    rd, fy, cy = scene.room_dims, scene.floor_y, scene.ceiling_y

    def mk(i: str, s: str, lo: list, hi: list) -> "SceneObject":   # PEP 8 E731: def, not lambda
        return SceneObject(i, s, BoundingBox(np.array(lo, float), np.array(hi, float)))

    return [
        mk("floor",   "FLOOR",   [0., fy - 0.1, 0.],            [rd["x"], fy,        rd["z"]]),
        mk("ceiling", "CEILING", [0., cy,       0.],            [rd["x"], cy + 0.1,  rd["z"]]),
        mk("wall_f",  "WALL",    [0., fy, -0.1],                [rd["x"], cy, 0.0]),
        mk("wall_b",  "WALL",    [0., fy, rd["z"]],             [rd["x"], cy, rd["z"] + 0.1]),
        mk("wall_l",  "WALL",    [-0.1, fy, 0.],                [0.0, cy, rd["z"]]),
        mk("wall_r",  "WALL",    [rd["x"], fy, 0.],             [rd["x"] + 0.1, cy, rd["z"]]),
    ]


# ── Tool 1: inspect_region (pure) ────────────────────────────────────────────
def inspect_region(region: Region, scene: Scene, obs: Dict[str, Any]) -> Dict[str, Any]:
    c = region.center
    rd = scene.room_dims
    near_floor   = abs(region.bbox_min[1] - scene.floor_y)  < 0.06
    near_ceiling = abs(region.bbox_max[1] - scene.ceiling_y) < 0.06
    near_wall    = min(c[0], rd["x"] - c[0], c[2], rd["z"] - c[2]) < 0.20
    feat = {
        "center": [round(float(v), 3) for v in c],
        "size_m": [round(float(v), 3) for v in region.size],
        "volume_m3": round(region.volume_m3, 3),
        "kind": region.kind,
        "semantic_hint": region.semantic,
        "near_floor": bool(near_floor),
        "near_ceiling": bool(near_ceiling),
        "near_wall": bool(near_wall),
        "note": region.note,
    }
    obs["features"] = feat
    return feat


# ── Tool 2: apply_physics (PROVE) — real contradiction engine ────────────────
def apply_physics(region: Region, scene: Scene, obs: Dict[str, Any]) -> Dict[str, Any]:
    from src.edge.phantom_lite.contradiction_engine import (
        ContradictionEngineFixed, PhysicsHypothesis)
    engine = ContradictionEngineFixed()
    scene_objs = [
        {"semantic": o.semantic,
         "bbox_min": o.bbox.min_pt.tolist(),
         "bbox_max": o.bbox.max_pt.tolist()}
        for o in _scene_objects(scene)
    ]
    hyp = PhysicsHypothesis(
        position=region.center,
        semantic=region.semantic,
        confidence=0.5,
        context={"room_bounds": scene.room_dims, "scene_objects": scene_objs,
                 "phone_position": [scene.room_dims["x"] / 2, 1.2, scene.room_dims["z"] / 2],
                 "input_tag": "RED"},
        acoustic_distance_m=None,
        floor_y=scene.floor_y, ceiling_y=scene.ceiling_y)
    tag, verdict, conf = engine.evaluate(hyp)
    out = {"verdict": verdict, "tag": tag, "confidence": round(float(conf), 3)}
    obs["physics"] = out
    return out


# ── Tool 3: acoustic_measure (MEASURE) — real forward/inverse DSP + SAS ───────
def _walk_arc(n: int = 12) -> List[np.ndarray]:
    # Smooth 3-D arc in open floor space (full-rank SAS, stable echo tracks).
    return [np.array([1.0 + 0.45 * np.cos(0.45 * i),
                      0.95 + 0.06 * i,
                      0.7 + 0.45 * np.sin(0.45 * i)]) for i in range(n)]


def acoustic_measure(region: Region, scene: Scene, obs: Dict[str, Any]) -> Dict[str, Any]:
    # Honest physics: a chirp echo localises a SURFACE, not the interior of a
    # solid. Decline volumes rather than fabricate a return.
    if region.kind == "volume":
        out = {"success": False,
               "reason": "acoustics reach occluded SURFACES, not solid interiors — "
                         "no single echo to triangulate here"}
        obs["acoustic"] = out
        return out
    try:
        from src.edge.sensing.acoustic_forward import sweep_measurements
        from src.edge.sensing.acoustic_chirp import ChirpConfig
        from src.edge.sensing.ism_filter import WallPlane
        from src.edge.sensing.sas_triangulator import cluster_and_triangulate_v3 as triangulate

        target = region.center
        walls = [WallPlane(1, 0, 0, 0, label="wall_x0"),
                 WallPlane(0, 0, 1, 0, label="wall_z0")]
        rng = np.random.default_rng(7)
        sas, errs_cm = sweep_measurements(_walk_arc(12), [target], walls,
                                          ChirpConfig(), rng)
        pts = triangulate(sas, floor_y=scene.floor_y)
        if not pts:
            out = {"success": False, "n_returns": len(sas),
                   "reason": "no echo track triangulated (insufficient usable returns)"}
            obs["acoustic"] = out
            return out
        best = min(pts, key=lambda p: float(np.linalg.norm(
            np.asarray(p.position, dtype=np.float64) - np.asarray(target, dtype=np.float64))))
        err_cm = float(np.linalg.norm(
            np.asarray(best.position, dtype=np.float64) - np.asarray(target, dtype=np.float64)) * 100.0)
        dsp_cm = float(np.mean(errs_cm)) if errs_cm else None
        out = {
            "success": err_cm < 8.0,
            "recovered_point": [round(float(v), 3) for v in best.position],
            "surface_error_cm": round(err_cm, 2),
            "dsp_recovery_cm": round(dsp_cm, 2) if dsp_cm is not None else None,
            "confidence": round(float(best.confidence), 3),
            "n_returns": len(sas),
        }
        if not out["success"]:
            out["reason"] = f"triangulated point {err_cm:.1f} cm from expected surface (>8 cm gate)"
        obs["acoustic"] = out
        return out
    except Exception as e:                      # never let a tool kill the agent
        out = {"success": False, "reason": f"acoustic pipeline error: {type(e).__name__}: {e}"}
        obs["acoustic"] = out
        return out


# ── Tool 4: generate_geometry (IMAGINE) — real VideoScene generation ─────────
def generate_geometry(region: Region, scene: Scene, obs: Dict[str, Any]) -> Dict[str, Any]:
    # Only imagine BOUNDED occluded volumes. A large never-scanned region is not
    # something to hallucinate — that is exactly what the robot should explore.
    MAX_IMAGINE_M3 = 1.5
    if region.volume_m3 > MAX_IMAGINE_M3:
        out = {"n_splats": 0, "tier": "none",
               "reason": f"region volume {region.volume_m3:.2f} m³ exceeds the "
                         f"{MAX_IMAGINE_M3} m³ imagine-cap — too large to generate "
                         "confidently; defer to exploration"}
        obs["generation"] = out
        return out
    try:
        from src.cloud.generation.videoscene_pipeline_fixed import generate_gaussians_for_region
        splats, tier = generate_gaussians_for_region(
            semantic=region.semantic,
            bbox_min=region.bbox_min, bbox_max=region.bbox_max,
            floor_y=scene.floor_y, ceiling_y=scene.ceiling_y,
            prompt=f"a {region.semantic.lower()} in an indoor room",
            simulate=SIMULATE, seed=42)
        out = {"n_splats": len(splats), "tier": tier,
               "reason": (f"generated {len(splats)} GREEN splats via tier='{tier}' "
                          "inside the occlusion bounds")}
        obs["generation"] = out
        return out
    except Exception as e:
        out = {"n_splats": 0, "tier": "none",
               "reason": f"generation error: {type(e).__name__}: {e}"}
        obs["generation"] = out
        return out


# ── Tool 5: plan_viewpoint (EXPLORE) — next-best-view for the robot ──────────
def plan_viewpoint(region: Region, scene: Scene, obs: Dict[str, Any]) -> Dict[str, Any]:
    # Geometric next-best-view: stand back from the region toward open floor at
    # robot height, facing the unknown. Info-gain heuristic = unresolved volume.
    rd = scene.room_dims
    room_center = np.array([rd["x"] / 2.0, scene.floor_y + 0.3, rd["z"] / 2.0])
    c = region.center.copy()
    c[1] = scene.floor_y + 0.3
    to_open = room_center - c
    n = float(np.linalg.norm(to_open))
    direction = to_open / n if n > 1e-6 else np.array([0., 0., 1.])
    standoff = 1.2
    waypoint = c + direction * standoff
    waypoint[0] = float(np.clip(waypoint[0], 0.2, rd["x"] - 0.2))
    waypoint[2] = float(np.clip(waypoint[2], 0.2, rd["z"] - 0.2))
    out = {
        "waypoint": [round(float(v), 3) for v in waypoint],
        "robot_height_m": round(scene.floor_y + 0.3, 3),
        "info_gain": round(region.volume_m3, 3),
        "reason": "next-best viewpoint to resolve the region from a clear vantage "
                  "(Mode B: robot drives here, re-scans, region re-enters the pipeline)",
    }
    obs["viewpoint"] = out
    return out


# ── Registry ─────────────────────────────────────────────────────────────────
@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[[Region, Scene, Dict[str, Any]], Dict[str, Any]]


TOOLS: Dict[str, ToolSpec] = {
    "inspect_region": ToolSpec(
        "inspect_region",
        "Return the region's geometry and occlusion features (size, volume, "
        "near floor/wall/ceiling, kind). Call FIRST. No side effects.",
        inspect_region),
    "apply_physics": ToolSpec(
        "apply_physics",
        "PROVE. Run the 8-law PHANTOM-LITE contradiction engine on the region. "
        "Returns verdict PROVEN/POSSIBLE/IMPOSSIBLE. PROVEN means physics alone "
        "determines the geometry (tag BLUE) — no measurement/generation needed.",
        apply_physics),
    "acoustic_measure": ToolSpec(
        "acoustic_measure",
        "MEASURE. Emit a smartphone LFM chirp, run the forward+inverse acoustic "
        "DSP and SAS triangulation to localise an occluded SURFACE behind an "
        "obstacle. Returns a recovered point + error (tag TEAL). Declines for "
        "solid interiors (no single surface echo).",
        acoustic_measure),
    "generate_geometry": ToolSpec(
        "generate_geometry",
        "IMAGINE. Generate plausible GREEN geometry for a bounded occluded volume "
        "with VideoScene, clamped to the occlusion bounds. Use only when physics "
        "and acoustics cannot reach it AND the region is small enough to imagine.",
        generate_geometry),
    "plan_viewpoint": ToolSpec(
        "plan_viewpoint",
        "EXPLORE. Compute a next-best-view waypoint so the robot can physically "
        "resolve a region that is too large/uncertain to prove, measure, or "
        "imagine. Leaves the region RED (open) for navigation safety.",
        plan_viewpoint),
}

# Tools the planner may choose between (inspect is implicit-first; finalize is terminal)
ACTION_TOOLS = ["inspect_region", "apply_physics", "acoustic_measure",
                "generate_geometry", "plan_viewpoint", "finalize"]
