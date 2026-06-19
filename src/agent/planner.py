"""
PHANTOM-ECHO REVEAL — Agentic Planner (ReAct loop over the pipeline's tools)
============================================================================

For each genuinely-unknown region the planner runs a Reason → Act → Observe
loop, choosing one tool per step until it can finalise the region with a
confidence tag. The decision procedure IS the project thesis:

    PROVE   → apply_physics   (PROVEN ⇒ BLUE)
    MEASURE → acoustic_measure (surface recovered ⇒ TEAL)
    IMAGINE → generate_geometry (splats produced ⇒ GREEN)
    EXPLORE → plan_viewpoint   (otherwise ⇒ RED + robot waypoint)

Two interchangeable policies decide the next tool:
  • DeterministicPolicy — offline, reproducible, no API key (default).
  • LLMPolicy           — Claude (claude-opus-4-8) via forced tool-use, enabled
                          by PHANTOM_AGENT_LLM=claude (+ ANTHROPIC_API_KEY).
                          Falls back to DeterministicPolicy on any error.

Both observe the SAME tool results, so the LLM never sees a hidden answer — it
sequences real tool calls exactly as the deterministic policy does.
"""

from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any

import numpy as np

from src.agent.tools import (
    TOOLS, ACTION_TOOLS, Region, Scene, default_scene)

logger = logging.getLogger("phantom.agent")

AGENT_MODEL = os.environ.get("PHANTOM_AGENT_MODEL", "claude-opus-4-8")
MAX_STEPS_PER_REGION = 6


@dataclass
class Decision:
    tool: str
    reasoning: str
    final_tag: Optional[str] = None     # only set when tool == "finalize"
    source: str = "deterministic"        # "deterministic" | "llm"


@dataclass
class Step:
    region_id: str
    step: int
    tool: str
    reasoning: str
    observation: Dict[str, Any]
    planner: str


@dataclass
class AgentResult:
    regions: Dict[str, str]              # region_id -> final tag
    tool_use_counts: Dict[str, int]
    steps: int
    transcript: List[Dict[str, Any]]
    planner: str
    elapsed_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Deterministic policy: the Prove→Measure→Imagine→Explore procedure ────────
class DeterministicPolicy:
    source = "deterministic"

    def choose(self, region: Region, history: List[Step], obs: Dict[str, Any]) -> Decision:
        done = {h.tool for h in history}

        if "inspect_region" not in done:
            return Decision("inspect_region",
                            "Start by inspecting the region's geometry and occlusion context.",
                            source=self.source)

        if "apply_physics" not in done:
            return Decision("apply_physics",
                            "PROVE first — can the 8 physical laws determine this region "
                            "outright, before any guessing or measurement?",
                            source=self.source)

        phys = obs.get("physics", {})
        if phys.get("verdict") == "PROVEN":
            return Decision("finalize",
                            f"Physics PROVED this geometry ({phys.get('tag')} prior, "
                            f"conf {phys.get('confidence')}). No measurement or generation "
                            "needed — tag BLUE.", final_tag="BLUE", source=self.source)

        # POSSIBLE or IMPOSSIBLE → physics can't settle it. A free-floating
        # IMPOSSIBLE verdict often means a hidden support we can hear → MEASURE.
        if "acoustic_measure" not in done:
            why = ("physics rejected a free-floating surface here, but an echo could "
                   "reveal the hidden support"
                   if phys.get("verdict") == "IMPOSSIBLE"
                   else "physics is inconclusive")
            return Decision("acoustic_measure",
                            f"MEASURE — {why}; fire the bat-sonar and triangulate the "
                            "occluded surface.", source=self.source)

        aco = obs.get("acoustic", {})
        if aco.get("success"):
            return Decision("finalize",
                            f"Acoustic SAS recovered the surface to "
                            f"{aco.get('surface_error_cm')} cm (DSP "
                            f"{aco.get('dsp_recovery_cm')} cm). Direct measurement — tag TEAL.",
                            final_tag="TEAL", source=self.source)

        if "generate_geometry" not in done:
            return Decision("generate_geometry",
                            f"Physics and sound couldn't reach it ({aco.get('reason','')}). "
                            "IMAGINE — generate plausible geometry inside the occlusion bounds.",
                            source=self.source)

        gen = obs.get("generation", {})
        if gen.get("n_splats", 0) > 0:
            return Decision("finalize",
                            f"Generated {gen['n_splats']} splats via tier='{gen['tier']}'. "
                            "Plausible completion within physics bounds — tag GREEN.",
                            final_tag="GREEN", source=self.source)

        if "plan_viewpoint" not in done:
            return Decision("plan_viewpoint",
                            f"Cannot prove, measure, or imagine confidently "
                            f"({gen.get('reason','')}). EXPLORE — plan a next-best viewpoint "
                            "for the robot (Mode B).", source=self.source)

        vp = obs.get("viewpoint", {})
        return Decision("finalize",
                        f"Left RED (open) for navigation safety; robot will drive to "
                        f"{vp.get('waypoint')} and re-scan. Honest unknown beats a guess.",
                        final_tag="RED", source=self.source)


# ── Optional LLM policy: Claude chooses the next tool via forced tool-use ────
class LLMPolicy:
    source = "llm"

    def __init__(self):
        import anthropic                      # raises ImportError if not installed
        self._client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from env
        self._fallback = DeterministicPolicy()

    _SYSTEM = (
        "You are the planner for PHANTOM-ECHO REVEAL, an occlusion-aware 3D "
        "reconstruction agent. For each unknown region you choose ONE tool at a "
        "time and observe its result, following the project's strict order: "
        "PROVE (apply_physics) → MEASURE (acoustic_measure) → IMAGINE "
        "(generate_geometry) → EXPLORE (plan_viewpoint). Always inspect_region "
        "first. Call finalize only once a tool has resolved the region, and set "
        "final_tag to BLUE (physics PROVEN), TEAL (acoustic surface recovered), "
        "GREEN (geometry generated), or RED (left open for the robot to explore). "
        "Prefer the cheapest method that works; never imagine what you can prove "
        "or measure, and never guess a region too large to imagine — explore it."
    )

    def _decide_tool_schema(self) -> Dict[str, Any]:
        return {
            "name": "decide",
            "description": "Choose the next tool to call for this region.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string",
                                  "description": "One sentence: why this tool now."},
                    "tool": {"type": "string", "enum": ACTION_TOOLS},
                    "final_tag": {"type": "string",
                                  "enum": ["BLUE", "TEAL", "GREEN", "RED"],
                                  "description": "Required only when tool == finalize."},
                },
                "required": ["reasoning", "tool"],
                "additionalProperties": False,
            },
        }

    def choose(self, region: Region, history: List[Step], obs: Dict[str, Any]) -> Decision:
        try:
            catalog = "\n".join(f"- {t.name}: {t.description}" for t in TOOLS.values())
            catalog += "\n- finalize: Conclude the region with a final_tag."
            state = {
                "region_id": region.region_id,
                "note": region.note,
                "observations_so_far": obs,
                "tools_already_called": [h.tool for h in history],
            }
            user = (f"Tools:\n{catalog}\n\nRegion state:\n"
                    f"{json.dumps(state, indent=2)}\n\nChoose the next tool.")
            resp = self._client.messages.create(
                model=AGENT_MODEL,
                max_tokens=1024,
                thinking={"type": "adaptive"},
                system=self._SYSTEM,
                tools=[self._decide_tool_schema()],
                tool_choice={"type": "tool", "name": "decide"},
                messages=[{"role": "user", "content": user}],
            )
            block = next(b for b in resp.content if b.type == "tool_use")
            inp = block.input
            tool = inp.get("tool")
            if tool not in ACTION_TOOLS:
                raise ValueError(f"LLM picked unknown tool {tool!r}")
            return Decision(tool, inp.get("reasoning", ""),
                            final_tag=inp.get("final_tag"), source=self.source)
        except Exception as e:
            logger.warning(f"LLM planner unavailable ({type(e).__name__}: {e}); "
                           "falling back to deterministic policy for this step.")
            d = self._fallback.choose(region, history, obs)
            d.source = "deterministic(fallback)"
            return d


def _make_policy():
    if os.environ.get("PHANTOM_AGENT_LLM", "").lower() == "claude":
        try:
            policy = LLMPolicy()
            logger.info(f"Agent planner: Claude LLM ({AGENT_MODEL}) via forced tool-use.")
            return policy
        except Exception as e:
            logger.warning(f"Claude planner requested but unavailable ({type(e).__name__}: "
                           f"{e}); using deterministic policy.")
    logger.info("Agent planner: deterministic Prove→Measure→Imagine policy (offline).")
    return DeterministicPolicy()


# ── The agent loop ───────────────────────────────────────────────────────────
def run_agent(scene: Optional[Scene] = None,
              emit: Optional[Callable[[Dict[str, Any]], None]] = None,
              policy=None) -> AgentResult:
    """Resolve every unknown region in `scene` with a tool-using agent.

    `emit` (optional) receives each step as a dict for live streaming.
    """
    scene = scene or default_scene()
    policy = policy or _make_policy()
    t0 = time.time()

    transcript: List[Step] = []
    region_tags: Dict[str, str] = {}
    tool_counts: Dict[str, int] = {}

    def _emit(ev: Dict[str, Any]):
        if emit:
            try:
                emit(ev)
            except Exception:
                pass

    _emit({"type": "agent_start", "n_regions": len(scene.regions),
           "planner": policy.source})

    for region in scene.regions:
        obs: Dict[str, Any] = {}
        history: List[Step] = []
        logger.info(f"── Region '{region.region_id}': {region.note}")

        for step_i in range(1, MAX_STEPS_PER_REGION + 1):
            decision = policy.choose(region, history, obs)
            logger.info(f"  [{decision.source}] step {step_i}: {decision.tool} — "
                        f"{decision.reasoning}")

            if decision.tool == "finalize":
                tag = decision.final_tag or "RED"
                region_tags[region.region_id] = tag
                tool_counts["finalize"] = tool_counts.get("finalize", 0) + 1
                step = Step(region.region_id, step_i, "finalize", decision.reasoning,
                            {"final_tag": tag}, decision.source)
                transcript.append(step)
                _emit({"type": "agent_step", **asdict(step)})
                break

            spec = TOOLS.get(decision.tool)
            if spec is None:
                logger.warning(f"  unknown tool {decision.tool!r}; skipping")
                continue
            observation = spec.fn(region, scene, obs)
            tool_counts[decision.tool] = tool_counts.get(decision.tool, 0) + 1
            step = Step(region.region_id, step_i, decision.tool, decision.reasoning,
                        observation, decision.source)
            history.append(step)        # per-region memory drives the next decision
            transcript.append(step)
            _emit({"type": "agent_step", **asdict(step)})
        else:
            # ran out of steps without finalising — record honest RED
            region_tags.setdefault(region.region_id, "RED")
            logger.warning(f"  region '{region.region_id}' hit step cap — left RED")

    elapsed = round(time.time() - t0, 3)
    result = AgentResult(
        regions=region_tags,
        tool_use_counts=tool_counts,
        steps=len(transcript),
        transcript=[asdict(s) for s in transcript],
        planner=policy.source,
        elapsed_s=elapsed,
    )
    _emit({"type": "agent_summary", "regions": region_tags,
           "tool_use_counts": tool_counts, "elapsed_s": elapsed})
    return result
