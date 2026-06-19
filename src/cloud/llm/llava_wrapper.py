"""
PHANTOM-ECHO REVEAL — LLaVA-NeXT-Video Scene Describer + Prompt Builder
llava_wrapper.py

Layer 3: Wraps LLaVA-NeXT-Video for:
    1. Generating natural-language scene descriptions from RGB frames
    2. Building VideoScene prompts (semantics + physics bounds)
    3. Stripping metric geometry claims from LLaVA output (Flaw 36 fix)

Flaw 36 fix (from ax.md):
    LLaVA hallucinates room dimensions in 4/10 cases.
    We keep only semantic content (object types, materials, colors)
    and replace all spatial/metric claims with acoustic ground truth.

Two modes:
    REAL  — loads LLaVA-NeXT-Video-7B-hf via transformers
    SYNTH — rule-based scene description from semantic labels (no GPU)
"""

import re
import logging
from typing import Optional, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Patterns to strip from LLaVA output (metric geometry claims)
_METRIC_PATTERNS = [
    r'\b\d+(?:\.\d+)?\s*(?:meter|metre|foot|feet|cm|inch|m)\b',
    r'\b(?:large|small|wide|narrow|huge|tiny)\s+room\b',
    r'\b\d+\s*(?:by|x)\s*\d+\b',  # "5 by 4", "3x4"
    r'\broom\s+(?:size|dimension|area)\b',
    r'\b(?:about|approximately|roughly)\s+\d+\b',
]
_METRIC_RE = re.compile('|'.join(_METRIC_PATTERNS), re.IGNORECASE)


def strip_metric_claims(text: str) -> str:
    """
    Remove all metric geometry claims from LLaVA output.
    Flaw 36 fix: prevents LLaVA from overriding acoustic measurements.
    """
    cleaned = _METRIC_RE.sub('[REDACTED-METRIC]', text)
    # Also strip sentences that are mostly metric (>2 redactions)
    sentences = cleaned.split('.')
    filtered = []
    for s in sentences:
        if s.count('[REDACTED-METRIC]') > 1:
            continue  # drop metric-heavy sentences
        filtered.append(s)
    return '.'.join(filtered).strip()


class LLaVASceneDescriber:
    """
    Wraps LLaVA-NeXT-Video for scene understanding.
    Falls back to rule-based description if model unavailable.
    """

    def __init__(self, model_id: str = "llava-hf/LLaVA-NeXT-Video-7B-hf",
                  device: str = "cpu"):
        self._model = None
        self._processor = None
        self._model_id = model_id
        self._device = device
        self._try_load_model()

    def _try_load_model(self) -> None:
        # v28: do NOT auto-download/load the 7B LLaVA model just because
        # `transformers` happens to be installed (it gets pulled in for the
        # Depth-Anything backend). On CPU this is a multi-GB download + an
        # unusably slow load that hangs the demo. Opt in explicitly with
        # `export PHANTOM_LLM_BACKEND=llava`; otherwise use the fast rule-based
        # scene describer (the demo never needs a 7B VLM).
        import os
        if os.environ.get("PHANTOM_LLM_BACKEND", "").lower() != "llava":
            logger.info("LLaVA disabled (rule-based scene describer). "
                        "Set PHANTOM_LLM_BACKEND=llava to load the 7B model.")
            return
        try:
            from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
            import torch
            logger.info(f"Loading LLaVA: {self._model_id}")
            self._processor = LlavaNextVideoProcessor.from_pretrained(self._model_id)
            self._model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                self._model_id,
                torch_dtype=torch.float16 if "cuda" in self._device else torch.float32,
                low_cpu_mem_usage=True,
            ).to(self._device)
            self._model.eval()
            logger.info("LLaVA loaded successfully")
        except Exception as e:
            logger.warning(f"LLaVA unavailable ({e}), using rule-based fallback")

    def describe_scene(self,
                        rgb_frame: np.ndarray,
                        semantic_labels: Optional[List[str]] = None) -> str:
        """
        Generate a scene description for VideoScene prompt conditioning.

        Args:
            rgb_frame:       (H, W, 3) uint8
            semantic_labels: list of detected semantic class names

        Returns:
            Clean scene description string (metric claims stripped)
        """
        if self._model is not None:
            raw = self._describe_real(rgb_frame)
        else:
            raw = self._describe_rule_based(semantic_labels or [])

        # Flaw 36 fix: strip metric claims
        clean = strip_metric_claims(raw)
        logger.debug(f"LLaVA description (stripped): {clean[:200]}")
        return clean

    def _describe_real(self, rgb_frame: np.ndarray) -> str:
        """Run real LLaVA inference."""
        try:
            from PIL import Image
            import torch
            img = Image.fromarray(rgb_frame)
            prompt = (
                "USER: <image>\nDescribe the objects visible in this indoor scene. "
                "Focus on furniture types, materials, colors, and spatial arrangement. "
                "Do NOT mention room dimensions or sizes.\nASSISTANT:"
            )
            inputs = self._processor(
                text=prompt, images=img, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=200)
            result = self._processor.decode(out[0], skip_special_tokens=True)
            # Extract after ASSISTANT:
            if "ASSISTANT:" in result:
                result = result.split("ASSISTANT:")[-1].strip()
            return result
        except Exception as e:
            logger.error(f"LLaVA real inference failed: {e}")
            return self._describe_rule_based([])

    def _describe_rule_based(self, semantic_labels: List[str]) -> str:
        """
        Rule-based scene description from semantic label list.
        Used when LLaVA is unavailable. Sufficient for VideoScene prompt.
        """
        label_descriptions = {
            "WALL":    "flat painted walls",
            "FLOOR":   "hardwood or tiled floor",
            "CEILING": "white ceiling",
            "SOFA":    "a fabric sofa against the wall",
            "CHAIR":   "a wooden chair",
            "TABLE":   "a wooden table",
            "MONITOR": "a computer monitor on the desk",
            "BOOKSHELF": "a bookshelf with books",
            "DOOR":    "a wooden door",
            "WINDOW":  "a window with natural light",
            "PLANT":   "a potted plant",
            "LAMP":    "a floor lamp",
        }

        if not semantic_labels:
            return (
                "An indoor living room scene with furniture against walls, "
                "hardwood flooring, and a white ceiling."
            )

        parts = []
        for lbl in semantic_labels:
            desc = label_descriptions.get(lbl.upper(), f"a {lbl.lower()}")
            parts.append(desc)

        return (
            f"An indoor scene containing: {', '.join(parts)}. "
            f"Neutral indoor lighting with diffuse shadows. "
            f"The scene has typical indoor materials: wood, fabric, painted plaster."
        )

    def build_videoscene_prompt(self,
                                  scene_description: str,
                                  semantic: str,
                                  bbox_min: List[float],
                                  bbox_max: List[float],
                                  acoustic_distance_m: Optional[float] = None,
                                  physics_hints: Optional[Dict[str, Any]] = None) -> str:
        """
        Build complete VideoScene generation prompt.

        Combines:
            - LLaVA scene description (semantic content, metric stripped)
            - Acoustic ground-truth distance (replaces metric claims)
            - Physics bounds from PHANTOM-LITE laws
            - Semantic affordance type

        Args:
            scene_description:    output of describe_scene()
            semantic:             target object class (CHAIR, SOFA, etc.)
            bbox_min/max:         physics-constrained bounding box [meters]
            acoustic_distance_m:  SAS-measured surface distance (ground truth)
            physics_hints:        dict of physics law hints (optional)

        Returns:
            Formatted prompt string for VideoScene API
        """
        bbox_str = (
            f"min=({bbox_min[0]:.2f},{bbox_min[1]:.2f},{bbox_min[2]:.2f}), "
            f"max=({bbox_max[0]:.2f},{bbox_max[1]:.2f},{bbox_max[2]:.2f})"
        )

        depth_str = (
            f"{acoustic_distance_m:.2f}m (acoustic measurement)"
            if acoustic_distance_m is not None
            else "unknown"
        )

        hints_str = ""
        if physics_hints:
            law_lines = []
            for law_name, verdict in physics_hints.items():
                law_lines.append(f"  - {law_name}: {verdict}")
            hints_str = "\nPhysics constraints:\n" + "\n".join(law_lines)

        prompt = (
            f"Generate a {semantic} in an indoor 3D scene.\n\n"
            f"Scene context: {scene_description}\n\n"
            f"Target region bounding box (world space, meters): {bbox_str}\n"
            f"Acoustic depth measurement: {depth_str}\n"
            f"{hints_str}\n\n"
            f"Requirements:\n"
            f"  - Object must fit entirely within the bounding box\n"
            f"  - Match the materials and style of the visible scene\n"
            f"  - Respect physics: gravity, structural support, no wall penetration\n"
            f"  - Output as 3D Gaussian splat representation\n"
        )

        return prompt

    def describe_from_gaussians(self,
                                gaussians: List[Dict[str, Any]],
                                rgb_frame: Optional[np.ndarray] = None) -> str:
        """Section 5.1 — Gaussian-aware scene description.

        Builds a meaningful scene description from the PHANTOM Gaussian tag
        distribution instead of the zero-info default. This gives VideoScene/FAISS
        a scene context that reflects what PHANTOM has actually sensed, rather than
        a generic 'indoor room' string.

        Tag semantics:
            WHITE  — directly sensed, visible surface
            TEAL   — acoustic bat-sonar, measured hidden surface
            GREEN  — VideoScene generated, confirmed plausible
            RED    — unknown occlusion, currently being revealed
            BLUE   — structural prior (floor/ceiling/walls)
            ORANGE — dynamic object (moving person/robot)

        Args:
            gaussians:  Current scene Gaussian list (from engine.all_gaussians)
            rgb_frame:  Optional current RGB frame for LLaVA real inference

        Returns:
            Scene description string
        """
        if not gaussians:
            return self._describe_rule_based([])

        from collections import Counter
        tags = Counter(g.get("tag", "RED") for g in gaussians)
        semantics = list({g.get("semantic", "UNKNOWN")
                          for g in gaussians
                          if g.get("semantic") not in (None, "UNKNOWN", "OTHER", "OCCLUDED_UNKNOWN")})

        n_total   = len(gaussians)
        n_sensed  = tags.get("WHITE", 0) + tags.get("TEAL", 0)
        n_gen     = tags.get("GREEN", 0)
        n_occl    = tags.get("RED", 0)
        n_dynamic = tags.get("ORANGE", 0)

        # If real LLaVA is available and we have an RGB frame, use it
        if self._model is not None and rgb_frame is not None:
            return self.describe_scene(rgb_frame, semantic_labels=semantics)

        # Gaussian-aware rule-based description
        parts = []
        coverage = round(100 * n_sensed / max(n_total, 1))
        parts.append(
            f"Indoor scene with {coverage}% direct sensor coverage "
            f"({n_sensed} sensed, {n_gen} generated, {n_occl} unknown regions"
            + (f", {n_dynamic} dynamic objects" if n_dynamic > 0 else "") + ")."
        )

        if semantics:
            parts.append(self._describe_rule_based(semantics[:8]))
        else:
            parts.append("Typical indoor scene: furniture and structural surfaces visible.")

        if n_occl > 0:
            parts.append(
                f"There are {n_occl} unknown Gaussians (RED) representing "
                f"occluded or unresolved regions awaiting generation."
            )
        if tags.get("TEAL", 0) > 0:
            parts.append(
                f"Acoustic bat-sonar has measured {tags['TEAL']} hidden surface "
                f"point(s) behind visible obstacles."
            )

        description = " ".join(parts)
        return strip_metric_claims(description)
