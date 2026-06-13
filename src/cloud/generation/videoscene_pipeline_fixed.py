"""
PHANTOM-ECHO REVEAL — VideoScene Pipeline (Flaw 43 fix: real API wired)
videoscene_pipeline_fixed.py

Replaces the stub `raise NotImplementedError` in the original
videoscene_pipeline.py with a three-tier fallback:

    Tier 1 — Real VideoScene API on RunPod A100
    Tier 2 — Stable Diffusion 3D inpainting via diffusers (if GPU)
    Tier 3 — Physics-consistent synthetic Gaussian library (always works)

The two bugs from the analysis are also fixed here:
    Bug 10 fix: validate_geometry_bounds and clamp_geometry_to_bounds
                now share the same ±5cm tolerance
    Bug 12 fix: law_gravity threshold is dynamic from scene floor_y,
                not hardcoded 0.3m
"""

import numpy as np
import time
import logging
import os
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

VIDEOSCENE_API_URL = os.environ.get("VIDEOSCENE_API_URL", "")
VIDEOSCENE_API_KEY = os.environ.get("VIDEOSCENE_API_KEY", "")
TOLERANCE_M = 0.05       # Bug 10 fix: unified ±5cm tolerance


# ── Geometry validation + clamping (Bug 10 fix: unified tolerance) ─────────
def validate_geometry_bounds(gaussians: List[Dict[str, Any]],
                               bbox_min: np.ndarray,
                               bbox_max: np.ndarray,
                               tolerance: float = TOLERANCE_M) -> bool:
    """
    Validate all Gaussians are within bbox ± tolerance.
    Bug 10 fix: tolerance is now also applied in clamp, so these are consistent.
    """
    if not gaussians:
        return True
    positions = np.array([g["position"] for g in gaussians], dtype=np.float32)
    lo = bbox_min - tolerance
    hi = bbox_max + tolerance
    in_bounds = np.all((positions >= lo) & (positions <= hi), axis=1)
    return bool(np.all(in_bounds))


def clamp_geometry_to_bounds(gaussians: List[Dict[str, Any]],
                               bbox_min: np.ndarray,
                               bbox_max: np.ndarray,
                               tolerance: float = TOLERANCE_M) -> List[Dict[str, Any]]:
    """
    Clamp Gaussian positions to bbox ± tolerance.
    Bug 10 fix: same tolerance as validate, not strict clip.
    """
    lo = bbox_min - tolerance
    hi = bbox_max + tolerance
    result = []
    for g in gaussians:
        g_copy = dict(g)
        pos = np.array(g["position"], dtype=np.float32)
        pos = np.clip(pos, lo, hi)
        g_copy["position"] = pos.tolist()
        result.append(g_copy)
    return result


# ── Physics-consistent synthetic Gaussian library (Tier 3) ───────────────
_SEMANTIC_CONFIGS = {
    "SOFA": {
        "dims":    [1.8, 0.85, 0.8],
        "color":   [0.4, 0.3, 0.25],
        "n_splats": 500,
        "support": "floor",
    },
    "CHAIR": {
        "dims":    [0.55, 0.9, 0.55],
        "color":   [0.35, 0.25, 0.15],
        "n_splats": 200,
        "support": "floor",
    },
    "TABLE": {
        "dims":    [1.2, 0.75, 0.6],
        "color":   [0.5, 0.35, 0.2],
        "n_splats": 300,
        "support": "floor",
    },
    "BOOKSHELF": {
        "dims":    [0.8, 1.8, 0.3],
        "color":   [0.45, 0.3, 0.2],
        "n_splats": 600,
        "support": "floor",
    },
    "MONITOR": {
        "dims":    [0.5, 0.35, 0.08],
        "color":   [0.1, 0.1, 0.1],
        "n_splats": 150,
        "support": "surface",
    },
    "PLANT": {
        "dims":    [0.3, 0.5, 0.3],
        "color":   [0.2, 0.5, 0.15],
        "n_splats": 180,
        "support": "floor",
    },
    "PAINTING": {
        "dims":    [0.6, 0.5, 0.02],
        "color":   [0.7, 0.6, 0.4],
        "n_splats": 100,
        "support": "wall",
    },
    "DEFAULT": {
        "dims":    [0.5, 0.5, 0.5],
        "color":   [0.5, 0.5, 0.5],
        "n_splats": 200,
        "support": "floor",
    },
}


def _generate_synthetic_gaussians(semantic: str,
                                    bbox_min: np.ndarray,
                                    bbox_max: np.ndarray,
                                    floor_y: float,
                                    ceiling_y: float,
                                    seed: int = 42) -> List[Dict[str, Any]]:
    """
    Tier 3: generate physics-consistent Gaussians from semantic template.

    Bug 12 fix: floor proximity check uses actual floor_y from scene,
    not hardcoded 0.3m threshold.
    """
    cfg = _SEMANTIC_CONFIGS.get(semantic.upper(), _SEMANTIC_CONFIGS["DEFAULT"])
    rng = np.random.default_rng(seed)
    n = cfg["n_splats"]
    base_color = np.array(cfg["color"])
    support = cfg["support"]

    # Compute valid placement bbox inside physics bounds
    center = (bbox_min + bbox_max) / 2
    half = np.array(cfg["dims"]) / 2

    # Floor-supported: snap bottom to floor_y (Bug 12 fix)
    if support == "floor":
        # Place object resting on floor (or on bbox bottom if higher)
        obj_floor = max(floor_y, bbox_min[1])
        obj_bottom = obj_floor
        obj_top = min(obj_bottom + cfg["dims"][1], bbox_max[1], ceiling_y)
        valid_min = np.array([
            max(bbox_min[0], center[0] - half[0]),
            obj_bottom,
            max(bbox_min[2], center[2] - half[2]),
        ])
        valid_max = np.array([
            min(bbox_max[0], center[0] + half[0]),
            obj_top,
            min(bbox_max[2], center[2] + half[2]),
        ])
    elif support == "wall":
        valid_min = bbox_min
        valid_max = bbox_max
    else:  # surface
        valid_min = bbox_min
        valid_max = bbox_max

    # Clamp to bbox
    valid_min = np.maximum(valid_min, bbox_min)
    valid_max = np.minimum(valid_max, bbox_max)

    if np.any(valid_max <= valid_min):
        # BUG-V22-8b FIX: collapsing to a single centre point put all n
        # splats at one interior coordinate (~10cm from any real surface).
        # Fall back to the full physics-bounded region box instead — its
        # faces are the best available surface estimate.
        logger.warning(f"Degenerate template bbox for {semantic} — "
                       f"sampling region box surface instead")
        lo = np.minimum(np.asarray(bbox_min, dtype=float),
                        np.asarray(bbox_max, dtype=float))
        hi = np.maximum(np.asarray(bbox_min, dtype=float),
                        np.asarray(bbox_max, dtype=float))
        valid_min, valid_max = lo, np.maximum(hi, lo + 1e-3)

    # BUG-V22-8 FIX: Gaussians represent SURFACES. Uniform volume sampling
    # put every generated point up to half the object inside it — 0% of GREEN
    # points were within 5cm of a true surface. Sample the box faces instead,
    # area-weighted, with outward face normals.
    ext = np.maximum(valid_max - valid_min, 1e-6)
    face_areas = np.array([ext[1]*ext[2], ext[1]*ext[2],   # x-, x+
                           ext[0]*ext[2], ext[0]*ext[2],   # y-, y+
                           ext[0]*ext[1], ext[0]*ext[1]])  # z-, z+
    face_normals = np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]],
                            dtype=np.float64)
    faces = rng.choice(6, size=n, p=face_areas/face_areas.sum())
    positions = rng.uniform(valid_min, valid_max, size=(n, 3))
    surf_normals = np.empty((n, 3))
    for i in range(n):
        ax = faces[i] // 2
        positions[i, ax] = valid_max[ax] if faces[i] % 2 else valid_min[ax]
        surf_normals[i] = face_normals[faces[i]]

    # Colors: base color + Gaussian noise for texture variation
    colors = np.clip(
        base_color + rng.normal(0, 0.05, size=(n, 3)),
        0, 1
    )

    # Normals: outward face normal + small jitter (BUG-V22-8)
    normals = surf_normals + rng.normal(0, 0.08, size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9

    scales = rng.uniform(0.02, 0.06, size=n)
    opacities = rng.uniform(0.7, 1.0, size=n)

    gaussians = []
    for i in range(n):
        gaussians.append({
            "position": positions[i].tolist(),
            "normal":   normals[i].tolist(),
            "color":    colors[i].tolist(),
            "scale":    float(scales[i]),
            "opacity":  float(opacities[i]),
            "tag":      "GREEN",
            "semantic": semantic,
        })

    return gaussians


# ── Real VideoScene API (Tier 1) ───────────────────────────────────────────
def _call_videoscene_api(prompt: str,
                          bbox_min: np.ndarray,
                          bbox_max: np.ndarray,
                          crop_a: Optional[np.ndarray] = None,
                          crop_b: Optional[np.ndarray] = None,
                          timeout_s: float = 3.0) -> Optional[List[Dict]]:
    """
    Call real VideoScene API on RunPod.
    Returns None if API unavailable or timeout.
    """
    if not VIDEOSCENE_API_URL or not VIDEOSCENE_API_KEY:
        return None

    try:
        import requests
        import base64

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
        }

        if crop_a is not None:
            payload["image_a"] = base64.b64encode(crop_a.tobytes()).decode()
        if crop_b is not None:
            payload["image_b"] = base64.b64encode(crop_b.tobytes()).decode()

        resp = requests.post(
            f"{VIDEOSCENE_API_URL}/generate",
            json=payload,
            headers={"Authorization": f"Bearer {VIDEOSCENE_API_KEY}"},
            timeout=timeout_s,
        )

        if resp.status_code == 200:
            data = resp.json()
            gaussians = data.get("gaussians", [])
            logger.info(f"VideoScene API returned {len(gaussians)} Gaussians")
            return gaussians
        else:
            logger.warning(f"VideoScene API error: {resp.status_code}")
            return None

    except Exception as e:
        logger.warning(f"VideoScene API call failed: {e}")
        return None


# ── Main generation function ───────────────────────────────────────────────
def generate_gaussians_for_region(
    semantic: str,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    floor_y: float = 0.0,
    ceiling_y: float = 2.5,
    prompt: Optional[str] = None,
    crop_a: Optional[np.ndarray] = None,
    crop_b: Optional[np.ndarray] = None,
    simulate: bool = False,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Main entry point: generate Gaussians for an occluded region.

    Three-tier fallback:
        Tier 1: Real VideoScene API (if env vars set)
        Tier 2: SD3/SD-turbo diffusers via _try_tier2_diffusion() (requires GPU)
        Tier 3: Physics-consistent synthetic library (always)

    Returns:
        (gaussians_list, tier_used)
    """
    t0 = time.time()

    if not simulate and VIDEOSCENE_API_URL:
        gaussians = _call_videoscene_api(prompt or f"Generate {semantic}", bbox_min, bbox_max, crop_a, crop_b)
        if gaussians is not None:
            gaussians = clamp_geometry_to_bounds(gaussians, bbox_min, bbox_max)
            logger.info(f"Tier 1 (API) generation: {len(gaussians)} splats in {time.time()-t0:.2f}s")
            return gaussians, "api"

    # Tier 2: local diffusion model (SD-turbo or SD3 via diffusers)
    # BUG-V19-5 FIX: Previously undocumented; now attempts real inference
    # when diffusers + torch are installed and a GPU or MPS device is available.
    # Falls back to Tier 3 gracefully if unavailable (no crash, no import error).
    if not simulate:
        tier2_result = _try_tier2_diffusion(
            semantic, bbox_min, bbox_max, floor_y, ceiling_y,
            prompt=prompt, crop_a=crop_a, seed=seed
        )
        if tier2_result is not None:
            tier2_result = clamp_geometry_to_bounds(tier2_result, bbox_min, bbox_max)
            logger.info(f"Tier 2 (diffusion) generation: {len(tier2_result)} splats "
                        f"in {time.time()-t0:.2f}s")
            return tier2_result, "diffusion"

    # Tier 3: synthetic
    gaussians = _generate_synthetic_gaussians(semantic, bbox_min, bbox_max, floor_y, ceiling_y, seed)
    gaussians = clamp_geometry_to_bounds(gaussians, bbox_min, bbox_max)

    if not validate_geometry_bounds(gaussians, bbox_min, bbox_max):
        logger.warning(f"Post-clamp validation failed for {semantic}")

    logger.info(f"Tier 3 (synthetic) generation: {len(gaussians)} splats in {time.time()-t0:.3f}s")
    return gaussians, "synthetic"


def _try_tier2_diffusion(semantic: str,
                          bbox_min: np.ndarray,
                          bbox_max: np.ndarray,
                          floor_y: float,
                          ceiling_y: float,
                          prompt: str = None,
                          crop_a=None,
                          seed: int = 42) -> list:
    """
    Tier 2: attempt local diffusion inference using diffusers.
    Returns list of gaussian dicts on success, None on any failure.
    Completely isolated — any import error or GPU error returns None.
    """
    try:
        import torch
        from diffusers import AutoPipelineForImage2Image
        from PIL import Image as PILImage
        import io

        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            return None   # no GPU — not worth running on CPU (too slow for demo)

        model_id = os.environ.get("PHANTOM_DIFFUSION_MODEL",
                                   "stabilityai/sd-turbo")

        # Load pipeline — cached after first call
        if not hasattr(_try_tier2_diffusion, "_pipe") or                 _try_tier2_diffusion._model_id != model_id:
            _try_tier2_diffusion._pipe = AutoPipelineForImage2Image.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None,
                use_safetensors=True,
            ).to(device)
            _try_tier2_diffusion._pipe.set_progress_bar_config(disable=True)
            _try_tier2_diffusion._model_id = model_id
            logger.info(f"Tier 2: loaded {model_id} on {device}")

        pipe = _try_tier2_diffusion._pipe
        gen  = torch.Generator(device=device).manual_seed(seed)

        # Build conditioning image from crop or blank
        if crop_a is not None:
            if isinstance(crop_a, np.ndarray):
                cond_img = PILImage.fromarray(
                    crop_a.astype(np.uint8) if crop_a.dtype != np.uint8 else crop_a
                ).resize((512, 512))
            else:
                cond_img = PILImage.new("RGB", (512, 512), (128, 128, 128))
        else:
            cond_img = PILImage.new("RGB", (512, 512), (128, 128, 128))

        final_prompt = prompt or f"Photorealistic {semantic} in indoor room, sharp, well-lit"

        # Single-step inference (sd-turbo uses num_inference_steps=1)
        out = pipe(
            prompt=final_prompt,
            image=cond_img,
            num_inference_steps=1,
            strength=0.5,
            guidance_scale=0.0,
            generator=gen,
        ).images[0]

        # Convert output image to depth-guided Gaussians
        # Use depth estimation heuristic from pixel brightness
        out_arr = np.array(out.convert("L"), dtype=np.float32) / 255.0
        bbox_center = (bbox_min + bbox_max) / 2.0
        bbox_extent = bbox_max - bbox_min

        # BUG-7 FIX: use semantic config splat count instead of hardcoded 60.
        # Tier 3 generates 200-600 Gaussians (SOFA=500, BOOKSHELF=600 etc).
        # 60 was too sparse for SPSR to close mesh holes for any large object.
        _cfg = _SEMANTIC_CONFIGS.get(semantic.upper(), _SEMANTIC_CONFIGS["DEFAULT"])
        n_splats = min(_cfg["n_splats"], 300)  # cap at 300 for Tier2 speed

        gaussians = []
        rng = np.random.default_rng(seed)
        H, W = out_arr.shape
        # Sample n_splats Gaussians from the generated image depth field
        for _ in range(n_splats):
            px = rng.integers(W // 4, 3 * W // 4)
            py = rng.integers(H // 4, 3 * H // 4)
            depth_val = float(out_arr[py, px])
            # Map pixel position + depth to 3D Gaussian position within bbox
            x = bbox_min[0] + (px / W) * bbox_extent[0]
            y = bbox_min[1] + depth_val * bbox_extent[1]
            z = bbox_min[2] + (py / H) * bbox_extent[2]
            pixel = np.array(out.getpixel((px, py)), dtype=np.float32) / 255.0
            gaussians.append({
                "position":   [float(x), float(y), float(z)],
                "normal":     [0.0, 1.0, 0.0],
                "color":      pixel[:3].tolist(),
                "scale":      float(min(bbox_extent) / 8.0),
                "opacity":    0.75,
                "confidence": 0.60,
                "tag":        "GREEN",
                "semantic":   semantic,
            })
        return gaussians

    except Exception as _e:
        logger.debug(f"Tier 2 unavailable: {_e}")
        return None
