"""
PHANTOM-ECHO REVEAL — Gaussian Decoder
gaussian_decoder.py

Decodes SVQ-compressed reveal responses from the cloud back into
Gaussian dicts usable by the viewer and scene graph.

Also handles:
    - Anchor-space → world-space transform
    - Tag validation
    - Color clamping and normal normalization
    - Duplicate filtering (if same region revealed twice)
"""

import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional

from src.cloud.compression.svq_endpoint import decompress_reveal_response
from src.shared.gaussian_format import TAG_ORDER, GaussianWire

logger = logging.getLogger(__name__)

VALID_TAGS = set(TAG_ORDER)


def decode_reveal_response(compressed_bytes: bytes,
                             anchor_transform: Optional[np.ndarray] = None
                             ) -> List[Dict[str, Any]]:
    """
    Decode SVQ-compressed /reveal response into Gaussian dicts.

    Args:
        compressed_bytes:  raw bytes from cloud /reveal endpoint
        anchor_transform:  (4,4) anchor-to-world transform (optional)
                           If provided, positions are transformed from
                           anchor-local to world space.

    Returns:
        List of Gaussian dicts ready for scene insertion
    """
    if not compressed_bytes:
        return []

    try:
        gaussians = decompress_reveal_response(compressed_bytes)
    except Exception as e:
        logger.error(f"SVQ decompress failed: {e}")
        # Try raw JSON fallback
        try:
            data = json.loads(compressed_bytes.decode())
            gaussians = data.get("gaussians", [])
        except Exception:
            return []

    if not gaussians:
        return []

    # Validate and clean each Gaussian
    cleaned = []
    for g in gaussians:
        g_clean = _validate_gaussian(g)
        if g_clean is None:
            continue

        # Apply anchor transform if provided
        if anchor_transform is not None:
            pos = np.array(g_clean["position"], dtype=np.float32)
            pos_h = np.append(pos, 1.0)
            g_clean["position"] = (anchor_transform @ pos_h)[:3].tolist()

            norm = np.array(g_clean["normal"], dtype=np.float32)
            norm_world = (anchor_transform[:3, :3] @ norm)
            n = np.linalg.norm(norm_world)
            g_clean["normal"] = (norm_world / (n + 1e-9)).tolist()

        cleaned.append(g_clean)

    logger.info(f"Decoded {len(cleaned)}/{len(gaussians)} valid Gaussians")
    return cleaned


def _validate_gaussian(g: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate and normalize one Gaussian dict.
    Returns None if the Gaussian is malformed.
    """
    try:
        pos   = [float(x) for x in g.get("position", [0, 0, 0])]
        norm  = [float(x) for x in g.get("normal",   [0, 1, 0])]
        color = [float(x) for x in g.get("color",    [0.5, 0.5, 0.5])]
        scale   = float(g.get("scale",   0.05))
        opacity = float(g.get("opacity", 1.0))
        tag     = str(g.get("tag", "GREEN"))
        semantic = str(g.get("semantic", "UNKNOWN"))

        if len(pos) != 3 or len(norm) != 3 or len(color) != 3:
            return None
        if any(np.isnan(x) or np.isinf(x) for x in pos + norm + color):
            return None

        # Normalize normal
        n_arr = np.array(norm, dtype=np.float32)
        n_len = np.linalg.norm(n_arr)
        if n_len < 1e-9:
            norm = [0.0, 1.0, 0.0]
        else:
            norm = (n_arr / n_len).tolist()

        # Clamp
        color   = [max(0.0, min(1.0, c)) for c in color]
        opacity = max(0.0, min(1.0, opacity))
        scale   = max(0.001, min(1.0, scale))
        tag     = tag if tag in VALID_TAGS else "GREEN"

        return {
            "position": pos,
            "normal":   norm,
            "color":    color,
            "scale":    scale,
            "opacity":  opacity,
            "tag":      tag,
            "semantic": semantic,
        }
    except Exception:
        return None


def filter_duplicates(existing: List[Dict[str, Any]],
                       incoming: List[Dict[str, Any]],
                       radius_m: float = 0.05) -> List[Dict[str, Any]]:
    """
    Remove incoming Gaussians that are too close to existing ones.
    Prevents double-generation artifacts when the same region is
    tapped twice.

    Uses a spatial hash grid for O(N) filtering.
    """
    if not existing or not incoming:
        return incoming

    CELL = radius_m
    occupied = set()

    for g in existing:
        pos = g["position"]
        key = (int(pos[0] / CELL), int(pos[1] / CELL), int(pos[2] / CELL))
        occupied.add(key)

    filtered = []
    for g in incoming:
        pos = g["position"]
        key = (int(pos[0] / CELL), int(pos[1] / CELL), int(pos[2] / CELL))
        if key not in occupied:
            filtered.append(g)
            occupied.add(key)

    removed = len(incoming) - len(filtered)
    if removed:
        logger.debug(f"Duplicate filter: removed {removed}/{len(incoming)} Gaussians")
    return filtered
