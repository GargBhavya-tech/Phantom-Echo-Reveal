"""
PHANTOM-ECHO REVEAL — SVQ Compression Endpoint Wiring
svq_endpoint.py

Wires Sub-Vector Quantization (from llm_cache.py) into the /reveal
API response pipeline. Generated Gaussians are compressed before
transmission to reduce WiFi payload from megabytes to ~100-300KB.

Target: 10,000 Gaussians × 13 bytes = 130KB (vs ~560KB uncompressed float32)

Usage:
    from src.cloud.compression.svq_endpoint import compress_reveal_response
    compressed_bytes = compress_reveal_response(gaussians_list)

    # Edge side:
    from src.cloud.compression.svq_endpoint import decompress_reveal_response
    gaussians_list = decompress_reveal_response(compressed_bytes)
"""

import numpy as np
import json
import base64
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Import SVQ implementation from llm_cache
from src.cloud.cache.llm_cache import (
    compress_gaussians, decompress_gaussians,
    CompressedGaussianCloud, TAG_MAP, TAG_INV
)


def gaussians_list_to_arrays(gaussians: List[Dict[str, Any]]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]
]:
    """
    Convert list of Gaussian dicts to numpy arrays for SVQ compression.

    Args:
        gaussians: list of dicts with keys: position, normal, color, scale, opacity, tag

    Returns:
        (positions, normals, colors, opacities, scales, tags)
    """
    if not gaussians:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, empty, np.zeros(0), np.zeros(0), []

    positions  = np.array([g.get("position", [0, 0, 0]) for g in gaussians], dtype=np.float32)
    normals    = np.array([g.get("normal", [0, 1, 0])   for g in gaussians], dtype=np.float32)
    colors     = np.array([g.get("color", [0.5, 0.5, 0.5]) for g in gaussians], dtype=np.float32)
    opacities  = np.array([g.get("opacity", 1.0)        for g in gaussians], dtype=np.float32)
    scales     = np.array([g.get("scale", 0.05)         for g in gaussians], dtype=np.float32)
    tags       = [g.get("tag", "GREEN")                  for g in gaussians]

    # Ensure normals are unit length
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    normals = normals / norms

    # Clip ranges
    colors    = np.clip(colors, 0, 1)
    opacities = np.clip(opacities, 0, 1)
    scales    = np.clip(scales, 0.001, 1.0)

    return positions, normals, colors, opacities, scales, tags


def compress_reveal_response(gaussians: List[Dict[str, Any]]) -> bytes:
    """
    Compress a list of Gaussian dicts to bytes for WiFi transmission.

    Uses SVQ: positions as float16, normals as uint8 codebook indices,
    colors as uint8, opacity+scale as uint8 log-quantized.

    Args:
        gaussians: list of Gaussian dicts from VideoScene

    Returns:
        JSON bytes (compressed representation + codebook)
        Real deployment: would use binary format for further size reduction
    """
    if not gaussians:
        return json.dumps({"n": 0, "gaussians": []}).encode()

    positions, normals, colors, opacities, scales, tags = gaussians_list_to_arrays(gaussians)

    compressed = compress_gaussians(positions, normals, colors, opacities, scales, tags)

    # Serialize as JSON (production: use numpy binary format)
    # Bug 8 fix: use base64 instead of hex to save 33% payload size
    payload = {
        "n": compressed.n_gaussians,
        "pos_f16":   base64.b64encode(compressed.positions_f16.tobytes()).decode('ascii'),
        "norm_codes": base64.b64encode(compressed.normal_codes.tobytes()).decode('ascii'),
        "colors_u8": base64.b64encode(compressed.colors_u8.tobytes()).decode('ascii'),
        "op_sc_u8":  base64.b64encode(compressed.opacity_scale_u8.tobytes()).decode('ascii'),
        "tags_u8":   base64.b64encode(compressed.tags_u8.tobytes()).decode('ascii'),
        "codebook":  base64.b64encode(compressed.normal_codebook.centroids.tobytes()).decode('ascii'),
        "cb_k":      compressed.normal_codebook.K,
        "cb_d":      compressed.normal_codebook.D,
        "ratio":     compressed.compression_ratio(),
        "size_kb":   compressed.size_kb(),
    }

    encoded = json.dumps(payload).encode()
    logger.info(
        f"SVQ compress: {len(gaussians)} Gaussians, "
        f"{len(encoded)/1024:.1f}KB payload "
        f"({compressed.compression_ratio():.1f}x ratio)"
    )
    return encoded


def decompress_reveal_response(data: bytes) -> List[Dict[str, Any]]:
    """
    Decompress bytes from /reveal endpoint back to Gaussian list.

    Args:
        data: bytes from compress_reveal_response

    Returns:
        list of Gaussian dicts
    """
    payload = json.loads(data.decode())

    n = payload.get("n", 0)
    if n == 0:
        return []

    # Reconstruct CompressedGaussianCloud
    from src.cloud.cache.llm_cache import SVQCodebook

    positions_f16 = np.frombuffer(base64.b64decode(payload["pos_f16"]), dtype=np.float16).reshape(n, 3)
    normal_codes  = np.frombuffer(base64.b64decode(payload["norm_codes"]), dtype=np.uint8)
    colors_u8     = np.frombuffer(base64.b64decode(payload["colors_u8"]), dtype=np.uint8).reshape(n, 3)
    op_sc_u8      = np.frombuffer(base64.b64decode(payload["op_sc_u8"]), dtype=np.uint8).reshape(n, 2)
    tags_u8       = np.frombuffer(base64.b64decode(payload["tags_u8"]), dtype=np.uint8)

    K, D = payload["cb_k"], payload["cb_d"]
    codebook_arr  = np.frombuffer(base64.b64decode(payload["codebook"]), dtype=np.float32).reshape(K, D)
    codebook      = SVQCodebook(centroids=codebook_arr, K=K, D=D)

    compressed = CompressedGaussianCloud(
        positions_f16=positions_f16,
        normal_codes=normal_codes,
        colors_u8=colors_u8,
        opacity_scale_u8=op_sc_u8,
        tags_u8=tags_u8,
        normal_codebook=codebook,
        n_gaussians=n,
        uncompressed_bytes=n * 11 * 4,
        compressed_bytes=len(data),
    )

    arrays = decompress_gaussians(compressed)

    gaussians = []
    for i in range(n):
        gaussians.append({
            "position": arrays["positions"][i].tolist(),
            "normal":   arrays["normals"][i].tolist(),
            "color":    arrays["colors"][i].tolist(),
            "opacity":  float(arrays["opacities"][i]),
            "scale":    float(arrays["scales"][i]),
            "tag":      arrays["tags"][i],
        })

    logger.info(f"SVQ decompress: {n} Gaussians from {len(data)/1024:.1f}KB")
    return gaussians


def estimate_payload_size_kb(n_gaussians: int) -> float:
    """Quick estimate of compressed payload size without running compression."""
    # ~13 bytes per Gaussian + 3KB codebook overhead
    return (n_gaussians * 13 + 3072) / 1024
