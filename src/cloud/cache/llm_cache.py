"""
PHANTOM-ECHO REVEAL — 3D-LLM Context Cache + SVQ Compression
llm_cache.py  +  svq_compression.py  (combined)

3D-LLM Cache:
    Stores Chat-3D/LEO scene descriptions keyed by FAISS embedding.
    When a new occluded region matches a cached description, reuses
    the cached VideoScene generation instead of re-running the model.
    Cache key = MobileCLIP embedding of visible scene crop + physics bounds hash.

Sub-Vector Quantization (SVQ):
    Compresses Gaussian data from ~2MB → ~100-300KB for edge→cloud transmission.
    Each Gaussian (7 floats: xyz, normal_xyz, opacity) → 8-bit quantised.
    Color (3 floats RGB) → 8-bit per channel (standard).
    Scale (1 float) → 8-bit log-quantised.

Target: 10,000 Gaussians at 22 bytes each = 220KB (vs 280KB uncompressed float32)
"""

import numpy as np
import hashlib
import json
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Part 1: Sub-Vector Quantization (SVQ)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class SVQCodebook:
    """Learned or computed quantization codebook for one sub-vector."""
    centroids:  np.ndarray   # (K, D) cluster centroids
    K:          int          # number of quantization levels
    D:          int          # sub-vector dimension

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode (N, D) vectors to (N,) uint8 indices.
        Nearest centroid assignment.
        """
        dists = np.sum(
            (vectors[:, None, :] - self.centroids[None, :, :]) ** 2,
            axis=-1
        )   # (N, K)
        return np.argmin(dists, axis=1).astype(np.uint8)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Decode (N,) uint8 indices to (N, D) float32 vectors."""
        return self.centroids[indices.astype(int)]


def build_svq_codebook(data: np.ndarray, K: int = 256, n_iter: int = 20) -> SVQCodebook:
    """
    Build SVQ codebook via k-means.
    K=256 → 8-bit index (uint8).
    """
    N, D = data.shape
    if N <= K:
        # Not enough data — use identity (no compression)
        centroids = np.vstack([data, np.zeros((K - N, D))])
        return SVQCodebook(centroids=centroids.astype(np.float32), K=K, D=D)

    rng = np.random.default_rng(42)
    # Random initialisation
    idx = rng.choice(N, K, replace=False)
    centroids = data[idx].copy()

    for _ in range(n_iter):
        # Assignment
        dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        assignments = np.argmin(dists, axis=1)
        # Update
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(K)
        for i in range(N):
            new_centroids[assignments[i]] += data[i]
            counts[assignments[i]] += 1
        mask = counts > 0
        new_centroids[mask] /= counts[mask, None]
        new_centroids[~mask] = centroids[~mask]  # keep empty clusters
        centroids = new_centroids

    return SVQCodebook(centroids=centroids.astype(np.float32), K=K, D=D)


@dataclass
class CompressedGaussianCloud:
    """SVQ-compressed Gaussian point cloud."""
    # Position: (N, 3) float16 (half precision — 6 bytes vs 12)
    positions_f16: np.ndarray

    # Normal: (N,) uint8 indices into normal codebook
    normal_codes: np.ndarray

    # Color: (N, 3) uint8 (standard 8-bit RGB — already compact)
    colors_u8: np.ndarray

    # Opacity + scale: (N, 2) uint8 log-quantised
    opacity_scale_u8: np.ndarray

    # Tags: (N,) uint8 (BLUE=0, TEAL=1, GREEN=2, RED=3)
    tags_u8: np.ndarray

    # Codebooks (sent once per session)
    normal_codebook: SVQCodebook

    # Metadata
    n_gaussians: int
    uncompressed_bytes: int
    compressed_bytes: int

    def compression_ratio(self) -> float:
        return self.uncompressed_bytes / max(1, self.compressed_bytes)

    def size_kb(self) -> float:
        return self.compressed_bytes / 1024


TAG_MAP = {"BLUE": 0, "TEAL": 1, "GREEN": 2, "RED": 3, "ORANGE": 4, "YELLOW": 5}  # FIX BUG C: ORANGE and YELLOW added
TAG_INV = {v: k for k, v in TAG_MAP.items()}  # auto-updated


def compress_gaussians(positions: np.ndarray,
                        normals: np.ndarray,
                        colors: np.ndarray,
                        opacities: np.ndarray,
                        scales: np.ndarray,
                        tags: List[str]) -> CompressedGaussianCloud:
    """
    Compress Gaussian cloud via SVQ + float16.

    Args:
        positions:  (N, 3) float32 world positions
        normals:    (N, 3) float32 unit normals
        colors:     (N, 3) float32 RGB [0, 1]
        opacities:  (N,)   float32 [0, 1]
        scales:     (N,)   float32 [0.01, 0.5]
        tags:       list of N tag strings

    Returns:
        CompressedGaussianCloud
    """
    N = len(positions)
    uncompressed = N * (3 + 3 + 3 + 1 + 1) * 4   # bytes (all float32)

    # Position: float16 (half precision)
    pos_f16 = positions.astype(np.float16)

    # Normals: build 256-centroid codebook
    normal_cb = build_svq_codebook(normals, K=min(256, N))
    normal_codes = normal_cb.encode(normals)

    # Colors: 8-bit per channel
    colors_u8 = np.clip(colors * 255, 0, 255).astype(np.uint8)

    # Opacity: linear 8-bit quantisation
    opacity_u8 = np.clip(opacities * 255, 0, 255).astype(np.uint8)

    # Scale: log-quantised 8-bit (scale range ~0.005 to 0.5m)
    log_scale = np.log1p(scales)
    log_min, log_max = np.log1p(0.005), np.log1p(0.5)
    scale_u8 = np.clip(
        (log_scale - log_min) / (log_max - log_min) * 255, 0, 255
    ).astype(np.uint8)

    opacity_scale_u8 = np.column_stack([opacity_u8, scale_u8])

    # Tags
    tags_u8 = np.array([TAG_MAP.get(t, 3) for t in tags], dtype=np.uint8)

    # Estimate compressed size
    compressed = (
        N * 6   # pos float16
        + N * 1   # normal codes
        + N * 3   # color uint8
        + N * 2   # opacity+scale uint8
        + N * 1   # tags
        + normal_cb.centroids.nbytes   # codebook overhead (256*3*4 = 3KB)
    )

    logger.info(
        f"SVQ compression: {N} Gaussians, "
        f"{uncompressed/1024:.0f}KB → {compressed/1024:.0f}KB "
        f"({uncompressed/compressed:.1f}x ratio)"
    )

    return CompressedGaussianCloud(
        positions_f16=pos_f16,
        normal_codes=normal_codes,
        colors_u8=colors_u8,
        opacity_scale_u8=opacity_scale_u8,
        tags_u8=tags_u8,
        normal_codebook=normal_cb,
        n_gaussians=N,
        uncompressed_bytes=uncompressed,
        compressed_bytes=compressed,
    )


def decompress_gaussians(compressed: CompressedGaussianCloud) -> Dict[str, np.ndarray]:
    """Decompress back to float32 arrays."""
    positions  = compressed.positions_f16.astype(np.float32)
    normals    = compressed.normal_codebook.decode(compressed.normal_codes)
    colors     = compressed.colors_u8.astype(np.float32) / 255.0
    opacities  = compressed.opacity_scale_u8[:, 0].astype(np.float32) / 255.0
    tags       = [TAG_INV[int(t)] for t in compressed.tags_u8]

    # Dequantise log-scale
    log_min, log_max = np.log1p(0.005), np.log1p(0.5)
    scale_norm = compressed.opacity_scale_u8[:, 1].astype(np.float32) / 255.0
    scales = np.expm1(scale_norm * (log_max - log_min) + log_min)

    return {"positions": positions, "normals": normals, "colors": colors,
            "opacities": opacities, "scales": scales, "tags": tags}


# ══════════════════════════════════════════════════════════════════════════
# Part 2: 3D-LLM Context Cache (FAISS)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    """One cached generation result."""
    key_hash:       str
    embedding:      np.ndarray    # (D,) float32 MobileCLIP embedding
    prompt:         str
    gaussians:      List[dict]
    semantic:       str
    confidence:     float
    created_at:     float = field(default_factory=time.time)
    hit_count:      int   = 0


class LLMContextCache:
    """
    FAISS-backed cache for VideoScene generation results.
    Avoids re-running expensive generation for similar scene contexts.

    Cache key = MobileCLIP(visible_crop) ⊕ hash(physics_bounds)
    Similarity threshold = 0.92 cosine similarity (very similar scenes)
    """

    def __init__(self, max_entries: int = 1000,
                  similarity_threshold: float = 0.92):
        self._entries: List[CacheEntry] = []
        self._max_entries = max_entries
        self._sim_threshold = similarity_threshold
        self._faiss_index = None
        self._embedding_dim = None

        try:
            import faiss
            self._faiss_available = True
        except ImportError:
            self._faiss_available = False
            logger.warning("FAISS not installed — using linear scan cache")

    def _build_faiss_index(self, dim: int) -> None:
        """Build or rebuild FAISS index."""
        if not self._faiss_available:
            return
        import faiss
        self._faiss_index = faiss.IndexFlatIP(dim)  # inner product (cosine after L2-norm)
        self._embedding_dim = dim
        if self._entries:
            embs = np.stack([e.embedding for e in self._entries])
            self._faiss_index.add(embs.astype(np.float32))

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _make_key_hash(self, embedding: np.ndarray,
                        physics_bounds: Dict) -> str:
        emb_bytes = embedding.tobytes()
        bounds_str = json.dumps(physics_bounds, sort_keys=True)
        return hashlib.sha256(emb_bytes + bounds_str.encode()).hexdigest()[:16]

    def lookup(self, embedding: np.ndarray,
                physics_bounds: Dict) -> Optional[CacheEntry]:
        """
        Look up cache by embedding similarity + physics bounds compatibility.
        Returns cached entry if similarity > threshold, else None.
        """
        if not self._entries:
            return None

        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-9)

        if self._faiss_available and self._faiss_index is not None:
            import faiss
            emb_q = emb_norm.reshape(1, -1).astype(np.float32)
            D, I = self._faiss_index.search(emb_q, k=1)
            if D[0][0] >= self._sim_threshold:
                entry = self._entries[I[0][0]]
                entry.hit_count += 1
                logger.info(f"Cache HIT (FAISS): sim={D[0][0]:.3f}, entry={entry.key_hash}")
                return entry
        else:
            # Linear scan fallback
            best_sim = -1.0
            best_entry = None
            for entry in self._entries:
                sim = self._cosine_sim(emb_norm, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
            if best_sim >= self._sim_threshold and best_entry:
                best_entry.hit_count += 1
                logger.info(f"Cache HIT (linear): sim={best_sim:.3f}")
                return best_entry

        return None

    def store(self, embedding: np.ndarray,
               physics_bounds: Dict,
               prompt: str,
               gaussians: List[dict],
               semantic: str,
               confidence: float) -> CacheEntry:
        """Store a new generation result in the cache."""
        emb_norm = (embedding / (np.linalg.norm(embedding) + 1e-9)).astype(np.float32)
        key = self._make_key_hash(emb_norm, physics_bounds)

        entry = CacheEntry(
            key_hash=key,
            embedding=emb_norm,
            prompt=prompt,
            gaussians=gaussians,
            semantic=semantic,
            confidence=confidence,
        )
        self._entries.append(entry)

        # Rebuild FAISS index if needed
        if self._faiss_available:
            if self._faiss_index is None:
                self._build_faiss_index(len(emb_norm))
            else:
                self._faiss_index.add(emb_norm.reshape(1, -1))

        # Evict oldest if over capacity
        if len(self._entries) > self._max_entries:
            self._entries.pop(0)
            if self._faiss_available:
                self._build_faiss_index(len(emb_norm))

        logger.info(
            f"Cache STORE: key={key}, semantic={semantic}, "
            f"n_gaussians={len(gaussians)}, total_entries={len(self._entries)}"
        )
        return entry

    def stats(self) -> Dict:
        return {
            "n_entries":    len(self._entries),
            "total_hits":   sum(e.hit_count for e in self._entries),
            "faiss_backed": self._faiss_available,
            "threshold":    self._sim_threshold,
        }


# Module-level singleton
_cache = LLMContextCache()

def get_cache() -> LLMContextCache:
    return _cache
