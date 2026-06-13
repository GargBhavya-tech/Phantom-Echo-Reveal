"""
PHANTOM-ECHO REVEAL — Semantic Cache Checker
cache_checker.py

Before generating anything, check if a semantically-similar region
was already generated in this or a previous session.

Cache hit = cosine_similarity(query_embedding, cached_embedding) > THRESHOLD

Two-level check:
    L1: in-process FAISS index (fast, ~1ms)
    L2: Redis persistent store (cross-session, ~5ms)

Returns the cached Gaussian list if hit, else None (→ generate fresh).
"""

import numpy as np
import logging
import json
import time
from typing import Optional, List, Dict, Any, Tuple

from src.edge.embedding.mobile_clip import get_embedder, EMBED_DIM

logger = logging.getLogger(__name__)

SIM_THRESHOLD_FAISS = 0.92    # strict: use cached result as-is
SIM_THRESHOLD_SOFT  = 0.85    # softer: use as seed for generation
MAX_CACHE_ENTRIES   = 10_000


class SemanticCacheChecker:
    """
    Two-level semantic cache for Gaussian generation results.

    Usage:
        checker = SemanticCacheChecker()
        hit = checker.query(semantic, bbox_min, bbox_max, rgb_crop)
        if hit:
            return hit  # cached Gaussians
        else:
            gaussians = generate(...)
            checker.store(semantic, bbox_min, bbox_max, rgb_crop, gaussians)
    """

    def __init__(self, redis_url: Optional[str] = None,
                  sim_threshold: float = SIM_THRESHOLD_FAISS):
        self._embedder = get_embedder()
        self._threshold = sim_threshold
        self._entries: List[Dict] = []     # {embedding, gaussians, semantic, ts}
        self._index = None                  # FAISS index (built lazily)
        self._index_dirty = True
        self._redis = self._init_redis(redis_url)

    def _init_redis(self, url: Optional[str]):
        if url is None:
            return None
        try:
            import redis
            r = redis.from_url(url, socket_connect_timeout=2,
                                    socket_timeout=2)
            r.ping()   # fail fast if connection refused
            r.ping()
            logger.info(f"Redis connected: {url}")
            return r
        except Exception as e:
            logger.info(f"Redis unavailable ({e}), using in-process cache only")
            return None

    # ── Query ──────────────────────────────────────────────────────────────
    def query(self,
               semantic: str,
               bbox_min: np.ndarray,
               bbox_max: np.ndarray,
               rgb_crop: Optional[np.ndarray] = None
               ) -> Optional[List[Dict[str, Any]]]:
        """
        Check if a semantically-similar generation exists in cache.

        Args:
            semantic:  object class name
            bbox_min/max: region bounds
            rgb_crop:  optional RGB crop for visual similarity

        Returns:
            Cached Gaussian list or None
        """
        embedding = self._build_query_embedding(semantic, bbox_min, bbox_max, rgb_crop)

        # L1: FAISS
        result = self._query_faiss(embedding)
        if result is not None:
            logger.info(f"Cache HIT (FAISS): {semantic}")
            return result

        # L2: Redis
        if self._redis is not None:
            result = self._query_redis(embedding, semantic)
            if result is not None:
                logger.info(f"Cache HIT (Redis): {semantic}")
                return result

        logger.debug(f"Cache MISS: {semantic}")
        return None

    def _query_faiss(self, embedding: np.ndarray) -> Optional[List[Dict]]:
        if len(self._entries) == 0:
            return None
        self._ensure_index()
        try:
            import faiss
            D, I = self._index.search(
                embedding.reshape(1, -1).astype(np.float32), 1
            )
            sim = float(D[0, 0])
            idx = int(I[0, 0])
            if sim >= self._threshold and 0 <= idx < len(self._entries):
                return self._entries[idx]["gaussians"]
        except Exception as e:
            logger.debug(f"FAISS query failed: {e}")
        return None

    def _query_redis(self, embedding: np.ndarray,
                      semantic: str) -> Optional[List[Dict]]:
        try:
            # Scan keys for same semantic
            pattern = f"phantom:cache:{semantic}:*"
            keys = list(self._redis.scan_iter(pattern, count=100))
            if not keys:
                return None
            best_sim = 0.0
            best_val = None
            for key in keys[:50]:   # check up to 50 candidates
                raw = self._redis.get(key)
                if raw is None:
                    continue
                entry = json.loads(raw)
                cached_emb = np.array(entry["embedding"], dtype=np.float32)
                sim = self._embedder.cosine_similarity(embedding, cached_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_val = entry
            if best_sim >= self._threshold and best_val:
                return best_val["gaussians"]
        except Exception as e:
            logger.debug(f"Redis query failed: {e}")
        return None

    # ── Store ──────────────────────────────────────────────────────────────
    def store(self,
               semantic: str,
               bbox_min: np.ndarray,
               bbox_max: np.ndarray,
               gaussians: List[Dict[str, Any]],
               rgb_crop: Optional[np.ndarray] = None,
               ttl_s: int = 3600) -> None:
        """
        Store a generated result in the cache.
        """
        embedding = self._build_query_embedding(semantic, bbox_min, bbox_max, rgb_crop)

        entry = {
            "embedding": embedding,
            "gaussians": gaussians,
            "semantic":  semantic,
            "ts":        time.time(),
        }

        # L1: in-process
        if len(self._entries) >= MAX_CACHE_ENTRIES:
            self._entries.pop(0)
        self._entries.append(entry)
        self._index_dirty = True

        # L2: Redis
        if self._redis is not None:
            try:
                key = f"phantom:cache:{semantic}:{time.time_ns()}"
                payload = {
                    "embedding": embedding.tolist(),
                    "gaussians": gaussians[:200],  # cap for Redis size
                    "semantic": semantic,
                    "ts": entry["ts"],
                }
                self._redis.setex(key, ttl_s, json.dumps(payload))
            except Exception as e:
                logger.debug(f"Redis store failed: {e}")

        logger.debug(f"Cached {len(gaussians)} Gaussians for {semantic}")

    # ── Internal ───────────────────────────────────────────────────────────
    def _build_query_embedding(self,
                                 semantic: str,
                                 bbox_min: np.ndarray,
                                 bbox_max: np.ndarray,
                                 rgb_crop: Optional[np.ndarray]) -> np.ndarray:
        """
        Build a combined embedding from semantic label + bbox size + visual crop.
        """
        # Text embedding of semantic label
        text_emb = self._embedder.embed_text(f"indoor {semantic.lower()} furniture object")

        # Bbox size embedding (normalized)
        size = np.clip(bbox_max - bbox_min, 0.01, 5.0)
        size_feat = np.tile(size / 5.0, EMBED_DIM // 3 + 1)[:EMBED_DIM].astype(np.float32)
        size_emb = self._embedder._normalize(size_feat)

        # Visual crop embedding
        if rgb_crop is not None and rgb_crop.size > 0:
            vis_emb = self._embedder.embed_image(rgb_crop)
            combined = 0.5 * text_emb + 0.3 * vis_emb + 0.2 * size_emb
        else:
            combined = 0.7 * text_emb + 0.3 * size_emb

        return self._embedder._normalize(combined)

    def _ensure_index(self) -> None:
        """Build/rebuild FAISS flat-IP index when dirty."""
        if not self._index_dirty or not self._entries:
            return
        try:
            import faiss
            vecs = np.stack(
                [e["embedding"] for e in self._entries]
            ).astype(np.float32)
            faiss.normalize_L2(vecs)
            index = faiss.IndexFlatIP(EMBED_DIM)
            index.add(vecs)
            self._index = index
            self._index_dirty = False
        except ImportError:
            logger.debug("faiss not installed, skipping L1 index")

    @property
    def size(self) -> int:
        return len(self._entries)
