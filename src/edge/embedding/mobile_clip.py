"""
PHANTOM-ECHO REVEAL — MobileCLIP Semantic Embedder
mobile_clip.py

Produces 512-d semantic embeddings from:
    - RGB image crops (visual branch)
    - Text labels (text branch)

Used by:
    cache_checker.py  — cosine similarity vs Redis cache
    affordance_router.py — FAISS semantic routing
    slot_lstm.py      — appearance matching

Three-tier backend:
    Tier 1: mobileclip package (3–15ms, preferred)
    Tier 2: openai/clip-vit-base-patch32 via transformers (~100ms)
    Tier 3: deterministic hash embedding (no GPU, always works)
"""

import numpy as np
import logging
import hashlib
from typing import Optional, List, Union

logger = logging.getLogger(__name__)

EMBED_DIM = 512


class MobileCLIPEmbedder:
    """
    Unified visual + text embedder with automatic backend selection.
    Singleton pattern: one instance shared across the process.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._backend = "hash"
        self._try_mobileclip()
        if self._backend == "hash":
            self._try_clip()
        self._initialized = True
        logger.info(f"MobileCLIPEmbedder: backend={self._backend}")

    def _try_mobileclip(self):
        try:
            import mobileclip
            self._model, _, self._processor = mobileclip.create_model_and_transforms(
                "mobileclip_s0", pretrained=True
            )
            self._model.eval()
            self._backend = "mobileclip"
        except Exception as e:
            logger.debug(f"MobileCLIP unavailable: {e}")

    def _try_clip(self):
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._model.eval()
            self._backend = "clip"
            logger.warning("Using CLIP (~100ms). MobileCLIP preferred for demo.")
        except Exception as e:
            logger.debug(f"CLIP unavailable: {e}")

    # ── Image embedding ────────────────────────────────────────────────────
    def embed_image(self, rgb_crop: np.ndarray) -> np.ndarray:
        """
        Embed an RGB crop → (512,) float32 unit vector.

        Args:
            rgb_crop: (H, W, 3) uint8

        Returns:
            (512,) float32 normalized embedding
        """
        if self._backend == "mobileclip":
            return self._embed_image_mobileclip(rgb_crop)
        elif self._backend == "clip":
            return self._embed_image_clip(rgb_crop)
        else:
            return self._embed_hash(rgb_crop.tobytes())

    def _embed_image_mobileclip(self, rgb_crop: np.ndarray) -> np.ndarray:
        try:
            import torch
            from PIL import Image
            img = Image.fromarray(rgb_crop)
            x = self._processor(img).unsqueeze(0)
            with torch.no_grad():
                feat = self._model.encode_image(x)
            v = feat.squeeze().cpu().numpy().astype(np.float32)
            return self._normalize(v)
        except Exception as e:
            logger.error(f"MobileCLIP image embed failed: {e}")
            return self._embed_hash(rgb_crop.tobytes())

    def _embed_image_clip(self, rgb_crop: np.ndarray) -> np.ndarray:
        try:
            import torch
            from PIL import Image
            img = Image.fromarray(rgb_crop)
            inputs = self._processor(images=img, return_tensors="pt")
            with torch.no_grad():
                feat = self._model.get_image_features(**inputs)
            v = feat.squeeze().cpu().numpy().astype(np.float32)
            return self._normalize(v)
        except Exception as e:
            logger.error(f"CLIP image embed failed: {e}")
            return self._embed_hash(rgb_crop.tobytes())

    # ── Text embedding ─────────────────────────────────────────────────────
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text label → (512,) float32 unit vector.
        """
        if self._backend == "mobileclip":
            return self._embed_text_mobileclip(text)
        elif self._backend == "clip":
            return self._embed_text_clip(text)
        else:
            return self._embed_hash(text.encode())

    def _embed_text_mobileclip(self, text: str) -> np.ndarray:
        try:
            import torch, mobileclip
            tokens = mobileclip.tokenize([text])
            with torch.no_grad():
                feat = self._model.encode_text(tokens)
            v = feat.squeeze().cpu().numpy().astype(np.float32)
            return self._normalize(v)
        except Exception as e:
            logger.error(f"MobileCLIP text embed failed: {e}")
            return self._embed_hash(text.encode())

    def _embed_text_clip(self, text: str) -> np.ndarray:
        try:
            import torch
            inputs = self._processor(text=[text], return_tensors="pt", padding=True)
            with torch.no_grad():
                feat = self._model.get_text_features(**inputs)
            v = feat.squeeze().cpu().numpy().astype(np.float32)
            return self._normalize(v)
        except Exception as e:
            logger.error(f"CLIP text embed failed: {e}")
            return self._embed_hash(text.encode())

    def embed_texts_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embed a list of texts → (N, 512) float32."""
        return np.stack([self.embed_text(t) for t in texts])

    # ── Similarity ─────────────────────────────────────────────────────────
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two unit vectors."""
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-9)

    def _embed_hash(self, data: bytes) -> np.ndarray:
        """
        Deterministic hash embedding (no GPU, always works).
        Produces consistent unit vector from any byte sequence.
        """
        digest = hashlib.sha256(data).digest()
        rng = np.random.default_rng(np.frombuffer(digest[:8], dtype=np.uint64)[0])
        v = rng.normal(0, 1, EMBED_DIM).astype(np.float32)
        return self._normalize(v)

    @property
    def backend(self) -> str:
        return self._backend


# Module-level singleton accessor
def get_embedder() -> MobileCLIPEmbedder:
    return MobileCLIPEmbedder()
