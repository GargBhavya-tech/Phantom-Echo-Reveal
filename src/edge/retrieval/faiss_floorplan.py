"""
PHANTOM-ECHO REVEAL — FAISS Floor-Plan Retrieval (Section 5.1 — Missing Component 1)
======================================================================================

Implements the FAISS_RETRIEVAL strategy from the affordance router using a real
in-memory FAISS IndexFlatL2 index. Replaces the previous Tier-3 fallback that
silently ignored the routing decision and always used random templates.

Architecture
------------
• 12 hand-authored semantic wall/furniture Gaussian templates (compact point clouds).
• Each template is embedded by MobileCLIPEmbedder.embed_text() → 512-d float32.
• Index type: IndexFlatL2 (exact nearest-neighbour, sufficient for N=12 templates).
• At query time: embed the incoming FAISS query string → nearest-neighbour lookup
  → retrieve the matching Gaussian template → scale/translate to the query bbox.

Offline fallback
----------------
If FAISS or MobileCLIP is unavailable (e.g. first-run before optional deps are
installed), the module falls back to the hash-based template selection that was
previously used. This is logged at WARNING level so the engineer knows to install
the optional dependencies.

Usage
-----
    from src.edge.retrieval.faiss_floorplan import FAISSFloorPlanRetriever

    retriever = FAISSFloorPlanRetriever()          # builds index once, lazy
    gaussians = retriever.retrieve(
        query="wall width=3.2m height=2.5m depth=0.2m vertical flat rectangular",
        bbox_min=np.array([0., 0., 0.]),
        bbox_max=np.array([3.2, 2.5, 0.2]),
        floor_y=0.0, ceiling_y=2.5,
        max_gaussians=200,
    )
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Compact semantic templates (Gaussian point clouds) ────────────────────────
# Each template is defined in a normalised [0,1]³ space.  retrieve() rescales
# them to the query bbox.  Color is semantic-class colour (not photo-realistic).

def _make_flat_wall(n: int = 120) -> np.ndarray:
    """Flat vertical wall in XY plane (Z≈0), X∈[0,1], Y∈[0,1]."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 1, (n, 2))
    z  = rng.uniform(0.0, 0.05, n)
    return np.column_stack([xy[:, 0], xy[:, 1], z])


def _make_box(n: int = 80) -> np.ndarray:
    """Hollow box surface, 6 faces."""
    rng = np.random.default_rng(1)
    faces = []
    for _ in range(6):
        uv = rng.uniform(0, 1, (n // 6, 2))
        face_pts = np.column_stack([uv[:, 0], uv[:, 1], np.zeros(len(uv))])
        faces.append(face_pts)
    pts = np.concatenate(faces)
    # Randomly rotate faces to form a box surface
    axes = [
        (0, 1, 2), (0, 2, 1), (1, 2, 0),
        (0, 1, 2), (0, 2, 1), (1, 2, 0),
    ]
    result = []
    for i, ax in enumerate(axes[:len(faces)]):
        p = faces[i][:, list(ax)]
        if i >= 3:
            p[:, ax.index(2)] = 1.0   # far face
        result.append(p)
    return np.clip(np.concatenate(result), 0, 1)


def _make_chair(n: int = 80) -> np.ndarray:
    """Chair: seat at y=0.45, backrest, 4 legs."""
    rng = np.random.default_rng(2)
    pts = []
    # seat
    xy = rng.uniform(0.15, 0.85, (n // 3, 2))
    pts.append(np.column_stack([xy[:, 0], np.full(n // 3, 0.45), xy[:, 1]]))
    # backrest
    xy2 = rng.uniform(0.15, 0.85, (n // 4, 2))
    pts.append(np.column_stack([xy2[:, 0], np.linspace(0.45, 0.95, n // 4), np.full(n // 4, 0.05)]))
    # legs (4 corners)
    for lx, lz in [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]:
        ys = np.linspace(0.0, 0.45, n // 16 + 1)
        pts.append(np.column_stack([np.full_like(ys, lx), ys, np.full_like(ys, lz)]))
    return np.clip(np.concatenate(pts), 0, 1)


def _make_table(n: int = 100) -> np.ndarray:
    """Table: flat top at y=0.75, 4 legs."""
    rng = np.random.default_rng(3)
    pts = []
    # surface
    xy = rng.uniform(0.05, 0.95, (n // 2, 2))
    pts.append(np.column_stack([xy[:, 0], np.full(n // 2, 0.75), xy[:, 1]]))
    for lx, lz in [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]:
        ys = np.linspace(0.0, 0.75, n // 16 + 1)
        pts.append(np.column_stack([np.full_like(ys, lx), ys, np.full_like(ys, lz)]))
    return np.clip(np.concatenate(pts), 0, 1)


def _make_shelf(n: int = 120) -> np.ndarray:
    """Shelf unit: 4 horizontal shelves + back wall."""
    rng = np.random.default_rng(4)
    pts = []
    for sh_y in [0.20, 0.42, 0.64, 0.86]:
        xy = rng.uniform(0.05, 0.95, (n // 5, 2))
        pts.append(np.column_stack([xy[:, 0], np.full(n // 5, sh_y), xy[:, 1]]))
    # back panel
    xy = rng.uniform(0.05, 0.95, (n // 5, 2))
    pts.append(np.column_stack([xy[:, 0], xy[:, 1], np.full(n // 5, 0.95)]))
    return np.clip(np.concatenate(pts), 0, 1)


def _make_sofa(n: int = 100) -> np.ndarray:
    """Sofa: deep seat, high backrest, two armrests."""
    rng = np.random.default_rng(5)
    pts = []
    # seat cushion (wide, shallow)
    xy = rng.uniform(0.0, 1.0, (n // 3, 2))
    pts.append(np.column_stack([xy[:, 0], np.full(n // 3, 0.42), xy[:, 1] * 0.55]))
    # backrest
    xb = rng.uniform(0.0, 1.0, (n // 4,))
    pts.append(np.column_stack([xb, np.linspace(0.42, 0.85, n // 4), np.full(n // 4, 0.52)]))
    # armrests
    for ax in [0.02, 0.96]:
        yz = rng.uniform(0.0, 0.55, (n // 8, 2))
        pts.append(np.column_stack([np.full(n // 8, ax), yz[:, 0] + 0.30, yz[:, 1]]))
    return np.clip(np.concatenate(pts), 0, 1)


def _make_door(n: int = 80) -> np.ndarray:
    """Door: tall flat panel with door-knob detail."""
    rng = np.random.default_rng(6)
    xy = rng.uniform(0.04, 0.96, (n - 5, 2))
    pts = [np.column_stack([xy[:, 0], xy[:, 1], np.zeros(n - 5)])]
    # door knob at (0.85, 0.50)
    ang = np.linspace(0, 2 * np.pi, 5)
    pts.append(np.column_stack([0.85 + 0.02 * np.cos(ang),
                                 0.50 + 0.02 * np.sin(ang),
                                 np.full(5, -0.05)]))
    return np.clip(np.concatenate(pts), 0, 1)


def _make_cabinet(n: int = 100) -> np.ndarray:
    """Cabinet: box with 2 doors."""
    rng = np.random.default_rng(7)
    # Body
    pts = [_make_box(n // 2)]
    # Door seam
    pts.append(np.column_stack([np.full(10, 0.5), np.linspace(0, 1, 10), np.zeros(10)]))
    return np.clip(np.concatenate(pts), 0, 1)


def _make_floor(n: int = 150) -> np.ndarray:
    """Floor: flat XZ plane at y=0."""
    rng = np.random.default_rng(8)
    xz = rng.uniform(0, 1, (n, 2))
    return np.column_stack([xz[:, 0], np.zeros(n), xz[:, 1]])


def _make_ceiling(n: int = 150) -> np.ndarray:
    """Ceiling: flat XZ plane at y=1."""
    rng = np.random.default_rng(9)
    xz = rng.uniform(0, 1, (n, 2))
    return np.column_stack([xz[:, 0], np.ones(n), xz[:, 1]])


def _make_plant(n: int = 80) -> np.ndarray:
    """Plant: pot (cylinder) + leafy sphere cloud."""
    rng = np.random.default_rng(10)
    # pot cylinder bottom
    ang = np.linspace(0, 2 * np.pi, 20)
    pot = np.column_stack([0.5 + 0.15 * np.cos(ang),
                            np.linspace(0.0, 0.25, 20),
                            0.5 + 0.15 * np.sin(ang)])
    # foliage sphere
    th = rng.uniform(0, 2 * np.pi, n - 20)
    ph = rng.uniform(-np.pi / 2, np.pi / 2, n - 20)
    r  = rng.uniform(0.2, 0.35, n - 20)
    foliage = np.column_stack([
        0.5 + r * np.cos(ph) * np.cos(th),
        0.45 + r * np.sin(ph) + 0.3,
        0.5 + r * np.cos(ph) * np.sin(th),
    ])
    return np.clip(np.concatenate([pot, foliage]), 0, 1)


def _make_monitor(n: int = 60) -> np.ndarray:
    """Monitor: flat rectangle on a small stand."""
    rng = np.random.default_rng(11)
    xy = rng.uniform(0.05, 0.95, (n - 10, 2))
    screen = np.column_stack([xy[:, 0], xy[:, 1] * 0.7 + 0.3, np.zeros(n - 10)])
    # stand
    ys = np.linspace(0.0, 0.30, 10)
    stand = np.column_stack([np.full(10, 0.5), ys, np.full(10, 0.0)])
    return np.clip(np.concatenate([screen, stand]), 0, 1)


# ── Template registry ─────────────────────────────────────────────────────────

_SEMANTIC_COLOR = {
    "WALL":     [0.72, 0.74, 0.78],
    "BOX":      [0.60, 0.55, 0.50],
    "CHAIR":    [0.55, 0.40, 0.30],
    "TABLE":    [0.65, 0.55, 0.42],
    "SHELF":    [0.58, 0.52, 0.45],
    "SOFA":     [0.45, 0.45, 0.62],
    "DOOR":     [0.70, 0.60, 0.50],
    "CABINET":  [0.60, 0.55, 0.48],
    "FLOOR":    [0.55, 0.58, 0.62],
    "CEILING":  [0.80, 0.80, 0.78],
    "PLANT":    [0.25, 0.60, 0.25],
    "MONITOR":  [0.15, 0.15, 0.20],
}

_TEMPLATES: List[Dict[str, Any]] = [
    {"semantic": "WALL",    "pts": _make_flat_wall,  "query": "wall vertical flat rectangular"},
    {"semantic": "BOX",     "pts": _make_box,         "query": "box cube object container"},
    {"semantic": "CHAIR",   "pts": _make_chair,       "query": "chair seat legs backrest furniture"},
    {"semantic": "TABLE",   "pts": _make_table,       "query": "table surface legs flat furniture"},
    {"semantic": "SHELF",   "pts": _make_shelf,       "query": "shelf unit storage horizontal levels"},
    {"semantic": "SOFA",    "pts": _make_sofa,        "query": "sofa couch seat backrest armrest"},
    {"semantic": "DOOR",    "pts": _make_door,        "query": "door flat panel vertical handle"},
    {"semantic": "CABINET", "pts": _make_cabinet,     "query": "cabinet storage box doors"},
    {"semantic": "FLOOR",   "pts": _make_floor,       "query": "floor flat horizontal surface"},
    {"semantic": "CEILING", "pts": _make_ceiling,     "query": "ceiling flat horizontal surface top"},
    {"semantic": "PLANT",   "pts": _make_plant,       "query": "plant pot foliage sphere organic"},
    {"semantic": "MONITOR", "pts": _make_monitor,     "query": "monitor screen display rectangle"},
]


# ── Embedder (lazy-loaded) ────────────────────────────────────────────────────

def _embed_text(text: str) -> Optional[np.ndarray]:
    """Try MobileCLIP text embedding; return None on failure."""
    try:
        from src.edge.embedding.mobile_clip import MobileCLIPEmbedder
        emb = MobileCLIPEmbedder()
        return emb.embed_text(text)
    except Exception as e:
        logger.debug(f"FAISS embed_text failed: {e}")
        return None


def _hash_embed(text: str, dim: int = 512) -> np.ndarray:
    """Deterministic hash-based pseudo-embedding for offline fallback.

    BUGFIX: the previous implementation did
        np.frombuffer(hash_bytes, dtype=np.float32)
    which REINTERPRETS arbitrary SHA-256 bytes as IEEE-754 floats. Most byte
    patterns decode to denormals, ±inf or NaN, so the "embedding" contained
    non-finite values and np.linalg.norm overflowed (RuntimeWarning) — and the
    resulting vector was meaningless for cosine similarity. We now read the
    bytes as UNSIGNED integers (always finite) and centre them to [-0.5, 0.5),
    giving a stable, normalisable pseudo-embedding.
    """
    h = hashlib.sha256(text.encode()).digest()
    raw = (h * ((dim // 32) + 1))[:dim]               # dim bytes, 0..255
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) / 255.0 - 0.5
    norm = float(np.linalg.norm(arr))
    return (arr / (norm + 1e-9)).astype(np.float32)


# ── Main class ────────────────────────────────────────────────────────────────

class FAISSFloorPlanRetriever:
    """FAISS-backed floor-plan / furniture geometry retrieval.

    Thread-safe: index is built once at construction and is read-only after.
    """

    def __init__(self):
        self._index  = None
        self._embeds: Optional[np.ndarray] = None   # (N, 512) query embeddings
        self._use_faiss = False
        self._build_index()

    def _build_index(self) -> None:
        """Build FAISS index from template query embeddings."""
        # Embed each template's canonical query string
        embed_fn = _embed_text
        embeds = []
        for t in _TEMPLATES:
            v = embed_fn(t["query"])
            if v is None:
                v = _hash_embed(t["query"])
            embeds.append(v.astype(np.float32))

        self._embeds = np.stack(embeds)   # (12, dim)

        try:
            import faiss
            dim = self._embeds.shape[1]
            self._index = faiss.IndexFlatL2(dim)
            self._index.add(self._embeds)
            self._use_faiss = True
            logger.info(
                f"FAISSFloorPlanRetriever: built IndexFlatL2 "
                f"({len(_TEMPLATES)} templates, dim={dim})"
            )
        except ImportError:
            logger.warning(
                "faiss-cpu not installed — FAISSFloorPlanRetriever falling back "
                "to cosine-similarity retrieval (pip install faiss-cpu to fix)"
            )

    def _query_embed(self, query: str) -> np.ndarray:
        v = _embed_text(query)
        if v is None:
            v = _hash_embed(query)
        return v.astype(np.float32)

    def _best_match(self, query_vec: np.ndarray) -> int:
        """Return the index of the best-matching template."""
        if self._use_faiss and self._index is not None:
            D, I = self._index.search(query_vec[np.newaxis], k=1)
            return int(I[0, 0])
        else:
            # Cosine similarity fallback
            norms = np.linalg.norm(self._embeds, axis=1) + 1e-9
            q_norm = np.linalg.norm(query_vec) + 1e-9
            scores = self._embeds @ query_vec / (norms * q_norm)
            return int(np.argmax(scores))

    def retrieve(self,
                 query: str,
                 bbox_min: np.ndarray,
                 bbox_max: np.ndarray,
                 floor_y: float = 0.0,
                 ceiling_y: float = 2.5,
                 max_gaussians: int = 200,
                 seed: int = 0) -> List[Dict[str, Any]]:
        """Retrieve the best-matching geometry template, scaled to bbox.

        Args:
            query:          FAISS query string (from affordance_router)
            bbox_min/max:   World-space bounding box to fill
            floor_y/ceiling_y: Room height constraints
            max_gaussians:  Maximum number of Gaussians to return
            seed:           RNG seed for stochastic jitter

        Returns:
            List of Gaussian dicts in PHANTOM wire format
        """
        try:
            q_vec = self._query_embed(query)
            idx   = self._best_match(q_vec)
            tmpl  = _TEMPLATES[idx]
            semantic = tmpl["semantic"]
            logger.info(
                f"FAISSFloorPlanRetriever: query='{query[:60]}' "
                f"→ template[{idx}] '{semantic}'"
            )

            # Generate template point cloud (in [0,1]³)
            rng = np.random.default_rng(seed)
            raw_pts = tmpl["pts"](n=min(max_gaussians, 200))

            # Rescale [0,1]³ → bbox
            extent = bbox_max - bbox_min
            # guard degenerate bboxes
            extent = np.where(extent < 0.05, 0.05, extent)
            pts = raw_pts * extent + bbox_min

            # Clamp Y to room bounds (physics law)
            pts[:, 1] = np.clip(pts[:, 1], floor_y, ceiling_y)

            # Add small random jitter for visual variety
            pts += rng.normal(0, 0.005, pts.shape)

            color = _SEMANTIC_COLOR.get(semantic, [0.65, 0.60, 0.55])
            gaussians = []
            for p in pts[:max_gaussians]:
                gaussians.append({
                    "position": p.tolist(),
                    "normal":   [0.0, 1.0, 0.0],
                    "color":    color,
                    "scale":    0.04,
                    "opacity":  0.90,
                    "confidence": 0.70,
                    "tag":      "GREEN",
                    "semantic": semantic,
                    "_faiss_retrieved": True,
                    "_template_idx": idx,
                })
            logger.info(
                f"FAISSFloorPlanRetriever: returning {len(gaussians)} Gaussians "
                f"for '{semantic}' from template[{idx}]"
            )
            return gaussians

        except Exception as e:
            logger.warning(f"FAISSFloorPlanRetriever.retrieve() failed: {e}")
            return []

    def retrieve_for_routing(self,
                              decision,
                              floor_y: float = 0.0,
                              ceiling_y: float = 2.5,
                              seed: int = 0) -> List[Dict[str, Any]]:
        """Convenience wrapper: takes a RoutingDecision from affordance_router."""
        bmin = decision.physics_bounds.min_pt
        bmax = decision.physics_bounds.max_pt
        query = decision.faiss_query or _build_default_query(decision.semantic, bmin, bmax)
        return self.retrieve(query, bmin, bmax, floor_y, ceiling_y, seed=seed)


def _build_default_query(semantic: str, bmin: np.ndarray, bmax: np.ndarray) -> str:
    w = bmax[0] - bmin[0]
    h = bmax[1] - bmin[1]
    d = bmax[2] - bmin[2]
    return f"{semantic.lower()} width={w:.1f}m height={h:.1f}m depth={d:.1f}m"


# Module-level singleton (lazy init on first import)
_retriever: Optional[FAISSFloorPlanRetriever] = None


def get_retriever() -> FAISSFloorPlanRetriever:
    """Return the module-level singleton retriever (built once)."""
    global _retriever
    if _retriever is None:
        _retriever = FAISSFloorPlanRetriever()
    return _retriever
