"""
PHANTOM-ECHO REVEAL — Shared Gaussian Data Structures
gaussian_format.py + payload_schema.py + constants.py (combined)

Single source of truth for:
  - Gaussian serialization format (edge ↔ cloud)
  - All numeric thresholds and configuration constants
  - Payload schema (pydantic-free, pure dataclass)
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json

# ── Confidence tag hierarchy ───────────────────────────────────────────────
TAG_WHITE  = "WHITE"   # ARKit high-confidence direct measurement
TAG_BLUE   = "BLUE"    # Physically proven by PHANTOM laws
TAG_TEAL   = "TEAL"    # Acoustically measured (SAS triangulation)
TAG_GREEN  = "GREEN"   # AI-generated (VideoScene)
TAG_YELLOW = "YELLOW"  # Physically probable (soft prior)
TAG_RED    = "RED"     # Unknown — beyond current knowledge
TAG_ORANGE = "ORANGE"  # Dynamic object (tracked separately)

TAG_ORDER = [TAG_WHITE, TAG_BLUE, TAG_TEAL, TAG_GREEN, TAG_YELLOW, TAG_RED, TAG_ORANGE]

# Log-odds sensor values per tag (for occupancy grid)
TAG_LOG_ODDS: Dict[str, float] = {
    TAG_WHITE:  3.0,
    TAG_BLUE:   2.5,
    TAG_TEAL:   2.0,
    TAG_GREEN:  1.5,
    TAG_YELLOW: 1.0,
    TAG_RED:    0.0,
    TAG_ORANGE: 0.5,   # BUG-3 FIX: was 0.0 — must match occupancy_grid.LOG_ODDS_SENSOR
                       # ("ORANGE": 0.5). A tracked moving person now contributes
                       # +0.5 occupancy to any costmap built from the shared table,
                       # not 0. Decays via LOG_ODDS_ORANGE_DECAY on next frame.
}

# ── All numeric constants from bible Section 24 ────────────────────────────
class Constants:
    # Acoustic
    SPEED_OF_SOUND_MS      = 343.0       # m/s at 20°C, 1 atm
    CHIRP_F_START_HZ       = 1_000.0
    CHIRP_F_END_HZ         = 22_000.0
    CHIRP_DURATION_S       = 0.02
    MIN_BASELINE_M         = 0.05        # minimum SAS baseline (5cm)
    ACOUSTIC_TOLERANCE_M   = 0.05        # ±5cm for PROVEN verdict

    # Confidence routing thresholds
    CONF_HIGH_THRESHOLD    = 0.75        # → WHITE
    CONF_LOW_THRESHOLD     = 0.40        # → RED

    # Semantic cache
    CACHE_SIM_THRESHOLD    = 0.85        # cosine similarity for cache HIT
    FAISS_SIM_THRESHOLD    = 0.92        # stricter FAISS cache threshold

    # Adaptive lambda scheduler
    LAMBDA_INIT            = 0.3
    LAMBDA_MIN             = 0.05
    LAMBDA_MAX             = 0.80
    LAMBDA_ALPHA           = 0.1         # increase rate
    LAMBDA_BETA            = 0.1         # decrease rate

    # Nav2 watchdog
    WATCHDOG_TIMEOUT_S     = 10.0
    WATCHDOG_MIN_PROGRESS  = 0.15        # meters

    # KPI targets
    KPI_F1_TARGET          = 0.97
    KPI_SEMANTIC_TARGET    = 0.93
    KPI_CHAMFER_TARGET_M   = 0.015       # 1.5cm

    # Atlas baseline
    ATLAS_F1               = 0.85
    ATLAS_SEMANTIC         = 0.80
    ATLAS_CHAMFER_M        = 0.05        # 5cm

    # DDGS
    DDGS_DISK_RATIO_THRESH = 0.1         # λ_min/λ_max < this → 2D disk
    DDGS_STRIDE_DEFAULT    = 4

    # Generation
    VIDEOSCENE_MAX_LATENCY_S = 3.0
    SVQ_TARGET_KB          = 300         # target payload size

    # SPSR
    SPSR_DEPTH             = 9
    SPSR_DENSITY_PCT       = 5.0

    # Atlanta-World regularization
    ATLANTA_LAMBDA         = 0.3
    ATLANTA_STRUCTURAL_THR = 0.85        # cos(angle) for axis snap


# ── Shared Gaussian wire format ────────────────────────────────────────────
@dataclass
class GaussianWire:
    """
    Minimal serializable Gaussian for edge↔cloud transmission.
    Use this everywhere instead of ad-hoc dicts.
    """
    position:   List[float]          # [x, y, z]
    normal:     List[float]          # [nx, ny, nz]
    color:      List[float]          # [r, g, b] in [0,1]
    scale:      float                # disk radius (meters)
    opacity:    float                # [0, 1]
    tag:        str                  # WHITE/BLUE/TEAL/GREEN/RED/ORANGE
    semantic:   str = "UNKNOWN"
    confidence: float = 1.0
    plane_d:    float = 0.0          # D in Ax+By+Cz+D=0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GaussianWire":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_numpy_position(self) -> np.ndarray:
        return np.array(self.position, dtype=np.float32)

    def to_numpy_normal(self) -> np.ndarray:
        return np.array(self.normal, dtype=np.float32)


# ── Scan payload (edge → cloud) ────────────────────────────────────────────
@dataclass
class ScanPayload:
    """Typed payload sent from edge to /scan endpoint."""
    session_id:        str
    frame_id:          int
    timestamp_s:       float
    depth_h:           int
    depth_w:           int
    depth_flat:        List[float]       # H*W float32
    confidence_flat:   List[int]         # H*W uint8
    rgb_flat:          List[int]         # H*W*3 uint8
    cam_to_world:      List[List[float]] # 4×4
    camera_intrinsics: Dict[str, float]  # fx, fy, cx, cy
    audio_flat:        Optional[List[float]] = None
    phone_position:    Optional[List[float]] = None  # [x, y, z]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "ScanPayload":
        d = json.loads(s)
        return cls(**d)


# ── Reveal payload (edge → cloud) ──────────────────────────────────────────
@dataclass
class RevealPayload:
    """Typed payload sent from edge to /reveal endpoint."""
    session_id:     str
    region_id:      str
    semantic:       str
    confidence_tag: str
    bbox_min:       List[float]   # [x, y, z]
    bbox_max:       List[float]   # [x, y, z]
    floor_y:        float = 0.0
    ceiling_y:      float = 2.5
    stereo_crop_a:  Optional[List[int]] = None   # flattened H*W*3 uint8
    stereo_crop_b:  Optional[List[int]] = None
    acoustic_point: Optional[List[float]] = None # [x, y, z] from SAS

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "RevealPayload":
        d = json.loads(s)
        return cls(**d)


# ── Reveal response (cloud → edge) ────────────────────────────────────────
@dataclass
class RevealResponse:
    """Response from /reveal — compressed or raw Gaussian list."""
    region_id:      str
    tag:            str
    gaussians:      List[Dict[str, Any]]   # list of GaussianWire dicts
    confidence:     float
    processing_ms:  float
    within_bounds:  bool = True
    compressed:     bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "RevealResponse":
        d = json.loads(s)
        return cls(**d)

    def get_gaussian_wires(self) -> List[GaussianWire]:
        return [GaussianWire.from_dict(g) for g in self.gaussians]
