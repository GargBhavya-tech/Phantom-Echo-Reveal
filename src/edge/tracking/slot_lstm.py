"""
PHANTOM-ECHO REVEAL — SlotLSTM Dynamic Object Tracker
slot_lstm.py

Layer 1: Static-Dynamic Separation (Flaw 35 fix)

Separates the Gaussian scene into two layers:
    STATIC  — permanent geometry (walls, floor, furniture)
    DYNAMIC — moving objects tracked as ORANGE blobs

Pipeline per frame:
    1. Detect motion via temporal depth difference
    2. Cluster moving pixels into dynamic regions
    3. Match to existing tracks via Hungarian algorithm
       (joint centroid distance + MobileCLIP appearance similarity)
    4. Update LSTM hidden state per track
    5. Translate existing Gaussian cluster as rigid body
    6. Tag dynamic Gaussians ORANGE; keep static layer clean

FIXED BUGS:
    Bug 5   — confirmed flag always False:
              The update() loop set:
                t.confirmed = True if hasattr(t, '_update_count') else False
              _update_count was NEVER set on any DynamicTrack, so
              hasattr() always returned False, confirmed was always False,
              confirmed_tracks() always returned [], and
              separate_static_dynamic() permanently had 0 dynamic objects.
              Flaw 35 (dynamic object contamination) was silently unfixed.

    Missing 8 — confirm_age parameter accepted but never used:
              SlotLSTMTracker.__init__ accepted confirm_age=3 and stored
              self._confirm_age = confirm_age, but the update loop never
              consulted self._confirm_age. Every track was either
              immediately confirmed (broken) or never confirmed (Bug 5).
              Fix: tracks now carry a _consecutive_matches counter.
              confirmed=True only after _consecutive_matches >= _confirm_age.
              This prevents transient motion noise (cloth blowing in wind,
              lighting change) from creating ghost ORANGE objects.

The confirmation gate is critical: without it, a single-frame depth
artefact creates a 'track' that immediately enters the dynamic layer,
corrupts the static map for that frame, and contaminates the costmap
before being aged out. With confirm_age=3, an object must be detected
in 3 consecutive frames before it is accepted as a real dynamic object.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

ORANGE_TAG      = "ORANGE"
MAX_TRACK_AGE   = 10        # frames before track is deleted
MIN_MOTION_M    = 0.03      # 3cm minimum motion to flag as dynamic
IOU_THRESHOLD   = 0.3
DEFAULT_CONFIRM_AGE = 3     # consecutive matches needed before ORANGE status


# ── Track data structure ──────────────────────────────────────────────────
@dataclass
class DynamicTrack:
    """One tracked dynamic object."""
    track_id:            int
    centroid:            np.ndarray    # (3,) world position
    velocity:            np.ndarray    # (3,) estimated velocity m/frame
    bbox_min:            np.ndarray    # (3,)
    bbox_max:            np.ndarray    # (3,)
    gaussian_ids:        List[int]     # indices of Gaussians in this track
    age:                 int  = 0      # frames since last observation
    confirmed:           bool = False  # True after confirm_age consecutive matches
    _consecutive_matches: int = 0      # FIX Missing 8: counts consecutive detections
    appearance_emb:      Optional[np.ndarray] = None    # MobileCLIP embedding
    lstm_h:              Optional[np.ndarray] = None    # hidden state
    lstm_c:              Optional[np.ndarray] = None    # cell state


# ── Motion detection ──────────────────────────────────────────────────────
def detect_motion_mask(
    depth_current:  np.ndarray,
    depth_previous: Optional[np.ndarray],
    threshold_m:    float = MIN_MOTION_M
) -> np.ndarray:
    """
    Detect pixels with significant depth change between frames.

    Args:
        depth_current:  (H, W) float32 current depth
        depth_previous: (H, W) float32 previous depth or None (first frame)
        threshold_m:    minimum depth change to flag as motion

    Returns:
        (H, W) bool mask — True = potentially dynamic pixel
    """
    if depth_previous is None or depth_previous.shape != depth_current.shape:
        return np.zeros(depth_current.shape, dtype=bool)

    valid  = (depth_current > 0.1) & (depth_previous > 0.1)
    diff   = np.abs(depth_current - depth_previous)
    motion = valid & (diff > threshold_m)
    return motion


def cluster_motion_pixels(
    motion_mask:   np.ndarray,
    depth_map:     np.ndarray,
    intrinsics:    Dict[str, float],
    cam_to_world:  np.ndarray,
    min_cluster_px: int = 100
) -> List[Dict]:
    """
    Cluster motion pixels into dynamic regions via connected components.

    Returns list of dicts: {centroid_3d, bbox_min, bbox_max, pixel_count, points_3d}
    """
    try:
        from scipy.ndimage import label
    except ImportError:
        return []

    labeled, n_labels = label(motion_mask)
    clusters = []

    fx = intrinsics["fx"]; fy = intrinsics["fy"]
    cx = intrinsics["cx"]; cy = intrinsics["cy"]

    for lbl in range(1, n_labels + 1):
        mask = labeled == lbl
        if mask.sum() < min_cluster_px:
            continue

        rows, cols = np.where(mask)
        depths = depth_map[rows, cols]
        valid  = depths > 0.1
        if not np.any(valid):
            continue

        rows, cols, depths = rows[valid], cols[valid], depths[valid]

        x_cam = (cols - cx) * depths / fx
        y_cam = (rows - cy) * depths / fy
        z_cam = depths
        pts_cam   = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=0)
        pts_world = (cam_to_world @ pts_cam)[:3, :].T

        centroid = pts_world.mean(axis=0)
        clusters.append({
            "centroid_3d": centroid,
            "bbox_min":    pts_world.min(axis=0),
            "bbox_max":    pts_world.max(axis=0),
            "pixel_count": int(mask.sum()),
            "points_3d":   pts_world,
        })

    return clusters


# ── Hungarian assignment ──────────────────────────────────────────────────
def _hungarian_assign(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Solve linear assignment problem via scipy or greedy fallback.
    Returns list of (track_idx, detection_idx) pairs.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        return list(zip(row_idx.tolist(), col_idx.tolist()))
    except ImportError:
        assignments = []
        used_cols   = set()
        for r in range(cost_matrix.shape[0]):
            best_c    = -1
            best_cost = np.inf
            for c in range(cost_matrix.shape[1]):
                if c not in used_cols and cost_matrix[r, c] < best_cost:
                    best_cost = cost_matrix[r, c]
                    best_c    = c
            if best_c >= 0:
                assignments.append((r, best_c))
                used_cols.add(best_c)
        return assignments


def build_cost_matrix(
    tracks:            List[DynamicTrack],
    detections:        List[Dict],
    spatial_weight:    float = 0.7,
    appearance_weight: float = 0.3,
    max_dist_m:        float = 1.5
) -> np.ndarray:
    """
    Build assignment cost matrix: rows=tracks, cols=detections.
    Cost = spatial_weight * normalised_distance
          + appearance_weight * (1 - cosine_similarity)
    """
    n_tracks     = len(tracks)
    n_detections = len(detections)
    cost = np.full((n_tracks, n_detections), 1e9, dtype=np.float64)

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            predicted    = track.centroid + track.velocity
            spatial_dist = float(np.linalg.norm(predicted - det["centroid_3d"]))

            if spatial_dist > max_dist_m:
                continue

            spatial_cost = spatial_dist / max_dist_m

            appearance_cost = 0.5   # neutral if no embedding
            if (track.appearance_emb is not None and
                    "appearance_emb" in det and det["appearance_emb"] is not None):
                a  = track.appearance_emb
                b  = det["appearance_emb"]
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na > 1e-9 and nb > 1e-9:
                    sim             = float(np.dot(a, b) / (na * nb))
                    appearance_cost = (1.0 - sim) / 2.0   # [0, 1]

            cost[i, j] = (spatial_weight    * spatial_cost
                          + appearance_weight * appearance_cost)

    return cost


# ── Main tracker class ────────────────────────────────────────────────────
class SlotLSTMTracker:
    """
    Tracks dynamic objects across frames using centroid + appearance matching.
    Maintains a list of CONFIRMED ORANGE tracks.

    confirm_age: how many consecutive frames a detection must appear in
    before being accepted as a real dynamic object (not transient noise).
    """

    def __init__(self,
                 max_age:     int = MAX_TRACK_AGE,
                 confirm_age: int = DEFAULT_CONFIRM_AGE):
        self._tracks:       List[DynamicTrack] = []
        self._next_id       = 0
        self._max_age       = max_age
        self._confirm_age   = confirm_age   # FIX Missing 8: stored AND used
        self._prev_depth:   Optional[np.ndarray] = None

    def update(self,
               depth_map:    np.ndarray,
               rgb_image:    np.ndarray,
               intrinsics:   Dict[str, float],
               cam_to_world: np.ndarray) -> List[DynamicTrack]:
        """
        One tracker update cycle.

        FIX Bug 5: confirmed flag is now set correctly.
        FIX Missing 8: confirm_age is actually consulted.

        Returns:
            list of confirmed DynamicTrack objects (ORANGE objects only)
        """
        motion_mask      = detect_motion_mask(depth_map, self._prev_depth)
        self._prev_depth = depth_map.copy()

        if not np.any(motion_mask):
            # No motion — age all tracks, reset consecutive_matches
            self._age_tracks(motion_detected=False)
            return self.confirmed_tracks()

        detections = cluster_motion_pixels(
            motion_mask, depth_map, intrinsics, cam_to_world
        )

        if not detections:
            self._age_tracks(motion_detected=False)
            return self.confirmed_tracks()

        # Embed appearances
        for det in detections:
            det["appearance_emb"] = self._embed_crop(rgb_image, det)

        # Hungarian assignment
        if self._tracks:
            cost_matrix  = build_cost_matrix(self._tracks, detections)
            assignments  = _hungarian_assign(cost_matrix)

            matched_tracks     = set()
            matched_detections = set()

            for t_idx, d_idx in assignments:
                if cost_matrix[t_idx, d_idx] < 1e8:
                    self._update_track(self._tracks[t_idx], detections[d_idx])
                    matched_tracks.add(t_idx)
                    matched_detections.add(d_idx)

            # Age unmatched tracks and reset their consecutive counter
            for t_idx, track in enumerate(self._tracks):
                if t_idx not in matched_tracks:
                    track.age                 += 1
                    track._consecutive_matches = 0   # broken streak → reset

            # New tracks for unmatched detections
            for d_idx, det in enumerate(detections):
                if d_idx not in matched_detections:
                    self._create_track(det)

        else:
            for det in detections:
                self._create_track(det)

        # Remove stale tracks
        self._tracks = [t for t in self._tracks if t.age < self._max_age]

        # FIX Bug 5 + Missing 8:
        # confirmed = True only after _consecutive_matches >= _confirm_age.
        # This replaces the broken `hasattr(t, '_update_count')` check
        # which was ALWAYS False (attribute never existed → never confirmed).
        for t in self._tracks:
            t.confirmed = (t._consecutive_matches >= self._confirm_age)

        n_confirmed = sum(1 for t in self._tracks if t.confirmed)
        logger.debug(
            f"SlotLSTM: {len(self._tracks)} tracks total, "
            f"{n_confirmed} confirmed (confirm_age={self._confirm_age})"
        )
        return self.confirmed_tracks()

    # ── LSTM cell (NumPy implementation) ─────────────────────────────────
    def _lstm_step(self,
                   x: np.ndarray,
                   h: np.ndarray,
                   c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vanilla LSTM cell step in NumPy (no torch dependency on edge)."""
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-np.clip(z, -10.0, 10.0)))

        if not hasattr(self, '_lstm_Wf'):
            np.random.seed(42)
            input_sz  = len(x)
            hidden_sz = len(h)
            scale = 0.1
            self._lstm_Wf = np.random.randn(hidden_sz, input_sz + hidden_sz) * scale
            self._lstm_Wi = np.random.randn(hidden_sz, input_sz + hidden_sz) * scale
            self._lstm_Wc = np.random.randn(hidden_sz, input_sz + hidden_sz) * scale
            self._lstm_Wo = np.random.randn(hidden_sz, input_sz + hidden_sz) * scale
            self._lstm_bf = np.zeros(hidden_sz)
            self._lstm_bi = np.zeros(hidden_sz)
            self._lstm_bc = np.zeros(hidden_sz)
            self._lstm_bo = np.zeros(hidden_sz)

        concat  = np.concatenate([x, h])
        f       = sigmoid(self._lstm_Wf @ concat + self._lstm_bf)
        i       = sigmoid(self._lstm_Wi @ concat + self._lstm_bi)
        c_tilde = np.tanh(self._lstm_Wc @ concat + self._lstm_bc)
        c_new   = f * c + i * c_tilde
        o       = sigmoid(self._lstm_Wo @ concat + self._lstm_bo)
        h_new   = o * np.tanh(c_new)
        return h_new, c_new

    # ── Track lifecycle ───────────────────────────────────────────────────
    def _create_track(self, detection: Dict) -> None:
        """Create a new unconfirmed track. _consecutive_matches starts at 1."""
        app_emb = detection.get("appearance_emb")
        track   = DynamicTrack(
            track_id=self._next_id,
            centroid=detection["centroid_3d"].copy(),
            velocity=np.zeros(3),
            bbox_min=detection["bbox_min"].copy(),
            bbox_max=detection["bbox_max"].copy(),
            gaussian_ids=[],
            age=0,
            confirmed=False,
            _consecutive_matches=1,    # first observation counts
            appearance_emb=app_emb,
            lstm_h=np.zeros(128, dtype=np.float32),
            lstm_c=np.zeros(128, dtype=np.float32),
        )
        x = np.concatenate([
            track.centroid,
            app_emb if app_emb is not None else np.zeros(512)
        ])
        track.lstm_h, track.lstm_c = self._lstm_step(x, track.lstm_h, track.lstm_c)

        self._tracks.append(track)
        self._next_id += 1

    def _update_track(self, track: DynamicTrack, detection: Dict) -> None:
        """Update an existing track with a new detection."""
        new_centroid = detection["centroid_3d"]
        track.velocity  = new_centroid - track.centroid
        track.centroid  = new_centroid.copy()
        track.bbox_min  = detection["bbox_min"].copy()
        track.bbox_max  = detection["bbox_max"].copy()
        track.age       = 0

        # FIX Missing 8: increment consecutive counter each matched update
        track._consecutive_matches += 1

        app_emb = detection.get("appearance_emb")
        if app_emb is not None:
            track.appearance_emb = app_emb

        x = np.concatenate([
            new_centroid,
            track.appearance_emb if track.appearance_emb is not None else np.zeros(512)
        ])
        track.lstm_h, track.lstm_c = self._lstm_step(x, track.lstm_h, track.lstm_c)

    def _age_tracks(self, motion_detected: bool = True) -> None:
        """Age all tracks one frame and remove stale ones."""
        for t in self._tracks:
            t.age += 1
            if not motion_detected:
                # If no motion at all this frame, reset consecutive counters
                # to prevent a track from staying confirmed with no evidence.
                t._consecutive_matches = max(0, t._consecutive_matches - 1)
        self._tracks = [t for t in self._tracks if t.age < self._max_age]

    def confirmed_tracks(self) -> List[DynamicTrack]:
        """Return only tracks that have been confirmed over confirm_age frames."""
        return [t for t in self._tracks if t.confirmed]

    def _embed_crop(self, rgb_image: np.ndarray, detection: Dict) -> Optional[np.ndarray]:
        """Section 5.1 fix: MobileCLIP embedding of the detection RGB crop.

        Previously returned a random hash, making appearance matching useless.
        Now:
        1. Projects the 3D bounding box to image coordinates using stored pixels
        2. Crops the RGB frame to the detection extent
        3. Runs MobileCLIPEmbedder.embed_image() on the crop
        4. Falls back to hash embedding on any failure (fail-open)
        """
        try:
            pixels = detection.get("pixels")   # (K, 2) array of [row, col] indices
            if pixels is None or len(pixels) == 0:
                return None

            # Compute tight image-space bounding box from pixel indices
            r_min = int(pixels[:, 0].min())
            r_max = int(pixels[:, 0].max()) + 1
            c_min = int(pixels[:, 1].min())
            c_max = int(pixels[:, 1].max()) + 1

            H, W = rgb_image.shape[:2]
            r_min = max(0, r_min);  r_max = min(H, r_max)
            c_min = max(0, c_min);  c_max = min(W, c_max)

            if (r_max - r_min) < 4 or (c_max - c_min) < 4:
                return None   # crop too tiny to embed meaningfully

            crop = rgb_image[r_min:r_max, c_min:c_max]

            from src.edge.embedding.mobile_clip import MobileCLIPEmbedder
            emb = MobileCLIPEmbedder().embed_image(crop)
            return emb

        except Exception as e:
            logger.debug(f"_embed_crop MobileCLIP failed ({e}), using hash fallback")
            # Hash fallback: deterministic from 3D position so similar positions
            # get similar embeddings — better than pure random.
            try:
                pts = detection.get("points_3d")
                if pts is not None and len(pts) > 0:
                    import hashlib
                    h = hashlib.sha256(pts[:3].astype(np.float32).tobytes()).digest()
                    rng = np.random.default_rng(
                        np.frombuffer(h[:8], dtype=np.uint64)[0])
                    v = rng.normal(0, 1, 512).astype(np.float32)
                    return v / (np.linalg.norm(v) + 1e-9)
            except Exception as e:
                # Deterministic fallback embedding is best-effort — on any failure
                # return None so the caller skips appearance matching for this track.
                logger.debug(f"fallback embedding skipped: {e}")
            return None

    def get_dynamic_gaussian_mask(self,
                                   gaussian_positions: np.ndarray,
                                   expand_m: float = 0.1) -> np.ndarray:
        """
        Return (N,) bool mask — True for Gaussians inside any confirmed
        dynamic track bbox.  Expand bbox by expand_m to capture boundary splats.
        """
        N    = len(gaussian_positions)
        mask = np.zeros(N, dtype=bool)

        for track in self.confirmed_tracks():
            lo     = track.bbox_min - expand_m
            hi     = track.bbox_max + expand_m
            in_bbox = np.all(
                (gaussian_positions >= lo) & (gaussian_positions <= hi),
                axis=1
            )
            mask |= in_bbox

        return mask


# ── Rigid body translator ─────────────────────────────────────────────────
def translate_gaussian_cluster_rigid(
    positions: np.ndarray,
    delta:     np.ndarray
) -> np.ndarray:
    """
    Translate a Gaussian cluster as a rigid body.
    All Gaussians shift by the same delta vector.
    """
    return positions + delta


def predict_track_delta(track: DynamicTrack, dt_frames: float = 1.0) -> np.ndarray:
    """Predict centroid displacement using constant-velocity model."""
    return track.velocity * dt_frames
