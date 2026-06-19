"""
PHANTOM-ECHO REVEAL — Synthetic Aperture Sonar (SAS) Triangulation
Layer 0: Multi-Modal Sensing (Acoustic Channel)

Uses multiple ISM-filtered echo measurements along the walking path
(virtual aperture created by IMU trajectory) to triangulate exact
3D positions of occluded surfaces.

Physics: Flaw 40 fix — single omnidirectional speaker has no bearing.
Solution: Walk phone along path → each position P_i + round-trip
          distance d_i defines a sphere. Intersect 3+ spheres → unique point Q.

Math (from Section 5.2 of project bible):
    ||Q - P_i||² = d_i²
    Subtract consecutive equations to get linear system:
    2(P_{i+1} - P_i)·Q = d_i² - d_{i+1}² - ||P_i||² + ||P_{i+1}||²
    Solve via least squares: Q = (A^T A)^{-1} A^T b
"""

import numpy as np
from scipy.optimize import least_squares
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

SPEED_OF_SOUND = 343.0  # m/s at 20°C
MIN_BASELINE_M = 0.05   # minimum 5cm baseline for valid triangulation


@dataclass
class SphereConstraint:
    """One sphere: phone at P_i, occluded surface at distance d_i."""
    position: np.ndarray   # (3,) world position
    distance: float        # d = c * t_echo / 2 [meters]
    snr_db: float          # measurement quality


@dataclass
class OccludedSurfacePoint:
    """A triangulated 3D position of an occluded surface."""
    position: np.ndarray        # (3,) world position [meters]
    confidence: float           # 0.0–1.0
    residual_m: float           # triangulation residual [meters]
    n_measurements: int         # number of sphere constraints used
    color_tag: str = "TEAL"     # always TEAL for acoustic measurements


def build_linear_system(spheres: List[SphereConstraint]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert list of sphere constraints into linear system Ax = b.

    For consecutive pairs (i, i+1):
        2(P_{i+1} - P_i)·Q = d_i² - d_{i+1}² - ||P_i||² + ||P_{i+1}||²

    Returns:
        A: (N-1, 3) matrix
        b: (N-1,)   right-hand side
    """
    n = len(spheres)
    A = np.zeros((n - 1, 3), dtype=np.float64)
    b = np.zeros(n - 1, dtype=np.float64)

    for i in range(n - 1):
        P_i = np.asarray(spheres[i].position,     dtype=np.float64)
        P_j = np.asarray(spheres[i + 1].position, dtype=np.float64)
        d_i = float(spheres[i].distance)
        d_j = float(spheres[i + 1].distance)

        A[i] = 2.0 * (P_j - P_i)
        b[i] = (d_i ** 2 - d_j ** 2
                - np.dot(P_i, P_i)
                + np.dot(P_j, P_j))

    return A, b



def _check_collinearity(spheres, rank):
    """
    Missing 2 fix: emit actionable diagnostic when SAS positions are collinear.

    When all phone positions lie on a single axis (e.g. pure Z walk),
    the SAS linear system is rank-1, which produces 0 triangulated points
    silently.  The existing rank<3 check returns None but gives no guidance.

    This function gives an actionable message: which axis the user walked
    along and what to do about it.
    """
    import numpy as np
    positions = np.array([s.position for s in spheres], dtype=np.float64)
    centred   = positions - positions.mean(axis=0)
    try:
        _, sv, Vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return  # SVD failed; let the rank check handle it

    dominant_axis = Vt[0]   # principal direction of motion
    axis_names    = ['X', 'Y', 'Z']
    dominant_name = axis_names[int(np.argmax(np.abs(dominant_axis)))]
    variance_ratio = sv[0] / (sv.sum() + 1e-12)

    if variance_ratio > 0.98 and rank < 3:
        logger.warning(
            f"SAS collinearity detected (rank={rank}, "
            f"variance_ratio={variance_ratio:.3f}): "
            f"phone positions are nearly collinear along {dominant_name}-axis. "
            f"The SAS linear system cannot triangulate from a straight-line walk. "
            f"FIX: Walk in a 2D path (XZ zigzag). "
            f"In code: generate_walk_sequence(axis='xz') not axis='z'. "
            f"In real deployment: user must not walk in a straight line."
        )

def triangulate_least_squares(spheres: List[SphereConstraint],
                               floor_y: float = 0.0) -> Optional[OccludedSurfacePoint]:
    """
    Triangulate a single occluded surface point from 3+ sphere constraints.

    Uses the linear system from consecutive-pair subtraction, then refines
    with nonlinear least squares for sub-centimeter accuracy.

    Args:
        spheres: list of SphereConstraint (at least 3 required)

    Returns:
        OccludedSurfacePoint or None if triangulation fails
    """
    if len(spheres) < 3:
        logger.warning(f"Need ≥3 sphere constraints, got {len(spheres)}")
        return None

    # Check baseline — Flaw 25 fix
    positions = np.array([s.position for s in spheres])
    baseline = np.max(np.linalg.norm(positions - positions[0], axis=1))
    if baseline < MIN_BASELINE_M:
        logger.warning(f"Baseline {baseline*100:.1f}cm < {MIN_BASELINE_M*100:.1f}cm — insufficient")
        return None

    # Linear least-squares initial estimate
    A, b = build_linear_system(spheres)
    try:
        Q_init, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError as e:
        logger.error(f"Linear system failed: {e}")
        return None

    if rank < 3:
        _check_collinearity(spheres, rank)
        logger.warning(f"Degenerate system: rank={rank} — 0 points triangulated")
        return None

    # Nonlinear refinement — minimise sum of squared sphere errors.
    # OVERFLOW FIX: do the residual math in float64 and bound the search to a
    # generous box around the initial estimate. On adversarial/ill-conditioned
    # geometry the unbounded LM step could shoot Q to ~1e20, where the internal
    # x.dot(x) in np.linalg.norm overflows float32 and emits a RuntimeWarning.
    # Bounding the step keeps Q physical and silences the warning at its source.
    Q0 = np.asarray(Q_init, dtype=np.float64).ravel()

    def sphere_residuals(Q):
        Q = np.asarray(Q, dtype=np.float64)
        return [float(np.linalg.norm(Q - np.asarray(s.position, dtype=np.float64)) - s.distance)
                for s in spheres]

    # ±50 m box around the linear estimate — far larger than any indoor scene,
    # so it never distorts a valid solution, but prevents numeric blow-up.
    _span = 50.0
    result = least_squares(
        sphere_residuals, Q0, method='trf', max_nfev=100,
        bounds=(Q0 - _span, Q0 + _span))

    Q_refined = result.x

    # FIX Bug 6 — Y-coordinate underground guard.
    # In adversarial geometry (constant-height walk, target at large lateral
    # distance) the nonlinear refinement residual can drive Y below floor.
    # The floor is always Y=0 in our coordinate system (acoustic sources are
    # above floor level; surfaces cannot be at negative Y).
    # Clamp Y to [floor_y, room_height]. floor_y inferred as minimum Y of
    # all phone positions (they sit on the floor surface, not below it).
    # BUG-CE10 FIX: use explicit floor_y instead of inferring from phone height.
    # Old code: floor_y = min(phone_y) → if phone held at 1.2m, floor = 1.2m,
    # and any surface at 0.5m gets clamped to 1.2m (completely wrong).
    room_h = floor_y + 10.0  # generous ceiling
    if Q_refined[1] < floor_y:
        logger.warning(
            f"SAS Y={Q_refined[1]:.3f}m below floor ({floor_y:.3f}m) — "
            f"clamping. Adversarial geometry or constant-height walk detected."
        )
        Q_refined = Q_refined.copy()
        Q_refined[1] = floor_y
    Q_refined[1] = float(np.clip(Q_refined[1], floor_y, room_h))

    rms_residual = float(np.sqrt(np.mean(result.fun ** 2)))

    # Confidence: high SNR + low residual + many measurements → high confidence
    mean_snr = np.mean([s.snr_db for s in spheres])
    confidence = _compute_confidence(mean_snr, rms_residual, len(spheres))

    logger.info(
        f"SAS triangulation: Q={Q_refined}, "
        f"residual={rms_residual*100:.1f}cm, "
        f"confidence={confidence:.2f}, "
        f"n_measurements={len(spheres)}"
    )

    return OccludedSurfacePoint(
        position=Q_refined,
        confidence=confidence,
        residual_m=rms_residual,
        n_measurements=len(spheres),
        color_tag="TEAL"
    )


def _compute_confidence(snr_db: float, residual_m: float, n_meas: int) -> float:
    """
    Heuristic confidence score for a triangulated point.
    HIGH (>0.75) → WHITE in PHANTOM-LITE terms (but acoustic = TEAL)
    """
    # SNR contribution: 0 at 0dB, 1 at 30dB+
    snr_score = min(1.0, max(0.0, snr_db / 30.0))

    # Residual contribution: 1 at 0cm, 0 at 5cm+
    residual_score = max(0.0, 1.0 - residual_m / 0.05)

    # Measurement count: 1 at 3, scales up to 1 at 10+
    count_score = min(1.0, (n_meas - 2) / 8.0)

    return 0.4 * snr_score + 0.4 * residual_score + 0.2 * count_score


# ── BUG-V22-SAS FIX: echo track association ─────────────────────────────────

def cluster_and_triangulate_v3(measurements: List[dict],
                                floor_y: float = 0.0,
                                gate_m: float = 0.35) -> List[OccludedSurfacePoint]:
    """
    BUG-V22-SAS FIX: v2's angular sub-clustering split walk-path measurements
    by *phone-position* bearing. On any normal walking trajectory consecutive
    positions exceed the 15° gate, so every sub-cluster ended up with <3
    constraints and triangulation NEVER fired — TEAL was silently always 0
    (verifiable in the v21 logs: "SAS v2: ... → 0 triangulated points").

    v3 replaces geometric clustering with nearest-neighbour echo TRACK
    association (the standard multi-target sonar approach):

      - Consecutive phone positions are centimetres apart, so the round-trip
        distance to the SAME physical surface changes slowly and smoothly.
      - For each new measurement, each echo distance is appended to the track
        whose last distance is closest (within gate_m). Unmatched echoes
        open new tracks.
      - Each track with ≥3 spatially-distinct constraints is triangulated
        with the existing least-squares solver (unchanged math).

    This is data association by temporal continuity instead of by bearing —
    correct for a moving virtual array, and O(n_measurements × n_tracks).
    """
    tracks: List[List[SphereConstraint]] = []
    for m in measurements:
        pos    = np.array(m["position"], dtype=np.float64)
        snr    = float(m.get("snr_db", 10.0))
        # OVERFLOW/NaN GUARD: drop non-finite or physically-impossible echo
        # ranges (a corrupt RIR peak can yield inf/NaN). Indoor round-trip
        # range is bounded well under 50 m; anything past that is spurious and
        # would otherwise poison the linear system (inf**2 → RuntimeWarning).
        echoes = [float(d) for d in m["distances"]
                  if np.isfinite(d) and 0.0 < float(d) < 50.0]

        # Predicted next distance per track (constant-velocity in range space:
        # the range-rate to a static target from a smoothly moving phone is
        # locally constant, so linear extrapolation beats last-value matching
        # and survives track crossings).
        preds = []
        for track in tracks:
            if len(track) >= 2:
                preds.append(2.0 * track[-1].distance - track[-2].distance)
            else:
                preds.append(track[-1].distance)

        # Globally-greedy joint assignment (small Hungarian substitute):
        # smallest |echo - predicted| pair first, one echo per track.
        pairs = sorted(
            ((abs(d - preds[ti]), ei, ti)
             for ei, d in enumerate(echoes) for ti in range(len(tracks))),
            key=lambda x: x[0])
        used_e: set = set(); used_t: set = set()
        assign: dict = {}
        for gap, ei, ti in pairs:
            # Mature tracks (>=2 points) have a reliable range-rate prediction,
            # so the gate tightens to 10cm; new tracks keep the loose gate.
            eff_gate = 0.10 if len(tracks[ti]) >= 2 else gate_m
            if gap > eff_gate or ei in used_e or ti in used_t:
                continue
            assign[ei] = ti
            used_e.add(ei); used_t.add(ti)

        for ei, d in enumerate(echoes):
            c = SphereConstraint(position=pos, distance=d, snr_db=snr)
            if ei in assign:
                tracks[assign[ei]].append(c)
            else:
                tracks.append([c])

    results: List[OccludedSurfacePoint] = []
    MIN_SEP_M = 0.05
    for track in tracks:
        unique: List[SphereConstraint] = []
        for c in track:
            if not any(np.linalg.norm(c.position - u.position) < MIN_SEP_M
                       for u in unique):
                unique.append(c)
        if len(unique) >= 3:
            point = triangulate_least_squares(unique, floor_y=floor_y)
            if point is not None:
                # Residual gate: a wrong association (mixed targets) produces
                # a point that does NOT satisfy its own sphere constraints.
                Q = np.array(point.position, dtype=np.float64)

                # Mirror disambiguation: when the phone is carried at constant
                # height the virtual array is PLANAR, so every solution Q has
                # a mirror Q' across the array plane with identical residuals.
                # Physical prior: occluded surfaces hidden behind furniture lie
                # BELOW phone carry height (gravity — objects rest on supports).
                pos_y   = np.array([c.position[1] for c in unique])
                y_plane = float(pos_y.mean())
                if float(pos_y.std()) < 0.05 and Q[1] > y_plane:
                    Q_mirror = Q.copy()
                    Q_mirror[1] = 2.0 * y_plane - Q[1]
                    if Q_mirror[1] >= floor_y - 0.05:
                        Q = Q_mirror
                        point.position = Q

                resid = float(np.mean([abs(np.linalg.norm(Q - c.position) - c.distance)
                                       for c in unique]))
                if resid <= 0.03:          # 3cm gate vs 8mm sensor noise
                    results.append(point)
                else:
                    logger.debug(f"SAS v3: rejected track, residual {resid*100:.1f}cm")

    logger.info(f"SAS v3: {len(measurements)} measurements → "
                f"{len(tracks)} echo tracks → {len(results)} triangulated points")
    return results
