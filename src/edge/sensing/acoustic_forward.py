"""
PHANTOM-ECHO REVEAL — Honest Acoustic Forward + Inverse Model
acoustic_forward.py   (v23 — integrity fix for report finding 3.1)

WHY THIS MODULE EXISTS
----------------------
The previous acoustic path was circular: it generated a fake echo
(`echo = ref_chirp*0.30 + noise`), ran the DSP, threw the result away
(`_ = detect_echo_peaks(...)`), and then set the SAS distance directly from
the known target position (`d = ||phone - target|| + noise`). The triangulator
recovered the target because the target *was* the input.

This module replaces that with the standard active-sonar simulation method
used to validate real systems:

    FORWARD  (synthesise what the microphone would record)
        - the target geometry is used ONLY to place the echo at its
          physically-correct round-trip delay  (this is a legitimate forward
          model — echo timing IS determined by distance)
        - confounding first-order echoes from the *visible* walls are mixed in
        - broadband measurement noise is added

    INVERSE  (recover distance from the recorded signal — the real estimator)
        - matched filter against the reference chirp
        - ISM subtraction of the predicted *visible* echoes (group-delay aware)
        - peak detection on the residual
        - distance = c * t_peak / 2     ← comes from the DETECTED peak time,
          never from the target coordinates

The distance handed to SAS therefore flows through the receiver chain and can
be wrong (noise, peak collision, failed subtraction). We log recovered-vs-true
error so the honesty is visible.

LIMITATION (stated plainly): this is a *controlled acoustic simulation*. It
does not model wave diffraction of the direct path around a solid occluder —
pyroomacoustics/image-source has no occlusion. Real-world deployment must be
validated on recorded phone audio. What this module proves is that the
ISM-subtract → matched-filter → triangulate estimator recovers occluded-surface
range from a physically-simulated, noise-corrupted, multipath signal — not that
a phone in a furnished room will. That validation is hardware work.
"""

from __future__ import annotations
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Sequence

from src.edge.sensing.acoustic_chirp import (
    ChirpConfig, generate_lfm_chirp, matched_filter)
from src.edge.sensing.ism_filter import (
    WallPlane, predict_first_order_arrival)
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

SPEED_OF_SOUND = 343.0  # m/s @ 20°C


@dataclass
class AcousticMeasurement:
    """Result of measuring one phone position acoustically."""
    position: np.ndarray          # (3,) phone world position
    distances: List[float]        # recovered one-way distances [m] (DSP output)
    snr_db: float                 # measured SNR of the residual peak
    true_distances: List[float]   # ground-truth ||P-T|| (for honest error report)
    recovered_errors_cm: List[float]


def _add_chirp(buf: np.ndarray, ref: np.ndarray, delay_samples: int, amp: float) -> None:
    """Place a scaled copy of the reference chirp at `delay_samples` into buf."""
    if delay_samples < 0:
        return
    end = delay_samples + len(ref)
    if delay_samples >= len(buf):
        return
    n = min(end, len(buf)) - delay_samples
    buf[delay_samples:delay_samples + n] += amp * ref[:n]


def simulate_received_signal(phone_pos: np.ndarray,
                             occluded_targets: Sequence[np.ndarray],
                             visible_walls: Sequence[WallPlane],
                             cfg: ChirpConfig,
                             rng: np.random.Generator,
                             buffer_s: float = 0.08,
                             occ_amp: float = 0.55,
                             wall_amp: float = 0.45,
                             noise_amp: float = 0.02) -> np.ndarray:
    """
    FORWARD MODEL. Synthesise the microphone recording for one chirp emission.

    The occluded-target geometry only sets echo *timing* (physical truth).
    Visible-wall echoes are confounders. Broadband noise is added. The receiver
    has to dig the occluded echo out of this.
    """
    ref = generate_lfm_chirp(cfg)
    fs = cfg.sample_rate
    L = int(buffer_s * fs) + len(ref) + 1
    received = np.zeros(L, dtype=np.float32)

    # direct blast (transmit leakage) at t≈0 — a strong confounder near zero lag
    _add_chirp(received, ref, delay_samples=0, amp=0.8)

    # confounding first-order echoes from KNOWN visible walls
    for w in visible_walls:
        t_vis = predict_first_order_arrival(phone_pos, phone_pos, w)
        _add_chirp(received, ref, int(round(t_vis * fs)), amp=wall_amp)

    # the occluded-surface echoes (timing = physical truth; the thing to recover)
    for tgt in occluded_targets:
        rt = 2.0 * float(np.linalg.norm(phone_pos - np.asarray(tgt, float)))
        delay = int(round((rt / SPEED_OF_SOUND) * fs))
        # mild range-dependent spreading loss (kept gentle so the estimator has
        # signal to work with — favorable-but-not-cheating SNR), lightly randomised
        d_one = max(0.2, rt / 2.0)
        atten = occ_amp * (1.0 / (1.0 + 0.15 * d_one ** 2)) * (1.0 + rng.normal(0, 0.05))
        _add_chirp(received, ref, delay, amp=float(atten))

    received += rng.normal(0, noise_amp, size=L).astype(np.float32)
    return received


def _predicted_visible_peak_times(phone_pos: np.ndarray,
                                  visible_walls: Sequence[WallPlane]) -> List[float]:
    return [predict_first_order_arrival(phone_pos, phone_pos, w) for w in visible_walls]


def measure_distances(phone_pos: np.ndarray,
                      occluded_targets: Sequence[np.ndarray],
                      visible_walls: Sequence[WallPlane],
                      cfg: ChirpConfig,
                      rng: np.random.Generator,
                      reject_window_s: float = 0.0008) -> AcousticMeasurement:
    """
    INVERSE MODEL (the real estimator).

    1. synthesise the received signal (forward model)
    2. matched filter against the reference chirp
    3. reject peaks that line up with predicted visible-wall echoes (ISM)
    4. surviving peaks → one-way distances via d = c·t/2

    Group-delay note: matched_filter uses mode='full', so an echo at sample
    `s` peaks at index `(M-1)+s`. We subtract (M-1) before converting to time —
    this is the latent ISM misalignment bug from the report, fixed here.
    """
    ref = generate_lfm_chirp(cfg)
    fs = cfg.sample_rate
    M = len(ref)

    received = simulate_received_signal(phone_pos, occluded_targets,
                                        visible_walls, cfg, rng)
    corr = matched_filter(received, ref)             # length L+M-1
    if corr.size == 0 or float(np.max(np.abs(corr))) < 1e-9:
        return AcousticMeasurement(np.asarray(phone_pos, float), [], 0.0, [], [])

    abs_corr = np.abs(corr)
    cmax = float(abs_corr.max())
    # peaks must be ≥ 30 cm apart in physical range
    min_gap = max(1, int((2 * 0.30 / SPEED_OF_SOUND) * fs))
    peak_idx, props = find_peaks(abs_corr, distance=min_gap,
                                 prominence=0.05 * cmax)

    # convert to corrected (group-delay-removed) arrival times
    cand_times, cand_amps = [], []
    for p in peak_idx:
        s = p - (M - 1)               # remove matched-filter group delay
        if s <= 0:
            continue                   # direct blast / pre-trigger
        cand_times.append(s / fs)
        cand_amps.append(abs_corr[p])

    # ISM rejection: drop peaks matching a predicted visible-wall arrival
    vis_times = _predicted_visible_peak_times(phone_pos, visible_walls)

    def is_visible(t: float) -> bool:
        return any(abs(t - tv) <= reject_window_s for tv in vis_times)

    distances, amps = [], []
    for t, a in zip(cand_times, cand_amps):
        if is_visible(t):
            continue
        distances.append(SPEED_OF_SOUND * t / 2.0)   # one-way range
        amps.append(a)

    # honest error reporting vs the geometry that generated the waveform
    true_d = [float(np.linalg.norm(phone_pos - np.asarray(t, float)))
              for t in occluded_targets]
    errs_cm = []
    for td in true_d:
        if distances:
            nearest = min(distances, key=lambda d: abs(d - td))
            errs_cm.append(abs(nearest - td) * 100.0)

    # SNR of the strongest surviving residual peak vs median background
    if amps:
        snr_db = float(20 * np.log10((max(amps) + 1e-9) /
                                     (np.median(abs_corr) + 1e-9)))
    else:
        snr_db = 0.0

    return AcousticMeasurement(
        position=np.asarray(phone_pos, float),
        distances=distances,
        snr_db=snr_db,
        true_distances=true_d,
        recovered_errors_cm=errs_cm,
    )


def sweep_measurements(phone_positions: Sequence[np.ndarray],
                       occluded_targets: Sequence[np.ndarray],
                       visible_walls: Sequence[WallPlane],
                       cfg: Optional[ChirpConfig] = None,
                       rng: Optional[np.random.Generator] = None
                       ) -> tuple[List[Dict], List[float]]:
    """
    Run the honest forward+inverse pipeline over a whole walk.

    Returns:
        sas_measurements : list of {"position","distances","snr_db"} for SAS
        recovery_errors_cm : per-position recovery error (for the dashboard/log)
    """
    if cfg is None:
        cfg = ChirpConfig()
    if rng is None:
        rng = np.random.default_rng(1234)

    sas, all_errs = [], []
    for p in phone_positions:
        m = measure_distances(np.asarray(p, float), occluded_targets,
                              visible_walls, cfg, rng)
        if m.distances:
            sas.append({"position": list(map(float, m.position)),
                        "distances": [float(d) for d in m.distances],
                        "snr_db": float(m.snr_db)})
        all_errs.extend(m.recovered_errors_cm)

    if all_errs:
        logger.info("Acoustic recovery error (DSP vs truth): "
                    f"mean={np.mean(all_errs):.2f}cm  max={np.max(all_errs):.2f}cm  "
                    f"({len(all_errs)} measurements)")
    return sas, all_errs
