"""
PHANTOM-ECHO REVEAL — Acoustic Chirp Emission & Processing
Layer 0: Multi-Modal Sensing (Acoustic Channel)

Emits LFM chirps via smartphone speaker, captures echoes via microphone,
and returns normalised room impulse response for downstream ISM filtering.

FIXED BUGS:
    Bug 3 — detect_echo_peaks crashes on empty RIR (zero-length array):
             np.max() on empty array raises ValueError with no identity.
             Fix: explicit len-check before computing prominence threshold.

    Bug 4 — NaN propagation in matched_filter:
             If received contains NaN, corr=NaN, norm=NaN, output=NaN.
             All downstream calls then produce NaN with no error message.
             Fix: NaN guard at entry; replace NaN samples with zeros and
             emit a WARNING so the problem is visible in logs.

ADDED Missing 4 — Input validation:
    Both matched_filter and detect_echo_peaks now guard against:
      - NaN / Inf inputs
      - Zero-length inputs
      - Inputs that are all-zero (produce no meaningful correlation)
"""

import numpy as np
from scipy.signal import chirp, correlate, find_peaks
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

SPEED_OF_SOUND = 343.0   # m/s at 20°C, 1 atm


@dataclass
class ChirpConfig:
    sample_rate: int   = 44100
    f_start:     float = 1000.0    # Hz — start of LFM sweep
    f_end:       float = 22000.0   # Hz — end of LFM sweep (near-ultrasonic)
    duration:    float = 0.02      # seconds — 20ms chirp
    amplitude:   float = 0.8      # 0.0–1.0


@dataclass
class EchoMeasurement:
    """Single echo measurement at one IMU position."""
    position:      np.ndarray    # (3,) world position of phone at chirp time
    arrival_times: np.ndarray    # (N,) arrival times of detected echo peaks [seconds]
    distances:     np.ndarray    # (N,) d = c * t / 2 [meters]
    snr_db:        float         # signal-to-noise ratio of measurement


def generate_lfm_chirp(cfg: ChirpConfig) -> np.ndarray:
    """
    Generate a Linear Frequency Modulated (LFM) chirp signal.

    Returns: (N,) float32 array, normalised to [-amplitude, +amplitude]
    """
    t = np.linspace(0, cfg.duration,
                    int(cfg.sample_rate * cfg.duration), endpoint=False)
    signal = chirp(t, f0=cfg.f_start, f1=cfg.f_end,
                   t1=cfg.duration, method='linear')
    return (signal * cfg.amplitude).astype(np.float32)


def _sanitize_audio(arr: np.ndarray, name: str) -> np.ndarray:
    """
    FIX Bug 4 / Missing 4 — Input sanitiser for audio arrays.

    Replaces NaN and Inf with 0.0 and emits a WARNING.
    Returns a float32 copy so the original is never mutated.

    Why zeros instead of raising:  the acoustic pipeline is best-effort.
    A single dropped mic frame should produce 0 echo detections
    (same as a room with no reflectors) rather than crashing the
    entire pipeline.  The WARNING in the log makes the problem visible
    for post-mortem without stopping the demo.
    """
    arr = np.asarray(arr, dtype=np.float32)
    bad = ~np.isfinite(arr)
    if np.any(bad):
        n_bad = int(bad.sum())
        logger.warning(
            f"acoustic_chirp: {name} contains {n_bad} NaN/Inf samples "
            f"({n_bad / len(arr) * 100:.1f}% of {len(arr)} total). "
            f"Replacing with 0.0. Check microphone capture pipeline."
        )
        arr = arr.copy()
        arr[bad] = 0.0
    return arr


def matched_filter(received: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Apply matched filter (cross-correlation) to isolate echo arrivals.

    FIX Bug 4: NaN inputs caused silent all-NaN propagation.
               Now guards at entry and emits WARNING.

    Args:
        received:  raw microphone recording (N_samples,) float32
        reference: original emitted chirp  (M_samples,) float32

    Returns:
        normalised cross-correlation (impulse response estimate),
        or zeros array of len(received)+len(reference)-1 on failure.
    """
    # Missing 4 — input validation
    received  = _sanitize_audio(received,  "received")
    reference = _sanitize_audio(reference, "reference")

    fallback_len = len(received) + len(reference) - 1

    if len(received) == 0 or len(reference) == 0:
        logger.warning("matched_filter: empty input array — returning zeros")
        return np.zeros(max(fallback_len, 1), dtype=np.float32)

    corr = correlate(received, reference, mode='full')

    # FIX Bug 4: NaN guard on norm.
    # np.max(np.abs(NaN)) = NaN → NaN + 1e-9 = NaN → corr/NaN = NaN
    # Instead: check for NaN in corr (can arise from NaN in inputs even
    # after sanitisation if scipy propagates differently), fall back to zeros.
    if not np.all(np.isfinite(corr)):
        logger.warning(
            "matched_filter: cross-correlation produced NaN/Inf "
            "(unexpected after sanitisation). Returning zeros."
        )
        return np.zeros(len(corr), dtype=np.float32)

    abs_max = np.max(np.abs(corr))
    if abs_max < 1e-12:
        # All-zero input (silence) — no peaks possible, return as-is
        logger.debug("matched_filter: near-zero signal — no echoes expected")
        return corr.astype(np.float32)

    return (corr / (abs_max + 1e-9)).astype(np.float32)


def detect_echo_peaks(
    rir: np.ndarray,
    sample_rate: int,
    min_distance_m: float = 0.3,
    prominence_threshold: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect peaks in the room impulse response corresponding to surface echoes.

    FIX Bug 3: empty RIR (zero-length array) caused ValueError in
               np.max(np.abs(rir)) — "zero-size array to reduction
               operation maximum which has no identity".
               Fix: explicit empty-array guard returns ([], []) safely.

    FIX Bug 4 / Missing 4: NaN guard at entry (should already be clean
               after matched_filter, but defensive check here too).

    Args:
        rir:                   normalised room impulse response
        sample_rate:           samples per second
        min_distance_m:        minimum physical distance between distinct surfaces
        prominence_threshold:  minimum peak prominence (fraction of max)

    Returns:
        arrival_times: (K,) array of peak arrival times [seconds]
        distances:     (K,) array of inferred distances  [meters]
        Both are empty arrays if no peaks are found.
    """
    EMPTY = (np.array([], dtype=np.float64), np.array([], dtype=np.float64))

    # Missing 4 — input validation
    rir = _sanitize_audio(rir, "rir")

    # FIX Bug 3 — empty array guard
    if len(rir) == 0:
        logger.warning(
            "detect_echo_peaks: received empty RIR (length=0). "
            "This can happen if subtract_visible_echoes clipped "
            "everything below the noise floor. Returning 0 peaks."
        )
        return EMPTY

    abs_rir = np.abs(rir)
    rir_max = float(np.max(abs_rir))

    if rir_max < 1e-12:
        # All-zero RIR — no meaningful peaks
        logger.debug("detect_echo_peaks: all-zero RIR — 0 peaks")
        return EMPTY

    min_sample_gap = int((2 * min_distance_m / SPEED_OF_SOUND) * sample_rate)
    min_sample_gap = max(1, min_sample_gap)   # guard: must be ≥ 1

    peaks, _ = find_peaks(
        abs_rir,
        distance=min_sample_gap,
        prominence=prominence_threshold * rir_max   # safe: rir_max > 1e-12
    )

    if len(peaks) == 0:
        return EMPTY

    # d = c * t / 2  (round-trip)
    arrival_times = peaks.astype(np.float64) / sample_rate
    distances     = SPEED_OF_SOUND * arrival_times / 2.0

    return arrival_times, distances


def compute_snr(rir: np.ndarray, signal_window: int = 200) -> float:
    """Estimate SNR of the strongest echo peak."""
    if len(rir) == 0 or not np.any(np.isfinite(rir)):
        return 0.0
    peak_idx    = int(np.argmax(np.abs(rir)))
    start       = max(0, peak_idx - signal_window // 2)
    end         = min(len(rir), peak_idx + signal_window // 2)
    signal_power = float(np.mean(rir[start:end] ** 2))
    noise_power  = float(np.mean(rir[:max(1, start)] ** 2)) + 1e-12
    snr_linear   = signal_power / noise_power
    return float(10 * np.log10(snr_linear + 1e-12))


def process_recording(
    recorded_audio: np.ndarray,
    phone_position: np.ndarray,
    cfg: Optional[ChirpConfig] = None
) -> EchoMeasurement:
    """
    Full pipeline: recorded audio → EchoMeasurement at a given phone position.

    Args:
        recorded_audio: microphone capture (N_samples,) float32
        phone_position: IMU world position (3,) at chirp emission
        cfg:            chirp configuration (uses defaults if None)

    Returns:
        EchoMeasurement with detected echo arrival times and distances.
        arrival_times and distances are empty arrays if no echoes detected.
    """
    if cfg is None:
        cfg = ChirpConfig()

    reference_chirp = generate_lfm_chirp(cfg)
    rir             = matched_filter(recorded_audio, reference_chirp)
    arrival_times, distances = detect_echo_peaks(rir, cfg.sample_rate)
    snr = compute_snr(rir)

    logger.debug(
        f"Position {np.round(phone_position, 3)}: "
        f"{len(arrival_times)} echoes detected, SNR={snr:.1f}dB"
    )

    return EchoMeasurement(
        position=phone_position,
        arrival_times=arrival_times,
        distances=distances,
        snr_db=snr,
    )
