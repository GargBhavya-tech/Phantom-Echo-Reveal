"""
PHANTOM-ECHO REVEAL — Edge-Local Image Source Method (ISM) Filter
Layer 0: Multi-Modal Sensing (Acoustic Channel)

Runs entirely on the smartphone GPU (Metal/Vulkan via numpy fallback).
Zero network dependency → microsecond-precision timing, no WiFi jitter.

Key insight from Flaw 39/42 fix:
    Cloud-based pyroomacoustics introduces 3ms WiFi jitter
    → 1.03m depth error at 343 m/s
    → completely unusable for sub-centimeter reconstruction
    Edge-local first-order ISM: O(N_walls) mirror operations, trivially fast.

Reference:
    Allen & Berkley (1979) "Image method for efficiently simulating
    small-room acoustics"
"""

import numpy as np
from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)

SPEED_OF_SOUND = 343.0  # m/s at 20°C


@dataclass
class WallPlane:
    """
    Axis-aligned or arbitrary plane: Ax + By + Cz + D = 0.
    Normal vector (A, B, C) must be unit-length.
    """
    A: float
    B: float
    C: float
    D: float
    label: str = "wall"   # 'floor', 'wall', 'ceiling', etc.

    def normal(self) -> np.ndarray:
        return np.array([self.A, self.B, self.C])


def mirror_source(speaker_pos: np.ndarray, wall: WallPlane) -> np.ndarray:
    """
    Compute the image (mirror) source of the speaker across a wall plane.

    Math:
        S' = S - 2 * (A*Sx + B*Sy + C*Sz + D) * (A, B, C)
        where (A, B, C) is the unit normal of the wall.

    Args:
        speaker_pos: (3,) speaker world position
        wall:        WallPlane with unit-normal

    Returns:
        (3,) mirror source position
    """
    n = wall.normal()
    dist_signed = wall.A * speaker_pos[0] + wall.B * speaker_pos[1] + wall.C * speaker_pos[2] + wall.D
    return speaker_pos - 2.0 * dist_signed * n


def predict_first_order_arrival(speaker_pos: np.ndarray,
                                 mic_pos: np.ndarray,
                                 wall: WallPlane) -> float:
    """
    Predict arrival time of the first-order reflection off a known visible wall.

    Math:
        t_predicted = ||S'_j - M|| / c
        where S'_j = mirror source, M = microphone position, c = 343 m/s

    Returns:
        predicted arrival time in seconds
    """
    s_mirror = mirror_source(speaker_pos, wall)
    path_length = np.linalg.norm(s_mirror - mic_pos)
    return path_length / SPEED_OF_SOUND


def build_first_order_rir(speaker_pos: np.ndarray,
                           mic_pos: np.ndarray,
                           visible_walls: List[WallPlane],
                           sample_rate: int,
                           rir_length: int,
                           reflection_coefficient: float = 0.7) -> np.ndarray:
    """
    Build a synthetic RIR containing only first-order reflections from
    known visible walls. This is what we SUBTRACT from the measured signal.

    Args:
        speaker_pos:          (3,) speaker position in world coords
        mic_pos:              (3,) microphone position (same phone)
        visible_walls:        list of confirmed visible wall planes from DDGS scene
        sample_rate:          audio sample rate (Hz)
        rir_length:           number of samples in output RIR
        reflection_coefficient: amplitude of each first-order reflection

    Returns:
        (rir_length,) float32 synthetic RIR for visible walls only
    """
    synthetic_rir = np.zeros(rir_length, dtype=np.float32)

    for wall in visible_walls:
        t_pred = predict_first_order_arrival(speaker_pos, mic_pos, wall)
        sample_idx = int(t_pred * sample_rate)

        if 0 <= sample_idx < rir_length:
            # Add a short Gaussian pulse at the predicted arrival time
            # Width ~0.5ms = standard chirp resolution
            pulse_width = max(1, int(0.0005 * sample_rate))
            for offset in range(-pulse_width, pulse_width + 1):
                idx = sample_idx + offset
                if 0 <= idx < rir_length:
                    gaussian = np.exp(-0.5 * (offset / (pulse_width / 3.0)) ** 2)
                    synthetic_rir[idx] += reflection_coefficient * gaussian

        logger.debug(
            f"Wall '{wall.label}': mirror arrival at {t_pred*1000:.2f}ms "
            f"(sample {sample_idx})"
        )

    return synthetic_rir


def subtract_visible_echoes(measured_rir: np.ndarray,
                             synthetic_visible_rir: np.ndarray,
                             alpha: float = 1.0) -> np.ndarray:
    """
    Subtract predicted first-order visible-surface echoes from the measured RIR.

    residual(t) = raw_echo_signal(t) - alpha * sum(predicted_first_order_arrivals)

    The residual contains ONLY occluded surface reflections.

    Args:
        measured_rir:         matched-filter output from acoustic_chirp.py
        synthetic_visible_rir: predicted RIR from build_first_order_rir()
        alpha:                subtraction weight (1.0 = full subtraction)

    Returns:
        residual RIR containing only occluded-surface echoes
    """
    # Align lengths
    min_len = min(len(measured_rir), len(synthetic_visible_rir))
    residual = measured_rir[:min_len].copy()
    residual -= alpha * synthetic_visible_rir[:min_len]

    # Soft-clip to suppress any subtraction artifacts below noise floor
    noise_floor = 0.01 * np.max(np.abs(residual) + 1e-9)
    residual = np.where(np.abs(residual) < noise_floor, 0.0, residual)

    logger.info(
        f"ISM subtraction: {len(measured_rir)} samples → "
        f"residual max amplitude {np.max(np.abs(residual)):.4f}"
    )
    return residual.astype(np.float32)


def extract_walls_from_scene(gaussian_scene: dict) -> List[WallPlane]:
    """
    Extract confirmed visible wall planes from the DDGS Gaussian scene
    (BLUE + WHITE Gaussians tagged as WALL/FLOOR/CEILING).

    Args:
        gaussian_scene: dict with keys 'gaussians' → list of Gaussian objects
                        each with fields: tag, semantic_tag, plane_normal, plane_d

    Returns:
        list of WallPlane objects for ISM filtering
    """
    walls = []
    for g in gaussian_scene.get("gaussians", []):
        if g.get("semantic_tag") in ("WALL", "FLOOR", "CEILING") and \
           g.get("confidence_tag") in ("WHITE", "BLUE", "TEAL"):

            n = np.array(g["plane_normal"])
            # Ensure unit normal
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            n = n / norm

            walls.append(WallPlane(
                A=float(n[0]),
                B=float(n[1]),
                C=float(n[2]),
                D=float(g["plane_d"]),
                label=g["semantic_tag"].lower()
            ))

    logger.info(f"Extracted {len(walls)} visible wall planes for ISM filtering")
    return walls
