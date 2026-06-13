"""
PHANTOM-ECHO REVEAL — Per-Vertex Semantic Labeler
semantic_labeler.py

Assigns FLOOR / WALL / CEILING / OBJECT labels to mesh vertices
based on normal direction and height. Required for the deliverable mesh
and for semantic accuracy KPI evaluation (>93% target).

Also outputs a per-vertex color map using the PHANTOM tag color scheme.
"""

import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

LABEL_FLOOR   = "FLOOR"
LABEL_WALL    = "WALL"
LABEL_CEILING = "CEILING"
LABEL_OBJECT  = "OBJECT"
LABEL_UNKNOWN = "UNKNOWN"

# Tag colors for visualization (RGB in [0,1])
LABEL_COLORS: Dict[str, Tuple[float, float, float]] = {
    LABEL_FLOOR:   (0.55, 0.45, 0.35),   # warm beige
    LABEL_WALL:    (0.75, 0.75, 0.80),   # light grey
    LABEL_CEILING: (0.90, 0.90, 0.95),   # near white
    LABEL_OBJECT:  (0.30, 0.60, 0.90),   # blue-ish
    LABEL_UNKNOWN: (0.50, 0.50, 0.50),   # neutral grey
}

# Normal thresholds
FLOOR_COS_THRESH   = 0.85    # normal.y > 0.85 → floor
CEILING_COS_THRESH = 0.85    # -normal.y > 0.85 → ceiling
WALL_COS_THRESH    = 0.70    # |normal.xz| > 0.70 → wall


def label_vertices(vertices: np.ndarray,
                    normals: np.ndarray,
                    floor_y: float = 0.0,
                    ceiling_y: float = 2.5,
                    floor_band_m: float = 0.15,
                    ceiling_band_m: float = 0.15) -> np.ndarray:
    """
    Assign semantic label to each mesh vertex.

    Uses a two-pass approach:
        Pass 1: Normal direction (geometric)
        Pass 2: Height refinement (floor/ceiling band correction)

    Args:
        vertices:       (N, 3) mesh vertex positions
        normals:        (N, 3) vertex normals (unit length)
        floor_y:        scene floor Y coordinate
        ceiling_y:      scene ceiling Y coordinate
        floor_band_m:   label as FLOOR if within this band of floor_y
        ceiling_band_m: label as CEILING if within this band of ceiling_y

    Returns:
        (N,) array of string labels
    """
    N = len(vertices)
    labels = np.full(N, LABEL_UNKNOWN, dtype=object)

    # Normalize normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    safe_normals = normals / (norms + 1e-9)

    up_dot   = safe_normals[:, 1]               # dot with (0,1,0)
    horiz    = np.sqrt(safe_normals[:, 0] ** 2 + safe_normals[:, 2] ** 2)

    # Pass 1: Normal-based classification
    is_floor   = up_dot  >  FLOOR_COS_THRESH
    is_ceiling = up_dot  < -CEILING_COS_THRESH
    is_wall    = horiz   >  WALL_COS_THRESH

    labels[is_floor]   = LABEL_FLOOR
    labels[is_ceiling] = LABEL_CEILING
    labels[is_wall]    = LABEL_WALL
    labels[~(is_floor | is_ceiling | is_wall)] = LABEL_OBJECT

    # Pass 2: Height band refinement
    y = vertices[:, 1]
    near_floor   = np.abs(y - floor_y)   < floor_band_m
    near_ceiling = np.abs(y - ceiling_y) < ceiling_band_m

    # Override: near floor → FLOOR regardless of normal (catches tilted mesh)
    labels[near_floor]   = LABEL_FLOOR
    labels[near_ceiling] = LABEL_CEILING

    # Stats
    unique, counts = np.unique(labels, return_counts=True)
    stats = dict(zip(unique, counts))
    total = N or 1
    logger.info(
        f"Semantic labeler: {N} vertices — "
        + ", ".join(f"{k}={v} ({100*v/total:.1f}%)" for k, v in stats.items())
    )

    return labels


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    """
    Convert label array to (N, 3) RGB color array for mesh visualization.
    """
    N = len(labels)
    colors = np.zeros((N, 3), dtype=np.float32)
    for i, lbl in enumerate(labels):
        colors[i] = LABEL_COLORS.get(lbl, LABEL_COLORS[LABEL_UNKNOWN])
    return colors


def compute_semantic_accuracy(pred_labels: np.ndarray,
                               gt_labels: np.ndarray) -> Dict[str, float]:
    """
    Compute per-class and overall semantic accuracy.

    Args:
        pred_labels: (N,) predicted label strings
        gt_labels:   (N,) ground truth label strings

    Returns:
        dict with overall accuracy + per-class accuracy
    """
    if len(pred_labels) != len(gt_labels):
        return {"error": "length mismatch"}

    N = len(pred_labels)
    correct = np.sum(pred_labels == gt_labels)
    overall = float(correct / N) if N > 0 else 0.0

    per_class = {}
    for cls in [LABEL_FLOOR, LABEL_WALL, LABEL_CEILING, LABEL_OBJECT]:
        gt_mask   = gt_labels == cls
        pred_mask = pred_labels == cls
        tp = np.sum(gt_mask & pred_mask)
        fn = np.sum(gt_mask & ~pred_mask)
        fp = np.sum(~gt_mask & pred_mask)
        recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        per_class[cls] = {
            "recall":    round(recall, 4),
            "precision": round(precision, 4),
            "support":   int(tp + fn),
        }

    return {"overall": round(overall, 4), "per_class": per_class,
            "kpi_pass": overall > 0.93}


def label_gaussians(gaussians: list,
                     floor_y: float = 0.0,
                     ceiling_y: float = 2.5) -> list:
    """
    Add semantic_label field to each Gaussian dict.

    Uses normal direction and height band, same as label_vertices.
    """
    for g in gaussians:
        pos    = np.array(g.get("position", [0, 0, 0]), dtype=np.float32)
        normal = np.array(g.get("normal",   [0, 1, 0]), dtype=np.float32)
        norm   = np.linalg.norm(normal)
        if norm > 1e-9:
            normal /= norm

        up_dot = float(normal[1])
        horiz  = float(np.sqrt(normal[0]**2 + normal[2]**2))
        y      = float(pos[1])

        if abs(y - floor_y) < 0.15 or up_dot > FLOOR_COS_THRESH:
            g["semantic_label"] = LABEL_FLOOR
        elif abs(y - ceiling_y) < 0.15 or up_dot < -CEILING_COS_THRESH:
            g["semantic_label"] = LABEL_CEILING
        elif horiz > WALL_COS_THRESH:
            g["semantic_label"] = LABEL_WALL
        else:
            g["semantic_label"] = LABEL_OBJECT

    return gaussians
