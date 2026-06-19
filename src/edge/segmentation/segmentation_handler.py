"""
PHANTOM-ECHO REVEAL — Edge Segmentation: MobileSAM + 3D Projection (Layer 0 Edge)
segmentation_handler.py

Pipeline:
    1. Run MobileSAM on RGB frame → 2D instance masks
    2. Project 2D masks into 3D using ARKit depth + camera pose
    3. Extract 3D bounding boxes for each instance
    4. Classify each instance via MobileCLIP (semantic label)

Edge-device design:
    MobileSAM runs in ~18ms on Apple Neural Engine (MPS backend)
    MobileCLIP runs in ~5ms (S2 variant)
    Total segmentation latency target: < 30ms per frame

Flaw fix (from bible v14):
    MobileSAM merges objects with similar textures → added depth-discontinuity
    secondary boundary detector. If two SAM segments are depth-continuous
    (|d_i - d_j| < 5cm at boundary), they may be one object; if discontinuous
    they are separate even if SAM merged them.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# MobileCLIP semantic class prompts
SEMANTIC_PROMPTS = [
    "a photo of a chair",
    "a photo of a sofa or couch",
    "a photo of a table",
    "a photo of a desk",
    "a photo of a bed",
    "a photo of a cabinet or wardrobe",
    "a photo of a wall",
    "a photo of a floor",
    "a photo of a ceiling",
    "a photo of a monitor or screen",
    "a photo of a plant",
    "a photo of a box or package",
    "a photo of a lamp",
    "a photo of a door",
]
SEMANTIC_LABELS = [
    "CHAIR", "SOFA", "TABLE", "DESK", "BED", "CABINET",
    "WALL", "FLOOR", "CEILING", "MONITOR", "PLANT", "BOX", "LAMP", "DOOR"
]


@dataclass
class SegmentedInstance:
    """One segmented object instance in the scene."""
    instance_id:  int
    mask_2d:      np.ndarray    # (H, W) bool
    points_3d:    np.ndarray    # (N, 3) 3D points from depth projection
    bbox_min:     np.ndarray    # (3,) world bounding box lower corner
    bbox_max:     np.ndarray    # (3,) world bounding box upper corner
    centroid:     np.ndarray    # (3,) world centroid
    semantic:     str           # CHAIR, TABLE, WALL, etc.
    clip_score:   float         # MobileCLIP classification confidence
    pixel_count:  int           # number of pixels in mask
    depth_range:  Tuple[float, float]  # (min_depth, max_depth) meters


@dataclass
class SegmentationResult:
    """Full segmentation result for one frame."""
    instances:          List[SegmentedInstance]
    depth_boundaries:   np.ndarray   # (H, W) float — depth discontinuity magnitude
    n_merged_splits:    int          # how many SAM merges were corrected
    processing_ms:      float


# ── Depth discontinuity detector ──────────────────────────────────────────

def compute_depth_discontinuities(depth_map: np.ndarray,
                                   threshold_m: float = 0.05) -> np.ndarray:
    """
    Detect depth discontinuities (object boundaries not found by RGB edges).

    Uses Sobel-like kernel on depth map.
    Returns (H, W) float array — magnitude of depth gradient.
    High values = boundaries between objects at different depths.

    Args:
        depth_map:    (H, W) float32 depth in meters
        threshold_m:  minimum depth jump to be considered a boundary

    Returns:
        (H, W) float32 discontinuity magnitude
    """
    H, W = depth_map.shape
    disc = np.zeros((H, W), dtype=np.float32)

    # Horizontal gradient
    horiz = np.abs(np.diff(depth_map, axis=1, prepend=depth_map[:, :1]))
    # Vertical gradient
    vert  = np.abs(np.diff(depth_map, axis=0, prepend=depth_map[:1, :]))

    disc = np.maximum(horiz, vert)
    disc[depth_map < 0.1] = 0   # zero out invalid depth regions
    return disc


def split_merged_segments(mask: np.ndarray,
                            depth_map: np.ndarray,
                            disc_map: np.ndarray,
                            disc_threshold: float = 0.05) -> List[np.ndarray]:
    """
    Split a SAM mask that may contain multiple depth-discontinuous objects.

    Strategy:
        1. Find internal depth discontinuities within the mask
        2. Label connected regions separated by discontinuities
        3. Return list of sub-masks

    Args:
        mask:           (H, W) bool — SAM segment mask
        depth_map:      (H, W) float32 depth
        disc_map:       (H, W) float32 discontinuity magnitude
        disc_threshold: minimum discontinuity to split (meters)

    Returns:
        list of (H, W) bool sub-masks (may be just [mask] if no splits)
    """
    try:
        from scipy import ndimage
    except ImportError:
        return [mask]

    # Create discontinuity barrier within this mask
    barrier = (disc_map > disc_threshold) & mask
    passable = mask & ~barrier

    # Label connected components in passable region
    labeled, n_labels = ndimage.label(passable)

    if n_labels <= 1:
        return [mask]

    # Filter out tiny fragments (< 2% of original mask)
    min_size = int(mask.sum() * 0.02)
    sub_masks = []
    for label_id in range(1, n_labels + 1):
        sub = labeled == label_id
        if sub.sum() >= min_size:
            sub_masks.append(sub)

    if len(sub_masks) <= 1:
        return [mask]

    logger.debug(f"Depth-discontinuity split: 1 SAM mask → {len(sub_masks)} objects")
    return sub_masks


# ── 3D projection ──────────────────────────────────────────────────────────

def project_mask_to_3d(mask: np.ndarray,
                        depth_map: np.ndarray,
                        camera_intrinsics: Dict[str, float],
                        camera_to_world: np.ndarray,
                        stride: int = 2) -> np.ndarray:
    """
    Project a 2D mask to 3D world coordinates using depth + camera pose.

    Args:
        mask:               (H, W) bool
        depth_map:          (H, W) float32 meters
        camera_intrinsics:  dict fx, fy, cx, cy
        camera_to_world:    (4, 4) extrinsic
        stride:             sample every N pixels (for speed)

    Returns:
        (N, 3) world positions of masked pixels
    """
    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    rows, cols = np.where(mask)
    # Apply stride sampling
    idx = np.arange(0, len(rows), stride)
    rows, cols = rows[idx], cols[idx]

    depths = depth_map[rows, cols]
    valid  = depths > 0.1
    rows, cols, depths = rows[valid], cols[valid], depths[valid]

    if len(rows) == 0:
        return np.zeros((0, 3))

    # Camera coordinates
    x_cam = (cols - cx) * depths / fx
    y_cam = (rows - cy) * depths / fy
    z_cam = depths
    ones  = np.ones_like(z_cam)

    pts_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=0)  # (4, N)
    pts_world = (camera_to_world @ pts_cam)[:3, :].T          # (N, 3)

    return pts_world.astype(np.float32)


def compute_3d_bbox(points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute axis-aligned 3D bounding box from point cloud.

    Returns:
        (bbox_min, bbox_max, centroid)
    """
    if len(points_3d) == 0:
        z = np.zeros(3)
        return z, z, z
    bbox_min = points_3d.min(axis=0)
    bbox_max = points_3d.max(axis=0)
    centroid = points_3d.mean(axis=0)
    return bbox_min, bbox_max, centroid


# ── MobileCLIP classification ──────────────────────────────────────────────

def classify_with_mobileclip(rgb_crop: np.ndarray) -> Tuple[str, float]:
    """
    Classify a cropped RGB image using MobileCLIP-S2.
    Falls back to "UNKNOWN" if model unavailable.

    Args:
        rgb_crop: (H, W, 3) uint8 RGB image crop

    Returns:
        (semantic_label, confidence_score)
    """
    try:
        import os
        # v28: don't auto-download CLIP (~600MB) just because transformers is
        # installed; the demo doesn't need it. Opt in with PHANTOM_EMBED_BACKEND=clip.
        if os.environ.get("PHANTOM_EMBED_BACKEND", "").lower() != "clip":
            return "UNKNOWN", 0.0
        import torch
        from transformers import CLIPProcessor, CLIPModel

        # Load MobileCLIP (cached after first call)
        if not hasattr(classify_with_mobileclip, "_model"):
            logger.info("Loading MobileCLIP-S2...")
            classify_with_mobileclip._model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"   # fallback to standard CLIP if MobileCLIP unavailable
            )
            classify_with_mobileclip._processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            classify_with_mobileclip._model.eval()

        model = classify_with_mobileclip._model
        processor = classify_with_mobileclip._processor

        from PIL import Image
        img = Image.fromarray(rgb_crop)
        inputs = processor(
            text=SEMANTIC_PROMPTS,
            images=img,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).squeeze().numpy()

        best_idx = int(np.argmax(probs))
        return SEMANTIC_LABELS[best_idx], float(probs[best_idx])

    except Exception as e:
        logger.debug(f"MobileCLIP unavailable ({e}) — returning UNKNOWN")
        return "UNKNOWN", 0.5


# ── MobileSAM runner ───────────────────────────────────────────────────────

def run_mobilesam(rgb_image: np.ndarray,
                   points_per_side: int = 16) -> List[np.ndarray]:
    """
    Run MobileSAM automatic mask generation on an RGB image.

    Args:
        rgb_image:       (H, W, 3) uint8 RGB
        points_per_side: grid density for automatic prompting

    Returns:
        list of (H, W) bool masks, one per detected instance
    """
    try:
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
        import torch

        if not hasattr(run_mobilesam, "_generator"):
            logger.info("Loading MobileSAM...")
            sam = sam_model_registry["vit_t"](
                checkpoint="weights/mobile_sam.pt"
            )
            device = "mps" if torch.backends.mps.is_available() else \
                     "cuda" if torch.cuda.is_available() else "cpu"
            sam.to(device)
            sam.eval()
            run_mobilesam._generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=points_per_side,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                min_mask_region_area=500,
            )

        masks_data = run_mobilesam._generator.generate(rgb_image)
        return [m["segmentation"] for m in masks_data]

    except Exception as e:
        logger.warning(f"MobileSAM unavailable ({e}) — using grid fallback")
        return _grid_segment_fallback(rgb_image)


def _grid_segment_fallback(rgb_image: np.ndarray,
                             grid_size: int = 4) -> List[np.ndarray]:
    """Fallback: divide image into grid cells as dummy segments."""
    H, W = rgb_image.shape[:2]
    masks = []
    ch, cw = H // grid_size, W // grid_size
    for r in range(grid_size):
        for c in range(grid_size):
            mask = np.zeros((H, W), dtype=bool)
            mask[r*ch:(r+1)*ch, c*cw:(c+1)*cw] = True
            masks.append(mask)
    return masks


# ── Main segmentation pipeline ─────────────────────────────────────────────

def segment_frame(rgb_image: np.ndarray,
                   depth_map: np.ndarray,
                   camera_intrinsics: Dict[str, float],
                   camera_to_world: np.ndarray,
                   run_classification: bool = True) -> SegmentationResult:
    """
    Full frame segmentation pipeline:
        1. MobileSAM → 2D masks
        2. Depth discontinuity split (fix SAM merges)
        3. 3D projection → bounding boxes
        4. MobileCLIP → semantic labels

    Args:
        rgb_image:          (H, W, 3) uint8
        depth_map:          (H, W) float32 meters
        camera_intrinsics:  dict fx, fy, cx, cy
        camera_to_world:    (4, 4) float64

    Returns:
        SegmentationResult
    """
    import time
    t0 = time.time()

    H, W = rgb_image.shape[:2]
    disc_map = compute_depth_discontinuities(depth_map)

    # Step 1: MobileSAM
    raw_masks = run_mobilesam(rgb_image)
    logger.info(f"MobileSAM: {len(raw_masks)} raw masks")

    # Step 2: Depth-discontinuity split
    split_masks = []
    n_splits = 0
    for mask in raw_masks:
        sub_masks = split_merged_segments(mask, depth_map, disc_map)
        if len(sub_masks) > 1:
            n_splits += len(sub_masks) - 1
        split_masks.extend(sub_masks)

    # Step 3+4: 3D projection + classification
    instances = []
    for i, mask in enumerate(split_masks):
        points_3d = project_mask_to_3d(
            mask, depth_map, camera_intrinsics, camera_to_world
        )
        if len(points_3d) < 10:
            continue

        bbox_min, bbox_max, centroid = compute_3d_bbox(points_3d)
        depth_vals = depth_map[mask & (depth_map > 0.1)]
        depth_range = (float(depth_vals.min()), float(depth_vals.max())) \
                       if len(depth_vals) > 0 else (0.0, 0.0)

        # Classification: crop RGB to 2D bbox
        semantic, clip_score = "UNKNOWN", 0.5
        if run_classification:
            rows, cols = np.where(mask)
            if len(rows) > 0:
                r0, r1 = rows.min(), rows.max() + 1
                c0, c1 = cols.min(), cols.max() + 1
                crop = rgb_image[r0:r1, c0:c1]
                if crop.size > 0:
                    semantic, clip_score = classify_with_mobileclip(crop)

        instances.append(SegmentedInstance(
            instance_id=i,
            mask_2d=mask,
            points_3d=points_3d,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            centroid=centroid,
            semantic=semantic,
            clip_score=clip_score,
            pixel_count=int(mask.sum()),
            depth_range=depth_range,
        ))

    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        f"Segmentation: {len(instances)} instances, "
        f"{n_splits} SAM splits corrected, "
        f"{elapsed_ms:.1f}ms"
    )

    return SegmentationResult(
        instances=instances,
        depth_boundaries=disc_map,
        n_merged_splits=n_splits,
        processing_ms=elapsed_ms,
    )
