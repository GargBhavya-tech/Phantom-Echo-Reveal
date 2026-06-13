"""
PHANTOM-ECHO REVEAL — Mode A: Interactive Tap Handler
tap_handler.py

Processes a user tap on the phone screen:
    1. Ray-cast tap pixel → 3D world point
    2. Find bounding RED/UNKNOWN voxel region around hit point
    3. Check semantic cache (mobile_clip + cache_checker)
    4. Build RevealPayload → send to cloud via cloud_client
    5. Decode SVQ response → create spatial anchor
    6. Insert decoded Gaussians into live scene

Full pipeline target: <3s from tap to Gaussians visible in viewer.
"""

import numpy as np
import asyncio
import time
import uuid
import logging
from typing import Optional, Dict, Any, List, Tuple

from src.edge.network.cloud_client import CloudClient
from src.edge.network.payload_builder import build_reveal_payload
from src.edge.network.gaussian_decoder import decode_reveal_response, filter_duplicates
from src.edge.embedding.cache_checker import SemanticCacheChecker
from src.edge.anchor.spatial_anchor import SpatialAnchorManager, SpatialAnchor
from src.edge.buffer.frame_buffer import FrameBuffer
from src.shared.gaussian_format import TAG_RED, TAG_GREEN, Constants

logger = logging.getLogger(__name__)

TAP_REVEAL_TIMEOUT_S = Constants.VIDEOSCENE_MAX_LATENCY_S   # 3.0s
SEARCH_RADIUS_M = 1.0     # search for RED voxels within 1m of tap point


class TapHandler:
    """
    Mode A: Interactive tap pipeline.

    One TapHandler instance lives for the session lifetime.
    Call handle_tap() from the UI thread (or async context).
    """

    def __init__(self,
                 cloud_client: CloudClient,
                 cache_checker: SemanticCacheChecker,
                 anchor_manager: SpatialAnchorManager,
                 frame_buffer: FrameBuffer,
                 session_id: str,
                 floor_y: float = 0.0,
                 ceiling_y: float = 2.5):
        self._client   = cloud_client
        self._cache    = cache_checker
        self._anchors  = anchor_manager
        self._frames   = frame_buffer
        self._session  = session_id
        self._floor_y  = floor_y
        self._ceiling_y = ceiling_y
        self._pending: Dict[str, bool] = {}   # region_id → in-flight

    async def handle_tap(self,
                          tap_u: float,
                          tap_v: float,
                          depth_map: np.ndarray,
                          cam_to_world: np.ndarray,
                          camera_intrinsics: Dict[str, float],
                          existing_gaussians: List[Dict],
                          semantic_hint: str = "UNKNOWN"
                          ) -> Tuple[List[Dict[str, Any]], Optional[SpatialAnchor]]:
        """
        Full Mode A tap pipeline.

        Args:
            tap_u, tap_v:       screen pixel coordinates (float)
            depth_map:          (H, W) float32 current depth frame
            cam_to_world:       (4, 4) current camera pose
            camera_intrinsics:  {fx, fy, cx, cy}
            existing_gaussians: current scene Gaussians (for duplicate filter)
            semantic_hint:      from segmentation or user selection

        Returns:
            (new_gaussians, anchor) — new Gaussians to insert + their anchor
        """
        t0 = time.time()
        region_id = str(uuid.uuid4())[:8]

        # ── Step 1: Ray-cast tap → 3D world point ─────────────────────────
        world_hit = self._raycast_tap(
            tap_u, tap_v, depth_map, cam_to_world, camera_intrinsics
        )
        if world_hit is None:
            logger.warning("Tap raycast missed — no depth at tap pixel")
            return [], None

        logger.info(f"Tap hit: {world_hit.round(3)} (semantic_hint={semantic_hint})")

        # ── Step 2: Compute reveal bbox around hit point ───────────────────
        bbox_min, bbox_max = self._compute_reveal_bbox(
            world_hit, existing_gaussians
        )

        # ── Step 3: Semantic cache check ───────────────────────────────────
        current_frame = self._frames.latest()
        rgb_crop = None
        if current_frame is not None:
            from src.edge.buffer.frame_buffer import FrameBuffer
            crop_a, _ = self._frames.extract_stereo_crops(
                current_frame, bbox_min, bbox_max
            )
            rgb_crop = crop_a

        cached = self._cache.query(semantic_hint, bbox_min, bbox_max, rgb_crop)
        if cached is not None:
            logger.info(f"Cache hit for {semantic_hint} — skipping cloud call")
            anchor = self._anchors.create_anchor(
                world_hit, semantic_hint, bbox_min, bbox_max
            )
            filtered = filter_duplicates(existing_gaussians, cached)
            elapsed = time.time() - t0
            logger.info(f"Mode A (cached): {len(filtered)} Gaussians in {elapsed:.2f}s")
            return filtered, anchor

        # ── Step 4: Guard against duplicate in-flight requests ─────────────
        key = f"{bbox_min.round(2)}"
        if self._pending.get(key):
            logger.warning("Duplicate tap ignored — reveal in flight")
            return [], None
        self._pending[key] = True

        try:
            # ── Step 5: Build payload ──────────────────────────────────────
            stereo_a = stereo_b = None
            if current_frame is not None:
                stereo_a, stereo_b = self._frames.extract_stereo_crops(
                    current_frame, bbox_min, bbox_max
                )

            payload = build_reveal_payload(
                session_id=self._session,
                region_id=region_id,
                semantic=semantic_hint,
                confidence_tag=TAG_RED,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                floor_y=self._floor_y,
                ceiling_y=self._ceiling_y,
                stereo_crop_a=stereo_a,
                stereo_crop_b=stereo_b,
            )

            # ── Step 6: Cloud call ─────────────────────────────────────────
            compressed = await asyncio.wait_for(
                self._client.reveal(payload.to_dict() if hasattr(payload, "to_dict") else
                                    vars(payload)),
                timeout=TAP_REVEAL_TIMEOUT_S
            )

            if compressed is None:
                logger.warning("Cloud reveal returned None — using synthetic fallback")
                compressed = self._synthetic_fallback(semantic_hint, bbox_min, bbox_max)

            # ── Step 7: Decode ─────────────────────────────────────────────
            new_gaussians = decode_reveal_response(compressed)
            new_gaussians = filter_duplicates(existing_gaussians, new_gaussians)

            # ── Step 8: Create anchor + cache store ────────────────────────
            anchor = self._anchors.create_anchor(
                world_hit, semantic_hint, bbox_min, bbox_max
            )
            anchor.gaussian_ids = list(range(
                len(existing_gaussians),
                len(existing_gaussians) + len(new_gaussians)
            ))
            self._cache.store(semantic_hint, bbox_min, bbox_max,
                               new_gaussians, rgb_crop)

            elapsed = time.time() - t0
            logger.info(
                f"Mode A: {len(new_gaussians)} Gaussians in {elapsed:.2f}s "
                f"(target <{TAP_REVEAL_TIMEOUT_S}s)"
            )
            return new_gaussians, anchor

        except asyncio.TimeoutError:
            logger.error(f"Tap reveal timed out after {TAP_REVEAL_TIMEOUT_S}s")
            fallback_bytes = self._synthetic_fallback(semantic_hint, bbox_min, bbox_max)
            return decode_reveal_response(fallback_bytes), None
        except Exception as e:
            logger.error(f"Tap reveal failed: {e}")
            return [], None
        finally:
            self._pending[key] = False

    def handle_tap_sync(self, tap_u, tap_v, depth_map, cam_to_world,
                         camera_intrinsics, existing_gaussians,
                         semantic_hint="UNKNOWN"):
        """Synchronous wrapper for non-async callers."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.handle_tap(
            tap_u, tap_v, depth_map, cam_to_world,
            camera_intrinsics, existing_gaussians, semantic_hint
        ))

    # ── Helpers ────────────────────────────────────────────────────────────
    def _raycast_tap(self,
                      u: float, v: float,
                      depth_map: np.ndarray,
                      cam_to_world: np.ndarray,
                      intrinsics: Dict[str, float]) -> Optional[np.ndarray]:
        """Back-project tap pixel through depth map to world 3D point."""
        H, W = depth_map.shape
        ui = int(np.clip(u, 0, W - 1))
        vi = int(np.clip(v, 0, H - 1))

        # Sample depth in 5×5 neighbourhood (tap is imprecise)
        r0 = max(0, vi - 2); r1 = min(H, vi + 3)
        c0 = max(0, ui - 2); c1 = min(W, ui + 3)
        patch = depth_map[r0:r1, c0:c1]
        valid = patch[patch > 0.1]
        if len(valid) == 0:
            return None
        depth = float(np.median(valid))

        fx = intrinsics["fx"]; fy = intrinsics["fy"]
        cx = intrinsics["cx"]; cy = intrinsics["cy"]

        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth

        pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        pt_world = (cam_to_world @ pt_cam)[:3]
        return pt_world.astype(np.float32)

    def _compute_reveal_bbox(self,
                               hit_point: np.ndarray,
                               existing_gaussians: List[Dict],
                               default_half: float = 0.4
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute reveal bounding box around the tap hit point.

        Uses nearby RED/UNKNOWN Gaussians to fit a tighter bbox.
        Falls back to a default cube if no nearby RED Gaussians.
        """
        if not existing_gaussians:
            half = np.array([default_half, default_half, default_half])
            return hit_point - half, hit_point + half

        # Find RED Gaussians within search radius
        positions = np.array([g["position"] for g in existing_gaussians],
                              dtype=np.float32)
        tags = [g.get("tag", "RED") for g in existing_gaussians]
        dists = np.linalg.norm(positions - hit_point, axis=1)

        red_mask = np.array([t == TAG_RED for t in tags])
        nearby = (dists < SEARCH_RADIUS_M) & red_mask

        if np.sum(nearby) < 5:
            # Not enough RED nearby, use default cube
            half = np.array([default_half, default_half, default_half])
            return hit_point - half, hit_point + half

        nearby_pts = positions[nearby]
        bbox_min = nearby_pts.min(axis=0)
        bbox_max = nearby_pts.max(axis=0)

        # Ensure minimum size
        size = bbox_max - bbox_min
        for i in range(3):
            if size[i] < 0.1:
                bbox_min[i] = hit_point[i] - 0.05
                bbox_max[i] = hit_point[i] + 0.05

        return bbox_min, bbox_max

    def _synthetic_fallback(self,
                              semantic: str,
                              bbox_min: np.ndarray,
                              bbox_max: np.ndarray) -> bytes:
        """Generate synthetic Gaussians locally if cloud is unreachable."""
        from src.cloud.generation.videoscene_pipeline_fixed import generate_gaussians_for_region
        from src.cloud.compression.svq_endpoint import compress_reveal_response
        gaussians, _ = generate_gaussians_for_region(
            semantic=semantic,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            floor_y=self._floor_y,
            ceiling_y=self._ceiling_y,
            simulate=True,
        )
        return compress_reveal_response(gaussians)

    def update_floor_ceiling(self, floor_y: float, ceiling_y: float) -> None:
        self._floor_y   = floor_y
        self._ceiling_y = ceiling_y
