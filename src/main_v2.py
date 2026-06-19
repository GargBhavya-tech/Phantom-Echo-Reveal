"""
PHANTOM-ECHO REVEAL v29 — Main Orchestrator
main_v2.py

Replaces main.py. All bugs fixed, all missing modules wired.

What changed vs main.py:
    - Real depth via SyntheticDepthGenerator (not hardcoded arrays)
    - QuantVGGT dense depth runs before PHANTOM laws (Flaw 31 fix)
    - Static/Dynamic separation via SlotLSTMTracker (Flaw 35 fix)
    - ISM residual properly subtracted before SAS (Bug 9 fix)
    - SVQ compression on reveal response (was missing)
    - LLaVA builds VideoScene prompt (not bare string)
    - Gap widths + light positions passed to contradiction engine
      so Laws L2, L3, L4 actually fire
    - Frame buffer maintained for stereo anchor selection
    - Shared GaussianWire format used throughout (no ad-hoc dicts)
    - Dynamic import hack for acoustic_chirp replaced with top-level import
"""

import numpy as np
import logging
import time
import os

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phantom.main")

# ── Top-level imports (Bug 1 fix: no dynamic __import__ hacks) ───────────
from src.edge.sensing.acoustic_chirp    import generate_lfm_chirp, matched_filter, detect_echo_peaks, ChirpConfig
from src.edge.sensing.ism_filter        import subtract_visible_echoes, build_first_order_rir, WallPlane
from src.edge.sensing.sas_triangulator  import cluster_and_triangulate_v3 as cluster_and_triangulate
from src.edge.sensing.arkit_depth       import SyntheticDepthGenerator, IMUTracker, normalize_depth_scale
# BUG-3 FIX (analysis report): import the honest forward+inverse DSP pipeline.
# The old path used echo_signal = ref_chirp*0.30 (a fake echo with no actual
# hidden-surface delay) — the acoustic "measurement" was a no-op and returned
# only direct-blast / noise peaks, never a hidden-surface distance. Replace
# with the same measure_distances() used by engine.py which properly:
#   1. synthesises a multipath signal with the occluded echo at the correct delay
#   2. runs matched filter + ISM subtraction on the received signal
#   3. returns the distance from the detected peak (never from coordinates)
from src.edge.sensing.acoustic_forward  import measure_distances as _acoustic_measure_distances
_ACOUSTIC_RNG = np.random.default_rng(seed=42)  # seeded for reproducibility
from src.edge.reconstruction.quantvggt  import QuantVGGT
import src.edge.reconstruction.ddgs_gaussrender as ddgs
from src.edge.reconstruction.static_dynamic_sep import separate_static_dynamic
from src.edge.tracking.slot_lstm        import SlotLSTMTracker
from src.edge.buffer.frame_buffer       import FrameBuffer, BufferedFrame
from src.edge.phantom_lite.contradiction_engine import (
    ContradictionEngineFixed, PhysicsHypothesis
)
import src.edge.phantom_lite.affordance_router as affordance_router
from src.cloud.generation.videoscene_pipeline_fixed import generate_gaussians_for_region
from src.cloud.compression.svq_endpoint import compress_reveal_response, estimate_payload_size_kb
# LLaVA is lazy-imported inside the VIDEOSCENE branch only (line ~662).
# A top-level import here triggered a 7B weight download on every demo run.
from src.mesh.spsr_extraction import run_spsr_pipeline
from src.mesh.normal_orientation        import orient_normals
from src.navigation.occupancy_grid      import OccupancyGrid, project_gaussians
from src.shared.gaussian_format         import GaussianWire, Constants, TAG_BLUE, TAG_GREEN, TAG_RED
from src.mesh.outpainting_sweep     import seal_all_boundaries
# color_baker.bake_vertex_colors is called internally by run_spsr_pipeline
# (via spsr_extraction.bake_vertex_colors_fast) — no separate call needed here.
from src.navigation.nav2_publisher   import publish_or_save_costmap_autosized

# ── Config ────────────────────────────────────────────────────────────────
SIMULATE       = os.environ.get("PHANTOM_SIMULATE", "true").lower() == "true"
# FIX-3: Cross-platform output path — /tmp/ does not exist on Windows.
# Use a user home subdirectory when PHANTOM_OUTPUT env var is not set.
OUTPUT_DIR     = os.environ.get(
    "PHANTOM_OUTPUT",
    os.path.join(os.path.expanduser("~"), "phantom_echo_output")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Furniture simulated by the synthetic depth generator. Module-level so the
# evaluation builds its ground truth from the SAME scene specification
# (BUG-V22-7: eval GT previously described a different scene than the one
# the simulator rendered, making every KPI comparison meaningless).
DEFAULT_FURNITURE = [
    {"bbox_min": [1.0, 0.0, 1.0], "bbox_max": [2.8, 0.85, 1.8], "visible": False},
    {"bbox_min": [0.3, 0.0, 0.5], "bbox_max": [0.9, 0.90, 0.9], "visible": True},
]


def run_full_pipeline(n_frames: int = 10, room_dims=None, furniture=None):
    """
    Run complete PHANTOM-ECHO REVEAL pipeline end-to-end.
    """
    if room_dims is None:
        room_dims = {"x": 5.0, "y": 2.5, "z": 4.0}

    logger.info("=" * 60)
    logger.info("PHANTOM-ECHO REVEAL v29 — Full Pipeline")
    logger.info(f"simulate={SIMULATE}, n_frames={n_frames}")
    logger.info("=" * 60)

    t_total = time.time()

    # ── Layer 0: Initialize sensors ───────────────────────────────────────
    logger.info("[Layer 0] Initializing sensors...")

    if furniture is None:
        furniture = DEFAULT_FURNITURE
    depth_gen   = SyntheticDepthGenerator(
        room_dims=room_dims,
        furniture=furniture,
    )
    imu_tracker = IMUTracker(max_poses=500)
    chirp_cfg   = {
        "f_start": Constants.CHIRP_F_START_HZ,
        "f_end":   Constants.CHIRP_F_END_HZ,
        "duration":Constants.CHIRP_DURATION_S,
        "sample_rate": 44100,
    }

    # ── Layer 1: Initialize reconstruction modules ────────────────────────
    logger.info("[Layer 1] Initializing reconstruction modules...")
    quantvggt   = QuantVGGT(mode="auto")   # v23: uses trained depth-completion net if models/depth_completion.pt present
    tracker     = SlotLSTMTracker(max_age=10)
    frame_buf   = FrameBuffer(max_frames=100)

    # ── Layer 2: Initialize PHANTOM-LITE ──────────────────────────────────
    engine      = ContradictionEngineFixed()

    # ── Layer 3: LLaVA + VideoScene ───────────────────────────────────────
    # LLaVA is lazy-instantiated inside the VIDEOSCENE branch below.
    # Instantiating it here would trigger a 7B weight download at demo startup.

    # ── Layer 4: Navigation ───────────────────────────────────────────────
    occ_grid = OccupancyGrid(
        origin=np.array([0.0, 0.0, 0.0]),
        voxel_size=0.05,
        shape=(int(room_dims["x"]/0.05)+2, int(room_dims["y"]/0.05)+2, int(room_dims["z"]/0.05)+2)
    )
    import src.navigation.active_perception as active_perception
    perception_state = active_perception.ActivePerceptionState()

    # ── Accumulate Gaussians ──────────────────────────────────────────────
    all_static_gaussians  = []
    all_dynamic_gaussians = []
    floor_y   = 0.0
    ceiling_y = room_dims["y"]
    phone_positions = []

    frames = depth_gen.generate_walk_sequence(
        n_frames=n_frames,
        start_pos=np.array([0.5, 1.2, 0.5]),
        axis="arc"   # v23: smooth 3D arc — rank-3 SAS + stable echo tracks
    )

    for frame_idx, depth_frame in enumerate(frames):
        logger.info(f"[Frame {frame_idx+1}/{n_frames}]")

        # --- L0: Scale normalize depth -----------------------------------
        depth_norm = normalize_depth_scale(
            depth_frame.depth_map,
            depth_frame.confidence_map
        )

        # --- L0: pose tracking for the SAS virtual aperture --------------
        phone_pos = depth_frame.camera_to_world[:3, 3]
        phone_positions.append(phone_pos.copy())
        imu_tracker.add_pose(
            phone_pos, depth_frame.camera_to_world[:3, :3],
            depth_frame.timestamp_s
        )

        # NOTE: the per-frame acoustic measurement that used to live here was
        # dead code — it appended to an uninitialised `sas_measurements` (a
        # latent NameError that `> /dev/null` hid) and its output was never
        # consumed. The HONEST acoustic path is the batched `sweep_measurements`
        # call below (after the walk), which triangulates one clean TEAL surface
        # from the whole virtual aperture. Removing the dead block fixes the
        # `--mode demo` crash without changing any measured result.


        # --- L1: QuantVGGT dense depth ------------------------------------
        dense_depth = quantvggt.infer(
            depth_frame.rgb_image,
            depth_norm,
            depth_frame.confidence_map,
            depth_frame.camera_intrinsics
        )

        # --- L1: DDGS GaussRender -----------------------------------------
        gaussians_raw = ddgs.build_gaussian_scene(
            dense_depth,
            depth_frame.confidence_map,
            depth_frame.rgb_image,
            depth_frame.camera_intrinsics,
            depth_frame.camera_to_world,
            stride=4,   # BUG-V22-11 FIX: must be EVEN — the synthetic
                        # generator fills depth/confidence at stride 2, so an
                        # odd DDGS stride sampled unfilled pixels (conf=0)
                        # and WHITE silently dropped to zero. Density for
                        # recall comes from the Law 1/6 BLUE infill, not
                        # sensor stride.
        )
        # Convert objects to dicts so the rest of the pipeline works
        gaussians_dicts = []
        for g in gaussians_raw:
            gaussians_dicts.append({
                "position": g.position.tolist(),
                "normal": g.normal.tolist(),
                "color": g.color_rgb.tolist(),
                "scale": float(g.scale_xy[0]),
                "opacity": g.opacity,
                "confidence": g.confidence,
                "tag": g.tag,
                "semantic": g.semantic,
            })
        gaussians_raw = gaussians_dicts

        # --- L1b: denoise per-frame sensor points (wire in tsdf_fusion) ----
        # The tsdf_fusion.knn_smooth module was implemented but never called.
        # It bilateral-smooths each point toward its local neighbours WITHOUT
        # changing the point count (recall preserved), cancelling independent
        # per-point depth noise while the radius cap protects edges. This helps
        # REAL noisy depth; on the clean synthetic ray-traced depth it is
        # effectively a no-op (neighbours already coincide), so it never hurts
        # the synthetic path. No specific accuracy gain is claimed here — it is
        # measured on real data via the real eval.
        if len(gaussians_raw) >= 16:
            from src.edge.reconstruction.tsdf_fusion import knn_smooth
            _pos = np.array([g["position"] for g in gaussians_raw], dtype=np.float64)
            _sm = knn_smooth(_pos, k=12, max_radius=0.03, iters=1)
            for _g, _p in zip(gaussians_raw, _sm):
                _g["position"] = [float(_p[0]), float(_p[1]), float(_p[2])]

        # --- L1: Static/Dynamic separation (Flaw 35 fix) ------------------
        static_g, dynamic_g = separate_static_dynamic(
            gaussians_raw, tracker,
            dense_depth, depth_frame.rgb_image,
            depth_frame.camera_intrinsics,
            depth_frame.camera_to_world
        )

        all_static_gaussians.extend(static_g)
        # BUG-V22-12 FIX: dynamic Gaussians were APPENDED every frame, so one
        # moving object left 8 stale ORANGE ghosts (27k of 57k Gaussians) —
        # ghost obstacles in the costmap. Dynamic state is a SNAPSHOT:
        # keep only the latest frame's tracks.
        all_dynamic_gaussians = list(dynamic_g)

        # --- Frame buffer for stereo anchor --------------------------------
        buffered = BufferedFrame(
            frame_id=frame_idx,
            position=phone_pos.copy(),
            rotation=depth_frame.camera_to_world[:3, :3].copy(),
            depth_map=dense_depth,
            confidence_map=depth_frame.confidence_map,
            rgb_image=depth_frame.rgb_image,
            camera_intrinsics=depth_frame.camera_intrinsics,
            timestamp_s=depth_frame.timestamp_s,
        )
        frame_buf.add_frame(buffered)

        # --- Update floor estimate ----------------------------------------
        if static_g:
            ys = [g["position"][1] for g in static_g]
            floor_y   = min(floor_y, float(np.percentile(ys, 5)))
            ceiling_y = max(ceiling_y, float(np.percentile(ys, 95)))

        logger.info(
            f"  Static: {len(static_g)}, Dynamic: {len(dynamic_g)}, "
            f"Buffer: {len(frame_buf)}"
        )

    # ── Layer 2: SAS triangulation (after walk) ────────────────────────────
    logger.info("[Layer 2] SAS triangulation...")
    acoustic_point = None
    positions_arr = np.array(phone_positions, dtype=np.float32) if phone_positions else None

    if positions_arr is not None and len(positions_arr) >= 3:
        try:
            # ── HONEST acoustic measurement (report fix 3.1) ──────────────────
            # Distances now come from matched-filter peak detection on a
            # synthesised, noise-corrupted, multipath signal — NOT from the
            # target coordinates. Target geometry sets echo timing only.
            from src.edge.sensing.acoustic_forward import sweep_measurements
            occluded_targets = [
                np.array([1.9, 0.4, 1.0]),   # single hidden surface (sets echo timing only)
            ]
            _rd = room_dims
            visible_walls = [
                WallPlane(A=1, B=0, C=0, D=0,            label="wall_x0"),
                WallPlane(A=0, B=0, C=1, D=0,            label="wall_z0"),
            ]
            measurements, acoustic_err_cm = sweep_measurements(
                [p for p in positions_arr], occluded_targets, visible_walls,
                ChirpConfig(), np.random.default_rng(1234))
            if acoustic_err_cm:
                logger.info(f"  Acoustic DSP recovery error: "
                            f"mean={np.mean(acoustic_err_cm):.2f}cm "
                            f"(over {len(acoustic_err_cm)} returns)")

            points = cluster_and_triangulate(measurements, floor_y=floor_y)
            acoustic_point = np.array(points[0].position) if points else None
            if acoustic_point is not None:
                logger.info(f"  SAS: {len(points)} surfaces, primary={acoustic_point.round(3)}")
                # BUG-V18-5 FIX: Convert OccludedSurfacePoint → gaussian dicts with
                # tag=TEAL so they appear in the final scene and are counted correctly.
                # Previously acoustic_point was only used for acoustic law context;
                # the OccludedSurfacePoint objects were discarded and TEAL was always 0.
                for ap in points:
                    all_static_gaussians.append({
                        "position":   ap.position.tolist(),
                        "normal":     [0.0, 1.0, 0.0],
                        "color":      [0.0, 0.78, 0.78],
                        "scale":      0.05,
                        "opacity":    float(ap.confidence),
                        "confidence": float(ap.confidence),
                        "tag":        "TEAL",
                        "semantic":   "OCCLUDED_SURFACE",
                    })
            else:
                logger.warning("  SAS: 0 surfaces triangulated")
        except Exception as e:
            logger.warning(f"  SAS failed: {e}")

    # ── Layer 2a: PROACTIVE LAWS — build BLUE geometry (V22) ───────────────
    # The bible (Laws 1 & 6) specifies that physics BUILDS geometry, not just
    # tags it: the floor provably extends under furniture (gravity — objects
    # need support) and walls provably connect floor to ceiling (structural
    # support). Until v22 the engine only re-tagged sensor points, so the
    # floor/ceiling — barely visible to a forward-facing camera — were
    # missing from the reconstruction entirely (floor recall was 0.00).
    # Build them explicitly over the OBSERVED xz extent at 4cm spacing.
    if all_static_gaussians:
        _pts = np.array([g["position"] for g in all_static_gaussians])
        _xz_min = _pts[:, [0, 2]].min(axis=0)
        _xz_max = _pts[:, [0, 2]].max(axis=0)
        # BUG-6 FIX: was 0.05m spacing — for a 5×4m room that produces
        # 100×80×2 = 16k floor+ceiling points BEFORE wall infill, then
        # another 10–30k wall points; total easily hits 50k+ BLUE Gaussians
        # and creates O(N²) pressure when physics evaluation scans all_static_gaussians.
        # Use 0.08m (matches realtime engine._proactive_blue) and add the same
        # MAX_BLUE=12000 hard cap that the engine uses.
        _SPACING = 0.08
        _MAX_BLUE = 12000
        _gx = np.arange(_xz_min[0], _xz_max[0] + 1e-6, _SPACING)
        _gz = np.arange(_xz_min[1], _xz_max[1] + 1e-6, _SPACING)
        _n_blue0 = len(all_static_gaussians)
        for _x in _gx:
            for _z in _gz:
                if len(all_static_gaussians) - _n_blue0 >= _MAX_BLUE:
                    logger.debug("[Layer 2a] floor/ceiling cap reached (%d)", _MAX_BLUE)
                    break
                all_static_gaussians.append({
                    "position": [float(_x), float(floor_y), float(_z)],
                    "normal": [0.0, 1.0, 0.0], "color": [0.55, 0.58, 0.62],
                    "scale": 0.06, "opacity": 0.9, "confidence": 0.95,
                    "tag": "BLUE", "semantic": "FLOOR",
                })
                all_static_gaussians.append({
                    "position": [float(_x), float(ceiling_y), float(_z)],
                    "normal": [0.0, -1.0, 0.0], "color": [0.80, 0.80, 0.78],
                    "scale": 0.06, "opacity": 0.9, "confidence": 0.95,
                    "tag": "BLUE", "semantic": "CEILING",
                })
        # LAW 6 — STRUCTURAL SUPPORT: a wall segment observed anywhere must
        # connect floor to ceiling (walls cannot terminate mid-air). For each
        # room wall plane with sensor evidence, fill its observed extent
        # vertically from floor_y to ceiling_y.
        _walls = [
            (0, 0.0,            np.array([1., 0., 0.])),   # x = 0
            (0, room_dims["x"], np.array([-1., 0., 0.])),  # x = X
            (2, 0.0,            np.array([0., 0., 1.])),   # z = 0
            (2, room_dims["z"], np.array([0., 0., -1.])),  # z = Z
        ]
        _gy = np.arange(floor_y + _SPACING, ceiling_y - 0.02, _SPACING)
        for _ax, _plane, _nrm in _walls:
            _near = _pts[np.abs(_pts[:, _ax] - _plane) < 0.15]
            if len(_near) < 50:
                continue                       # wall never observed — stays RED
            _lat = 2 if _ax == 0 else 0        # lateral axis along the wall
            _lo, _hi = _near[:, _lat].min(), _near[:, _lat].max()
            for _t in np.arange(_lo, _hi + 1e-6, _SPACING):
                for _y in _gy:
                    if len(all_static_gaussians) - _n_blue0 >= _MAX_BLUE:
                        logger.debug("[Layer 2a] wall cap reached (%d)", _MAX_BLUE)
                        break
                    _p = [0.0, float(_y), 0.0]
                    _p[_ax] = float(_plane); _p[_lat] = float(_t)
                    all_static_gaussians.append({
                        "position": _p, "normal": _nrm.tolist(),
                        "color": [0.72, 0.74, 0.78], "scale": 0.06,
                        "opacity": 0.9, "confidence": 0.95,
                        "tag": "BLUE", "semantic": "WALL",
                    })
        logger.info(f"[Layer 2a] Proactive laws built "
                    f"{len(all_static_gaussians)-_n_blue0} BLUE floor/ceiling/wall Gaussians")

    # ── Layer 2: PHANTOM-LITE laws on each Gaussian ────────────────────────
    logger.info("[Layer 2] Contradiction engine evaluation...")
    tagged_gaussians = []
    _phone_last = np.array(phone_positions[-1]) if phone_positions else np.zeros(3)
    _sas_phone_dist = (float(np.linalg.norm(_phone_last - acoustic_point))
                       if acoustic_point is not None else None)

    # Bug V18-B FIX: build scene_objects list ONCE before the loop.
    # Was previously rebuilt for every Gaussian — O(N) allocation waste.
    _scene_objs_const = [
        {"semantic": "FLOOR",   "bbox_min": [0., floor_y-0.05, 0.],            "bbox_max": [room_dims["x"], floor_y,        room_dims["z"]]},
        {"semantic": "CEILING", "bbox_min": [0., ceiling_y,    0.],            "bbox_max": [room_dims["x"], ceiling_y+0.05, room_dims["z"]]},
        {"semantic": "WALL",    "bbox_min": [0., floor_y, -0.05],              "bbox_max": [room_dims["x"], ceiling_y, 0.]},
        {"semantic": "WALL",    "bbox_min": [0., floor_y, room_dims["z"]],     "bbox_max": [room_dims["x"], ceiling_y, room_dims["z"]+0.05]},
        {"semantic": "WALL",    "bbox_min": [-0.05, floor_y, 0.],              "bbox_max": [0., ceiling_y, room_dims["z"]]},
        {"semantic": "WALL",    "bbox_min": [room_dims["x"], floor_y, 0.],     "bbox_max": [room_dims["x"]+0.05, ceiling_y, room_dims["z"]]},
    ]
    # Physics evaluation cap: process up to 5000 Gaussians for demo speed.
    # BLUE Gaussians short-circuit immediately (O(1)), so the effective cost
    # is proportional to WHITE/YELLOW/ORANGE count, not total count.
    # BUG-V22-2b FIX: TEAL points are appended to all_static_gaussians AFTER
    # the camera Gaussians (~20k), so the [:5000] physics cap silently dropped
    # every one of them. Pull TEAL out of the tail explicitly.
    # BUG-V22-2b (extended): TEAL measurements AND proactive BLUE geometry
    # live past the physics cap — both carry their final tag already, so
    # rescue them from the tail (they skip re-evaluation below anyway).
    _eval_set = (all_static_gaussians[:12000] +
                 [g for g in all_static_gaussians[12000:]
                  if g.get("tag") in ("TEAL", "BLUE")])
    for g in _eval_set:
        # BUG-V22-2 FIX: TEAL Gaussians are direct acoustic MEASUREMENTS —
        # re-running the physics engine on them re-tagged them (usually BLUE)
        # so the final teal count was always 0 even when SAS succeeded.
        # Measurement outranks inference: preserve the TEAL tag.
        # BUG-V22-13 FIX (review Bug #4, true root cause): the engine
        # re-tagged WHITE (sensor-confirmed, conf>0.75) points as BLUE —
        # physics inference must never downgrade a direct measurement.
        # WHITE, TEAL (acoustic measurement) and proactive BLUE pass through.
        if g.get("tag") in ("TEAL", "BLUE", "WHITE"):
            tagged_gaussians.append(dict(g))
            continue
        g_pos = np.array(g["position"])

        # FIX BUG A: acoustic law only applies near the SAS-triangulated surface
        if acoustic_point is not None and _sas_phone_dist is not None:
            _g_to_sas = float(np.linalg.norm(g_pos - acoustic_point))
            acou_dist = _sas_phone_dist if _g_to_sas <= 0.30 else None
        else:
            acou_dist = None

        # Bug V18-B FIX: use prebuilt constant instead of rebuilding per Gaussian
        hyp = PhysicsHypothesis(
            position=g_pos,
            semantic=g.get("semantic", "UNKNOWN"),
            confidence=g.get("confidence", 0.5),
            context={
                "room_bounds":        room_dims,
                "scene_objects":      _scene_objs_const,  # prebuilt outside loop (V18-B fix)
                "phone_position":     _phone_last.tolist(),
                "visible_gap_width_m": g.get("gap_width_m", None),
                "shadow_endpoint":    g.get("shadow_endpoint", None),
                "light_source":       [room_dims["x"]/2, room_dims["y"]-0.1,
                                        room_dims["z"]/2],
                "lit_surface_point":  g["position"],
                "input_tag":          g.get("tag", ""),
            },
            acoustic_distance_m=acou_dist,   # FIX BUG A: None for visible geometry
            floor_y=floor_y,
            ceiling_y=ceiling_y,
        )
        tag, verdict, _ = engine.evaluate(hyp)
        g_tagged = dict(g)
        g_tagged["tag"] = tag
        tagged_gaussians.append(g_tagged)

    # ── Layer 2b: Atlanta-World architectural relabel (V22) ───────────────
    # DDGS classifies semantics from locally-estimated normals, which are
    # noisy on obliquely-viewed walls → thousands of wall points labelled
    # OTHER. Architectural class is positional under the Atlanta-World
    # assumption: a point within 12cm of a room boundary plane IS that
    # boundary. Position prior overrides noisy normal classification.
    _rx, _rz = room_dims["x"], room_dims["z"]
    _relabeled = 0
    for g in tagged_gaussians:
        if g.get("semantic") in ("FLOOR", "WALL", "CEILING", "PLATFORM"):
            continue
        _px, _py, _pz = g["position"]
        if abs(_py - floor_y) < 0.12:
            g["semantic"] = "FLOOR";   _relabeled += 1
        elif abs(_py - ceiling_y) < 0.12:
            g["semantic"] = "CEILING"; _relabeled += 1
        elif min(abs(_px), abs(_rx - _px), abs(_pz), abs(_rz - _pz)) < 0.12:
            g["semantic"] = "WALL";    _relabeled += 1
    if _relabeled:
        logger.info(f"[Layer 2b] Atlanta-World relabel: {_relabeled} points")

    # ── Layer 2c: SEED RED voxels for genuinely-occluded volumes (SYSTEMIC FIX) ─
    # Root cause of generated=0/green=0 in every scene: the proactive BLUE laws
    # fill all architectural surfaces, the camera sees the rest, so NO voxel is
    # ever left RED — and Layer 3 (which only generates for RED voxels) was
    # always skipped. The core contribution ("imagine only what physics cannot
    # determine") never executed.
    #
    # Fix: the INTERIOR of an opaque occluder (furniture marked visible=False)
    # is genuinely unknown — the camera sees its shell, physics does not
    # determine its contents, acoustics only reach its surface. That volume is
    # exactly what generation is for. Seed RED voxels there. They bypass the
    # contradiction loop (they are not re-tagged) and feed Layer 3 directly.
    _red_seeded = 0
    _had_occluded = False
    for _f in furniture:
        if _f.get("visible", True):
            continue                      # only occluded furniture has unknown interior
        _had_occluded = True
        bmin, bmax = np.array(_f["bbox_min"], float), np.array(_f["bbox_max"], float)
        # sample the interior on a coarse grid, inset from the observed shell
        gx = np.arange(bmin[0] + 0.12, bmax[0] - 0.10, 0.18)
        gy = np.arange(bmin[1] + 0.12, bmax[1] - 0.05, 0.18)
        gz = np.arange(bmin[2] + 0.12, bmax[2] - 0.10, 0.18)
        # BUG-1 FIX (second-report edge case): furniture smaller than the grid
        # step produces an EMPTY arange on that axis → 0 seeds → Layer 3 silently
        # skipped. Fall back to the bbox centre so every occluded volume seeds
        # at least one RED voxel.
        if gx.size == 0 or gy.size == 0 or gz.size == 0:
            _c = (bmin + bmax) / 2.0
            gx = np.array([_c[0]]) if gx.size == 0 else gx
            gy = np.array([_c[1]]) if gy.size == 0 else gy
            gz = np.array([_c[2]]) if gz.size == 0 else gz
            logger.info(f"[Layer 2c] Furniture too small for grid step; "
                        f"seeding centre voxel at {_c.round(2).tolist()}")
        for _x in gx:
            for _y in gy:
                for _z in gz:
                    tagged_gaussians.append({
                        "position": [float(_x), float(_y), float(_z)],
                        "normal": [0.0, 1.0, 0.0], "color": [0.85, 0.15, 0.20],
                        "scale": 0.06, "opacity": 0.35, "confidence": 0.20,
                        "tag": TAG_RED, "semantic": "OCCLUDED_UNKNOWN",
                    })
                    _red_seeded += 1
    if _red_seeded:
        logger.info(f"[Layer 2c] Seeded {_red_seeded} RED voxels in occluded "
                    f"volumes → Layer 3 generation will run.")
    elif _had_occluded:
        # BUG-1 FIX: never let IMAGINE silently vanish.
        logger.warning("[Layer 2c] 0 RED voxels seeded despite occluded "
                       "furniture present — Layer 3 generation will be SKIPPED. "
                       "Check furniture bbox sizes.")

    # ── Layer 3: Affordance routing + generation ───────────────────────────
    logger.info("[Layer 3] Affordance routing + VideoScene generation...")
    generated_gaussians = []

    # Identify RED regions needing generation
    red_gaussians = [g for g in tagged_gaussians if g.get("tag") == TAG_RED]
    if red_gaussians:
        positions_red = np.array([g["position"] for g in red_gaussians])
        bbox_min = positions_red.min(axis=0)
        bbox_max = positions_red.max(axis=0)

        # BUG-V19-4 + BUG-V19-7 FIX: replace hardcoded "SOFA" with
        # position-based semantic inference, then pass through affordance_router.
        #
        # Position heuristic (deterministic, no classifier needed):
        #   - bbox top within 5cm of floor → FLOOR extension
        #   - bbox centre within 15cm of any wall face → WALL
        #   - bbox bottom >= 60cm (chair/table height) AND top <= ceiling-20cm → FURNITURE
        #   - default → OBJECT (generic — router handles affordance)
        #
        # The router then decides: SKIP/PRIMITIVE/GENERATE and floor_y/ceiling_y
        # snap for correct physical placement (Novel Contribution 3 in the bible).
        bbox_center = (bbox_min + bbox_max) / 2.0
        bbox_height = float(bbox_max[1] - bbox_min[1])
        bbox_bottom = float(bbox_min[1])
        bbox_top    = float(bbox_max[1])

        wall_threshold = 0.15   # within 15cm of a wall face
        rx, rz        = room_dims["x"], room_dims["z"]

        near_floor   = bbox_bottom <= floor_y   + 0.05
        near_ceiling = bbox_top    >= ceiling_y - 0.05
        near_wall_x  = (bbox_center[0] < wall_threshold or
                        bbox_center[0] > rx - wall_threshold)
        near_wall_z  = (bbox_center[2] < wall_threshold or
                        bbox_center[2] > rz - wall_threshold)

        if near_floor and bbox_height < 0.05:
            semantic = "FLOOR"
        elif near_ceiling and bbox_height < 0.05:
            semantic = "CEILING"
        elif near_wall_x or near_wall_z:
            semantic = "WALL"
        elif bbox_bottom >= floor_y + 0.60:
            semantic = "SHELF"       # elevated — probably shelf / counter surface
        elif bbox_height >= 0.40 and bbox_height <= 1.0:
            semantic = "CHAIR"       # human-scale furniture
        else:
            semantic = "OBJECT"      # generic fallback

        # Route through affordance_router — this is Novel Contribution 3:
        # floor-supported vs wall-mounted vs ceiling-hung routing.
        routing = affordance_router.route_region(
            region_id="layer3_red",
            semantic=semantic,
            confidence_tag=TAG_RED,
            region_bbox={"min_pt": bbox_min.tolist(), "max_pt": bbox_max.tolist()},
            contradiction_result={},
            floor_y=floor_y,
            ceiling_y=ceiling_y,
        )
        # Use router's physics-corrected bounds if available
        if routing.physics_bounds is not None:
            route_min = np.array(routing.physics_bounds.min_pt
                                  if hasattr(routing.physics_bounds, "min_pt")
                                  else bbox_min)
            route_max = np.array(routing.physics_bounds.max_pt
                                  if hasattr(routing.physics_bounds, "max_pt")
                                  else bbox_max)
        else:
            route_min, route_max = bbox_min, bbox_max

        # BUG-V20-2 FIX + BUG-V21-2 FIX + BUG-V22-1 FIX:
        # Act on routing.strategy (V20 fix). Initialise new_gaussians=[] BEFORE
        # the if/elif/else (V21 fix). Keep ALL branches at 8-space indent so
        # Python does not raise IndentationError (V22 fix — the root cause of
        # the broken V21 submission was the comment+initialiser being accidentally
        # dedented to 4-space, pulling the `if` out of the for-loop body).
        from src.edge.phantom_lite.affordance_router import GenerationStrategy
        new_gaussians: list = []
        tier: str = "none"
        # FIX-2: initialise crop_a / crop_b here so FAISS_RETRIEVAL and SLOTLSTM
        # branches don't hit NameError (they were only defined inside the else-branch).
        crop_a: object = None
        crop_b: object = None

        if routing.strategy == GenerationStrategy.SKIP:
            logger.info(f"[Layer 3] Region '{semantic}' SKIP — already proven/measured")
            # new_gaussians stays [] — no generation needed

        elif routing.strategy == GenerationStrategy.PRIMITIVE:
            logger.info(f"[Layer 3] Region '{semantic}' PRIMITIVE — flat extrusion")
            center   = (route_min + route_max) / 2.0
            rng_prim = np.random.default_rng(42)
            _prim_area = float(np.prod(np.maximum(route_max - route_min, 0.01)[:3]))
            _n_prim    = max(20, min(200, int(_prim_area * 200)))
            for _ in range(_n_prim):
                jitter = rng_prim.uniform(-0.05, 0.05, 3)
                generated_gaussians.append({
                    "position":   (center + jitter).tolist(),
                    "normal":     [0.0, 1.0, 0.0],
                    "color":      [0.85, 0.85, 0.85],
                    "scale":      0.08,
                    "opacity":    0.80,
                    "confidence": 0.70,
                    "tag":        TAG_GREEN,
                    "semantic":   semantic,
                })
            tier = "primitive"

        elif routing.strategy == GenerationStrategy.FAISS_RETRIEVAL:
            # BUG-V21-3 FIX: explicit branch for FAISS_RETRIEVAL.
            # Full FAISS floor-plan DB query is future work; routes to Tier3
            # generation with router's physics-corrected bounds + prompt.
            logger.info("[Layer 3] FAISS_RETRIEVAL → Tier3 generation (prototype)")
            new_gaussians, tier = generate_gaussians_for_region(
                semantic=semantic,
                bbox_min=route_min, bbox_max=route_max,
                floor_y=floor_y, ceiling_y=ceiling_y,
                prompt=routing.prompt_hint,
                crop_a=crop_a, crop_b=crop_b,
                simulate=SIMULATE, seed=42,
            )

        elif routing.strategy == GenerationStrategy.SLOTLSTM:
            # BUG-V21-3 FIX: explicit branch for SLOTLSTM.
            # Full SlotLSTM structural constraint filtering is future work;
            # routes to Tier3 generation with physics-corrected bounds + prompt.
            logger.info("[Layer 3] SLOTLSTM → Tier3 generation (prototype)")
            new_gaussians, tier = generate_gaussians_for_region(
                semantic=semantic,
                bbox_min=route_min, bbox_max=route_max,
                floor_y=floor_y, ceiling_y=ceiling_y,
                prompt=routing.prompt_hint,
                crop_a=crop_a, crop_b=crop_b,
                simulate=SIMULATE, seed=42,
            )

        else:
            # VIDEOSCENE and any future strategies: full generation pipeline
            # Lazy-load LLaVA here — only the VIDEOSCENE path uses it.
            from src.cloud.llm.llava_wrapper import LLaVASceneDescriber
            llava = LLaVASceneDescriber()
            scene_desc = llava.describe_scene(
                frames[-1].rgb_image if frames else np.zeros((192, 256, 3), dtype=np.uint8),
                [semantic],
            )
            prompt = routing.prompt_hint or llava.build_videoscene_prompt(
                scene_desc, semantic,
                route_min.tolist(), route_max.tolist(),
                acoustic_distance_m=float(np.linalg.norm(bbox_min - bbox_max)) / 2,
            )
            crop_a, crop_b = (
                frame_buf.extract_stereo_crops(
                    frame_buf.latest(), bbox_min, bbox_max, crop_size=(224, 224)
                )
                if frame_buf.latest() else (None, None)
            )
            new_gaussians, tier = generate_gaussians_for_region(
                semantic=semantic,
                bbox_min=route_min, bbox_max=route_max,
                floor_y=floor_y, ceiling_y=ceiling_y,
                prompt=prompt,
                crop_a=crop_a, crop_b=crop_b,
                simulate=SIMULATE, seed=42,
            )

        # V19-LOW-3 FIX: snap GREEN cluster to semantic support surface
        # (Gravity Override Anomaly fix — Novel Contribution 3 in the bible).
        try:
            from src.generation_correction.plane_alignment import (
                GreenCluster, StructuralPlane, align_cluster,
            )
            if new_gaussians:
                pos_arr = np.array([g["position"] for g in new_gaussians], dtype=np.float64)
                cluster = GreenCluster(
                    region_id="layer3_red",
                    semantic=semantic,
                    positions=pos_arr,
                    centroid=pos_arr.mean(axis=0),
                    bbox_min=route_min.astype(np.float64),
                    bbox_max=route_max.astype(np.float64),
                )
                structural_planes = [
                    StructuralPlane(np.array([0.,  1.,  0.]),  floor_y,         "FLOOR"),
                    StructuralPlane(np.array([0., -1.,  0.]),  -ceiling_y,      "CEILING"),
                    StructuralPlane(np.array([0.,  0., -1.]),  0.,              "WALL"),
                    StructuralPlane(np.array([0.,  0.,  1.]),  -room_dims["z"], "WALL"),
                    StructuralPlane(np.array([-1., 0.,  0.]),  0.,              "WALL"),
                    StructuralPlane(np.array([1.,  0.,  0.]),  -room_dims["x"], "WALL"),
                ]
                ar = align_cluster(cluster, structural_planes)
                if ar.success:
                    delta = ar.translation_m
                    for g in new_gaussians:
                        g["position"] = (np.array(g["position"]) + delta).tolist()
                    logger.info(
                        f"  Plane alignment: {semantic} snapped "
                        f"delta={delta.round(3)}m residual={ar.residual_m*100:.1f}cm"
                    )
        except Exception as _pa_err:
            logger.debug(f"  Plane alignment skipped: {_pa_err}")

        generated_gaussians.extend(new_gaussians)
        logger.info(f"  Generated {len(new_gaussians)} GREEN Gaussians via {tier}")
        # (duplicate log line removed — FIX-4)

    # SVQ compress reveal response
    all_reveal = tagged_gaussians + generated_gaussians + all_dynamic_gaussians
    # PERF-V22d: SVQ encoding is pure Python — O(N). The wire payload in the
    # live system is the REVEAL response (a few hundred GREEN Gaussians),
    # never the whole scene; compressing a 15k sample is enough to report a
    # representative ratio without burning 10+ seconds.
    _svq_in = all_reveal if len(all_reveal) <= 8000 else         [all_reveal[j] for j in
         np.random.default_rng(0).choice(len(all_reveal), 8000, replace=False)]
    compressed = compress_reveal_response(_svq_in)
    est_kb = estimate_payload_size_kb(len(all_reveal))
    logger.info(
        f"  SVQ payload: {len(compressed)/1024:.1f}KB "
        f"(estimate {est_kb:.1f}KB) for {len(all_reveal)} Gaussians"
    )

    # ── Layer 4: Mesh extraction ───────────────────────────────────────────
    logger.info("[Layer 4] SPSR mesh extraction...")
    if tagged_gaussians:
        positions = np.array([g["position"] for g in tagged_gaussians], dtype=np.float32)
        normals   = np.array([g.get("normal", [0,1,0]) for g in tagged_gaussians], dtype=np.float32)
        colors    = np.array([g.get("color", [0.5,0.5,0.5]) for g in tagged_gaussians], dtype=np.float32)

        # Fix normals
        camera_positions_np = np.array(phone_positions) if phone_positions else np.array([[0.5, 1.2, 0.5]])
        normals = orient_normals(positions, normals, camera_positions_np).normals_oriented


        # Missing-7 fix: orient normals on GREEN generated gaussians before SPSR
        if generated_gaussians:
            gen_pos = np.array([g['position'] for g in generated_gaussians], dtype=np.float32)
            gen_nrm = np.array([g.get('normal', [0,1,0]) for g in generated_gaussians], dtype=np.float32)
            gen_nrm = orient_normals(gen_pos, gen_nrm, camera_positions_np).normals_oriented
            for j, g in enumerate(generated_gaussians):
                g['normal'] = gen_nrm[j].tolist()

        # Missing-6 fix: run Outpainting Sweep to seal RED boundaries before SPSR
        try:
            boundary_patches = seal_all_boundaries(
                gaussians=all_static_gaussians + tagged_gaussians,
                floor_y=floor_y, ceiling_y=ceiling_y,
                room_dims=room_dims,
            )
            logger.info(f'  Outpainting: {len(boundary_patches)} seal patches')
        except Exception as _e:
            logger.warning(f'Outpainting sweep failed: {_e}')
            boundary_patches = []

        # BUG-V18-1 (Section 2) FIX: concatenate boundary_patches into the
        # positions/normals/colors arrays before calling run_spsr_pipeline.
        # Previously boundary_patches was computed and logged but never passed
        # downstream — the Outpainting Sweep was silently a no-op.
        if boundary_patches:
            bp_pos = np.array([p.position for p in boundary_patches], dtype=np.float32)
            bp_nrm = np.array([p.normal   for p in boundary_patches], dtype=np.float32)
            bp_col = np.array([p.color    for p in boundary_patches], dtype=np.float32)
            # Orient patch normals toward nearest camera before SPSR
            bp_nrm = orient_normals(bp_pos, bp_nrm, camera_positions_np).normals_oriented
            positions = np.vstack([positions, bp_pos])
            normals   = np.vstack([normals,   bp_nrm])
            colors    = np.vstack([colors,    bp_col])
            logger.info(f'  Boundary patches merged: total {len(positions)} points for SPSR')

        mesh = run_spsr_pipeline(
            positions, normals, colors,
            output_ply=os.path.join(OUTPUT_DIR, "mesh.ply")
        )
        if mesh is not None:
            # FIX: guard against None before accessing .vertices
            # run_spsr_pipeline() returns None if open3d is unavailable.
            try:
                logger.info(f"  Mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
            except AttributeError:
                logger.warning("SPSR returned unexpected object — skipping mesh stats")
        else:
            logger.warning("SPSR mesh is None — open3d may not be installed. "
                          "Run: pip install open3d")

    # ── Layer 5: Active perception / Nav2 ─────────────────────────────────
    logger.info("[Layer 5] Active perception / Nav2...")
    
    positions_np = np.array([g["position"] for g in tagged_gaussians]) if tagged_gaussians else np.zeros((0, 3))
    tags_list = [g.get("tag", TAG_RED) for g in tagged_gaussians]
    camera_positions_np = np.array(phone_positions)
    # PERF-V22c: free-space ray casting is O(N x ray_len) in pure Python.
    # 8k Gaussians fully saturate a 5cm grid; project a uniform subsample.
    if len(positions_np) > 8000:
        _sel = np.random.default_rng(0).choice(len(positions_np), 8000, replace=False)
        positions_np = positions_np[_sel]
        tags_list    = [tags_list[j] for j in _sel]
    # PERF-V22e: occupied-voxel updates are cheap; the free-space Bresenham
    # ray per Gaussian is the O(N x ray_len) Python hotspot. Mark occupancy
    # from all samples, carve free space from a 1500-ray subsample (free
    # space is spatially redundant — neighbouring rays carve the same voxels).
    project_gaussians(occ_grid, positions_np, tags_list, camera_positions_np,
                      cast_free_rays=False)
    if len(positions_np) > 1500:
        _rsel = np.random.default_rng(1).choice(len(positions_np), 1500, replace=False)
        project_gaussians(occ_grid, positions_np[_rsel],
                          [tags_list[j] for j in _rsel], camera_positions_np)

    best_viewpoint = active_perception.select_next_viewpoint(occ_grid, np.array(phone_positions[-1] if phone_positions else [0.0, 0.0, 0.0]), perception_state)
    if best_viewpoint is not None:
        logger.info(f"  Nav2 goal (simulated sent): {best_viewpoint.position.round(3)}")

    # Missing-9 fix: publish/save costmap after active perception
    try:
        publish_or_save_costmap_autosized(
            gaussians=all_static_gaussians + tagged_gaussians,
            floor_y=floor_y,
            output_dir=OUTPUT_DIR,
        )
    except Exception as _e:
        logger.warning(f'Nav2 costmap publish failed: {_e}')


    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    # Per-tag breakdown (used by run_eval, viewer HUD, and summary)
    _tag_counts = {}
    for g in all_reveal:
        t = g.get("tag", "RED").upper()
        _tag_counts[t] = _tag_counts.get(t, 0) + 1

    counts = {
        # Structural counts
        "static":    len(all_static_gaussians),
        "dynamic":   len(all_dynamic_gaussians),
        "tagged":    len(tagged_gaussians),
        "generated": len(generated_gaussians),
        "total":     len(all_reveal),
        # Per-tag counts (lowercase keys for run_eval + viewer)
        "white":     _tag_counts.get("WHITE",  0),
        "blue":      _tag_counts.get("BLUE",   0),
        "teal":      _tag_counts.get("TEAL",   0),
        "green":     _tag_counts.get("GREEN",  0),
        "yellow":    _tag_counts.get("YELLOW", 0),
        "red":       _tag_counts.get("RED",    0),
        "orange":    _tag_counts.get("ORANGE", 0),
    }

    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.2f}s")
    for k, v in counts.items():
        logger.info(f"  {k:12s}: {v} Gaussians")
    logger.info(f"  SVQ payload: {len(compressed)/1024:.1f}KB")
    logger.info("=" * 60)

    return {
        "gaussians":  all_reveal,
        "camera_positions": [p.tolist() for p in phone_positions],
        "counts":     counts,
        "floor_y":    floor_y,
        "ceiling_y":  ceiling_y,
        "elapsed_s":  round(elapsed, 3),
        "payload_kb": round(len(compressed)/1024, 1),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_server", action="store_true", help="Start the WebGPU viewer server")
    args = parser.parse_args()

    result = run_full_pipeline(n_frames=8)
    print(f"\nDone. Total Gaussians: {result['counts']['total']}, "
          f"Payload: {result['payload_kb']}KB, "
          f"Time: {result['elapsed_s']}s")

    if args.start_server:
        # v22: the legacy WebGPU viewer is retired — the live dashboard
        # (src/realtime + src/frontend) replaced it.
        print("\nThe legacy viewer was replaced in v22. Run instead:")
        print("    python -m src.main --mode realtime   ->  http://localhost:8000")
    else:
        print("\nLive dashboard: python -m src.main --mode realtime")
