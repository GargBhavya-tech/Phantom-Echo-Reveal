# PHANTOM-ECHO REVEAL

- **Problem Statement Number** - 09
- **Problem Statement Title** - Occlusion-Aware 3D Scene Reconstruction in Partially Observable Real-World Environments
- **Team name** - Chole Bhhature
- **Team members (Names)** - Bhavya Garg
- **Institute/College Name** - IIIT Bangalore, 26/C, Electronics City Phase 1, Hosur Road, Bengaluru 560100
- **Final Presentation Google Drive Link** - *(upload `PHANTOM_ECHO_REVEAL_Presentation.pptx` as PDF to Google Drive, set link sharing to "Anyone with the link", paste here)*
- **Full Submission Demo Video Link** - *(record the live dashboard demo below, upload to YouTube as unlisted, paste here)*
- **Setup & Result Reproducibility Video Link** - *(screen-record the Quick Start + `--mode eval` steps below, upload to YouTube as unlisted, paste here)*

### Project Artefacts

- **Technical Documentation** - [`docs/technical.md`](docs/technical.md) · [`docs/user_guide.md`](docs/user_guide.md) · [`docs/ax.md`](docs/ax.md)
- **Source Code** - [`src/`](src/) (pipeline, real-time backend, frontend, eval & benchmark code)
- **Models Used** -
  - MobileSAM — https://huggingface.co/dhkim2810/MobileSAM
  - MobileCLIP-S2 — https://huggingface.co/apple/MobileCLIP-S2
  - LLaVA-NeXT-Video-7B — https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf
  - Stable Video Diffusion (VideoScene Tier-2 fallback) — https://huggingface.co/stabilityai/stable-video-diffusion-img2vid
  - *(All open-weight. Simulate mode — the default — requires no model downloads.)*
- **Models Published** - None (no new model weights were trained; all novel contributions are algorithmic)
- **Datasets Used** - ScanNet (DDGS pre-training reference + real-eval instructions) — https://github.com/ScanNet/ScanNet
- **Datasets Published** - None (synthetic scenes are generated procedurally at runtime by `src/edge/sensing/arkit_depth.py`)

### Attribution

Built from scratch for this hackathon. Algorithmic ideas build on published research —
3D Gaussian Splatting (Kerbl et al. 2023), Screened Poisson Surface Reconstruction
(Kazhdan & Hoppe 2013), the Image Source Method (Allen & Berkley 1979), and Synthetic
Aperture Sonar triangulation — all re-implemented in this repo. No existing open-source
project was used as a code base. Open-weight models used are listed above.

---

## Core Philosophy: Prove → Measure → Imagine

Most 3D reconstruction systems treat occluded geometry as a prediction problem.
PHANTOM-ECHO REVEAL treats it as an **elimination problem**:

```
PROVE    Apply 8 physical laws → eliminate impossible geometries → BLUE
MEASURE  Smartphone bat-sonar (LFM chirp + ISM + SAS) → TEAL
IMAGINE  VideoScene generation only for what remains genuinely unknown → GREEN
UNKNOWN  Kept as RED — robot explores autonomously (Mode B dual-trigger)
```

Every Gaussian carries a colour-coded confidence tag:

| Tag | Colour | Source |
|-----|--------|--------|
| WHITE  | ⬜ | ARKit high-confidence sensor |
| BLUE   | 🔵 | 8 physics laws (PHANTOM-LITE) |
| TEAL   | 🩵 | Acoustic SAS triangulation |
| GREEN  | 🟢 | VideoScene AI generation |
| YELLOW | 🟡 | Soft structural prior |
| RED    | 🔴 | Unknown — open in nav map |
| ORANGE | 🟠 | SlotLSTM dynamic track |

---

## KPI Results (reproducible with one command)

```bash
python -m src.main --mode eval        # all 3 scenes, ~2 min, CPU-only
```

| Metric (3-scene synthetic benchmark¹) | PHANTOM v22 | Target | Met |
|---|---|---|---|
| F1 @ 5cm (observed surfaces) | **0.903** | ≥ 0.85 | ✓ |
| F1 @ 10cm | **0.930** | — | ✓ |
| Precision @ 5cm | **0.99** | — | ✓ |
| Semantic accuracy | **99.9%** | ≥ 93% | ✓ |
| Reconstruction error (median dist. to true surface) | **0.01 cm** | < 1.5cm | ✓ |
| Live TEAL acoustic triangulation | **<10cm err, 0 ghosts (20/20 trials)** | — | ✓ |
| End-to-end live scan (8 frames, CPU, no models) | **~14 s** | — | ✓ |

Per-scene results in `output/eval_results.json` (living_room 0.916 / office
0.890 / bedroom 0.903).

> ¹ **Integrity note:** the v21 evaluation could not run (three crash bugs)
> and its ground truth described a different scene than the simulator
> rendered, so earlier headline KPIs were retracted and the evaluation was
> rebuilt from scratch (BUG-V22-3…7, `docs/technical.md`). The v22 numbers
> above come from real engineering fixes, each documented and reproducible:
> a depth-convention bug (ray-length stored as z-depth) that injected a
> systematic 3-4cm error into every point (BUG-V22-10), surface- instead of
> volume-sampled generation (BUG-V22-8), depth-edge halo rejection
> (BUG-V22-9), and the bible's proactive Laws 1 & 6 finally BUILDING
> floor/ceiling/wall geometry instead of only re-tagging sensor points.
> For real-data validation use dataset mode (held-out-frame protocol).

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — set PHANTOM_SIMULATE=true for offline mode (no cloud API needed)
```

### 3. Run the demo pipeline
```bash
python -m src.main --mode demo
```
Output: `output/scene_gaussians.ply`, `output/scene_mesh.ply`, `output/costmap.npy`, `output/summary.json`

### 4. Open the live dashboard
```bash
python -m src.main --mode realtime
# Then open: http://localhost:8000
```

### 5. Run evaluation
```bash
python -m src.main --mode eval --scenes living_room_01 office_01 bedroom_01
```

### 6. Atlas baseline comparison
```bash
python -m src.eval.atlas_baseline
```


---

## NEW in v22 — Real-Time Live Dashboard (backend + frontend)

The Phase 2 submission adds a real-time layer: the pipeline now runs
frame-by-frame and streams Gaussians over WebSocket to a browser dashboard
while the scan is happening.

```bash
python -m src.main --mode realtime
# open http://localhost:8000 in Chrome
```

What you see live:
1. **Start live scan** — DDGS Gaussians appear frame-by-frame, colour-coded by certainty (WHITE/BLUE/YELLOW/RED).
2. **TEAL appears mid-scan** — as soon as 3+ chirp baselines exist, SAS v3 triangulates the occluded surface and a TEAL marker pops into the scene (bat-sonar, live).
3. **ORANGE dynamic layer** — replaced every frame, never enters the static scene.
4. **Tap-to-Reveal (Mode A)** — toggle ✦, click any RED cluster → `POST /api/reveal` → GREEN Gaussians appear, affordance-routed and plane-snapped, in <1s (simulate) / <3s (cloud).
5. **Nav2 costmap + KPI panel + pipeline stage tracker** update as each layer completes.

**Real-data input (v22.1):** the dashboard's source selector also accepts
real data — `python scripts/get_real_dataset.py` fetches a real Redwood RGB-D
living-room scene and the engine scores itself against a **held-out frame it
never saw** (median error / F1 reported live); or upload **your own photo**
(open-weight Depth Anything V2 monocular depth → RGB point cloud, visual-only
since single photos have no ground truth). See RUN_INSTRUCTIONS.md.

Architecture: `src/realtime/engine.py` (incremental pipeline, event emitter) →
`src/realtime/server.py` (FastAPI + WebSocket hub, snapshot replay for late
joiners) → `src/frontend/index.html` (Three.js viewer + dashboard, zero build
step). All pipeline modules are reused unchanged — the engine is an orchestrator,
not a fork.

v22 also fixes **BUG-V22-SAS**: v2 angular clustering silently produced 0 TEAL
points on any walking trajectory. v3 uses range-rate echo track association +
planar-array mirror disambiguation; verified 20/20 triangulations <10cm error,
0 ghosts (see `docs/technical.md`).

---

## Architecture — 7 Layers

```
Layer 0  ARKit depth + LFM acoustic chirp + ISM filter + SAS triangulation
Layer 1  QuantVGGT dense depth → DDGS 2D planar disk Gaussians (triple-tagged)
Layer 2  PHANTOM-LITE: 8 physics laws → BLUE proven + TEAL measured + RED unknown
Layer 3  VideoScene generation (3-tier) + Semantic Affordance Router
Layer 4  Dual output: navigation map (open RED) + deliverable mesh (SPSR sealed)
Layer 5  ROS2 Nav2 + adaptive-λ information-gain reward + Mode B auto-trigger
Layer 6  WebGPU gsplat.js viewer + QR code for judge phones
```

---

## Demo Script (2 minutes)

```
[0:00–0:20]  Live scan — DDGS builds colour-coded scene in real time
[0:20–0:35]  Acoustic bat-sonar — TEAL Gaussian appears behind sofa
             "Sound bounced around the sofa. 23ms. 343 m/s. 3.9m."
[0:35–0:50]  Static/Dynamic separation — ORANGE hand tracked, table stays clean
[0:50–1:00]  Judges scan QR code → WebGPU viewer on their phones
[1:00–1:20]  Tap to Reveal — judge taps RED box → GREEN objects appear < 3s
[1:20–1:40]  Mode B — robot hits RED zone → pauses → GREEN fills → resumes
[1:40–2:00]  Atlas vs PHANTOM comparison table
```

---

## Project Structure

```
src/
├── main.py                      ← Single canonical entry point (demo|eval|realtime)
├── realtime/                    ← v22: streaming engine + FastAPI/WebSocket server
├── frontend/                    ← v22: Three.js live dashboard (single file)
├── main_v2.py                   ← Full pipeline implementation
├── edge/
│   ├── sensing/                 ← acoustic_chirp, ism_filter, sas_triangulator, arkit_depth
│   ├── reconstruction/          ← ddgs_gaussrender, quantvggt, static_dynamic_sep
│   ├── phantom_lite/            ← contradiction_engine (8 laws), affordance_router
│   ├── tracking/                ← slot_lstm
│   ├── segmentation/            ← segmentation_handler (MobileSAM), ray_registration
│   ├── embedding/               ← mobile_clip, cache_checker
│   ├── buffer/                  ← frame_buffer (stereo anchor)
│   ├── anchor/                  ← spatial_anchor, ray_registration
│   ├── network/                 ← cloud_client, payload_builder, gaussian_decoder
│   ├── app/                     ← tap_handler, auto_trigger
│   └── (viewer: see src/frontend + src/realtime — v22 live dashboard)
├── cloud/
│   ├── api/                     ← server.py (Flask, all bugs fixed)
│   ├── generation/              ← videoscene_pipeline_fixed (3-tier fallback)
│   ├── compression/             ← svq_endpoint
│   ├── llm/                     ← llava_wrapper
│   └── cache/                   ← llm_cache (FAISS)
├── mesh/
│   ├── spsr_extraction.py       ← SPSR + batched color baking
│   ├── color_baker.py           ← NEW: UV-mapped vertex color baking
│   ├── normal_orientation.py    ← viewpoint flip + MST propagation
│   ├── outpainting_sweep.py     ← seals RED boundaries for deliverable mesh
│   ├── atlanta_world.py         ← soft normal regularization
│   └── semantic_labeler.py      ← per-vertex FLOOR/WALL/CEILING/PLATFORM labels
├── navigation/
│   ├── nav2_publisher.py        ← publishes/saves costmap (wired in Layer 4)
│   ├── occupancy_grid.py        ← Binary Bayes Filter
│   ├── active_perception.py     ← adaptive-λ information-gain reward
│   ├── global_costmap.py        ← static mesh → OccupancyGrid
│   ├── local_costmap.py         ← dynamic bboxes
│   ├── nav2_bridge.py           ← ROS2 Nav2 integration
│   └── nav2_watchdog.py         ← 10s stall → fallback mode
├── generation_correction/
│   └── plane_alignment.py       ← continuous plane-to-point alignment
├── eval/
│   ├── run_eval.py              ← honest evaluation (calls real pipeline)
│   ├── atlas_baseline.py        ← NEW: Atlas vs PHANTOM comparison table
│   ├── evaluate_real.py         ← real-hardware evaluation harness
│   └── check_model_hashes.py   ← verify HuggingFace model pins
└── shared/
    └── gaussian_format.py       ← data structures, constants, tag definitions

simulation/
├── hackathon_room.world         ← Gazebo SDF world (table, sofa, box)
├── phantom_robot.urdf           ← Differential drive robot URDF
├── nav2_params.yaml             ← Nav2 tuned for indoor cluttered env
└── demo_launch.py               ← Launches Gazebo + Nav2 + pipeline

output/
├── scene_gaussians.ply          ← Point cloud (Layer 4)
├── scene_mesh.ply               ← Watertight mesh (SPSR, Layer 4)
├── costmap.npy                  ← 2D occupancy grid (nav fallback)
├── eval_results.json            ← Latest eval run
└── atlas_baseline.json          ← Atlas vs PHANTOM comparison
```

---

## Real ScanNet Evaluation

```python
# In src/eval/run_eval.py, replace _build_ground_truth() with:
import open3d as o3d

def load_scannet_gt(scene_path: str):
    mesh = o3d.io.read_triangle_mesh(f"{scene_path}/_vh_clean_2.ply")
    pcd  = mesh.sample_points_uniformly(5000)
    return np.asarray(pcd.points), np.asarray(pcd.normals)
```

ScanNet download: https://github.com/ScanNet/ScanNet

---

## System Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| Python | 3.10+ | 3.11.6 |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU sim) | RTX 3080 |
| OS | Linux / macOS / WSL2 | Ubuntu 22.04 |
| Browser | Chrome 113+ | Chrome 124 |
| ROS2 (optional) | Humble | Humble + Gazebo 11 |

---

## Novel Contributions

1. **Physics-First Contradiction** — 8 physical laws eliminate impossible geometries before any generation. World first.
2. **Smartphone Acoustic SAS** — Edge-local ISM + SAS triangulation for occluded surfaces. Zero extra hardware.
3. **Semantic Affordance Routing** — Continuous plane-to-point alignment prevents gravity-override anomaly.
4. **Dual Output Architecture** — Navigation map (open RED) + deliverable mesh (sealed) from one Gaussian scene.
5. **46-Flaw Engineering History** — 46 distinct bugs identified and fixed across 17 versions before implementation.

---

## License

MIT — see LICENSE
#   P h a n t o m - E c h o - R e v e a l  
 