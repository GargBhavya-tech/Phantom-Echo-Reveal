# PHANTOM-ECHO REVEAL

- **Problem Statement Number** - 09
- **Problem Statement Title** - Occlusion-Aware 3D Scene Reconstruction in Partially Observable Real-World Environments
- **Team name** - Chole Bhhature
- **Team members (Names)** - Bhavya Garg
- **Institute/College Name** - IIIT Bangalore, 26/C, Electronics City Phase 1, Hosur Road, Bengaluru 560100
- **Final Presentation Google Drive Link** -  https://drive.google.com/file/d/19BzMjPqj--rZNBhccx6RqT0lctmszCTz/view?usp=sharing
- **Full Submission Demo Video Link** -  https://youtu.be/yK_UO12oUd8
- **Setup & Result Reproducibility Video Link** -  https://youtu.be/-cx3FQUgTkc

## Video Disclosure

Both submission videos use AI-generated text-to-speech narration, read 
from a script written entirely by the team. No other part of either 
video is AI-generated: all screen recordings, system outputs, terminal 
results, dashboard interactions, and the live demo are genuine, 
unedited computation. Editing was limited to standard cuts and 
speed-ups of idle wait time (e.g. package installation).

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

### Runtime Compliance (open-weight only — no proprietary AI in the runtime)

**The default runtime calls no proprietary AI API.** Every result in this
submission — the headline F1@5cm 0.957, the acoustic triangulation, the demo,
and the agentic Prove→Measure→Imagine planner — is produced by deterministic
code plus open-weight models (MobileSAM, MobileCLIP-S2, LLaVA-NeXT, SVD). The
`anthropic` package is **not** in `requirements.txt`, so no Claude/GPT call can
execute in a clean install.

The agent planner ships an *optional* `LLMPolicy` (`src/agent/planner.py`) that
can use Claude via forced tool-use, but it is **off by default**, requires the
user to set `PHANTOM_AGENT_LLM=claude` + install `anthropic` + supply their own
key, and falls back to the deterministic policy on any error. It is included
only to show the architecture is policy-agnostic; it is **not** part of the
evaluated runtime and is never invoked by `reproduce.sh`, the tests, or any demo
mode. To hard-disable it, leave `PHANTOM_AGENT_LLM` unset (the default).

Separately: an AI coding assistant (Claude Code) was used during *development*
as a critic and pair-programmer — commit co-author trailers reflect this
honestly. All architecture, physics modelling, and design decisions are the
team's own. See [`docs/ax.md`](docs/ax.md) for the full reflection.

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

## KPI Results

PHANTOM-ECHO REVEAL reports **two** evaluations and is explicit about the
difference, because the distinction is the whole credibility story:

| Evaluation | What it proves | Where |
|---|---|---|
| **A. Real-data held-out** (blind) | Generalisation to data the system never saw | `output/real_data_eval.json` |
| **B. Synthetic closed-loop** | Internal consistency / pipeline fidelity | `output/eval_results.json` |

### A. Real-Data Held-Out Validation — the honest, non-circular number

```bash
python -m src.eval.run_real_eval --dataset datasets/redwood_sample --frames 4
```

Reconstruct from real Redwood RGB-D frames 1–4, then score against **frame 5,
which the pipeline never observed** (back-projected real depth = ground truth).
This is a blind test — GT and prediction share no generator.

The primary metric scores the **directly-sensed reconstruction**
(WHITE sensor + TEAL acoustic + RED unknown + YELLOW prior) against the sensed
held-out depth — a like-for-like comparison. Generative GREEN completion and the
axis-aligned BLUE structural prior (a rectangular-room assumption that does not
hold on this real scene) are reported separately under `full_scene`, not mixed
into the reconstruction-accuracy headline. GT depth is capped at 5m (standard
indoor-RGBD practice). On real data the sensed reconstruction is denoised with
**TSDF-style k-NN surface fusion** (`src/edge/reconstruction/tsdf_fusion.py`):
per-frame depth noise is cancelled by averaging multiple observations of each
surface, without dropping coverage.

| Metric (real Redwood, held-out frame) | Sensed recon | Full scene | Target | Met |
|---|---|---|---|---|
| Median reconstruction error | **0.98 cm** | 0.94 cm | < 2 cm | ✓ |
| Recall @ 5cm | **0.994** | 0.998 | — | ✓ |
| F1 @ 10cm | **0.998** | 0.926 | ≥ 0.85 | ✓ |
| F1 @ 5cm | **0.957** | 0.786 | ≥ 0.85 | ✓ (sensed) |
| Precision @ 5cm | **0.922** | 0.648 | — | — |

> All five numbers are the committed `output/real_data_eval.json` (regenerate
> with the command above).

**Read this honestly:** on real, noisy PrimeSense depth the system reconstructs
~99% of the observed surface (recall 0.994) to a **0.98 cm median error**, with
**F1 @ 5cm = 0.957** and **F1 @ 10cm = 0.998** on the directly-sensed layer.
This blind, non-circular result is the strongest number in the project. The v29
`_fill_depth_holes` fix — preserve every valid sensor reading, NN-fill only the
missing pixels, and skip the synthetic-room densifier (which assumed a
rectangular room and injected ~1 m error on real scenes) — took real held-out
F1@5cm from 0.77 → **0.957** and median error 1.76 → **0.98 cm**. The
`full_scene` column folds in generative GREEN and the axis-aligned BLUE
structural prior, which is invalid on this non-rectangular real room, so its 5cm
precision drops to 0.65; we report it separately rather than blending it into
the headline.

### B. Synthetic Closed-Loop Benchmark — internal consistency only

```bash
python -m src.main --mode eval        # all 3 scenes, ~1 min, CPU-only
```

| Metric (3-scene synthetic benchmark¹) | PHANTOM | Target | Met |
|---|---|---|---|
| F1 @ 5cm (observed surfaces, mean) | **0.858** | ≥ 0.85 | ~ (1/3 scenes — see note) |
| F1 @ 10cm (mean) | **0.879** | ≥ 0.85 | ✓ |
| Precision @ 5cm (mean) | **0.987** | — | ✓ |
| Semantic accuracy (mean) | **95.7%** | ≥ 93% | ✓ |
| Reconstruction error (analytic dist. to true surface)² | **~0.0 cm** | < 1.5cm | ✓ |
| End-to-end live scan (8–12 frames, CPU, no models) | **~10–35 s** | — | ✓ |

> **5cm-F1 honesty note:** the synthetic mean F1@5cm is **0.858**; only
> living_room (0.901) clears 0.85 at the 5 cm threshold — office (0.840) and
> bedroom (0.833) clear it only at 10 cm. `output/eval_results.json` therefore
> reports `all_kpis_met: false`, and we do not paper over it. The number to
> quote is the **real** blind held-out F1@5cm of **0.957** (Section A); this
> synthetic table is a self-consistency check, not the accuracy headline.

Per-scene F1@5cm in `output/eval_results.json` (living_room 0.901 / office
0.840 / bedroom 0.833). `coverage` (0.37–0.58) is reported per scene: a partial
walk only observes part of the room surfaces, and recall is computed over that
observed region — it does not claim to reconstruct never-scanned geometry.

> ² The synthetic reconstruction error is measured *analytically* against the
> exact planes/boxes the simulator rendered from, so it is a near-zero **lower
> bound by construction**, not a generalisation claim. Use the real-data median
> error (1.5 cm) above as the headline accuracy figure.

> **Generation (GREEN) and acoustic SAS (TEAL) pillars:** in the synthetic
> closed-loop scene the 8 physics laws fully explain the room, so no RED
> survives and generation correctly stays idle (elimination-first by design).
> Both pillars fire in the **live dashboard** and in the **real-data run**
> (the held-out scan above reveals CHAIR/SHELF GREEN clusters and triangulates
> TEAL surfaces). Watch them live with `python -m src.main --mode realtime`.

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

> **Everything below runs offline on CPU.** Simulate mode is the default — no
> cloud API, no model downloads, and the real-data evaluation reads the
> `datasets/redwood_sample/` frames committed in this repo.

### 0. One command (recommended)
```bash
./reproduce.sh        # Linux/macOS  — installs deps, runs the pipeline,
                      # the held-out eval, and the full test suite
# Windows:  .\reproduce.ps1
```

### 1. Manual setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
No `.env` is required for the default offline mode. Cloud VideoScene
generation is optional — only then do you `cp .env.example .env` and fill in
the keys.

### 2. Reproduce the headline KPI (blind, real-data, non-circular)
```bash
python -m src.eval.run_real_eval --dataset datasets/redwood_sample --frames 4
#  -> output/real_data_eval.json
#     F1@5cm 0.957 · F1@10cm 0.998 · median recon err 1.0 cm  (sensed layer)
```
This reconstructs from real Redwood RGB-D frames 1–4 and scores against the
held-out frame 5 the pipeline never saw. **What this number measures:** the
fidelity of the *directly-sensed* reconstruction (WHITE sensor + RED unknown).
The acoustic (TEAL) and generative (GREEN) layers are reported **separately**
under `full_scene` — see *Honest scope of each metric* below.

### 3. Run the demo pipeline + live dashboard
```bash
python -m src.main --mode demo        # writes output/summary.json
python -m src.main --mode realtime    # live dashboard at http://localhost:8000
```

### 4. Agentic planner (Prove → Measure → Imagine as tool use)
```bash
python -m src.main --mode agent       # -> output/agent_trace.json
```
This is where the **acoustic / TEAL** path is exercised end-to-end on the
synthetic scene: a hidden surface is recovered by the forward+inverse sonar
model and triangulated (see *Honest scope* below).

### 5. Synthetic closed-loop benchmark + baseline
```bash
python -m src.main --mode eval --scenes living_room_01 office_01 bedroom_01
python -m src.eval.atlas_baseline
```

### 6. Run the test suite
```bash
python -m pytest tests/ -q            # 45 tests: physics laws + integrity
```

---

### Honest scope of each metric (read this — it is the credibility story)

| Layer | Tag | Where it is proven | Where it is **not** claimed |
|---|---|---|---|
| Direct sensing | WHITE / RED | **Real held-out KPI** (`real_data_eval.json`, F1@5cm 0.957) | — |
| Acoustic occlusion recovery | TEAL | **Synthetic forward+inverse sonar** (`--mode agent`, `acoustic_forward.py`) | Not yet measured on real phone audio — see `acoustic_forward.py` limitations |
| Generative completion | GREEN | Reported under `full_scene` in `real_data_eval.json` | Excluded from the headline accuracy number |

The real-data headline is deliberately scored on the directly-sensed layer
only, so it is a clean sensed-vs-sensed comparison. The acoustic channel — the
project's signature contribution — is validated in a **controlled physical
simulation** (matched-filter → ISM subtraction → SAS triangulation on a
noise-corrupted, multipath signal), not on recorded hardware audio. That
hardware validation is stated as future work, not as a measured result.


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
Layer 6  Three.js/WebGL viewer + QR code for judge phones
Agent    Prove→Measure→Imagine planner that calls Layers 0–5 as TOOLS (below)
```

## Agentic Layer — Prove → Measure → Imagine as tool use

The pipeline above is wrapped by a **tool-using agent** (`src/agent/`) that turns
the project's philosophy into an explicit reasoning loop. For each genuinely
**unknown** region it runs a Reason → Act → Observe loop, choosing one pipeline
**tool** per step and deciding the cheapest method that resolves the region:

| Tool (wraps a real module) | Step | Resolves to |
|---|---|---|
| `apply_physics` → contradiction engine (8 laws) | **PROVE** | BLUE if PROVEN |
| `acoustic_measure` → forward/inverse DSP + SAS | **MEASURE** | TEAL if a surface is recovered |
| `generate_geometry` → VideoScene | **IMAGINE** | GREEN if bounded & generatable |
| `plan_viewpoint` → next-best-view | **EXPLORE** | RED + a robot waypoint otherwise |

```bash
python -m src.main --mode agent          # offline, deterministic, reproducible
#   floor_contact  -> BLUE   (physics proved it)
#   hidden_surface -> TEAL   (bat-sonar recovered the occluded surface, ~0.2 cm)
#   sofa_interior  -> GREEN  (generated within occlusion bounds)
#   far_void       -> RED    (too large to imagine — robot explores; waypoint emitted)
# -> output/agent_trace.json   (full per-step reasoning + tool observations)
```

The planner has two interchangeable policies over the **same** tool surface:
a **deterministic** Prove→Measure→Imagine procedure (the default — no API key,
fully reproducible) and an optional **Claude planner** (`claude-opus-4-8` via
forced tool-use) enabled with `PHANTOM_AGENT_LLM=claude` + `ANTHROPIC_API_KEY`,
which falls back to the deterministic policy on any error. The agent never sees
a hidden answer — it sequences real tool calls and interprets their results
(e.g. acoustics honestly *decline* a solid interior, so the agent moves on to
generation). Locked by `tests/test_integrity.py::test_agent_tool_use_resolves_all_paths`.

---

## Demo Script (2 minutes)

```
[0:00–0:20]  Live scan — DDGS builds colour-coded scene in real time
[0:20–0:35]  Acoustic bat-sonar — TEAL Gaussian appears behind sofa
             "Sound bounced around the sofa. 23ms. 343 m/s. 3.9m."
[0:35–0:50]  Static/Dynamic separation — ORANGE hand tracked, table stays clean
[0:50–1:00]  Judges scan QR code → Three.js/WebGL viewer on their phones
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
| OS | Linux / macOS / Windows / WSL2 | Ubuntu 22.04 · Windows 11 |
| Browser | Chrome 113+ | Chrome 124 |
| ROS2 (optional) | Humble | Humble + Gazebo 11 |

---

## Novel Contributions

1. **Physics-First Contradiction** — 8 physical laws eliminate impossible geometries before any generation. World first.
2. **Smartphone Acoustic SAS** — Edge-local ISM + SAS triangulation for occluded surfaces. Zero extra hardware.
3. **Semantic Affordance Routing** — Continuous plane-to-point alignment prevents gravity-override anomaly.
4. **Dual Output Architecture** — Navigation map (open RED) + deliverable mesh (sealed) from one Gaussian scene.
5. **46-Flaw Engineering History** — 46 distinct bugs identified and fixed across 17 versions before implementation.
6. **Agentic Tool-Use Planner** — the Prove→Measure→Imagine philosophy executed as a real Reason→Act→Observe agent that calls the physics, acoustic, generation, and next-best-view modules as tools and decides per region which to use (deterministic offline, or Claude `claude-opus-4-8` via forced tool-use). See `src/agent/`.

---

## License

MIT — see LICENSE

---

## API Endpoints (live dashboard)

With `python -m src.main --mode realtime` running, interactive API docs are
auto-served by FastAPI at **http://localhost:8000/docs** (Swagger UI).

| Method | Path | Purpose |
|---|---|---|
| GET  | `/` | Three.js dashboard |
| WS   | `/ws` | Live event stream (snapshot replay on connect) |
| POST | `/api/scan/start` | Begin a scan (`synthetic` or `dataset` source) |
| POST | `/api/scan/stop` | Cancel a running scan gracefully |
| POST | `/api/reveal` | Mode A tap-to-reveal a RED region |
| POST | `/api/mode_b` | Mode B robot auto-reveal within a radius |
| POST | `/api/agent` | Run the Prove→Measure→Imagine tool-using agent (streams reasoning to the log panel) |
| POST | `/api/photo` | Upload a photo -> monocular-depth point cloud |
| GET  | `/api/state` | Engine state + live tag counts |
| GET  | `/api/kpis` | KPI table (eval results + Atlas baseline) |
| GET  | `/api/scene/export` | Download reconstructed mesh/PLY |
| GET  | `/health` | Server health + uptime (smoke-test probe) |

> Demo-mode auth: set `PHANTOM_DEMO_TOKEN` to require an `X-Demo-Token` header
> on `/api/reveal` and `/api/mode_b` (prevents injection on shared-WiFi demos).
