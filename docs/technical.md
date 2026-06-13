# PHANTOM-ECHO REVEAL — Technical Documentation

**Problem Statement 09: Occlusion-Aware 3D Scene Reconstruction**
**Team: Chole Bhhature | Bhavya Garg | IIIT Bangalore**
**Samsung ennovateX AX Hackathon 2026**

---

## 1. Architecture Overview

PHANTOM-ECHO REVEAL is a 7-layer occlusion-aware 3D reconstruction system
built on a single philosophical principle:

> **Prove → Measure → Imagine** (in that strict order, never skip ahead)

Instead of predicting what is hidden, the system eliminates what is physically
impossible, measures what acoustics can reach, and only generates what remains.

### Layer Stack

```
Layer 6: WebGPU gsplat.js Demo Viewer (src/edge/ui/viewer.html)
    ↑
Layer 5: Active Perception — ROS2 Nav2 navigation hints
    ↑
Layer 4: Dual Output
    ├── Navigation Map (open RED boundaries for robot safety)
    └── Deliverable Mesh (sealed GREEN geometry for AR/VR)
    ↑
Layer 3: Smart Generation + Semantic Affordance Router
    ├── SKIP        → BLUE/TEAL already proven
    ├── PRIMITIVE   → floor/ceiling/box
    ├── FAISS       → wall retrieval from floor-plan DB
    ├── SlotLSTM    → structured furniture (chair/table/sofa)
    └── VideoScene  → full diffusion generation (GREEN)
    ↑
Layer 2: PHANTOM-LITE Contradiction Engine (8 Physics Laws)
    ↑
Layer 1: DDGS GaussRender (2D planar disk Gaussians)
    ↑
Layer 0: Multi-Modal Sensing
    ├── ARKit LiDAR depth + RGB (visible surfaces)
    └── Acoustic Bat-Sonar SAS (occluded surfaces)
```

---


## 1b. Architecture Diagram

> Click any node in the HTML viewer for more detail. The SVG version below is embedded for static docs.

```
LAYER 0 ── Multi-Modal Sensing
  ┌─────────────────┐   ┌──────────────────────────────┐
  │  ARKit / ARCore  │   │  Acoustic bat-sonar (LFM)    │
  │  500 depth pts   │   │  chirp → ISM filter → SAS    │
  └────────┬────────┘   └──────────────┬───────────────┘
           └──────────────┬────────────┘
                          ▼
LAYER 1 ── DDGS GaussRender
  ┌───────────────────────────────┐   ┌──────────────────┐
  │  2D planar disk Gaussians     │──▶│  SlotLSTM tracker│
  │  QuantVGGT densification      │   │  static/dynamic  │──▶ ORANGE
  └───────────────────────────────┘   └──────────────────┘
                          ▼
LAYER 2 ── PHANTOM-LITE Contradiction Engine (8 physics laws)
  ┌───────────────────────────────────────────────────────┐
  │  gravity · no-penetration · structural support        │
  │  occlusion boundary · perspective continuity          │
  │  acoustic SAS · surface normal · compactness          │
  │                              → BLUE / YELLOW / RED    │
  └───────────────────────────────────────────────────────┘
                          ▼
LAYER 3 ── Semantic Affordance Router + VideoScene
  ┌──────────────────────────────┐
  │  SKIP (already BLUE/TEAL)    │
  │  FAISS retrieve (wall/floor) │──▶ GREEN Gaussians
  │  VideoScene diffusion (3s)   │
  └──────────────────────────────┘
                          ▼
LAYER 4 ── Dual Output
  ┌─────────────────────────┐   ┌──────────────────────┐
  │  Navigation map          │   │  Deliverable mesh    │
  │  RED boundaries open     │   │  SPSR sealed + norml │
  └─────────────────────────┘   └──────────────────────┘
                          ▼
LAYER 5 ── Active Perception (ROS2 Nav2)
  Mode A: human taps RED zone → robot moves → resurvey
  Mode B: robot resolves own blind spots autonomously
                          ▼
LAYER 6 ── WebGPU gsplat.js Viewer
  QR code → judge phone → tap occluded object → reveal < 3s
```

**Confidence tag legend:**

| Tag | Color | Source | When assigned |
|-----|-------|--------|---------------|
| WHITE | ⬜ White | ARKit direct | High-confidence ARKit measurement |
| BLUE | 🔵 Blue | 8 physics laws | Physically certain — no generation |
| TEAL | 🩵 Teal | Acoustic SAS | Bat-sonar triangulation |
| GREEN | 🟢 Green | VideoScene AI | Generated within physics bounds |
| YELLOW | 🟡 Yellow | Soft priors | Structurally probable |
| RED | 🔴 Red | None | Unknown — open in nav map |
| ORANGE | 🟠 Orange | SlotLSTM | Tracked dynamic object |

---
## 2. The 4-Category Confidence System

Every Gaussian in the output has exactly one of these tags:

| Tag | Color | Source | Meaning |
|-----|-------|--------|---------|
| **PROVEN** | 🔵 Blue | 8 Physics Laws | Physically certain — no generation needed |
| **MEASURED** | 🩵 Teal | Acoustic SAS | Ground-truth from bat-sonar triangulation |
| **IMAGINED** | 🟢 Green | VideoScene AI | AI-generated within physics bounds |
| **UNKNOWN** | 🔴 Red | — | Unresolvable — left open in navigation map |

This is the key differentiator from all prior work. Previous systems
(NeRF, 3DGS, MonST3R) generate geometry everywhere and tag nothing.
PHANTOM-ECHO REVEAL knows what it knows.

---

## 3. Acoustic Bat-Sonar (Key Innovation)

### Why acoustics?

ARKit LiDAR cannot see behind furniture. Cameras cannot see around corners.
Acoustics diffract and scatter — a smartphone speaker can hear behind a sofa.

### How it works (3 steps):

**Step 1 — Emit LFM chirp**
A 20ms linear frequency-modulated chirp (1 kHz → 22 kHz) is emitted from
the smartphone speaker. This is the bat's echolocation pulse.

**Step 2 — ISM Filter (edge-local, zero WiFi dependency)**
The room impulse response is captured by the microphone. A first-order
Image Source Method (ISM) filter subtracts all predicted echoes from
*visible* surfaces (known from DDGS Gaussians). The residual contains
only occluded surface reflections.

Critical design decision: ISM runs entirely on the smartphone GPU
(no cloud call). Cloud pyroomacoustics introduces 3ms WiFi jitter
= 1.03m depth error at 343 m/s. Unacceptable for sub-centimeter work.

**Step 3 — SAS Triangulation**
The user walks the phone along a path (creating a synthetic aperture).
Each position + round-trip echo distance defines a sphere. Three or more
intersecting spheres → unique 3D point on the occluded surface.

Math:
```
||Q - P_i||² = d_i²
Consecutive-pair subtraction → linear system:
2(P_{i+1} - P_i)·Q = d_i² - d_{i+1}² - ||P_i||² + ||P_{i+1}||²
Solve: Q = (AᵀA)⁻¹Aᵀb, refine with Levenberg-Marquardt
```

Achieves ~1.2cm residual error in simulation.

---

## 4. PHANTOM-LITE Contradiction Engine (8 Physics Laws)

The engine takes a geometry hypothesis and runs it through 8 laws.
Any single IMPOSSIBLE verdict rejects the hypothesis immediately.

| Law | Name | Description |
|-----|------|-------------|
| L1 | Gravity | Objects must rest on a supporting surface |
| L2 | Occlusion Geometry | Gap width ≥ visible object projected width |
| L3 | Shadow Geometry | Cast shadow constrains occluder height |
| L4 | Light Propagation | Lit surface → unobstructed LoS to light |
| L5 | Acoustic Mirror | Echo distance = exact surface position |
| L6 | No Penetration | Solid objects cannot interpenetrate |
| L7 | Support | Every object needs structural support |
| L8 | Symmetry Prior | Man-made rooms are typically rectangular |

Laws L1, L6, L7 run on every hypothesis (universal).
Laws L3, L4, L5 run only when the corresponding sensor data is available.

---

## 5. DDGS GaussRender (Layer 1)

Standard 3D Gaussian Splatting uses 3D ellipsoid Gaussians. When
converted to a point cloud for Screened Poisson Surface Reconstruction
(SPSR), normals are inconsistent → meshing fails.

DDGS forces each Gaussian to be a **2D planar disk** (scale_z = 0):
- Disk plane normal = surface normal (by construction)
- Point cloud normals are consistent
- SPSR produces clean, watertight mesh

This is what makes the Layer 4 deliverable mesh watertight and
directly usable in AR/VR without post-processing cleanup.

---

## 6. Semantic Affordance Router (Layer 3)

The router prevents over-generation by matching each occluded region
to the appropriate generation strategy:

```
BLUE/TEAL  → SKIP        (physics already proved it — never generate)
FLOOR/CEILING → PRIMITIVE (trivial flat extrusion)
WALL       → FAISS_RETRIEVAL (retrieve from floor-plan DB)
CHAIR/TABLE/SOFA/DESK → SLOTLSTM (structural affordance filter)
UNKNOWN    → VIDEOSCENE  (full diffusion with physics prompt)
```

SlotLSTM constraints for CHAIR example:
- Seat height: 0.38–0.55m from floor
- 4 legs (or pedestal)
- Backrest extends 0.30–0.50m above seat

These constraints are encoded into the VideoScene prompt — not as a
post-hoc filter, but as hard bounds the model must respect.

---

## 7. KPI Results

Evaluated on simulation (Blender ground-truth) and cross-dataset
(ReplicaCAD, Matterport3D):

| Metric | Target | Achieved | Atlas Baseline | Improvement |
|--------|--------|----------|---------------|-------------|
| F1 Score | > 0.97 | Run `python -m src.eval.run_eval` | 0.85 | — |
| Semantic Accuracy | > 93% | Run `python -m src.eval.run_eval` | 80% | — |
| Reconstruction Error | < 1.5cm | Run `python -m src.eval.run_eval` | 5.0cm | — |

> Numbers are generated live by `run_eval.py` which calls `run_full_pipeline()` and
> measures Chamfer F1 against synthetic ground truth. This is a simulation prototype
> — results reflect what the pipeline actually produces, not aspirational targets.

---

## 8. Installation

### Requirements
- Python 3.10+
- Node.js 18+ (for WebGPU viewer)
- Ubuntu 22.04 / macOS 13+ / Windows 11

### Quick start (simulation mode — no hardware needed)

```bash
git clone https://github.com/YOUR_USERNAME/phantom-echo-reveal
cd phantom-echo-reveal
pip install -r requirements.txt
python -m src.main --mode demo
# Open src/edge/ui/viewer.html in Chrome/Firefox for 3D viewer
```

### Run evaluation

```bash
python -m src.eval.evaluate --mode simulation --scene living_room_01
# Results written to eval_results.json
```

### Full hardware mode (iPhone 12 Pro+ required)

See [User Guide](user_guide.md) for iOS ARKit integration instructions.

---

## 9. Open Source Attribution

This project builds on the following open-source libraries and models:

| Component | Source | License |
|-----------|--------|---------|
| FusionSegNet v5 | Internal prior work (Bhavya Garg) | MIT |
| LLaVA-NeXT-Video | HuggingFace: llava-hf/LLaVA-NeXT-Video-7B-hf | Apache 2.0 |
| MobileSAM | HuggingFace: dhkim2810/MobileSAM | Apache 2.0 |
| MobileCLIP | HuggingFace: apple/MobileCLIP-S2 | Apple ML Research License |
| VideoScene | HuggingFace: stabilityai/stable-video-diffusion-img2vid | CC-BY-NC-4.0 |
| gsplat | GitHub: nerfstudio-project/gsplat | Apache 2.0 |
| pyroomacoustics | GitHub: LCAV/pyroomacoustics | MIT |
| Open3D | open3d.org | MIT |
| numpy, scipy | numpy.org, scipy.org | BSD |
| FastAPI | fastapi.tiangolo.com | MIT |

---

## 10. Datasets Used

| Dataset | Source | License | Usage |
|---------|--------|---------|-------|
| ReplicaCAD | Meta Research | CC-BY 4.0 | Evaluation ground truth |
| Matterport3D | Matterport | Research License | Cross-dataset eval |
| ScanNet v2 | Technical University Munich | CC-BY-NC-SA | Cross-dataset eval |

No proprietary datasets were used. All evaluation datasets are publicly
available with compatible research licenses.

---

## 11. File Structure

```
phantom-echo-reveal/
├── src/
│   ├── edge/
│   │   ├── sensing/
│   │   │   ├── acoustic_chirp.py      # LFM chirp emission + processing
│   │   │   ├── ism_filter.py          # Edge-local Image Source Method
│   │   │   └── sas_triangulator.py    # Synthetic Aperture Sonar
│   │   ├── reconstruction/
│   │   │   └── ddgs_gaussrender.py    # 2D planar disk Gaussians (Layer 1)
│   │   ├── phantom_lite/
│   │   │   ├── contradiction_engine.py # 8 Physics Laws (Layer 2)
│   │   │   └── affordance_router.py    # Semantic routing (Layer 3)
│   │   └── ui/
│   │       └── viewer.html            # WebGL/WebGPU 3D viewer (Layer 6)
│   ├── cloud/
│   │   └── generation/
│   │       └── videoscene_pipeline.py  # Cloud generation + validation
│   ├── eval/
│   │   └── evaluate.py                # F1, semantic acc, recon error
│   └── main.py                        # Full pipeline orchestrator
├── docs/
│   ├── technical.md                   # This file
│   ├── ax.md                          # AI/agentic workflow reflection
│   └── user_guide.md                  # Setup + usage guide
├── requirements.txt
├── LICENSE
└── README.md
```

---

## v22 — Real-Time Streaming Architecture (Phase 2)

### Overview

v21 was a batch pipeline: `run_full_pipeline()` processed all frames, then wrote
files. v22 adds a real-time layer that runs the SAME modules incrementally and
streams every intermediate result to a browser dashboard over WebSocket.

```
┌────────────────────────── backend (Python) ───────────────────────────┐
│ src/realtime/engine.py        RealtimeEngine (worker thread)          │
│   per frame: sense → ISM → QuantVGGT → DDGS → static/dyn split        │
│              → PHANTOM-LITE physics tagging → emit "frame" event      │
│   ≥3 baselines: SAS v3 triangulation → emit "teal" event (live!)      │
│   after walk:  RED bbox → affordance route → generation → "green"     │
│               costmap → "costmap" · SVQ + KPIs → "summary"            │
│ src/realtime/server.py        FastAPI                                 │
│   WS  /ws              event stream, snapshot replay for late joiners │
│   POST /api/scan/start  POST /api/reveal   GET /api/state /api/kpis   │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ JSON events (compact wire format)
┌───────────────────────────────▼───────────────────────────────────────┐
│ src/frontend/index.html   Three.js viewer + dashboard (no build step) │
│   7 per-tag point clouds · orbit camera · tap-to-reveal raycasting    │
│   legend w/ live counts · stage tracker · KPI table · costmap canvas  │
└────────────────────────────────────────────────────────────────────────┘
```

Design decisions:
- **Engine = orchestrator, not fork.** All v21 modules are imported unchanged;
  the engine only re-sequences them and adds event emission. One code path
  for batch eval and live demo.
- **Thread → asyncio bridge.** The engine runs in a worker thread; events cross
  into the FastAPI event loop via `asyncio.run_coroutine_threadsafe`.
- **Snapshot replay.** The engine keeps a bounded replay buffer; a judge who
  scans the QR mid-demo receives the scene built so far, then live updates.
- **Dynamic layer replacement.** ORANGE Gaussians are replaced per frame on the
  wire (`dynamic` field) — moving objects never accumulate or enter the
  static scene (Flaw 35 invariant preserved in streaming).
- **Wire compaction.** `{p:[xyz], c:[rgb], t:tag, s:scale}` with 3-decimal
  rounding; ≤1500 static Gaussians per frame event (RED/TEAL never dropped).

### BUG-V22-SAS — acoustic triangulation was silently dead

Discovered during Phase 2 integration testing: `cluster_and_triangulate_v2`
returned **0 points on every realistic walking trajectory** (visible in v21
logs: `SAS v2: 1 distance clusters → 5 angular sub-clusters → 0 triangulated
points`). Root cause: the angular sub-split clustered constraints by
*phone-position bearing from the path centroid*. Consecutive walk positions
exceed the 15° gate, so every sub-cluster ended with <3 constraints — below
the triangulation minimum. TEAL, one of the two flagship novelties, was a
no-op.

Fix (`cluster_and_triangulate_v3`):
1. **Echo track association by range-rate continuity** — the round-trip
   distance to a static target from a smoothly moving phone changes slowly;
   each new echo is jointly (globally-greedy, one echo per track) assigned to
   the track whose linearly-extrapolated next distance is closest. Gate: 35cm
   for young tracks, 10cm once a track has a velocity estimate.
2. **Planar-array mirror disambiguation** — a phone carried at constant height
   forms a planar virtual array, so every solution has an equal-residual
   mirror across the array plane. We apply a gravity prior: occluded surfaces
   behind furniture lie below carry height → choose the lower solution.
3. **Self-consistency residual gate** — a wrongly-associated track triangulates
   to a point that violates its own sphere constraints; tracks with mean
   residual >3cm are rejected (sensor noise is 8mm).

Validation (6 random trials × {5,6,8,12} frames, 2 hidden targets):
**20/20 triangulated points within 10cm of a true target, 0 ghosts** (v2: 0
points ever). Both `src/main_v2.py` and the real-time engine now use v3.

### Event protocol

| type | payload | when |
|------|---------|------|
| `frame` | `gaussians[]`, `dynamic[]`, `counts` | after each processed frame |
| `teal` | `gaussians[]` | SAS triangulation succeeds mid-scan |
| `green` | `gaussians[]`, `tier` | Layer 3 generation done |
| `reveal_result` | `gaussians[]`, `semantic`, `latency_ms` | Mode A tap |
| `costmap` | `w`, `h`, `resolution`, `data[]` | Layer 4/5 done |
| `summary` | `counts`, `elapsed_s`, `payload_kb`, `kpis` | pipeline complete |
| `stage` / `log` | `msg` | progress + diagnostics |

### v22 evaluation rewrite (BUG-V22-3 … BUG-V22-7)

Phase 2 integration testing revealed the v21 evaluation could never have
produced its advertised numbers — it crashed on three separate bugs before
emitting a result, and its ground truth described a scene the simulator never
rendered:

| Bug | Symptom | Fix |
|-----|---------|-----|
| V22-3 | `_build_ground_truth` returns `(vis_pts, occ_pts)` 4-col arrays; caller unpacked as `(points, labels)` → KD-tree ValueError | merge sets, split coords/labels |
| V22-4 | occluded-object semantic ids 2–9 collide with CEILING/PLATFORM ids | aligned to the 5-class scheme |
| V22-5 | pointwise `compute_semantic_accuracy` called on unaligned arrays → dict error → `round()` crash | nearest-neighbour label protocol |
| V22-6 | `chamfer_distance` returns a tuple; `round(cd*100)` crashed | use gt→pred direction explicitly |
| V22-7 | GT scene ≠ simulated scene; 800-pt GT (~27cm spacing) made precision metrics meaningless | GT built from the same scene spec the simulator renders; analytic surface distance (planes + box SDFs); dense observed-region recall |

Consequence: earlier headline KPIs were retracted and replaced with
reproducible ones (see README). We consider surfacing this before submission
— rather than letting judges discover it in the reproducibility video — the
correct engineering call.

Additional v22 pipeline fixes: BUG-V22-2/2b (TEAL measurements were re-tagged
by the physics engine and then dropped by the 5000-Gaussian cap — teal count
was always 0), PERF-V22 (active-perception ray-marching subsampled: minutes →
seconds), dynamic-layer per-frame replacement in the streaming engine.

### v22.1 — Real-data input modes

Two new input sources, selectable in the dashboard header:

**Real RGB-D dataset** (`source:"dataset"`): `scripts/get_real_dataset.py`
downloads the public Redwood RGB-D sample (real PrimeSense captures of a
living room) into `datasets/redwood_sample/{color,depth,pose}`; any folder in
that layout works (TUM RGB-D, ScanNet exports, ARKit recordings).
The engine consumes it through the existing `RealDepthGenerator`, normalises
world coordinates into PHANTOM's room frame, disables the acoustic layer
(datasets carry no microphone stream), and reserves the last frame as a
held-out evaluation target: reconstruction quality is measured against a
frame the system never saw (`real_data_eval` in the summary event —
median error, F1@5cm, F1@10cm). This is the honest "how good is the model on
real data" number.

**Photo upload** (`POST /api/photo`): a single photo from any phone/laptop is
passed through open-weight monocular depth (Depth Anything V2 Small,
`depth-anything/Depth-Anything-V2-Small-hf`), back-projected with generic
intrinsics, and streamed as a true-RGB point cloud (`PHOTO` wire tag, vertex
colours). Visual-only by design: a single photo has no ground truth, no metric
scale, and no second view — so no KPI is reported. The UI states this rather
than inventing a score.

### v22 final — accuracy engineering (F1 0.51 → 0.90)

Honest, documented fixes; every number reproduces with `--mode eval`:

| Fix | What was wrong | Effect |
|-----|----------------|--------|
| **BUG-V22-10** | Simulator stored Euclidean **ray length** in the depth map while back-projection assumed ARKit **z-depth** — a systematic error growing toward image corners (up to ~4cm) baked into every single point | precision@5cm 0.61 → 0.97, median surface error 3.0cm → 0.2cm |
| **BUG-V22-8/8b** | Generated GREEN Gaussians sampled the object's interior **volume** (Gaussians represent surfaces); degenerate template boxes collapsed all 200 splats onto one interior point | GREEN precision 0.00 → surface-sampled with outward normals |
| **BUG-V22-9** | The depth densifier (NN-fill + boxcar) smears depth across object boundaries → "halo" points floating between surfaces | edge-pixel rejection (standard RGB-D practice) |
| **Proactive Laws 1 & 6** | The bible specifies physics **builds** geometry (floor extends under furniture, walls connect floor↔ceiling); the engine only re-tagged sensor points, so the floor — barely visible to a forward-facing camera — was entirely missing (floor recall 0.00) | BLUE floor/ceiling/wall planes built over observed extent; recall 0.46 → 0.85; live `infill` event in the dashboard |
| **PERF-V22b/c/d** | Frontier ray-marching, free-space Bresenham casting, and SVQ encoding were O(N) pure Python over the whole cloud | eval runs in ~40s/scene on CPU |

Final 3-scene benchmark: **F1@5cm 0.902 · F1@10cm 0.931 · semantic 96.8% ·
median error 0.06cm** (`output/eval_results.json`).

The legacy v17 WebGPU viewer (`src/edge/ui/viewer.html`, `webgpu_server.py`)
is retired and excluded from the submission package — it displayed hardcoded
demo KPIs, which conflicts with the project's reproducibility standard.

### v22.2 — external review triage

An external code review raised 6 "critical" bugs. Each was verified against
the code with experiments before acting — verdicts:

| Claim | Verdict | Action |
|---|---|---|
| #1 recall gate (25cm proximity) is invalid; camera-sphere gating lifts F1 to 0.97 | **Refuted by experiment**: a 4m camera-sphere gate includes surfaces behind the camera / outside the FOV — measured recall fell 0.85 → 0.31, the opposite of the claim. The proximity gate measures accuracy of reconstructed regions; `coverage` is reported alongside it transparently. | `camera_positions` now exposed in results for anyone to re-run the comparison |
| #2 ORANGE accumulates 8× in batch | **Confirmed** | BUG-V22-12: dynamic layer is a per-frame snapshot (27k → 3k) |
| #3 TEAL dead due to v1 import | **False** (v3 explicitly imported) — TEAL low count is the known 2-target geometry, tracked separately | — |
| #4 WHITE = 0 | **Confirmed, different root cause**: the contradiction engine re-tagged sensor-confirmed WHITE as BLUE (a measurement must not be downgraded by inference), compounded by an odd DDGS stride sampling unfilled simulator pixels | BUG-V22-11 (even stride) + BUG-V22-13 (WHITE passthrough) |
| #5 ISM dict/object key mismatch kills acoustics | **False for the main path** — `main_v2` constructs `WallPlane` objects directly; `extract_walls_from_scene` is not in the pipeline | — |
| #6 orphaned `t_min` crashes normal orientation | **False** — no such code exists; orientation logs show successful flips every run | — |

Also fixed from the review's edge-case list: SPSR <100-point guard, and a new
Atlanta-World positional relabel pass (Layer 2b) after observing oblique-wall
normal noise mislabelling wall points as OTHER (office semantic 0.84 → 0.998).

Final 3-scene benchmark after v22.2: **F1@5cm 0.903 · F1@10cm 0.930 ·
semantic 99.9% · median error 0.01cm** — all KPIs met, all reproducible.

### v22.3 — production-audit pass

- **Walk-sway experiment:** raising hand-sway 5cm → 15cm (proposed to improve
  SAS conditioning) was tested and REJECTED: 7/7 ghost triangulations vs
  20/20 good at 5cm. The planar-array mirror prior requires near-planar
  positions and beats the better-conditioned unconstrained system. Comment
  preserved in `arkit_depth.py`.
- **KPI panel provenance:** the dashboard panel is now explicitly titled
  "3-scene benchmark (pre-computed)"; it switches to "LIVE (held-out real
  frame)" in dataset mode and to "n/a (no ground truth)" in photo mode —
  a judge can never mistake benchmark numbers for live ones.
- **Known limitation (documented, not hidden):** static/dynamic separation is
  tuned on synthetic motion; a person moving through a REAL dataset capture
  can bake into the static cloud (ghosting). Mitigation path: temporal voxel
  consistency filtering — listed as future work in README.
