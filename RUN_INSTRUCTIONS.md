# PHANTOM-ECHO REVEAL v22 — How to Run Everything

**Every mode works offline, on CPU, with no model downloads in the default simulate setting.**

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.10 or newer | `python --version` to check |
| Chrome or Edge | Any modern browser works for the dashboard |
| ~500MB disk | Only for `requirements.txt`; torch/open3d are optional extras |
| GPU | Not required — all critical paths run on CPU |

---

## Step 0 — Install dependencies (one time)

```bash
# Minimal install — enough for the live dashboard and all tests
pip install numpy scipy scikit-learn opencv-python-headless fastapi "uvicorn[standard]" websockets pydantic python-dotenv

# OR: full install from requirements.txt (adds open3d, torch stubs, etc.)
pip install -r requirements.txt
```

> **Windows / venv:** if you have a `venv/` folder, activate it first:
> `venv\Scripts\activate` then run any command below.

---

## Mode 1 — Live Real-Time Dashboard ← start here for the demo

```bash
python -m src.main --mode realtime
```

Then open **http://localhost:8000** in Chrome.

### What happens internally

1. **FastAPI + Uvicorn start** on port 8000 (`src/realtime/server.py`).
2. A **WebSocket hub** (`Hub`) connects the browser to the engine. Any WebSocket client that joins late receives a full snapshot replay so it never sees a blank scene.
3. The **Three.js** frontend (`src/frontend/index.html`) opens a WebSocket and renders incoming Gaussian splats in real time — each Gaussian is a coloured point whose colour = its certainty tag.

### In the dashboard — what each button does

| Button | What it triggers | What you'll see |
|---|---|---|
| **▶ Start live scan** | `POST /api/scan/start` → `engine.start_scan()` runs in a background thread | Room builds frame-by-frame (WHITE → BLUE tags appear live as each frame is processed) |
| **Source dropdown** | `synthetic` / `Real RGB-D` / `My photo` | Changes which data the engine processes (see Modes 3b and 3c below) |
| **Frames dropdown** | Sets `n_frames` (6 / 8 / 12 / 20) | More frames = more acoustic baselines = more chance of TEAL splats appearing |
| **✦ Tap-to-Reveal** | Enables click-selection on the 3D canvas; click any cluster → `POST /api/reveal` | RED cluster fills with GREEN AI-generated Gaussians in < 1s |
| **🧠 Agent** | `POST /api/agent` | Runs the Prove→Measure→Imagine planner; each decision appears live in the log panel below the 3D view |
| **🔊 Sonar** | `GET /api/sonar_demo` | Opens the standalone acoustic bat-sonar animated demo in a new tab |
| **↺ Reset** | `POST /api/reset` | Clears the scene + HUD + Three.js geometry without restarting the server |

### What the colours mean

| Colour | Tag | Meaning |
|---|---|---|
| ⬜ WHITE | SENSOR | Raw depth reading from the RGB-D or synthetic sensor — not yet physics-validated |
| 🔵 BLUE | PROVEN | Physics laws (L1–L8) confirmed this surface is geometrically consistent — no measurement needed |
| 🟡 YELLOW | POSSIBLE | Physics says this is geometrically plausible but not proven — kept as a candidate |
| 🟠 ORANGE | APPROXIMATE | Near-certain but with a wider uncertainty margin |
| 🩵 TEAL | ACOUSTIC | Bat-sonar DSP recovered a hidden surface at this position — measured, not guessed |
| 🟢 GREEN | GENERATED | AI-generated geometry (VideoScene pipeline) filled an occluded zone |
| 🔴 RED | UNKNOWN | Physics says this is impossible or unresolvable — left open for the robot to explore |

### Phone / tablet access

```bash
# Find your LAN IP (Windows)
ipconfig
# Then scan the QR code in the right sidebar, or open:
http://<your-ip>:8000
```

The QR code in the right sidebar points to `http://<lan-ip>:8000/m` — a lightweight mobile page where you can take a photo and immediately see it reconstructed on the big screen.

---

## Mode 2 — Batch Pipeline (file output)

```bash
python -m src.main --mode demo
```

### What happens internally

Runs `src/main_v2.py::run_full_pipeline()` — the full 7-layer pipeline in batch mode:

1. **Layer 1 — Reconstruction:** Synthetic depth frames are fed through `QuantVGGT` → `DDGSGaussRender` → produces a list of 3D Gaussian splats.
2. **Layer 2 — PHANTOM-LITE:** Each Gaussian is passed through the `ContradictionEngineFixed` with all 8 physics laws (L1 Gravity → L8 Symmetry). Output: a tag (WHITE/BLUE/TEAL/GREEN/YELLOW/RED) per Gaussian.
3. **Layer 3 — Generation:** RED clusters are sent to the `VideoScene` generation pipeline (Tier 1/2/3 depending on region size). Fills gaps with plausible GREEN Gaussians.
4. **Layer 4 — Proactive Fill:** `_proactive_blue()` fills floor, ceiling, and visible walls with BLUE Gaussians even if the sensor didn't hit them — room completeness.
5. **Layer 5 — Navigation:** `OccupancyGrid` projects the Gaussian cloud into a 2D top-down costmap. Output: `output/costmap.npy` and a Nav2-compatible PGM.
6. **Summary:** writes `output/summary.json` with per-tag Gaussian counts, elapsed time, and KPI numbers.

### Output files

```
output/
  summary.json          ← tag counts, elapsed time, KPIs
  scene_mesh.ply        ← reconstructed mesh (open in MeshLab or Blender)
  scene_gaussians.ply   ← Gaussian cloud (same data as the dashboard)
  costmap.npy           ← 2D Nav2 costmap array
```

---

## Mode 3a — Agent Mode (Prove → Measure → Imagine)

```bash
python -m src.main --mode agent
```

### What happens internally

Runs `src/agent/planner.py::run_agent()` — a tool-using ReAct loop that resolves every unknown region in the scene:

For each unknown region, the agent:
1. **Calls `inspect_region`** — reads the Gaussian geometry, occlusion context, and bounding box.
2. **Calls `apply_physics`** — runs the ContradictionEngine. If PROVEN → finalise as **BLUE**.
3. **Calls `acoustic_measure`** — fires the bat-sonar DSP (`acoustic_forward.measure_distances`). If surface recovered → finalise as **TEAL**.
4. **Calls `generate_geometry`** — runs the VideoScene pipeline inside the region bounds. If splats generated → finalise as **GREEN**.
5. **Calls `plan_viewpoint`** — computes the next-best-view waypoint for the robot. Finalise as **RED** (honest unknown, send robot to explore).

**Output:** `output/agent_trace.json` — a full per-region reasoning transcript with tool names, reasoning strings, and observations.

### Optional: Claude LLM planner

```bash
# Set these, then run agent mode again
set PHANTOM_AGENT_LLM=claude
set ANTHROPIC_API_KEY=sk-ant-...

python -m src.main --mode agent
```

The Claude planner (`claude-opus-4-8`) calls the same tools via forced tool-use — the LLM never sees the answer, it sequences real DSP and physics calls. Falls back to the deterministic policy on any API error.

---

## Mode 3b — Real RGB-D Dataset

```bash
python scripts/get_real_dataset.py    # one time — downloads ~5MB Redwood living-room sequence
python -m src.main --mode realtime
```

In the dashboard: pick **"Real RGB-D dataset"** → **▶ Start live scan**.

### What happens differently

- The engine loads real sensor frames from `datasets/redwood_sample/` instead of generating synthetic depth.
- Depth holes in real sensor data are filled using `_fill_depth_holes()` (nearest-neighbour propagation — real values are never overwritten).
- After the scan, the KPI panel shows the held-out-frame score: the pipeline reconstructs from frames 1…N and is evaluated against a frame it never saw.
- Output: `output/real_data_eval.json` — the **headline KPI** (F1@5cm = 0.957 vs Atlas 0.823).

---

## Mode 3c — Your Own Photo

```bash
pip install torch transformers pillow    # one time, ~2.5GB download (CPU-only torch is fine)
```

In the dashboard: pick **"My photo (upload)"** → **▶ Start** → choose a photo.  
Or scan the QR code with your phone → take a photo directly.

### What happens internally

1. Photo is uploaded to `uploads/` (size limit: 20MB).
2. `DepthProMonocular` (open-weight depth model, ~100MB, cached after first run) estimates a dense depth map from the single image.
3. Depth map → point cloud → Gaussian splats, coloured and streamed to the dashboard.

> **Note:** Photo mode is visual-only. A single photo has no ground truth, so no accuracy score is computed. Use Mode 3b for measurable results.

---

## Mode 4 — Evaluation / KPI Reproduction

```bash
python -m src.main --mode eval
```

Runs `src/eval/run_eval.py` across 3 synthetic scenes (`living_room_01`, `office_01`, `bedroom_01`) and writes `output/eval_results.json`.

> These are **self-consistency** numbers on synthetic scenes. The real headline metric is in `output/real_data_eval.json` (Mode 3b). The eval output explicitly labels itself as synthetic to avoid misleading judges.

```bash
python -m src.eval.atlas_baseline    # compare against the Atlas baseline
```

---

## Mode 5 — Single Frame (fast smoke test)

```bash
python -m src.main --mode single-frame
```

Runs the full pipeline on exactly 1 synthetic depth frame. Useful for verifying the setup works before a longer scan. Completes in < 5 seconds.

---

## Mode 6 — Tests (verify the codebase)

```bash
python -m pytest tests/ -v
```

**Expected: 49 passed in ~4 seconds.** Covers:
- `test_arkitscenes.py` — ARKit camera/IMU maths (4 tests)
- `test_integrity.py` — acoustic DSP honesty, no circular measurements, no hardcoded KPIs (15 tests)
- `test_physics_laws.py` — all 8 physics laws (L1–L8) with 30 unit tests

---

## Docker (one-command containerised run)

```bash
# Build (runs tests at image-build time — fails fast if anything is broken)
docker build -t phantom-echo .

# Run the live dashboard
docker run -p 8000:8000 phantom-echo
# → open http://localhost:8000

# Run evaluation inside the container
docker run phantom-echo python -m src.eval.run_real_eval \
    --dataset datasets/redwood_sample --frames 4

# Run tests inside the container
docker run phantom-echo python -m pytest tests/ -q
```

The Docker image runs in `PHANTOM_SIMULATE=true` mode — fully offline, no GPU, no cloud keys.

---

## Reproduce All Committed Artifacts (one command)

```bash
# Windows
powershell -ExecutionPolicy Bypass -File .\reproduce.ps1

# Mac / Linux
bash reproduce.sh
```

Runs in order:
1. **Integrity tests** — verifies acoustic DSP is honest, no circular measurements
2. **Demo pipeline** — regenerates `output/summary.json` (TEAL/BLUE/GREEN counts)
3. **Synthetic eval** — regenerates `output/eval_results.json` (self-consistency benchmark)

---

## Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `PHANTOM_SIMULATE` | `true` | `true` = synthetic room, `false` = real sensor |
| `PHANTOM_OUTPUT` | `output` | Directory for all output files |
| `PHANTOM_STREAM_CAP` | `1500` | Max Gaussians streamed per frame (reduce if browser lags) |
| `PORT` | `8000` | Server port for the dashboard |
| `PHANTOM_DEMO_TOKEN` | _(unset)_ | When set, `/api/reveal` and `/api/mode_b` require `X-Demo-Token: <value>` header (useful for shared-WiFi demos) |
| `PHANTOM_AGENT_LLM` | _(unset)_ | Set to `claude` (+ `ANTHROPIC_API_KEY`) to use Claude as the agent planner |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `uvicorn not found` | `pip install "uvicorn[standard]" fastapi websockets` |
| Port 8000 already in use | `set PORT=8001` (Windows) then rerun; open `http://localhost:8001` |
| Dashboard is blank after opening | Click **▶ Start live scan** — the scene doesn't auto-start |
| No TEAL splats appear | Use 8 or 12 frames in the dropdown; TEAL needs ≥ 3 acoustic baselines |
| `open3d` import warning | Optional dependency; `pip install open3d` only if you want mesh export |
| Photo upload fails with torch error | `pip install torch transformers pillow` |
| "No real dataset at …" error | Run `python scripts/get_real_dataset.py` first |
| Tests fail on Windows with `RuntimeError: daemon process` | Expected — `_KD_WORKERS=1` is already set for Windows; update and rerun |
| `real_data_eval.json not found` on KPI panel | Run a scan in **Real RGB-D dataset** mode first; the file is written after the scan |
