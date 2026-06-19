# PHANTOM-ECHO REVEAL — User Guide & Setup Instructions

**Problem Statement 09 | Team Chole Bhhature | IIIT Bangalore**

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | 3.11 recommended |
| pip | 23+ | `pip install --upgrade pip` |
| Git | 2.40+ | |
| RAM | 8 GB+ | 16 GB for full VideoScene mode |
| OS | Ubuntu 22.04 / macOS 13+ / Windows 11 | |
| (Optional) GPU | CUDA 11.8+ | For real VideoScene inference |
| (Optional) iPhone | 12 Pro or newer | For hardware acoustic + ARKit |

---

## Quick Start (Simulation Mode — No Hardware)

Everything runs in simulation. No iPhone, no GPU, no Redis required.

### Step 1 — Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/phantom-echo-reveal.git
cd phantom-echo-reveal
pip install -r requirements.txt
```

### Step 2 — Run the pipeline demo

```bash
python -m src.main --mode demo --output output/
```

Expected output (abridged — exact counts/timing vary by machine):
```
[INFO] phantom.main: PHANTOM-ECHO REVEAL v29 — Full Pipeline
[INFO] phantom.main: [Layer 2] SAS v3: 8 measurements → 1 triangulated point
[INFO] phantom.main: [Layer 2a] Proactive laws built BLUE floor/ceiling/wall Gaussians
[INFO] phantom.main: [Layer 2c] Seeded 144 RED voxels in occluded volumes → generation will run
[INFO] phantom.main: [Layer 3]   Generated 200 GREEN Gaussians via template
[INFO] phantom.main: Pipeline complete in ~10–35s
  white: 918   blue: 27441   teal: 1   green: 200   red: 144
```
Writes `output/summary.json`, `output/mesh.ply`, `output/costmap.npy`.

### Step 3 — Open the live 3D dashboard

```bash
python -m src.main --mode realtime          # → http://localhost:8000
```
Open `http://localhost:8000` in Chrome/Firefox/Edge and click **▶ Start live scan**.

You will see:
- ⬜ White — high-confidence ARKit sensor points
- 🔵 Blue — physics-built floor/ceiling/wall priors
- 🩵 Teal — acoustic bat-sonar measured points
- 🟢 Green — AI/template-generated occluded geometry
- 🟡 Yellow — physically-probable soft prior
- 🔴 Red — unknown regions (left open in the nav map)

Use the layer toggles to show/hide each confidence category.
Drag to orbit, scroll to zoom.

---

## Run Evaluation

```bash
# A. Synthetic 3-scene self-consistency benchmark (closed-loop)
python -m src.main --mode eval          # → output/eval_results.json

# B. Blind real-data held-out test (the honest, non-circular headline)
python -m src.eval.run_real_eval --dataset datasets/redwood_sample --frames 4
#                                        → output/real_data_eval.json
```

Honest KPI summary (committed artifacts; targets per README):
```
                        Synthetic (closed-loop)     Real held-out (blind)
F1 @ 5cm   (target≥0.85)  0.858 mean (1/3 @5cm)      0.957   ✓
F1 @ 10cm  (target≥0.85)  0.879 mean   ✓             0.998   ✓
Semantic   (target≥0.93)  0.957  ✓                   —
Recon err  (target<1.5cm) ~0.0cm (circular)          0.98cm  ✓
```
The synthetic benchmark reports `all_kpis_met: false` (office/bedroom clear 0.85
only at 10 cm). Quote the **real held-out** column as the accuracy headline.

---

## Module-Level Testing

Run individual module tests to verify each layer independently:

```bash
# Test acoustic chirp generation
python -c "
from src.edge.sensing.acoustic_chirp import ChirpConfig, generate_lfm_chirp
import numpy as np
cfg = ChirpConfig()
chirp = generate_lfm_chirp(cfg)
print(f'Chirp generated: {len(chirp)} samples, max amplitude={np.max(np.abs(chirp)):.3f}')
"

# Test SAS triangulation
python -c "
from src.edge.sensing.sas_triangulator import triangulate_least_squares, SphereConstraint
import numpy as np
# True point at [2.5, 0.75, 2.0]
true_point = np.array([2.5, 0.75, 2.0])
constraints = []
for i in range(6):
    pos = np.array([0.5 + i*0.3, 1.2, 0.5])
    d = np.linalg.norm(true_point - pos)
    constraints.append(SphereConstraint(pos, d + np.random.normal(0, 0.005), 20.0))
result = triangulate_least_squares(constraints)
print(f'True: {true_point}, Estimated: {result.position.round(3)}, Error: {np.linalg.norm(result.position-true_point)*100:.1f}cm')
"

# Test PHANTOM-LITE contradiction engine
python -c "
from src.edge.phantom_lite.contradiction_engine import *
import numpy as np
hyp = Hypothesis('test_chair', BoundingBox(np.array([1.,0.,1.]), np.array([1.6,0.9,1.6])), 'CHAIR', 'ACOUSTIC')
ctx = {'floor_y': 0.0, 'scene_objects': [], 'camera_pos': np.array([2.5,1.5,2.0]),
       'room_bbox': BoundingBox(np.zeros(3), np.array([5,2.5,4]))}
result = run_contradiction_engine(hyp, ctx)
print(f'Verdict: {result.final_verdict.value}, Tag: {result.final_tag}, Confidence: {result.final_confidence:.2f}')
"
```

---

## Hardware Mode (iPhone + Real Acoustics)

> **Hardware mode requires an iPhone 12 Pro or newer with ARKit LiDAR.**

### iOS App Setup

The iOS companion app (Swift/ARKit) is located in `ios/` (separate repo).
It sends ARKit depth frames and acoustic recordings to the Python pipeline
over local WiFi.

```bash
# Start the cloud API server
uvicorn src.cloud.api.server:app --host 0.0.0.0 --port 8000

# The iOS app connects to http://<your_ip>:8000
# and streams depth frames + audio in real-time
```

### Acoustic calibration

Before each session, run room calibration:

1. Stand in the centre of the room
2. Tap **Calibrate** in the iOS app
3. The system emits 3 calibration chirps and measures room impulse response
4. Visible walls are automatically extracted from ARKit DDGS Gaussians
5. ISM filter is calibrated — ready for occluded region scanning

### Walking protocol for SAS

For best triangulation accuracy:
- Walk a **curved / zigzag** path (an arc or XZ zigzag), NOT a straight line —
  a collinear walk makes the SAS linear system rank-deficient and triangulates
  0 points (the code warns about this in `sas_triangulator._check_collinearity`).
- Cover at least ~1.5m of baseline with lateral variation
- Hold phone at roughly constant height (1.0–1.2m) — the planar-array mirror
  prior assumes near-constant carry height
- Walk at a steady pace (0.3–0.5 m/s); a chirp is emitted automatically every 200ms

---

## Output Files

After running the pipeline, `output/` contains:

| File | Description |
|------|-------------|
| `scene_gaussians.ply` | Point cloud with normals — feed to SPSR for mesh |
| `summary.json` | Tag distribution, KPI summary, processing time |
| `eval_results.json` | Full KPI evaluation results (if run) |

### Meshing from PLY (optional)

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("output/scene_gaussians.ply")
pcd.estimate_normals()
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
o3d.io.write_triangle_mesh("output/scene_mesh.obj", mesh)
print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
```

---

## Reproducing Published Results

To exactly reproduce the evaluation results in the submission:

```bash
# One command regenerates every output/ artifact and runs the integrity suite:
./reproduce.sh            # Linux/macOS
powershell -ExecutionPolicy Bypass -File .\reproduce.ps1   # Windows

# Or compare the synthetic aggregate directly:
python -c "
import json
d = json.load(open('output/eval_results.json'))
print(f'mean_f1:       {d[\"mean_f1\"]:.4f}  (expected ~0.858)')
print(f'mean_semantic: {d[\"mean_semantic\"]:.4f} (expected ~0.957)')
print(f'all_kpis_met:  {d[\"all_kpis_met\"]}')
"
```

Generation seeds are fixed (numpy seed=42) so the synthetic benchmark is
deterministic on a given machine/library set. Quote the real held-out number
(`output/real_data_eval.json`) as the headline.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Run from the project root: `cd phantom-echo-reveal && python -m src.main`

**`ImportError: open3d not found`**
Install separately: `pip install open3d` (may need `--pre` on Apple Silicon)

**`WebGL ERROR` in viewer**
Use Chrome 113+ or Firefox 112+. Enable hardware acceleration in browser settings.

**Acoustic triangulation returns 0 points**
Ensure at least 3 phone positions with >5cm baseline between them.
Check that `distances` array is non-empty in acoustic measurements.

**Synthetic F1 differs slightly across machines**
The synthetic benchmark depends on numpy/scipy versions (KD-tree, RNG). Small
drift is expected; the committed `output/eval_results.json` (mean F1@5cm 0.858)
is the reference. The accuracy headline is the real held-out number, not this.

---

## v22 — Live Dashboard Guide

### Start

```bash
pip install -r requirements.txt
python -m src.main --mode realtime          # → http://localhost:8000
```

### Using the dashboard

1. **▶ Start live scan** — choose 6/8/12 frames; watch the room build up
   frame-by-frame, colour-coded by certainty.
2. **Legend** — live per-tag counts; uncheck a tag to hide that layer
   (e.g. hide WHITE/BLUE to see only what the system *inferred*).
3. **TEAL toast** — appears mid-scan when bat-sonar triangulates an occluded
   surface.
4. **✦ Tap-to-Reveal** — toggle, then click a RED cluster in the 3D view.
   GREEN Gaussians appear where physics + generation place them. Every
   connected browser sees the reveal simultaneously.
5. **Navigation** — drag to orbit, scroll to zoom.
6. **Costmap panel** — Output A (Nav2 occupancy grid); red cells = high cost.

### Demo over WiFi (judge phones)

```bash
python -m src.main --mode realtime
# find your LAN IP (e.g. 192.168.1.7) and share http://192.168.1.7:8000
```
Late joiners automatically receive the scene built so far (snapshot replay).

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| Dashboard loads but empty | click ▶ Start live scan |
| "Backend unreachable" toast | server not running / wrong port |
| No TEAL | use ≥6 frames (SAS needs ≥3 well-spread baselines) |
| No mesh.ply | `pip install open3d` (optional, mesh step degrades gracefully) |
