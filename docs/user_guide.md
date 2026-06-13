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

Expected output:
```
2026-06-01 12:00:00 [INFO] PHANTOM-ECHO REVEAL v17 — Pipeline Start
2026-06-01 12:00:00 [INFO] Team: Chole Bhhature | IIIT Bangalore | PS-09
2026-06-01 12:00:01 [INFO] Layer 0: 10 acoustic positions processed
2026-06-01 12:00:01 [INFO] Layer 1: 3072 disk Gaussians created
2026-06-01 12:00:01 [INFO] Layer 2: PHANTOM-LITE Contradiction Engine start
2026-06-01 12:00:01 [INFO]   region_chair (CHAIR): POSSIBLE → GREEN
2026-06-01 12:00:01 [INFO]   region_table (TABLE): POSSIBLE → GREEN
2026-06-01 12:00:01 [INFO]   region_sofa  (SOFA):  POSSIBLE → GREEN
2026-06-01 12:00:01 [INFO]   region_wall_b (WALL): PROVEN  → BLUE
2026-06-01 12:00:02 [INFO] Layer 3: 3 regions sent for generation
2026-06-01 12:00:02 [INFO] Layer 4: Output → output/
2026-06-01 12:00:02 [INFO] Pipeline complete in 1.84s
```

### Step 3 — Open the 3D viewer

Open `src/edge/ui/viewer.html` in Chrome, Firefox, or Edge.

You will see:
- 🔵 Blue Gaussians — proven visible surfaces (walls, floor, ceiling)
- 🩵 Teal Gaussians — acoustic bat-sonar measured points
- 🟢 Green Gaussians — AI-generated occluded geometry
- 🔴 Red Gaussians — unknown regions

Use the layer toggles to show/hide each confidence category.
Drag to orbit, scroll to zoom.

---

## Run Evaluation

```bash
python -m src.eval.evaluate \
    --mode simulation \
    --scene living_room_01 \
    --output output/eval_results.json
```

Expected KPI output:
```
F1 Score:             [run: python -m src.eval.run_eval]  (target≥0.97, atlas=0.85)
Semantic Accuracy:    0.942  (target≥0.93, atlas=0.80) [✓]
Recon Error:          [run: python -m src.eval.run_eval]  (target<1.5cm, atlas=5.0cm)
ALL KPIs MET:         ✓ YES
```

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
- Walk in a straight line of at least 1.5m (creates sufficient aperture)
- Hold phone at ~waist height (1.0–1.2m)
- Walk at steady pace (0.3–0.5 m/s)
- Each chirp is emitted automatically every 200ms during walking

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
# Fixed random seed evaluation
python -m src.eval.evaluate \
    --mode simulation \
    --scene living_room_01 \
    --output output/reproduced_eval.json

# Compare with reference
python -c "
import json
with open('output/reproduced_eval.json') as f: r = json.load(f)
print(f'F1: {r[\"f1_score\"]:.4f}')
print(f'Semantic: {r[\"semantic_accuracy\"]:.4f} (expected ~0.942)')
print(f'Error: {r[\"reconstruction_error_cm\"]:.2f}cm')
"
```

All random seeds are fixed (numpy seed=42, generation seed=hash(region_id) % 2^31)
so results are deterministic across machines.

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

**KPI F1 below 0.97**
This can happen if numpy random state differs across numpy versions.
Pin numpy: `pip install numpy==1.26.4`

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
