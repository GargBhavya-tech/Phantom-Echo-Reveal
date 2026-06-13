# How to Run PHANTOM-ECHO REVEAL v22 (Windows / Mac / Linux)

## 1. Requirements
- Python 3.10 or newer  (check: `python --version`)
- Chrome / Edge browser
- No GPU, no model downloads needed (simulate mode is the default)

## 2. Setup (one time, ~2 min)
Open a terminal INSIDE this folder (the one containing `src/` and `README.md`):

```bash
pip install numpy scipy scikit-learn opencv-python-headless fastapi "uvicorn[standard]" websockets pydantic python-dotenv
```

(Or `pip install -r requirements.txt` for the full set — open3d/torch are
optional and only needed for mesh export / real models.)

## 3. Run the LIVE real-time dashboard  ← the main demo

```bash
python -m src.main --mode realtime
```

Then open  →  **http://localhost:8000**  in Chrome.

In the dashboard:
1. Click **▶ Start live scan** — watch the room build up frame by frame,
   colour-coded by certainty (WHITE/BLUE/YELLOW/RED).
2. Mid-scan a toast appears: **TEAL** = bat-sonar measured an occluded surface.
3. Drag to orbit, scroll to zoom. Uncheck tags in the legend to hide layers.
4. Click **✦ Tap-to-Reveal**, then click a RED cluster in the 3D view →
   GREEN AI-generated Gaussians appear in under a second.
5. Right panel shows pipeline stages, KPIs, the Nav2 costmap, and a live log.

To show it on a phone: find your laptop's LAN IP (`ipconfig` on Windows) and
open `http://<your-ip>:8000` from the phone on the same WiFi.

## 3b. Run on REAL data

**Real RGB-D dataset (recommended — gives a real accuracy score):**
```bash
python scripts/get_real_dataset.py        # downloads a real living-room scene (~5MB)
python -m src.main --mode realtime
```
In the dashboard pick **"Real RGB-D dataset"** as the source → Start live scan.
After the scan, the log + KPI panel show the held-out-frame score: the system
reconstructs from frames 1..N and is graded against a frame it never saw.

**Your own photo:**
```bash
pip install torch transformers pillow     # one time, ~2.5GB (CPU is fine)
```
Pick **"My photo (upload)"** as the source → Start → choose a photo from your
computer (or open the dashboard from your phone via your LAN IP and upload
straight from the camera). First run downloads the open-weight depth model
(~100MB). Photo mode is visual-only — a single photo has no ground truth, so
there is no score; use the dataset mode for measurable comparisons.

## 4. Run the batch pipeline (files output)

```bash
python -m src.main --mode demo
```
Outputs PLY / costmap / summary into `~/phantom_echo_output/` (or set PHANTOM_OUTPUT).

## 5. Reproduce the KPI numbers in the README
(Expected: F1@5cm ≈ 0.90, semantic ≈ 97%, median error < 0.1cm across 3 scenes)

```bash
python -m src.main --mode eval
python -m src.eval.atlas_baseline     # comparison table
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `uvicorn not found` | `pip install "uvicorn[standard]" fastapi websockets` |
| Port 8000 busy | `set PORT=8001` (Windows) then rerun; open localhost:8001 |
| Dashboard empty | click ▶ Start live scan |
| No TEAL appears | use 8 or 12 frames in the dropdown |
| `open3d` warning | optional; `pip install open3d` only if you want mesh.ply |
| Photo upload error about torch | `pip install torch transformers pillow` |
| "No real dataset at ..." | run `python scripts/get_real_dataset.py` first |
