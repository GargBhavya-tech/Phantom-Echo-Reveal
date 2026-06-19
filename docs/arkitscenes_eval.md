# Evaluating on real Apple ARKitScenes (LiDAR / iPhone-iPad) data

This adds a **second, real-data** evaluation track using Apple's
[ARKitScenes](https://github.com/apple/ARKitScenes) dataset — RGB-D captured
with the actual iPad/iPhone LiDAR sensor. It makes the slides' "ARKit camera
depth" claim literally true instead of aspirational.

> **License note:** ARKitScenes is released by Apple under **CC BY-NC-SA 4.0
> (non-commercial)**. Use it for the hackathon's research/demo evaluation and
> attribute Apple. Keep it as an *evaluation* dataset, not shipped code.

## 1. Download one or two scenes (on your own machine — needs internet)

```bash
git clone https://github.com/apple/ARKitScenes
cd ARKitScenes
# grab the low-res RGB-D + poses for a single video_id (small subset)
python download_data.py raw \
  --video_id 40753679 \
  --download_dir /tmp/ar_raw \
  --raw_dataset_assets lowres_wide lowres_depth lowres_wide_intrinsics lowres_wide.traj
```

This yields `/tmp/ar_raw/.../40753679_frames/` with `lowres_wide/`,
`lowres_depth/`, `lowres_wide_intrinsics/`, and `lowres_wide.traj`.

## 2. Convert to the PHANTOM layout

```bash
python -m src.edge.sensing.arkitscenes_loader \
  --src /tmp/ar_raw/.../40753679_frames \
  --out datasets/arkit_sample \
  --stride 10 --max-frames 6
```

`--stride 10` deliberately takes frames ~1s apart so the held-out frame sees a
**different viewpoint** — a harder, more honest test than near-static
consecutive frames.

## 3. Run the held-out eval (identical to the Redwood path)

```bash
python -m src.eval.run_real_eval --dataset datasets/arkit_sample --frames 4
#  -> output/real_data_eval.json
```

## What to expect (and how to report it)

ARKit LiDAR depth is **256×192 and noisier** than Redwood's high-res Kinect
depth, so the F1 will likely land **below** the Redwood 0.957 — that is normal
and is the honest cost of using real phone data. Report it as a *second* result:

> Headline (Redwood high-res RGB-D): F1@5cm **0.957**
> Real Apple LiDAR (ARKitScenes, iPad Pro): F1@5cm **0.xx** — validated on the
> exact sensor class we deploy to.

A slightly lower number on real phone data is *more* convincing than a high
number on a Kinect-era stand-in, and it fits the deck's existing honesty story.

## Verification without download

The adapter is unit-tested (`tests/test_arkitscenes.py`) and validated
end-to-end against a faithful synthetic fixture
(`tests/fixtures/make_arkitscenes_fixture.py`) so the code path is proven before
you download anything:

```bash
python tests/fixtures/make_arkitscenes_fixture.py --out /tmp/99999999_frames --n 6
python -m src.edge.sensing.arkitscenes_loader --src /tmp/99999999_frames --out /tmp/arkit_sample --stride 1 --max-frames 6
python -m src.eval.run_real_eval --dataset /tmp/arkit_sample --frames 4
```
