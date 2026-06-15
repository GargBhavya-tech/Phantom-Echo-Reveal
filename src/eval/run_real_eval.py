"""
PHANTOM-ECHO REVEAL — Real-Data Held-Out Evaluation
===================================================

This is the project's ONLY non-circular metric. Unlike `run_eval.py` (which
scores the reconstruction against the same procedural simulator that produced
it — a closed-loop consistency check), this script:

  1. Loads a REAL RGB-D sequence (Redwood / ScanNet-style) from disk.
  2. Reconstructs the scene from frames 1..N.
  3. Scores the reconstruction against a held-out frame N+1 the pipeline
     NEVER saw during reconstruction (back-projected depth = ground truth).

The held-out-frame protocol is implemented in `RealtimeEngine._held_out_eval`;
this driver runs a dataset-mode scan and captures the resulting metric, then
writes it to `output/real_data_eval.json` as a committed submission artifact.

Usage:
    python -m src.eval.run_real_eval \
        --dataset datasets/redwood_sample --frames 4 --output output
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path

os.environ.setdefault("PHANTOM_SIMULATE", "true")

from src.realtime.engine import RealtimeEngine

logger = logging.getLogger("phantom.real_eval")


def run_real_eval(dataset_path: str, n_frames: int, output_dir: str) -> dict:
    if not os.path.isdir(dataset_path):
        raise SystemExit(
            f"Dataset not found at '{dataset_path}'. "
            f"Run: python scripts/get_real_dataset.py")

    captured: dict = {}

    def _capture(ev: dict):
        # The engine attaches the held-out result to the final summary event's
        # KPI block (kpis.real_data_eval) when running a dataset scan.
        if ev.get("type") == "summary":
            kpis = ev.get("kpis", {}) or {}
            if "real_data_eval" in kpis:
                captured["real_data_eval"] = kpis["real_data_eval"]
        if ev.get("type") == "stage" and ev.get("status") == "error":
            captured["error"] = ev.get("msg")

    engine = RealtimeEngine(emit=_capture)
    t0 = time.time()
    started = engine.start_scan(
        n_frames=n_frames, frame_delay_s=0.0,
        source="dataset", dataset_path=dataset_path)
    if not started:
        raise SystemExit("engine refused to start (already running?)")

    # Drive the scan to completion (no event loop needed — synchronous join).
    if engine._thread is not None:
        engine._thread.join()

    elapsed = round(time.time() - t0, 2)
    if "error" in captured:
        raise SystemExit(f"scan failed: {captured['error']}")
    if "real_data_eval" not in captured:
        raise SystemExit(
            "no held-out metric was produced — the dataset likely has too few "
            "frames (need at least n_frames+1 with valid depth).")

    result = {
        "protocol": "held-out frame (real RGB-D, never seen during reconstruction)",
        "dataset": dataset_path,
        "frames_reconstructed": n_frames,
        "frames_held_out": 1,
        "engine_state": engine.state,
        "n_scene_gaussians": len(engine.all_gaussians),
        "wall_time_s": elapsed,
        "metric": captured["real_data_eval"],
        "note": (
            "This is a BLIND metric: ground truth is a real depth frame the "
            "reconstruction never observed. Contrast with output/eval_results.json "
            "which is a closed-loop synthetic consistency check."
        ),
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, "real_data_eval.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    m = captured["real_data_eval"]
    logger.info("=" * 60)
    logger.info("REAL-DATA HELD-OUT EVALUATION (blind, non-circular)")
    logger.info("=" * 60)
    logger.info(f"  Dataset:          {dataset_path}")
    logger.info(f"  GT points:        {m.get('n_gt_points')}")
    logger.info(f"  Median recon err: {m.get('recon_err_cm')} cm")
    logger.info(f"  F1 @ 5cm:         {m.get('f1_5cm')}")
    logger.info(f"  F1 @ 10cm:        {m.get('f1_10cm')}")
    logger.info(f"  Precision @ 5cm:  {m.get('precision_5cm')}")
    logger.info(f"  Recall @ 5cm:     {m.get('recall_5cm')}")
    logger.info(f"\nResults saved -> {out_path}")
    return result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="PHANTOM-ECHO REVEAL real-data held-out eval")
    parser.add_argument("--dataset", default=os.path.join("datasets", "redwood_sample"))
    parser.add_argument("--frames", type=int, default=4,
                        help="frames used for reconstruction (1 more is held out)")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    run_real_eval(args.dataset, args.frames, args.output)


if __name__ == "__main__":
    main()
