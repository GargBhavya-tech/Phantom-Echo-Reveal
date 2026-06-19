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
  4. Exports the Gaussian point cloud as a PLY file.
  5. Runs Screened Poisson Surface Reconstruction (SPSR) via open3d to
     produce a watertight mesh, then computes Chamfer distance against the
     GT point cloud for an independent mesh-quality metric.

The held-out-frame protocol is implemented in `RealtimeEngine._held_out_eval`;
this driver runs a dataset-mode scan and captures the resulting metric, then
writes it to `output/real_data_eval.json` as a committed submission artifact.

Usage:
    python -m src.eval.run_real_eval \\
        --dataset datasets/redwood_sample --frames 4 --output output
"""

import os
import json
import time
import argparse
import logging
import numpy as np
from pathlib import Path

os.environ.setdefault("PHANTOM_SIMULATE", "true")

from src.realtime.engine import RealtimeEngine

logger = logging.getLogger("phantom.real_eval")


# ── SPSR Mesh Export (open3d) ────────────────────────────────────────────────

def _export_ply_and_mesh(gaussians: list, output_dir: str) -> dict:
    """
    Export Gaussian point cloud → PLY, then run Screened Poisson Surface
    Reconstruction (SPSR) via open3d to produce a watertight mesh PLY.

    Returns a dict with paths and mesh stats, or an error message.
    """
    result = {}
    try:
        import open3d as o3d

        pts = np.array([g["position"] for g in gaussians], dtype=np.float64)
        norms = np.array(
            [g.get("normal", [0.0, 1.0, 0.0]) for g in gaussians], dtype=np.float64)
        colors = np.array(
            [g.get("color", [0.5, 0.5, 0.5]) for g in gaussians], dtype=np.float64)
        colors = np.clip(colors, 0.0, 1.0)

        pcd = o3d.geometry.PointCloud()
        pcd.points  = o3d.utility.Vector3dVector(pts)
        pcd.normals = o3d.utility.Vector3dVector(norms)
        pcd.colors  = o3d.utility.Vector3dVector(colors)

        # Remove statistical outliers before meshing (improves SPSR quality)
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        logger.info(f"SPSR: {len(pcd.points)} Gaussians → "
                    f"{len(pcd_clean.points)} after outlier removal")

        # Export point cloud PLY
        ply_path = os.path.join(output_dir, "scene_gaussians_real.ply")
        o3d.io.write_point_cloud(ply_path, pcd_clean)
        result["point_cloud_ply"] = ply_path
        result["n_points_exported"] = int(len(pcd_clean.points))

        # Run SPSR (depth=9 ≈ 1.6cm voxels for a 2m scene)
        logger.info("SPSR: running Screened Poisson Surface Reconstruction (depth=9)…")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_clean, depth=9, width=0, scale=1.1, linear_fit=False)

        # Trim low-density artefacts (open water boundaries)
        densities_np = np.asarray(densities)
        keep_mask = densities_np > np.percentile(densities_np, 5)
        mesh = mesh.select_by_index(np.where(keep_mask)[0])

        mesh_path = os.path.join(output_dir, "mesh_real_spsr.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        result["mesh_ply"] = mesh_path
        result["n_triangles"] = int(len(mesh.triangles))
        result["n_vertices"]  = int(len(mesh.vertices))
        logger.info(f"SPSR mesh: {len(mesh.vertices)} vertices, "
                    f"{len(mesh.triangles)} triangles → {mesh_path}")

    except Exception as e:
        logger.warning(f"SPSR export failed (non-fatal): {e}")
        result["spsr_error"] = str(e)

    return result


def _chamfer_mesh_vs_gt(mesh_ply: str, gt_points: np.ndarray,
                         output_dir: str) -> dict:
    """
    Sample points from the SPSR mesh and compute one-sided Chamfer distance
    (mesh→GT) as an independent mesh-quality metric.
    """
    result = {}
    try:
        import open3d as o3d
        from scipy.spatial import cKDTree

        mesh = o3d.io.read_triangle_mesh(mesh_ply)
        if len(mesh.triangles) == 0:
            return {"chamfer_error": "empty mesh"}

        # Sample ~8000 points from mesh surface
        mesh_pcd = mesh.sample_points_uniformly(number_of_points=8000)
        pred_pts = np.asarray(mesh_pcd.points)

        gt_tree = cKDTree(gt_points)
        d_m2g, _ = gt_tree.query(pred_pts, k=1, workers=1)

        pred_tree = cKDTree(pred_pts)
        d_g2m, _ = pred_tree.query(gt_points, k=1, workers=1)

        p5  = float((d_m2g < 0.05).mean())
        r5  = float((d_g2m < 0.05).mean())
        p10 = float((d_m2g < 0.10).mean())
        r10 = float((d_g2m < 0.10).mean())

        result = {
            "mesh_f1_5cm":       round(2 * p5  * r5  / (p5  + r5  + 1e-9), 4),
            "mesh_f1_10cm":      round(2 * p10 * r10 / (p10 + r10 + 1e-9), 4),
            "mesh_precision_5cm": round(p5,  4),
            "mesh_recall_5cm":    round(r5,  4),
            "mesh_chamfer_cm":    round(float(np.median(d_m2g)) * 100, 2),
            "n_mesh_samples":     int(len(pred_pts)),
        }
        logger.info(f"Mesh Chamfer: {result['mesh_chamfer_cm']} cm  "
                    f"F1@5cm={result['mesh_f1_5cm']}  F1@10cm={result['mesh_f1_10cm']}")
    except Exception as e:
        logger.warning(f"Chamfer mesh eval failed (non-fatal): {e}")
        result["chamfer_error"] = str(e)

    return result


# ── GT point cloud reconstruction from held-out frame ───────────────────────

def _gt_points_from_frame(frame, world_offset: np.ndarray) -> np.ndarray:
    """Back-project the held-out frame depth into room-frame coordinates.
    Applies the same world_offset subtraction used for Gaussian positions so
    that GT points align with the mesh (which is in the shifted room frame)."""
    fx = frame.camera_intrinsics["fx"]
    fy = frame.camera_intrinsics["fy"]
    cx = frame.camera_intrinsics["cx"]
    cy = frame.camera_intrinsics["cy"]
    d  = frame.depth_map

    vs, us = np.where((d > 0.1) & (d < 5.0))
    step = max(1, len(us) // 8000)
    vs, us = vs[::step], us[::step]
    zz = d[vs, us].astype(np.float64)
    pc = np.stack([(us - cx) * zz / fx,
                   (vs - cy) * zz / fy,
                   zz,
                   np.ones_like(zz)])
    return (frame.camera_to_world @ pc)[:3].T - world_offset


# ── Main evaluation driver ───────────────────────────────────────────────────

def run_real_eval(dataset_path: str, n_frames: int, output_dir: str) -> dict:
    if not os.path.isdir(dataset_path):
        raise SystemExit(
            f"Dataset not found at '{dataset_path}'. "
            f"Run: python scripts/get_real_dataset.py")

    captured: dict = {}

    def _capture(ev: dict):
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

    if engine._thread is not None:
        engine._thread.join()

    elapsed = round(time.time() - t0, 2)
    if "error" in captured:
        raise SystemExit(f"scan failed: {captured['error']}")
    if "real_data_eval" not in captured:
        raise SystemExit(
            "no held-out metric was produced — the dataset likely has too few "
            "frames (need at least n_frames+1 with valid depth).")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Export PLY + SPSR mesh ───────────────────────────────────────────────
    logger.info("Exporting Gaussian point cloud and running SPSR meshing…")
    mesh_result = _export_ply_and_mesh(engine.all_gaussians, output_dir)

    # ── Chamfer distance: mesh vs GT held-out frame ──────────────────────────
    chamfer_result = {}
    if "mesh_ply" in mesh_result:
        logger.info("Computing Chamfer distance: mesh vs held-out GT…")
        try:
            from src.edge.sensing.real_dataset_loader import RealDepthGenerator
            loader = RealDepthGenerator(dataset_path)
            all_frames = loader.generate_walk_sequence(n_frames=n_frames + 1)
            if len(all_frames) >= n_frames + 1:
                held_out = all_frames[-1]
                # Apply same world_offset the engine used so GT aligns with mesh
                world_offset = getattr(engine, "_world_offset", np.zeros(3))
                gt_pts = _gt_points_from_frame(held_out, world_offset)
                chamfer_result = _chamfer_mesh_vs_gt(
                    mesh_result["mesh_ply"], gt_pts, output_dir)
        except Exception as e:
            logger.warning(f"GT reload for Chamfer failed (non-fatal): {e}")

    # ── Assemble result ──────────────────────────────────────────────────────
    m = captured["real_data_eval"]
    result = {
        "protocol": "held-out frame (real RGB-D, never seen during reconstruction)",
        "dataset": dataset_path,
        "frames_reconstructed": n_frames,
        "frames_held_out": 1,
        "engine_state": engine.state,
        "n_scene_gaussians": len(engine.all_gaussians),
        "wall_time_s": elapsed,
        "metric": m,
        "mesh": {**mesh_result, **chamfer_result},
        "note": (
            "BLIND metric: ground truth is a real depth frame the "
            "reconstruction never observed. Contrast with output/eval_results.json "
            "which is a closed-loop synthetic consistency check."
        ),
    }

    out_path = os.path.join(output_dir, "real_data_eval.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

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
    if chamfer_result and "mesh_f1_5cm" in chamfer_result:
        logger.info(f"  Mesh F1 @ 5cm:    {chamfer_result.get('mesh_f1_5cm')}")
        logger.info(f"  Mesh Chamfer:     {chamfer_result.get('mesh_chamfer_cm')} cm")
    logger.info(f"\nResults saved → {out_path}")
    if "point_cloud_ply" in mesh_result:
        logger.info(f"Point cloud  → {mesh_result['point_cloud_ply']}")
    if "mesh_ply" in mesh_result:
        logger.info(f"SPSR mesh    → {mesh_result['mesh_ply']}")
    return result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="PHANTOM-ECHO REVEAL real-data held-out eval")
    parser.add_argument("--dataset", default=os.path.join("datasets", "redwood_sample"))
    parser.add_argument("--frames", type=int, default=4,
                        help="frames used for reconstruction (1 more is held out)")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    run_real_eval(args.dataset, args.frames, args.output)


if __name__ == "__main__":
    main()
