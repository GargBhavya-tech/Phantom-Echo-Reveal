"""
PHANTOM-ECHO REVEAL — ARKitScenes adapter
==========================================

Converts an Apple **ARKitScenes** low-res RGB-D capture (real iPad/iPhone LiDAR
data) into the simple ``color/ depth/ pose/ intrinsics.json`` layout that
``RealDepthGenerator`` already understands. After conversion you can run the
existing held-out evaluation unchanged:

    python -m src.edge.sensing.arkitscenes_loader \
        --src   /path/to/<video_id>_frames \
        --out   datasets/arkit_sample \
        --stride 10 --max-frames 6
    python -m src.eval.run_real_eval --dataset datasets/arkit_sample --frames 4

WHY A CONVERTER (not a direct loader): keeping the eval path identical to the
Redwood path means the ARKitScenes result is produced by *exactly* the same
code that produces the headline number — no special-casing, no risk of an
apples-to-oranges comparison.

ARKitScenes raw low-res format (per Apple's DATA.md):
  <video_id>_frames/
    lowres_wide/            <video_id>_<timestamp>.png   RGB 256x192 uint8
    lowres_depth/           <video_id>_<timestamp>.png   depth uint16 PNG, millimetres
    confidence/             <video_id>_<timestamp>.png   uint8 (0 low .. 2 high)  [optional]
    lowres_wide_intrinsics/ <video_id>_<timestamp>.pincam  "w h fx fy cx cy"
    lowres_wide.traj        per line: "ts  rx ry rz  tx ty tz"  (axis-angle radians, metres)

IMPORTANT coordinate-frame note:
  ARKitScenes .traj encodes the camera pose in the ARKit world frame as a
  rotation (axis-angle) + translation that map WORLD -> CAMERA (i.e. it is the
  extrinsic / camera_from_world). The PHANTOM pipeline expects camera_to_world,
  so we INVERT it here. Getting this wrong silently wrecks every physics law and
  the held-out reprojection, so it is unit-tested in tests/test_arkitscenes.py.

License: ARKitScenes is released by Apple under CC BY-NC-SA 4.0 (NON-COMMERCIAL).
Use it for the hackathon's research/demo evaluation only, and attribute Apple.
"""
from __future__ import annotations

import os
import re
import glob
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
except Exception:                                   # pragma: no cover
    cv2 = None


# ── low-level parsers ────────────────────────────────────────────────────────

def parse_pincam(path: str) -> Dict[str, float]:
    """Parse a .pincam line: 'width height fx fy cx cy'."""
    with open(path, "r", encoding="utf-8") as f:
        vals = f.read().strip().split()
    w, h, fx, fy, cx, cy = (float(v) for v in vals[:6])
    return {"width": w, "height": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy}


def axis_angle_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: axis-angle (3,) in radians -> rotation matrix (3,3)."""
    rvec = np.asarray(rvec, dtype=np.float64)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def traj_line_to_camera_to_world(line: str) -> Tuple[str, np.ndarray]:
    """Parse one .traj line -> (timestamp_key, 4x4 camera_to_world matrix).

    The raw line stores world->camera (rotation Rcw, translation tcw). We invert
    to camera->world so the matrix can be consumed directly as camera_to_world.
    """
    parts = line.strip().split()
    ts = parts[0]
    rvec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    tvec = np.array([float(parts[4]), float(parts[5]), float(parts[6])])

    R_cw = axis_angle_to_matrix(rvec)          # world -> camera rotation
    # Build world->camera, then invert to camera->world.
    T_cw = np.eye(4)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = tvec
    T_wc = np.linalg.inv(T_cw)                  # camera -> world
    return f"{round(float(ts), 3):.3f}", T_wc


def _frame_id_from_name(path: str) -> Optional[str]:
    """Extract the '<timestamp>' token from '<video_id>_<timestamp>.png'."""
    base = os.path.basename(path)
    m = re.match(r".*_(\d+\.?\d*)\.(png|jpg)$", base)
    return m.group(1) if m else None


def _nearest_pose(poses: Dict[str, np.ndarray], ts: str) -> np.ndarray:
    """Return the camera_to_world whose timestamp key is nearest to `ts`.

    ARKitScenes RGB/depth timestamps don't always exactly equal a .traj
    timestamp, so we snap to the closest pose (Apple's own dataloader does the
    same kind of nearest/interpolated lookup)."""
    if ts in poses:
        return poses[ts]
    keys = np.array([float(k) for k in poses.keys()])
    target = float(ts)
    nearest = poses[f"{keys[np.argmin(np.abs(keys - target))]:.3f}"]
    return nearest


# ── converter ────────────────────────────────────────────────────────────────

def convert(src_frames_dir: str, out_dir: str,
            stride: int = 10, max_frames: int = 6) -> int:
    """Convert an ARKitScenes <video_id>_frames dir to color/depth/pose layout.

    Returns the number of frames written. Picks frames `stride` apart so the
    held-out frame sees a genuinely different viewpoint (unlike near-static
    consecutive frames), which makes the eval more honest, not less.
    """
    if cv2 is None:
        raise RuntimeError("opencv (cv2) is required: pip install opencv-python-headless")

    rgb_dir   = os.path.join(src_frames_dir, "lowres_wide")
    depth_dir = os.path.join(src_frames_dir, "lowres_depth")
    intr_dir  = os.path.join(src_frames_dir, "lowres_wide_intrinsics")
    conf_dir  = os.path.join(src_frames_dir, "confidence")
    # .traj may be named lowres_wide.traj (raw) or color.traj (3dod sample).
    traj_path = None
    for cand in ("lowres_wide.traj", "color.traj"):
        if os.path.exists(os.path.join(src_frames_dir, cand)):
            traj_path = os.path.join(src_frames_dir, cand)
            break
    if traj_path is None:
        raise FileNotFoundError(f"No .traj (lowres_wide.traj/color.traj) in {src_frames_dir}")

    # depth_densified is the 3dod-sample equivalent of lowres_depth.
    if not os.path.isdir(depth_dir):
        alt = os.path.join(src_frames_dir, "depth_densified")
        if os.path.isdir(alt):
            depth_dir = alt
    if not os.path.isdir(intr_dir):
        alt = os.path.join(src_frames_dir, "color_intrinsics")
        if os.path.isdir(alt):
            intr_dir = alt

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    if not depth_files:
        raise FileNotFoundError(f"No depth PNGs found in {depth_dir}")

    # Parse the whole trajectory once.
    poses: Dict[str, np.ndarray] = {}
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                key, mat = traj_line_to_camera_to_world(line)
                poses[key] = mat
    if not poses:
        raise ValueError(f"No poses parsed from {traj_path}")

    os.makedirs(os.path.join(out_dir, "color"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "pose"), exist_ok=True)

    chosen = depth_files[::stride][:max_frames]
    written = 0
    saved_intr = None
    for i, dpath in enumerate(chosen):
        fid = _frame_id_from_name(dpath)
        if fid is None:
            logger.warning("skip unparseable depth name %s", dpath)
            continue

        # locate matching RGB + intrinsics by the same timestamp token
        rgb_match = glob.glob(os.path.join(rgb_dir, f"*_{fid}.png")) or \
                    glob.glob(os.path.join(rgb_dir, f"*{fid}*.png"))
        intr_match = glob.glob(os.path.join(intr_dir, f"*_{fid}.pincam")) or \
                     glob.glob(os.path.join(intr_dir, f"*{fid}*.pincam"))
        if not rgb_match or not intr_match:
            logger.warning("frame %s missing rgb/intrinsics; skipping", fid)
            continue

        rgb = cv2.imread(rgb_match[0], cv2.IMREAD_COLOR)
        depth16 = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)   # uint16 mm
        if rgb is None or depth16 is None:
            logger.warning("frame %s failed to read; skipping", fid)
            continue

        # depth is uint16 millimetres → keep as 16-bit mm PNG (loader divides /1000)
        intr = parse_pincam(intr_match[0])
        if saved_intr is None:
            saved_intr = {"fx": intr["fx"], "fy": intr["fy"],
                          "cx": intr["cx"], "cy": intr["cy"]}

        pose = _nearest_pose(poses, fid)

        cv2.imwrite(os.path.join(out_dir, "color", f"{i:05d}.png"), rgb)
        cv2.imwrite(os.path.join(out_dir, "depth", f"{i:05d}.png"), depth16)
        np.savetxt(os.path.join(out_dir, "pose", f"{i:05d}.txt"), pose)
        written += 1

    if saved_intr is None:
        raise RuntimeError("No frames written — check the source directory layout")

    with open(os.path.join(out_dir, "intrinsics.json"), "w", encoding="utf-8") as f:
        json.dump(saved_intr, f, indent=2)

    logger.info("ARKitScenes → %s : wrote %d frames (intrinsics fx=%.1f cx=%.1f)",
                out_dir, written, saved_intr["fx"], saved_intr["cx"])
    print(f"Converted {written} ARKitScenes frames → {out_dir}")
    return written


def _main():
    ap = argparse.ArgumentParser(description="Convert ARKitScenes raw frames to PHANTOM layout")
    ap.add_argument("--src", required=True, help="<video_id>_frames directory")
    ap.add_argument("--out", default="datasets/arkit_sample")
    ap.add_argument("--stride", type=int, default=10,
                    help="take every Nth frame (larger = more viewpoint change)")
    ap.add_argument("--max-frames", type=int, default=6)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    convert(args.src, args.out, stride=args.stride, max_frames=args.max_frames)


if __name__ == "__main__":
    _main()
