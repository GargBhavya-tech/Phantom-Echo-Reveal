"""
Generate a small, faithful **ARKitScenes-format** fixture (no network needed).

This writes a <video_id>_frames directory with the exact file layout, dtypes and
conventions of a real ARKitScenes low-res capture:
  - lowres_wide/<vid>_<ts>.png        RGB 256x192 uint8
  - lowres_depth/<vid>_<ts>.png       depth uint16 PNG in MILLIMETRES
  - confidence/<vid>_<ts>.png         uint8 (0/1/2)
  - lowres_wide_intrinsics/<vid>_<ts>.pincam   "w h fx fy cx cy"
  - lowres_wide.traj                  "ts rx ry rz tx ty tz" (axis-angle rad, metres)

The geometry is a simple textured planar wall at a known depth, viewed from a
SLOWLY TRANSLATING + ROTATING camera, so the converted dataset exercises the
same reconstruction + held-out reprojection path as real data. It is NOT used as
a headline result — only to prove the loader/converter is correct end-to-end so
you can trust it the moment you download real Apple frames.
"""
from __future__ import annotations
import os, argparse
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

W, H = 256, 192
FX = FY = 211.9          # typical ARKit lowres focal length
CX, CY = 127.5, 95.5
VIDEO_ID = "99999999"


def _axis_angle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle (3,) radians (inverse of loader's Rodrigues)."""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))
    if theta < 1e-9:
        return np.zeros(3)
    rx = (R[2, 1] - R[1, 2])
    ry = (R[0, 2] - R[2, 0])
    rz = (R[1, 0] - R[0, 1])
    v = np.array([rx, ry, rz])
    return v / (2.0 * np.sin(theta)) * theta


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def generate(out_frames_dir: str, n_frames: int = 6) -> None:
    if cv2 is None:
        raise RuntimeError("opencv required: pip install opencv-python-headless")
    rgb_dir   = os.path.join(out_frames_dir, "lowres_wide")
    depth_dir = os.path.join(out_frames_dir, "lowres_depth")
    conf_dir  = os.path.join(out_frames_dir, "confidence")
    intr_dir  = os.path.join(out_frames_dir, "lowres_wide_intrinsics")
    for d in (rgb_dir, depth_dir, conf_dir, intr_dir):
        os.makedirs(d, exist_ok=True)

    # A frontal wall at z=2.0m in world coords, plus a floor — enough structure
    # for the pipeline's physics + reprojection to have something to score.
    traj_lines = []
    for i in range(n_frames):
        ts = 1000.0 + i * 0.5
        # camera_to_world: small translate + small yaw, like a real slow scan
        t_wc = np.array([0.02 * i, 1.1, 0.0])          # camera position in world
        R_wc = _rot_y(np.deg2rad(2.0 * i))             # camera->world rotation
        # Build world->camera (what .traj stores) = inverse of camera->world
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        rvec = _axis_angle(R_cw)
        traj_lines.append(f"{ts:.3f} {rvec[0]:.6f} {rvec[1]:.6f} {rvec[2]:.6f} "
                          f"{t_cw[0]:.6f} {t_cw[1]:.6f} {t_cw[2]:.6f}")

        # Render a depth map of a wall ~2m ahead with a textured RGB.
        depth_m = np.full((H, W), 2.0, dtype=np.float32)
        # add a closer box region (occluder-ish) so depth isn't a flat plane
        depth_m[60:130, 90:170] = 1.4
        depth_mm = (depth_m * 1000.0).astype(np.uint16)

        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[..., 0] = (np.linspace(0, 255, W)[None, :].repeat(H, 0)).astype(np.uint8)
        rgb[..., 1] = (np.linspace(0, 255, H)[:, None].repeat(W, 1)).astype(np.uint8)
        rgb[60:130, 90:170] = (200, 120, 60)
        conf = np.full((H, W), 2, dtype=np.uint8)

        tag = f"{VIDEO_ID}_{ts:.5f}"
        cv2.imwrite(os.path.join(rgb_dir,   tag + ".png"), rgb)
        cv2.imwrite(os.path.join(depth_dir, tag + ".png"), depth_mm)
        cv2.imwrite(os.path.join(conf_dir,  tag + ".png"), conf)
        with open(os.path.join(intr_dir, tag + ".pincam"), "w", encoding="utf-8") as f:
            f.write(f"{W} {H} {FX} {FY} {CX} {CY}\n")

    with open(os.path.join(out_frames_dir, "lowres_wide.traj"), "w", encoding="utf-8") as f:
        f.write("\n".join(traj_lines) + "\n")
    print(f"Wrote synthetic ARKitScenes fixture ({n_frames} frames) → {out_frames_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/99999999_frames")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    generate(a.out, a.n)
