"""
Download a real indoor RGB-D scene and convert it to PHANTOM's dataset format.

Source: Open3D sample of the Redwood RGB-D dataset (5 frames of a real living
room captured with a PrimeSense sensor, with camera odometry). Public dataset,
~5 MB download, no registration.

Usage:
    python scripts/get_real_dataset.py
Creates:
    datasets/redwood_sample/{color/, depth/, pose/, intrinsics.json}

Then in the dashboard pick "Real RGB-D dataset" and Start live scan,
or: curl -X POST localhost:8000/api/scan/start -H "Content-Type: application/json"
        -d '{"source":"dataset","n_frames":4}'
"""

import os
import json
import zipfile
import urllib.request

URL = ("https://github.com/isl-org/open3d_downloads/releases/download/"
       "20220201-data/SampleRedwoodRGBDImages.zip")
OUT = os.path.join("datasets", "redwood_sample")


def parse_odometry_log(path):
    """Redwood .log format: 'i j k' header line + 4 lines of a 4x4 matrix."""
    poses = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i + 4 < len(lines) + 1:
        try:
            rows = [list(map(float, lines[i + 1 + r].split())) for r in range(4)]
        except (ValueError, IndexError):
            break
        poses.append(rows)
        i += 5
    return poses


def main():
    os.makedirs("datasets", exist_ok=True)
    zpath = os.path.join("datasets", "redwood_sample.zip")
    if not os.path.exists(zpath):
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, zpath)
    print("Extracting...")
    tmp = os.path.join("datasets", "_redwood_raw")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmp)

    # locate extracted root (zip may nest a folder)
    root = tmp
    for cand in os.listdir(tmp):
        if os.path.isdir(os.path.join(tmp, cand)) and "Redwood" in cand:
            root = os.path.join(tmp, cand)

    import shutil
    for sub in ("color", "depth", "pose"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    colors = sorted(f for f in os.listdir(os.path.join(root, "color")))
    depths = sorted(f for f in os.listdir(os.path.join(root, "depth")))
    for i, (c, d) in enumerate(zip(colors, depths)):
        shutil.copy(os.path.join(root, "color", c),
                    os.path.join(OUT, "color", f"{i:06d}{os.path.splitext(c)[1]}"))
        shutil.copy(os.path.join(root, "depth", d),
                    os.path.join(OUT, "depth", f"{i:06d}{os.path.splitext(d)[1]}"))

    # poses from odometry log (camera-to-world)
    log = None
    for f in os.listdir(root):
        if f.endswith(".log"):
            log = os.path.join(root, f)
    poses = parse_odometry_log(log) if log else []
    import numpy as np
    for i in range(len(colors)):
        M = np.array(poses[i]) if i < len(poses) else np.eye(4)
        np.savetxt(os.path.join(OUT, "pose", f"{i:06d}.txt"), M)

    # PrimeSense intrinsics (Redwood standard)
    with open(os.path.join(OUT, "intrinsics.json"), "w") as f:
        json.dump({"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5}, f)

    print(f"Done: {len(colors)} frames in {OUT}")
    print('Now run the dashboard and choose "Real RGB-D dataset".')


if __name__ == "__main__":
    main()
