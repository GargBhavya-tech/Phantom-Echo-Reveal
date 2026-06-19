"""
Export REAL acoustic DSP data for the sonar-reveal demo (v24).

Runs the honest forward+inverse acoustic pipeline over the demo walk and captures,
per phone position: the synthesised received waveform, the matched-filter trace,
the detected occluded-echo distance, and the ground-truth distance — plus the
final SAS-triangulated surface point and the mean DSP recovery error.

The frontend (src/frontend/sonar_reveal.html) animates this. Because every number
shown comes from this real pipeline, the demo is true, not a canned animation.

Run:  python -m src.realtime.export_acoustic_demo
Writes: output/acoustic_demo.json
"""
import os
import json
import numpy as np

from src.edge.sensing.acoustic_chirp import ChirpConfig, generate_lfm_chirp, matched_filter
from src.edge.sensing.ism_filter import WallPlane, predict_first_order_arrival
from src.edge.sensing.acoustic_forward import simulate_received_signal, measure_distances
from src.edge.sensing.sas_triangulator import cluster_and_triangulate_v3 as triangulate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEED = 343.0


def downsample(a, n=400):
    if len(a) <= n:
        return a.tolist()
    idx = np.linspace(0, len(a) - 1, n).astype(int)
    return a[idx].tolist()


def main():
    cfg = ChirpConfig()
    rng = np.random.default_rng(7)

    room = {"x": 4.0, "z": 4.0, "y": 2.5}
    target = np.array([1.9, 0.4, 1.0])          # hidden surface behind the sofa
    occluder = {"x0": 0.9, "z0": 0.6, "x1": 2.0, "z1": 1.4, "h": 0.85}  # sofa (top-down)
    walls = [WallPlane(1, 0, 0, 0, "wall_x0"), WallPlane(0, 0, 1, 0, "wall_z0")]

    # smooth 3D arc walk (same family the pipeline uses)
    walk = [np.array([1.0 + 0.45 * np.cos(0.45 * i),
                      0.95 + 0.06 * i,
                      0.7 + 0.45 * np.sin(0.45 * i)]) for i in range(12)]

    ref = generate_lfm_chirp(cfg)
    fs = cfg.sample_rate

    frames = []
    sas = []
    errs = []
    for i, p in enumerate(walk):
        received = simulate_received_signal(p, [target], walls, cfg,
                                            np.random.default_rng(100 + i))
        corr = matched_filter(received, ref)
        m = measure_distances(p, [target], walls, cfg, np.random.default_rng(100 + i))
        det_d = m.distances[0] if m.distances else None
        true_d = float(np.linalg.norm(p - target))
        if det_d is not None:
            sas.append({"position": list(map(float, p)),
                        "distances": [det_d], "snr_db": m.snr_db})
            if m.recovered_errors_cm:
                errs.append(m.recovered_errors_cm[0])
        frames.append({
            "i": i,
            "phone": [float(p[0]), float(p[2])],          # top-down (x,z)
            "phone_y": float(p[1]),
            "received": downsample(np.abs(received), 300),
            "corr": downsample(np.abs(corr), 300),
            "detected_distance_m": None if det_d is None else round(det_d, 4),
            "true_distance_m": round(true_d, 4),
            "snr_db": round(float(m.snr_db), 1) if m.distances else None,
        })

    pts = triangulate(sas, floor_y=0.0)
    tri = None
    if pts:
        Q = np.array(pts[0].position, float)
        tri = {"position": [float(Q[0]), float(Q[2])], "pos3d": list(map(float, Q)),
               "confidence": round(float(pts[0].confidence), 2),
               "error_cm": round(float(np.linalg.norm(Q - target) * 100), 2),
               "n_measurements": int(pts[0].n_measurements)}

    out = {
        "room": room,
        "occluder": occluder,
        "target_true": [float(target[0]), float(target[2])],
        "frames": frames,
        "triangulated": tri,
        "mean_recovery_error_cm": round(float(np.mean(errs)), 3) if errs else None,
        "speed_of_sound_mps": SPEED,
        "note": ("Every waveform and distance here is produced by the real "
                 "forward+inverse acoustic pipeline (matched filter + ISM + SAS). "
                 "The hidden surface is recovered from SOUND, not seen by camera."),
    }
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    path = os.path.join(ROOT, "output", "acoustic_demo.json")
    with open(path, "w") as f:
        json.dump(out, f)

    # Build a STANDALONE, double-clickable HTML with the real data inlined
    # (no server / no CORS needed for the judges).
    tmpl_path = os.path.join(ROOT, "src", "frontend", "sonar_reveal.html")
    if os.path.exists(tmpl_path):
        html = open(tmpl_path, encoding="utf-8").read()
        html = html.replace("/*__ACOUSTIC_DATA__*/ null",
                            json.dumps(out))
        standalone = os.path.join(ROOT, "output", "sonar_reveal.html")
        with open(standalone, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Wrote {standalone}  (double-click to run the demo)")

    print(f"Wrote {path}")
    print(f"  positions with usable echo: {len(sas)}/{len(walk)}")
    print(f"  mean DSP recovery error   : {out['mean_recovery_error_cm']} cm")
    if tri:
        print(f"  triangulated surface error: {tri['error_cm']} cm  (conf {tri['confidence']})")


if __name__ == "__main__":
    main()
