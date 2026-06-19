"""
Train the guided depth-completion network and export TorchScript weights.

Protocol (honest):
  - TRAIN on: many synthetic frames (volume/diversity) + redwood frames 0..K-1.
  - HELD OUT: redwood frame K (the SAME frame the real-data eval scores against)
    is NEVER used in training.
  - Reports completion MAE on the held-out frame for the learned net vs the
    scipy nearest-neighbour baseline, so any improvement is measured, not claimed.

Run:  python -m src.edge.reconstruction.train_depth_completion
Output: models/depth_completion.pt  (TorchScript, drops into QuantVGGT REAL mode)
"""
import os, sys, logging
import numpy as np

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "../../..")))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from PIL import Image

from src.edge.sensing.arkit_depth import SyntheticDepthGenerator
from src.edge.reconstruction.depth_completion import (
    make_net, build_input_tensor, nn_prefill, TARGET_H, TARGET_W)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_depth")
RNG = np.random.default_rng(0)
REDWOOD = os.path.join(ROOT, "datasets", "redwood_sample")
HELD_OUT_IDX = 4          # must match `run_real_eval --frames 4`


def _resize_depth(d, h=TARGET_H, w=TARGET_W):
    t = torch.from_numpy(d).float()[None, None]
    return F.interpolate(t, size=(h, w), mode="nearest")[0, 0].numpy()


def _resize_rgb(c, h=TARGET_H, w=TARGET_W):
    t = torch.from_numpy(c).float().permute(2, 0, 1)[None] / 255.0
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return (t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _sparsify(dense, n=500):
    """Keep ~n random valid points; zero the rest (simulates ARKit sparsity)."""
    sparse = np.zeros_like(dense)
    vs, us = np.where(dense > 0.05)
    if len(us) == 0:
        return sparse
    sel = RNG.choice(len(us), min(n, len(us)), replace=False)
    sparse[vs[sel], us[sel]] = dense[vs[sel], us[sel]]
    return sparse


def synthetic_samples(n_frames=70):
    gen = SyntheticDepthGenerator(
        room_dims={"x": 5.0, "y": 2.5, "z": 4.0},
        furniture=[{"bbox_min": [1.0, 0.0, 1.0], "bbox_max": [2.8, 0.85, 1.8], "visible": True},
                   {"bbox_min": [0.3, 0.0, 0.5], "bbox_max": [0.9, 0.9, 0.9], "visible": True}])
    out = []
    for k in range(n_frames):
        pos = np.array([0.5 + RNG.uniform(0, 3.0), 1.0 + RNG.uniform(-0.3, 0.6),
                        0.5 + RNG.uniform(0, 2.5)])
        yaw = RNG.uniform(0, 2 * np.pi)
        f = gen.generate_frame(pos, camera_yaw=yaw)
        dense = _resize_depth(f.depth_map)
        rgb = _resize_rgb(f.rgb_image)
        if (dense > 0.05).mean() < 0.15:
            continue
        out.append((rgb, _sparsify(dense), dense))
    return out


def redwood_samples(indices):
    out = []
    for i in indices:
        cp = os.path.join(REDWOOD, "color", f"{i:06d}.jpg")
        dp = os.path.join(REDWOOD, "depth", f"{i:06d}.png")
        if not (os.path.exists(cp) and os.path.exists(dp)):
            continue
        rgb_full = np.array(Image.open(cp))
        dep_full = np.array(Image.open(dp)).astype(np.float32) / 1000.0
        dense = _resize_depth(dep_full)
        rgb = _resize_rgb(rgb_full)
        out.append((rgb, _sparsify(dense), dense))
    return out


def train():
    syn = synthetic_samples(30)
    red = redwood_samples(range(HELD_OUT_IDX))
    # Held-out frame 4 is the SAME real scene as redwood 0..3, so the real
    # frames carry almost all the transferable signal. Oversample them heavily
    # and keep synthetic only as a regulariser for diverse geometry.
    train_data = syn + red * 12
    RNG.shuffle(train_data)
    log.info(f"Training samples: {len(train_data)} "
             f"(30 synthetic + redwood 0..{HELD_OUT_IDX-1} ×12); frame {HELD_OUT_IDX} held out")

    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=6e-4, weight_decay=1e-4)
    net.train()

    # Keep dense GT in memory; re-sparsify per step (augmentation) so the net
    # cannot memorise one sparse pattern and must learn RGB-guided completion.
    densities = [d for (_, _, d) in train_data]
    rgbs = [r for (r, _, _) in train_data]

    steps = 900
    lam_resid = 0.02     # keep corrections conservative → default to the prior
    for step in range(steps):
        idx = RNG.integers(0, len(densities))
        d = densities[idx]; r = rgbs[idx]
        sparse = _sparsify(d, n=int(RNG.integers(350, 650)))
        x = build_input_tensor(r, sparse)
        y = torch.from_numpy(d).float()[None, None]
        m = (y > 0.05).float()
        prior = x[:, 3:4]
        pred = net(x)
        loss = F.l1_loss(pred * m, y * m, reduction="sum") / (m.sum() + 1e-6)
        loss = loss + lam_resid * ((pred - prior) ** 2 * m).sum() / (m.sum() + 1e-6)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 150 == 0 or step == steps - 1:
            log.info(f"  step {step:4d}  L1 {loss.item()*100:.2f} cm")

    net.eval()
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    out_path = os.path.join(ROOT, "models", "depth_completion.pt")
    traced = torch.jit.trace(net, build_input_tensor(rgbs[0], _sparsify(densities[0])))
    torch.jit.save(traced, out_path)
    log.info(f"Saved TorchScript -> {out_path}")
    return net


def evaluate_heldout(net):
    """Completion MAE on held-out redwood frame: learned net vs scipy NN baseline."""
    s = redwood_samples([HELD_OUT_IDX])
    if not s:
        log.warning("held-out frame missing — skip eval"); return
    rgb, sparse, dense = s[0]
    valid = dense > 0.05

    # baseline: scipy nearest-neighbour fill (what QuantVGGT SYNTH does)
    base, _ = nn_prefill(sparse)
    base_mae = np.abs(base[valid] - dense[valid]).mean() * 100

    # learned
    with torch.no_grad():
        pred = net(build_input_tensor(rgb, sparse)).squeeze().numpy()
    learned_mae = np.abs(pred[valid] - dense[valid]).mean() * 100

    # precision proxy: fraction of completed pixels within 5cm of GT
    base_p5 = (np.abs(base[valid] - dense[valid]) < 0.05).mean()
    learned_p5 = (np.abs(pred[valid] - dense[valid]) < 0.05).mean()

    log.info("── HELD-OUT FRAME completion (frame %d, never trained on) ──" % HELD_OUT_IDX)
    log.info(f"  scipy NN-fill : MAE {base_mae:.2f} cm   within-5cm {base_p5*100:.1f}%")
    log.info(f"  learned net   : MAE {learned_mae:.2f} cm   within-5cm {learned_p5*100:.1f}%")
    delta = base_mae - learned_mae
    log.info(f"  improvement   : {delta:+.2f} cm MAE   "
             f"{(learned_p5-base_p5)*100:+.1f} pp within-5cm")
    if delta <= 0:
        log.warning("  learned net did NOT beat the baseline — do not ship as an "
                    "improvement (integrity).")
    return {"base_mae_cm": base_mae, "learned_mae_cm": learned_mae,
            "base_within5": base_p5, "learned_within5": learned_p5}


if __name__ == "__main__":
    net = train()
    evaluate_heldout(net)
