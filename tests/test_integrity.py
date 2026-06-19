"""
PHANTOM-ECHO REVEAL — Integrity regression tests (v23)

These tests exist because of the audit. They lock in the fixes so they cannot
silently regress:

  - the acoustic channel must RECOVER distance through DSP, not read the target
  - the old circular pattern must not reappear in the pipelines
  - the repo must not ship merge-conflict markers
  - the synthetic eval must disclose that it is synthetic

Run:  python -m pytest tests/ -q     (or:  python tests/test_integrity.py)
Only numpy + scipy are required.
"""
import os
import re
import json
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── 1. Acoustic measurement flows through DSP, not the answer key ────────────
def test_acoustic_distance_is_dsp_recovered_not_fed():
    from src.edge.sensing.acoustic_forward import measure_distances
    from src.edge.sensing.acoustic_chirp import ChirpConfig
    from src.edge.sensing.ism_filter import WallPlane

    phone = np.array([1.0, 1.1, 0.7])
    target = np.array([1.9, 0.4, 1.0])
    walls = [WallPlane(1, 0, 0, 0), WallPlane(0, 0, 1, 0)]
    rng = np.random.default_rng(0)

    m = measure_distances(phone, [target], walls, ChirpConfig(), rng)
    assert m.distances, "no distance recovered from the simulated echo"

    true_d = float(np.linalg.norm(phone - target))
    nearest = min(m.distances, key=lambda d: abs(d - true_d))

    # It must be CLOSE to truth (the estimator works)...
    assert abs(nearest - true_d) < 0.05, f"recovery error too large: {abs(nearest-true_d)*100:.1f}cm"
    # ...but NOT bit-identical to the analytic value: a DSP-recovered range is
    # quantised to the audio sample grid (c/fs/2 ≈ 0.39cm), so equality to
    # machine precision would mean the code bypassed the receiver chain.
    assert abs(nearest - true_d) > 1e-6, "distance equals analytic truth exactly — DSP was bypassed (circular!)"


def test_acoustic_sas_recovers_known_surface():
    from src.edge.sensing.acoustic_forward import sweep_measurements
    from src.edge.sensing.acoustic_chirp import ChirpConfig
    from src.edge.sensing.ism_filter import WallPlane
    from src.edge.sensing.sas_triangulator import cluster_and_triangulate_v3 as tri

    target = np.array([1.9, 0.4, 1.0])
    walk = [np.array([1.0 + 0.45 * np.cos(0.45 * i),
                      0.95 + 0.06 * i,
                      0.7 + 0.45 * np.sin(0.45 * i)]) for i in range(12)]
    walls = [WallPlane(1, 0, 0, 0), WallPlane(0, 0, 1, 0)]
    sas, errs = sweep_measurements(walk, [target], walls, ChirpConfig(),
                                   np.random.default_rng(5))
    pts = tri(sas, floor_y=0.0)
    assert pts, "SAS triangulated 0 surfaces from honest acoustic returns"
    err_cm = min(np.linalg.norm(np.array(p.position) - target) * 100 for p in pts)
    assert err_cm < 5.0, f"triangulation error {err_cm:.1f}cm exceeds 5cm"


# ── 2. The old circular acoustic pattern must be gone from the pipelines ──────
def test_no_circular_acoustic_in_pipelines():
    bad = re.compile(r"distances?\.append\(\s*max\(\s*0\.05\s*,\s*"
                     r"float\(np\.linalg\.norm\(\s*phone")
    for rel in ["src/realtime/engine.py", "src/main_v2.py"]:
        txt = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        # the discarded-DSP smell: assigning detect_echo_peaks(...) to `_`
        assert "_ = detect_echo_peaks(" not in txt, f"{rel}: DSP output still discarded"
        assert not bad.search(txt), f"{rel}: circular 'distance = ||phone-target||' pattern present"


# ── 3. No unresolved merge-conflict markers anywhere ─────────────────────────
SCAN_DIRS = ("src", "tests", "docs")
_TEXT_EXT  = (".py", ".md", ".txt", ".html", ".json", ".yaml")


def _iter_repo_text_files():
    """Yield PROJECT-OWNED text files only — never third-party trees (venv/ etc.).

    Uses an ALLOWLIST of source dirs + root-level docs rather than a denylist.
    The old denylist (".git/__pycache__/uploads/datasets") omitted venv/, so on
    any machine with a local virtualenv on disk this test scanned scipy/sympy/
    IPython, whose docstrings contain bare '=======' underlines, and reported
    them as merge-conflict markers — the integrity suite failed spuriously.
    """
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, _, files in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for fn in files:
                if fn.endswith(_TEXT_EXT):
                    yield os.path.join(dirpath, fn)
    for fn in os.listdir(ROOT):                     # root-level docs only
        p = os.path.join(ROOT, fn)
        if os.path.isfile(p) and fn.endswith((".md", ".txt")):
            yield p


def test_no_merge_conflict_markers():
    # A genuine conflict carries angle-bracket markers ('<<<<<<< ' / '>>>>>>> ').
    # A bare '=======' line is a common docstring/heading underline, so it is
    # only treated as a conflict marker when the SAME file also has an
    # angle-bracket marker. This removes the false positives the audit found.
    angle = re.compile(r"^(<<<<<<< |>>>>>>> )")
    offenders = []
    for p in _iter_repo_text_files():
        try:
            lines = open(p, encoding="utf-8", errors="ignore").read().splitlines()
        except Exception:
            continue
        if any(angle.match(ln) for ln in lines):   # bare '=======' is not enough
            offenders.append(p)
    assert not offenders, f"merge-conflict markers in: {offenders}"


# ── 3b. README KPIs must match the committed eval artifact (anti-drift) ───────
def test_readme_kpi_matches_artifact():
    """The README's synthetic headline F1 must match output/eval_results.json.

    Locks the doc↔artifact consistency the audit flagged (README claimed 0.903
    while the committed JSON said 0.858 with all_kpis_met=false). If the eval is
    regenerated and the mean shifts, this fails until the README is updated.
    """
    p = os.path.join(ROOT, "output", "eval_results.json")
    if not os.path.exists(p):
        return  # eval not run on this machine yet
    mean_f1 = json.load(open(p)).get("mean_f1")
    if mean_f1 is None:
        return
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    token3 = f"{mean_f1:.3f}"        # e.g. "0.858"
    token2 = f"{mean_f1:.2f}"        # e.g. "0.86"
    assert token3 in readme or token2 in readme, (
        f"README does not cite the committed synthetic mean_f1={mean_f1}. "
        "Update the README KPI table to match output/eval_results.json.")


# ── 4. The synthetic benchmark must disclose that it is synthetic ────────────
def test_synthetic_eval_is_disclosed():
    p = os.path.join(ROOT, "output", "eval_results.json")
    if not os.path.exists(p):
        return  # eval not run yet; nothing to check
    d = json.load(open(p))
    note = (d.get("evaluation_note", "") + d.get("note", "")).lower()
    assert "synthetic" in note, "eval_results.json must disclose it is a synthetic/self-consistency benchmark"


# ── 5. Bugs from the v24 audit (locked in) ───────────────────────────────────
def test_bug1_wall_planes_have_correct_sign():
    # malformed D=-rd[...] planes (plane placed 10m outside the room) must be gone
    for rel in ["src/main_v2.py", "src/realtime/engine.py"]:
        txt = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        assert 'D=-_rd["x"]' not in txt and 'D=-rd["x"]' not in txt, f"{rel}: malformed wall sign"
        assert 'D=-_rd["z"]' not in txt and 'D=-rd["z"]' not in txt, f"{rel}: malformed wall sign"


def test_bug2_possible_is_not_tagged_green():
    txt = open(os.path.join(ROOT, "src/edge/phantom_lite/contradiction_engine.py"),
               encoding="utf-8").read()
    # the POSSIBLE fallback must not emit GREEN (GREEN is generation-only)
    assert 'final_tag="GREEN"' not in txt, "POSSIBLE still tagged GREEN"


def test_bug3_log_odds_tables_fully_unified():
    # v27: previously this only checked ORANGE and gave false confidence while
    # BLUE/TEAL/GREEN/YELLOW still diverged (GREEN 1.5 vs 0.7). Check ALL keys.
    from src.shared.gaussian_format import TAG_LOG_ODDS
    from src.navigation.occupancy_grid import LOG_ODDS_SENSOR
    keys = set(TAG_LOG_ODDS) | set(LOG_ODDS_SENSOR)
    mismatches = {k: (TAG_LOG_ODDS.get(k), LOG_ODDS_SENSOR.get(k))
                  for k in keys if TAG_LOG_ODDS.get(k) != LOG_ODDS_SENSOR.get(k)}
    assert not mismatches, f"log-odds tables diverge on {mismatches}"


def test_bug_ceiling_hung_not_impossible():
    import numpy as np
    from src.edge.phantom_lite.contradiction_engine import (
        law_gravity, CEILING_HUNG_SEMANTICS, PhysicsVerdict)

    class _G:
        def __init__(self, y): self.min_pt = np.array([2., y, 2.]); self.max_pt = np.array([2.3, y+0.3, 2.3])
    class _H:
        def __init__(self, sem, y): self.semantic = sem; self.geometry = _G(y); self.confidence = 0.5

    assert "CHANDELIER" in CEILING_HUNG_SEMANTICS
    assert law_gravity(_H("CHANDELIER", 2.2), 0.0, []).verdict != PhysicsVerdict.IMPOSSIBLE, \
        "chandelier @2.2m still IMPOSSIBLE"
    assert law_gravity(_H("BOX", 2.2), 0.0, []).verdict == PhysicsVerdict.IMPOSSIBLE, \
        "floating box wrongly allowed"


def test_tsdf_denoising_is_wired():
    txt = open(os.path.join(ROOT, "src/main_v2.py"), encoding="utf-8").read()
    assert "knn_smooth" in txt, "tsdf_fusion.knn_smooth implemented but never called"


def test_bug4_leg_count_allows_non_four():
    from src.edge.phantom_lite.affordance_router import SLOTLSTM_CONSTRAINTS
    for k in ("CHAIR", "TABLE"):
        lo, hi = SLOTLSTM_CONSTRAINTS[k]["leg_count"]
        assert lo < hi, f"{k} leg_count min==max — stools/office chairs rejected"


def test_bug6_wall_mounted_not_impossible():
    import numpy as np
    from src.edge.phantom_lite.contradiction_engine import (
        law_gravity, WALL_MOUNTED_SEMANTICS, PhysicsVerdict)

    class _G:
        def __init__(self, y): self.min_pt = np.array([1., y, 1.]); self.max_pt = np.array([1.2, y+0.3, 1.2])
    class _H:
        def __init__(self, sem, y): self.semantic = sem; self.geometry = _G(y); self.confidence = 0.5

    assert "CLOCK" in WALL_MOUNTED_SEMANTICS
    clock = law_gravity(_H("CLOCK", 1.5), 0.0, [])
    assert clock.verdict != PhysicsVerdict.IMPOSSIBLE, "wall clock still IMPOSSIBLE"
    # control: a genuinely floating object must STILL be impossible
    lamp = law_gravity(_H("LAMP", 1.5), 0.0, [])
    assert lamp.verdict == PhysicsVerdict.IMPOSSIBLE, "floating object wrongly allowed"


def test_agent_tool_use_resolves_all_paths():
    # The agentic layer must drive the real pipeline modules as tools and
    # exercise every resolution path: PROVE→BLUE, MEASURE→TEAL, IMAGINE→GREEN,
    # EXPLORE→RED. Runs the deterministic (offline) planner.
    from src.agent import run_agent
    res = run_agent()
    tags = set(res.regions.values())
    assert {"BLUE", "TEAL", "GREEN", "RED"}.issubset(tags), \
        f"agent did not exercise all four resolution paths: {res.regions}"
    used = set(res.tool_use_counts)
    assert {"apply_physics", "acoustic_measure", "generate_geometry",
            "plan_viewpoint"}.issubset(used), \
        f"agent did not call all pipeline tools: {used}"
    assert len(res.regions) == 4 and res.steps > 0


def test_systemic_generation_runs_in_eval():
    # The whole point: occluded volumes are seeded RED so Layer 3 generates.
    import json
    p = os.path.join(ROOT, "output", "eval_living_room_01.json")
    if not os.path.exists(p):
        return
    t = json.load(open(p))["tag_distribution"]
    assert t["red"] > 0, "no RED voxels — generation will be skipped (systemic bug)"
    assert t["generated"] > 0 and t["green"] > 0, "generation pillar did not run in eval"
    assert t["teal"] > 0, "acoustic triangulation produced 0 TEAL in eval"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} integrity tests passed")
    # Propagate failure as a non-zero exit code so reproduce.sh (set -e) and CI
    # actually catch a red suite. Previously this always exited 0, so a failing
    # test was silently masked by the one-command reproduce wrapper.
    sys.exit(0 if passed == len(fns) else 1)
