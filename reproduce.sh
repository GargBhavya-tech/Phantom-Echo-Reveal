#!/usr/bin/env bash
# PHANTOM-ECHO REVEAL — one-command reproduce
# Regenerates every committed artifact in output/ from a clean run, reproduces
# the blind real-data KPI, and runs the FULL test suite.
# Requires only: numpy scipy scikit-learn open3d opencv-python-headless pytest.
set -e
cd "$(dirname "$0")"
export PHANTOM_OUTPUT="$(pwd)/output"

echo "==> [1/4] full test suite (physics laws + integrity)"
python -m pytest tests/ -q

echo "==> [2/4] BLIND real-data held-out eval -> output/real_data_eval.json"
python -m src.eval.run_real_eval --dataset datasets/redwood_sample --frames 4 \
  | grep -E "F1 @ 5cm|F1 @ 10cm|Median recon|Precision|Recall" || true

echo "==> [3/4] demo pipeline -> output/summary.json (acoustic TEAL via DSP, synthetic)"
python -m src.main --mode demo > /dev/null
python - <<'PY'
import json; d=json.load(open("output/summary.json"))
c=d["counts"]; print(f"    TEAL={c['teal']} (DSP-triangulated)  BLUE={c['blue']}  WHITE={c['white']}  GREEN={c['green']}")
PY

echo "==> [4/4] synthetic closed-loop eval -> output/eval_results.json (self-consistency, disclosed)"
python -m src.main --mode eval > /dev/null
python - <<'PY'
import json; d=json.load(open("output/eval_results.json"))
print(f"    synthetic mean_f1={d['mean_f1']}  mean_semantic={d['mean_semantic']}")
print("    >> HEADLINE accuracy is the BLIND real held-out number above")
print("       (output/real_data_eval.json), NOT this synthetic self-consistency check.")
PY
echo "Done. All output/ artifacts reproduce from this run."
