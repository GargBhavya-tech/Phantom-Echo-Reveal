# PHANTOM-ECHO REVEAL — one-command reproduce (Windows PowerShell mirror of reproduce.sh)
# Regenerates every committed artifact in output/ from a clean run and runs the
# integrity test suite. Requires only: numpy scipy scikit-learn.
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\reproduce.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:PHANTOM_OUTPUT = Join-Path $PSScriptRoot "output"

# Prefer the project venv interpreter if present, else fall back to `python`.
$py = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "==> [1/3] integrity tests (lock in the audit fixes)"
& $py tests/test_integrity.py
if ($LASTEXITCODE -ne 0) { throw "integrity tests FAILED (exit $LASTEXITCODE)" }

Write-Host "==> [2/3] demo pipeline -> output/summary.json (honest acoustic TEAL via DSP)"
& $py -m src.main --mode demo | Out-Null
& $py -c "import json; d=json.load(open('output/summary.json')); c=d['counts']; print(f\"    TEAL={c['teal']} (DSP-triangulated)  BLUE={c['blue']}  WHITE={c['white']}  GREEN={c['green']}\")"

Write-Host "==> [3/3] synthetic eval -> output/eval_results.json (SELF-CONSISTENCY, disclosed)"
& $py -m src.main --mode eval | Out-Null
& $py -c "import json; d=json.load(open('output/eval_results.json')); print(f\"    synthetic mean_f1={d['mean_f1']}  mean_semantic={d['mean_semantic']}  all_kpis_met={d['all_kpis_met']}\"); print('    NOTE:', d['evaluation_note'][:90], '...'); print('    >> Headline accuracy is the REAL held-out number in output/real_data_eval.json')"

Write-Host "Done. All artifacts in output/ now reproduce from this run."
