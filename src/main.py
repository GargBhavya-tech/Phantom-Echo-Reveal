"""
PHANTOM-ECHO REVEAL v22 — Single Canonical Entry Point
Delegates entirely to main_v2.run_full_pipeline() so that
`--mode demo` and `--mode eval` use the SAME code path.

Bug-1 fix: dual entry-point confusion eliminated.
"""

import argparse, json, logging, subprocess, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="PHANTOM-ECHO REVEAL Pipeline")
    parser.add_argument("--mode", choices=["demo", "eval", "single-frame", "realtime"], default="demo")
    parser.add_argument("--output", default="output")
    parser.add_argument("--scenes", nargs="+",
                        default=["living_room_01", "office_01", "bedroom_01"])
    args = parser.parse_args()

    if args.mode == "demo":
        import os
        os.environ.setdefault("PHANTOM_OUTPUT", args.output)
        from src.main_v2 import run_full_pipeline
        summary = run_full_pipeline()
        print(json.dumps(summary, indent=2, default=str))
        print(f"\nViewer: open src/edge/ui/viewer.html in Chrome/Firefox")

    elif args.mode == "eval":
        # Bug-CE7 fix: call run_eval (real pipeline), not old evaluate.py
        result = subprocess.run(
            [sys.executable, "-m", "src.eval.run_eval",
             "--output", args.output,
             "--scenes"] + args.scenes,
            capture_output=False
        )
        sys.exit(result.returncode)

    elif args.mode == "realtime":
        # v22: live streaming dashboard (FastAPI + WebSocket + Three.js)
        import os
        os.environ.setdefault("PHANTOM_OUTPUT", args.output)
        import uvicorn
        print("PHANTOM-ECHO REVEAL — live dashboard at http://localhost:8000")
        uvicorn.run("src.realtime.server:app", host="0.0.0.0", port=8000)

    elif args.mode == "single-frame":
        from src.main_v2 import run_full_pipeline
        summary = run_full_pipeline(n_frames=1)
        # BUG-V18-5 FIX: run_full_pipeline returns 'counts' dict, not 'n_gaussians_visible'
        _counts = summary.get('counts', {})
        _total  = sum(_counts.values()) if _counts else 0
        logger.info(f"Single frame: {_total} Gaussians {_counts}")


if __name__ == "__main__":
    main()
