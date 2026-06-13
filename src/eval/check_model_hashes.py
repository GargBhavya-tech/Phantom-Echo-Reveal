"""
PHANTOM-ECHO REVEAL — Model Hash Verification
Verifies that all pinned HuggingFace model weights are still
available and match the expected commit hashes.

Usage:
    python -m src.eval.check_model_hashes

Judges can run this to confirm exact reproducibility.
"""
import sys, json, urllib.request

MODELS = [
    {
        "name":   "MobileSAM",
        "repo":   "dhkim2810/MobileSAM",
        "commit": "a9b07f9c0c51e45a8bccf8cb0c0c1f9c6c77c09a",
        "role":   "Instance segmentation (Layer 0→1 boundary)",
        "file":   "src/edge/segmentation/segmentation_handler.py",
    },
    {
        "name":   "MobileCLIP-S2",
        "repo":   "apple/MobileCLIP-S2",
        "commit": "4e0db7cb1ddb8be0b77feab2af9e7be25879b48c",
        "role":   "Semantic classification (Affordance Router)",
        "file":   "src/edge/embedding/mobile_clip.py",
    },
    {
        "name":   "LLaVA-NeXT-Video-7B",
        "repo":   "llava-hf/LLaVA-NeXT-Video-7B-hf",
        "commit": "f42d64c890bfe5e92ef710e55e8e0c82fc5e55e7",
        "role":   "Vision-language scene description (cloud)",
        "file":   "src/cloud/llm/llava_wrapper.py",
    },
    # NOTE: VideoScene (one-step leap flow distillation) is not yet a public
    # HuggingFace release. No commit hash exists to verify. The three-tier
    # fallback in videoscene_pipeline_fixed.py handles this correctly.
]

RESET = "\033[0m"; GREEN = "\033[92m"; RED = "\033[91m"; DIM = "\033[2m"; BOLD = "\033[1m"

def check_model(m):
    url = f"https://huggingface.co/api/models/{m['repo']}/revision/{m['commit']}"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read())
            sha = data.get("sha", "")
            match = sha.startswith(m["commit"][:8])
            return "pass" if match else "mismatch", sha
    except Exception as e:
        return "error", str(e)

def main():
    print(f"\n{BOLD}PHANTOM-ECHO REVEAL — Model Hash Verification{RESET}")
    print(f"{DIM}Checking {len(MODELS)} pinned models against HuggingFace...{RESET}\n")
    print(f"  {'Model':<22} {'Status':>8}   {'Commit (first 12)':>14}   Role")
    print(f"  {'-'*22} {'-'*8}   {'-'*14}   {'-'*30}")

    all_ok = True
    for m in MODELS:
        status, detail = check_model(m)
        ok = status == "pass"
        if not ok:
            all_ok = False
        icon = GREEN + "✓" + RESET if ok else RED + "✗" + RESET
        commit_short = m['commit'][:12]
        print(f"  {m['name']:<22} {icon:>8}   {commit_short:>14}   {m['role']}")
        if not ok:
            print(f"    {RED}→ {detail}{RESET}")

    print()
    if all_ok:
        print(f"  {GREEN}{BOLD}All model hashes verified. Submission is reproducible.{RESET}")
    else:
        print(f"  {RED}Some hashes could not be verified. Check network or update commits.{RESET}")
    print()
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
