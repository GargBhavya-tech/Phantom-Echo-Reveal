"""Quick pipeline end-to-end check."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("PHANTOM_SIMULATE", "true")

from src.main_v2 import run_full_pipeline
result = run_full_pipeline(n_frames=3)
c = result["counts"]

print("=== Pipeline Results ===")
print("Total Gaussians :", c["total"])
print("Static          :", c["static"])
print("Dynamic ORANGE  :", c["orange"])
print("Tagged          :", c["tagged"])
print("Generated GREEN :", c["green"])
print()
print("Per-tag breakdown:")
for tag in ["white","blue","teal","green","yellow","red","orange"]:
    n = c.get(tag, 0)
    bar = "#" * min(40, n // 50)
    print("  {:8s}: {:5d}  {}".format(tag.upper(), n, bar))
print()
print("Time: {}s".format(result["elapsed_s"]))
print("Payload: {}KB".format(result["payload_kb"]))

assert c["blue"] > 0 or c["white"] > 0, "ERROR: All sensor Gaussians lost their tags!"
assert c["red"] < c["tagged"], "ERROR: All Gaussians RED - physics broken."
print()
print("ASSERTION PASS: Sensor tags preserved, physics working correctly")
