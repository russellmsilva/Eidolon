"""
numpy/sandbox verification script -- run this on JarvisLabs with
`conda activate nosumina`, from the same directory as trace_tools.py (or
adjust the sys.path.insert below).

Checks that a trivial candidate importing numpy runs cleanly through the
existing run_candidate(..., use_sandbox=True) bwrap path, with no bind
errors. Set USE_SANDBOX = False below only for a quick local plumbing
check on a machine without bwrap -- that skips the actual thing this
script exists to verify, so treat a local-only pass as informative, not
as a substitute for the real JarvisLabs bwrap run.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")  # adjust if trace_tools.py lives elsewhere
from trace_tools import run_candidate, ALLOWED_IMPORTS

USE_SANDBOX = False  # flip to False only for a plumbing check without bwrap

print("ALLOWED_IMPORTS:", sorted(ALLOWED_IMPORTS))
assert "numpy" in ALLOWED_IMPORTS, "numpy not in ALLOWED_IMPORTS"

import numpy
print(f"[env check] numpy resolves to: {numpy.__file__}")
print(f"[env check] sys.executable:    {sys.executable}")

# Current candidate contract: a top-level `class GameModel` with a
# `predict(self, grid_before, action, previous_state) -> (predicted_grid_after,
# goal, state)` method -- not a bare predict_next_state(grid, action) function.
candidate_code = '''
import numpy as np

class GameModel:
    def predict(self, grid_before, action, previous_state):
        # Trivial identity transform -- just enough to prove numpy actually
        # imported and ran an array op inside the sandbox, nothing more.
        arr = np.array(grid_before, dtype=int)
        arr = arr + 0
        return arr.tolist(), False, previous_state
'''

tmpdir = Path(tempfile.mkdtemp(prefix="numpy_sandbox_check_"))
candidate_path = tmpdir / "candidate_numpy_check.py"
candidate_path.write_text(candidate_code)

records = [
    {"step": 0, "action": "ACTION_UP", "grid_before": [[1, 2], [3, 4]]},
    {"step": 1, "action": "ACTION_DOWN", "grid_before": [[0, 0], [0, 0]]},
]

print(f"\nRunning candidate at {candidate_path} through run_candidate(use_sandbox={USE_SANDBOX})...")
try:
    results = run_candidate(candidate_path, records, use_sandbox=USE_SANDBOX)
except Exception as e:
    print(f"\n[FAIL] run_candidate raised {type(e).__name__}: {e}")
    print(
        "\nIf this looks like a bwrap bind error (numpy's compiled .so files or "
        "data living outside the conda_prefix bind), compare the numpy path "
        "printed above against conda_prefix in build_bwrap_command's --ro-bind "
        "list. If numpy resolves to somewhere NOT under .../envs/nosumina, add "
        "its directory to the --ro-bind list explicitly."
    )
    sys.exit(1)

print("\n[raw results]")
for step, r in sorted(results.items()):
    print(f"  step {step}: {r}")

ok = True
for step, r in results.items():
    if "error" in r:
        ok = False
        print(f"\n[FAIL] step {step} errored inside sandbox: {r['error']}")

expected = {0: [[1, 2], [3, 4]], 1: [[0, 0], [0, 0]]}
for step, r in results.items():
    if "prediction" in r and r["prediction"] != expected[step]:
        ok = False
        print(f"\n[FAIL] step {step} prediction mismatch: got {r['prediction']}, expected {expected[step]}")
    if "prediction" in r and r.get("goal") is not False:
        ok = False
        print(f"\n[FAIL] step {step} goal mismatch: got {r.get('goal')!r}, expected False")

if ok:
    print(
        "\n[PASS] candidate imported numpy and ran cleanly inside the bwrap "
        "sandbox; predictions match the expected identity output."
    )
else:
    print("\n[FAIL] see errors above.")
    sys.exit(1)