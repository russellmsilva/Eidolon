"""
Step 3 verification script. Run this on JarvisLabs with `conda activate
eidolon`, from the same directory as trace_tools.py, WITH use_sandbox=True
(the default) to actually exercise bwrap. This file also runs standalone
with USE_SANDBOX=False for a quick local plumbing check on a machine
without bwrap -- see the constant below.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")  # adjust if trace_tools.py lives elsewhere
from trace_tools import run_candidate

USE_SANDBOX = False  # flip to False only for a plumbing check without bwrap


def write_candidate(tmpdir, name, source):
    path = tmpdir / name
    path.write_text(source)
    return path


def check(label, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return condition


all_ok = True
tmpdir = Path(tempfile.mkdtemp(prefix="step3_runner_check_"))

# ---------------------------------------------------------------------------
# Case 1: trivial identity GameModel -- always returns the input grid
# unchanged, goal=False, empty state carried through unmodified.
# ---------------------------------------------------------------------------
identity_source = '''
class GameModel:
    def predict(self, grid_before, action, previous_state):
        return grid_before, False, previous_state
'''
identity_path = write_candidate(tmpdir, "identity_candidate.py", identity_source)

records = [
    {"step": 0, "action": "ACTION_UP",    "grid_before": [[1, 0], [0, 0]]},
    {"step": 1, "action": "ACTION_LEFT",  "grid_before": [[1, 2], [0, 0]]},
    {"step": 2, "action": "ACTION_DOWN",  "grid_before": [[1, 2], [3, 0]]},
    {"step": 3, "action": "ACTION_RIGHT", "grid_before": [[1, 2], [3, 4]]},
]

print("\n=== Case 1: trivial identity GameModel ===")
results_1 = run_candidate(identity_path, records, use_sandbox=USE_SANDBOX)
print("raw results:", results_1)

all_ok &= check(
    "all 4 rows returned (no premature abort)",
    set(results_1.keys()) == {0, 1, 2, 3},
)
all_ok &= check(
    "every row's prediction equals its own grid_before (identity)",
    all(results_1[r["step"]].get("prediction") == r["grid_before"] for r in records),
)
all_ok &= check(
    "every row reports goal=False",
    all(results_1[r["step"]].get("goal") is False for r in records),
)
all_ok &= check(
    "no row reports an error",
    all("error" not in results_1[r["step"]] for r in records),
)

# ---------------------------------------------------------------------------
# Case 2: deliberately crashing GameModel -- raises on the row whose action
# is ACTION_CRASH. Confirms abort-on-crash: rows before the crash succeed,
# the crashing row reports an error, and every row after it is simply
# absent from the results (no attempt to keep going with stale state).
# ---------------------------------------------------------------------------
crash_source = '''
class GameModel:
    def predict(self, grid_before, action, previous_state):
        if action == "ACTION_CRASH":
            raise ValueError("deliberate crash for abort-on-crash test")
        return grid_before, False, previous_state
'''
crash_path = write_candidate(tmpdir, "crash_candidate.py", crash_source)

crash_records = [
    {"step": 0, "action": "ACTION_UP",    "grid_before": [[1, 0], [0, 0]]},
    {"step": 1, "action": "ACTION_LEFT",  "grid_before": [[1, 2], [0, 0]]},
    {"step": 2, "action": "ACTION_CRASH", "grid_before": [[1, 2], [3, 0]]},
    {"step": 3, "action": "ACTION_RIGHT", "grid_before": [[1, 2], [3, 4]]},
]

print("\n=== Case 2: deliberately crashing GameModel (abort-on-crash) ===")
results_2 = run_candidate(crash_path, crash_records, use_sandbox=USE_SANDBOX)
print("raw results:", results_2)

all_ok &= check(
    "rows before the crash (0, 1) returned successfully",
    0 in results_2 and "prediction" in results_2[0]
    and 1 in results_2 and "prediction" in results_2[1],
)
all_ok &= check(
    "the crashing row (2) reports an error",
    2 in results_2 and "error" in results_2[2],
)
all_ok &= check(
    "the row after the crash (3) is absent entirely (no stale-state continuation)",
    3 not in results_2,
)
all_ok &= check(
    "exactly 3 rows total in results (0, 1, 2 -- not 4)",
    len(results_2) == 3,
)

print()
print("ALL PASS" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)