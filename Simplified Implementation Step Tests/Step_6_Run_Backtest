import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
from trace_tools import run_backtest, load_row_failure_counts

USE_SANDBOX = False  # this scratch container has no bwrap; flip True on JarvisLabs


def check(label, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return condition


all_ok = True
tmpdir = Path(tempfile.mkdtemp(prefix="step6_backtest_check_"))


def write_candidate(name, source):
    path = tmpdir / name
    path.write_text(source)
    return path


# ===========================================================================
# Part 1: hand-traced multi-round count accumulation
#
# 6-row trace, alternating NOP/INC actions: NOP is identity (grid unchanged),
# INC adds 1 to the single cell. Rows 0,2,4 are NOP (identity candidate gets
# these right); rows 1,3,5 are INC (identity candidate gets these wrong).
# Round 1 & 2 run the identity candidate (counts should accumulate to 1, then
# 2, on the failing rows); round 3 runs a candidate that implements the real
# rule correctly (all counts should reset to 0).
# ===========================================================================
print("=== Part 1: hand-traced multi-round count accumulation ===")

records = []
v = 0
actions = ["NOP", "INC", "NOP", "INC", "NOP", "INC"]
for i, action in enumerate(actions):
    before = [[v]]
    if action == "INC":
        v = (v + 1) % 16
    after = [[v]]
    records.append({"step": i, "action": action, "grid_before": before, "grid_after": after})

print("records:", records)

identity_candidate = write_candidate("identity.py", '''
class GameModel:
    def predict(self, grid_before, action, previous_state):
        return grid_before, False, previous_state
''')

fixed_candidate = write_candidate("fixed.py", '''
class GameModel:
    def predict(self, grid_before, action, previous_state):
        if action == "INC":
            v = (grid_before[0][0] + 1) % 16
            return [[v]], False, previous_state
        return grid_before, False, previous_state
''')

counts_path = tmpdir / "row_failure_counts.json"

# --- Round 1: identity candidate ---
r1 = run_backtest(identity_candidate, records, boundary=6, counts_path=counts_path, use_sandbox=USE_SANDBOX)
expected_pass_r1 = [True, False, True, False, True, False]
actual_pass_r1 = [s["passed"] for s in r1["scored"]]
all_ok &= check("round 1: per-row pass/fail matches hand-check", actual_pass_r1 == expected_pass_r1)
all_ok &= check("round 1: n_pass/n_total/accuracy correct", r1["n_pass"] == 3 and r1["n_total"] == 6 and r1["accuracy"] == 0.5)
all_ok &= check("round 1: longest streak is 1 (alternating pass/fail)", r1["streak"] == 1)

counts_r1 = load_row_failure_counts(counts_path)
expected_counts_r1 = {"0": 0, "1": 1, "2": 0, "3": 1, "4": 0, "5": 1}
actual_counts_r1 = {k: v["count"] for k, v in counts_r1.items()}
all_ok &= check("round 1: counts match hand-trace exactly, row for row", actual_counts_r1 == expected_counts_r1)
all_ok &= check(
    "round 1: actual_grid/actual_goal populated for every row",
    all("actual_grid" in v and "actual_goal" in v for v in counts_r1.values()),
)
all_ok &= check(
    "round 1: predicted_grid matches identity candidate's (wrong) output on failing rows",
    counts_r1["1"]["predicted_grid"] == [[0]] and counts_r1["3"]["predicted_grid"] == [[1]],
)

# --- Round 2: identity candidate again (same failures -> counts increment further) ---
r2 = run_backtest(identity_candidate, records, boundary=6, counts_path=counts_path, use_sandbox=USE_SANDBOX)
counts_r2 = load_row_failure_counts(counts_path)
expected_counts_r2 = {"0": 0, "1": 2, "2": 0, "3": 2, "4": 0, "5": 2}
actual_counts_r2 = {k: v["count"] for k, v in counts_r2.items()}
all_ok &= check("round 2: counts accumulate correctly (failing rows now at 2)", actual_counts_r2 == expected_counts_r2)
all_ok &= check(
    "round 2: actual_grid unchanged from round 1 (ground truth never overwritten)",
    all(counts_r1[k]["actual_grid"] == counts_r2[k]["actual_grid"] for k in counts_r1),
)

# --- Round 3: fixed (correct) candidate -> everything should pass, counts reset ---
r3 = run_backtest(fixed_candidate, records, boundary=6, counts_path=counts_path, use_sandbox=USE_SANDBOX)
all_ok &= check("round 3: fixed candidate passes every row", r3["n_pass"] == 6 and r3["accuracy"] == 1.0)
all_ok &= check("round 3: streak equals full row range (6)", r3["streak"] == 6)
all_ok &= check("round 3: failures list is empty", r3["failures"] == [])

counts_r3 = load_row_failure_counts(counts_path)
expected_counts_r3 = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
actual_counts_r3 = {k: v["count"] for k, v in counts_r3.items()}
all_ok &= check("round 3: all counts reset to 0 after a fully-correct round", actual_counts_r3 == expected_counts_r3)
all_ok &= check(
    "round 3: predicted_grid now overwritten to match the correct output",
    counts_r3["1"]["predicted_grid"] == [[1]] and counts_r3["3"]["predicted_grid"] == [[2]],
)

# ===========================================================================
# Part 2: goal discounting
# ===========================================================================
print("\n=== Part 2: goal discounting ===")

goal_records = [
    {  # true positive: predicted_goal correct, grid deliberately wrong -> should still PASS (discounted)
        "step": 100, "action": "GOAL_TP_WRONGGRID",
        "grid_before": [[1]], "grid_after": [[9]],
        "levels_completed_before": 0, "levels_completed_after": 1,
    },
    {  # false positive: predicts goal, shouldn't have -> FAIL even though grid happens to match
        "step": 101, "action": "GOAL_FP",
        "grid_before": [[9]], "grid_after": [[9]],
        "levels_completed_before": 0, "levels_completed_after": 0,
    },
    {  # false negative: doesn't predict goal, grid also wrong -> FAIL (not discounted)
        "step": 102, "action": "GOAL_FN",
        "grid_before": [[9]], "grid_after": [[3]],
        "levels_completed_before": 0, "levels_completed_after": 1,
    },
    {  # true negative + correct grid -> ordinary PASS
        "step": 103, "action": "GOAL_TN_CORRECT",
        "grid_before": [[3]], "grid_after": [[3]],
        "levels_completed_before": 0, "levels_completed_after": 0,
    },
]

goal_candidate = write_candidate("goal_candidate.py", '''
class GameModel:
    def predict(self, grid_before, action, previous_state):
        if action == "GOAL_TP_WRONGGRID":
            return [[5]], True, previous_state
        if action == "GOAL_FP":
            return grid_before, True, previous_state
        if action == "GOAL_FN":
            return grid_before, False, previous_state
        return grid_before, False, previous_state
''')

goal_counts_path = tmpdir / "row_failure_counts_goal.json"
rg = run_backtest(goal_candidate, goal_records, boundary=4, counts_path=goal_counts_path, use_sandbox=USE_SANDBOX)
by_step = {s["step"]: s for s in rg["scored"]}

all_ok &= check(
    "true positive (wrong grid) still passes -- grid discounted",
    by_step[100]["passed"] is True and by_step[100]["goal_discounted"] is True,
)
all_ok &= check(
    "false positive fails even with a grid that happens to match",
    by_step[101]["passed"] is False and by_step[101]["goal_discounted"] is True,
)
all_ok &= check(
    "false negative fails and is NOT discounted (predicted_goal was False)",
    by_step[102]["passed"] is False and by_step[102]["goal_discounted"] is False,
)
all_ok &= check(
    "ordinary true negative with correct grid passes normally",
    by_step[103]["passed"] is True and by_step[103]["goal_discounted"] is False,
)

# ===========================================================================
# Part 3: crash / abort-on-crash handling flows through correctly
# ===========================================================================
print("\n=== Part 3: crash handling ===")

crash_records = [
    {"step": 0, "action": "OK", "grid_before": [[0]], "grid_after": [[0]]},
    {"step": 1, "action": "CRASH", "grid_before": [[0]], "grid_after": [[0]]},
    {"step": 2, "action": "OK", "grid_before": [[0]], "grid_after": [[0]]},
]
crash_candidate = write_candidate("crash_candidate.py", '''
class GameModel:
    def predict(self, grid_before, action, previous_state):
        if action == "CRASH":
            raise ValueError("boom")
        return grid_before, False, previous_state
''')
crash_counts_path = tmpdir / "row_failure_counts_crash.json"
rc = run_backtest(crash_candidate, crash_records, boundary=3, counts_path=crash_counts_path, use_sandbox=USE_SANDBOX)
by_step_c = {s["step"]: s for s in rc["scored"]}

all_ok &= check("row before the crash passes normally", by_step_c[0]["passed"] is True)
all_ok &= check(
    "crashing row fails with its real error message",
    by_step_c[1]["passed"] is False and "boom" in by_step_c[1]["error"],
)
all_ok &= check(
    "row after the crash fails with a 'no result' error, not silently skipped",
    by_step_c[2]["passed"] is False and "no result" in by_step_c[2]["error"],
)
all_ok &= check("crash test: exactly 1 pass out of 3", rc["n_pass"] == 1 and rc["n_total"] == 3)

print()
print("ALL PASS" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)