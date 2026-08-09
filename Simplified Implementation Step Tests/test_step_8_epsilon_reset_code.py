import importlib.util
import json

spec = importlib.util.spec_from_file_location('trace_tools', 'trace_tools.py')
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)

RUN_KWARGS = dict(use_sandbox=False, cpu_seconds=10, mem_mb=512, per_call_seconds=2, overall_timeout=60)


def make_trace(n):
    """
    n independent rows. grid_before = [[i % 16]], grid_after = [[(i+1) % 16]].
    No goal transitions (levels_completed omitted -> actual_goal always False),
    keeping this test focused purely on Step 8's accept/reject/commit/revert
    logic, not Step 6/7's goal-discounting (already covered elsewhere).
    """
    records = []
    for i in range(n):
        records.append({
            "step": i, "action": "a",
            "grid_before": [[i % 16]], "grid_after": [[(i + 1) % 16]],
        })
    return records


def make_candidate(correct_steps):
    """
    A GameModel whose predict() tracks its own call-count via previous_state
    (independent of grid content), so we get EXACT control over which of the
    N calls in a given backtest replay are correct -- run_candidate
    instantiates the class fresh and replays records[:boundary] in order for
    every single backtest call, so call-index i always equals the row's
    position in that boundary slice.
    """
    correct_steps_repr = repr(set(correct_steps))
    return f"""
class GameModel:
    def predict(self, grid_before, action, previous_state):
        i = previous_state.get("i", 0)
        correct_steps = {correct_steps_repr}
        if i in correct_steps:
            new_val = (grid_before[0][0] + 1) % 16
        else:
            new_val = (grid_before[0][0] + 2) % 16  # deliberately wrong
        return [[new_val]], False, {{"i": i + 1}}
"""


def read_counts(path):
    with open(path) as f:
        return json.load(f)


import tempfile, os

workdir = tempfile.mkdtemp(prefix="nosumina_step8_test_")
counts_path = os.path.join(workdir, "row_failure_counts.json")
best_counts_path = os.path.join(workdir, "row_failure_counts_best.json")


def reset_counts():
    for p in (counts_path, best_counts_path):
        if os.path.exists(p):
            os.unlink(p)


# ============================================================
# Test 1: chunk 1 (no baseline) -- round 1 accepted outright regardless
# of its own accuracy, since current_best_accuracy starts None.
# ============================================================
reset_counts()
records = make_trace(20)
low_acc_candidate = make_candidate(range(2))  # 2/20 = 10%, deliberately bad

def round_builder_1(round_n, current_best_code, current_best_n_pass, current_best_n_total):
    assert current_best_code is None, "chunk 1 round 1 should see current_best_code=None"
    return low_acc_candidate

result1 = tt.run_chunk_rounds(round_builder_1, records, boundary=20,
                               counts_path=counts_path, best_counts_path=best_counts_path,
                               baseline_code=None, max_rounds=1, **RUN_KWARGS)
assert result1["current_best_code"] == low_acc_candidate
assert abs(result1["current_best_accuracy"] - 0.10) < 1e-9
assert result1["log"][0]["event"] == "accept"
assert result1["log"][0]["threshold"] is None
print("Test 1 (chunk 1, no baseline, round 1 accepted unconditionally): PASS")


# ============================================================
# Test 2: chunk 2's baseline recompute + round loop, replicating the
# design doc's worked example: baseline 80%, round1 60% (rejected),
# round2 78% (accepted).
# ============================================================
reset_counts()
records50 = make_trace(50)
baseline_code = make_candidate(range(40))          # 40/50 = 80%
round1_code = make_candidate(range(30))            # 30/50 = 60%
round2_code = make_candidate(list(range(30)) + list(range(30, 69))[:9])  # need exactly 39/50
round2_code = make_candidate(range(39))            # 39/50 = 78%

calls = []
def round_builder_2(round_n, current_best_code, current_best_n_pass, current_best_n_total):
    calls.append(round_n)
    return {1: round1_code, 2: round2_code}[round_n]

result2 = tt.run_chunk_rounds(round_builder_2, records50, boundary=50,
                               counts_path=counts_path, best_counts_path=best_counts_path,
                               baseline_code=baseline_code, max_rounds=2, **RUN_KWARGS)

assert calls == [1, 2], f"expected round_builder called for rounds 1,2, got {calls}"
log = result2["log"]
assert log[0]["event"] == "baseline_recompute" and abs(log[0]["accuracy"] - 0.80) < 1e-9
assert log[1]["event"] == "reject_substitution" and log[1]["round"] == 1
assert abs(log[1]["candidate_accuracy"] - 0.60) < 1e-9
assert abs(log[1]["threshold"] - 0.75) < 1e-9  # 0.80 - 0.05
assert log[2]["event"] == "accept" and log[2]["round"] == 2
assert abs(log[2]["candidate_accuracy"] - 0.78) < 1e-9
assert abs(log[2]["prev_best_accuracy"] - 0.80) < 1e-9  # compared against baseline, NOT round1's rejected 60%
assert abs(log[2]["threshold"] - 0.75) < 1e-9
assert result2["current_best_code"] == round2_code
assert abs(result2["current_best_accuracy"] - 0.78) < 1e-9
assert result2["early_stopped"] is False
assert result2["rounds_run"] == 2

# row_failure_counts.json must now match round2 (accepted)'s own mutations,
# i.e. be byte-identical (as JSON) to row_failure_counts_best.json.
assert read_counts(counts_path) == read_counts(best_counts_path)
print("Test 2 (baseline 80%, round1 60% rejected, round2 78% accepted): PASS")


# ============================================================
# Test 2b: same scenario, --max-rounds 3, round3 76% (within 5pts of
# round2's 78%) -> accepted, becomes chunk's final code.
# ============================================================
reset_counts()
round3_code = make_candidate(range(38))  # 38/50 = 76%
calls3 = []
def round_builder_3(round_n, current_best_code, current_best_n_pass, current_best_n_total):
    calls3.append(round_n)
    return {1: round1_code, 2: round2_code, 3: round3_code}[round_n]

result3 = tt.run_chunk_rounds(round_builder_3, records50, boundary=50,
                               counts_path=counts_path, best_counts_path=best_counts_path,
                               baseline_code=baseline_code, max_rounds=3, **RUN_KWARGS)
assert calls3 == [1, 2, 3]
assert result3["current_best_code"] == round3_code
assert abs(result3["current_best_accuracy"] - 0.76) < 1e-9
assert result3["log"][-1]["event"] == "accept" and result3["log"][-1]["round"] == 3
assert abs(result3["log"][-1]["threshold"] - 0.73) < 1e-9  # 0.78 - 0.05, compared to round2 not baseline
print("Test 2c (max_rounds=3, round3 76% accepted against round2's 78%): PASS")


# ============================================================
# Test 2c: same scenario, --max-rounds 1 -> only round 1 runs (60%,
# rejected against the 80% baseline); chunk ends with the BASELINE code,
# not round1's rejected candidate, and no round 2 ever gets built.
# ============================================================
reset_counts()
calls1 = []
def round_builder_1r(round_n, current_best_code, current_best_n_pass, current_best_n_total):
    calls1.append(round_n)
    return round1_code

result1r = tt.run_chunk_rounds(round_builder_1r, records50, boundary=50,
                                counts_path=counts_path, best_counts_path=best_counts_path,
                                baseline_code=baseline_code, max_rounds=1, **RUN_KWARGS)
assert calls1 == [1]
assert result1r["current_best_code"] == baseline_code  # round1 rejected -> baseline carried forward
assert abs(result1r["current_best_accuracy"] - 0.80) < 1e-9
assert result1r["rounds_run"] == 1
print("Test 2d (max_rounds=1, round1 rejected, chunk ends with baseline code): PASS")


# ============================================================
# Test 3: row_failure_counts.json commit/revert, tested explicitly.
# Compare two INDEPENDENT fresh runs (separate files, so neither mutates
# state the other depends on): (A) baseline recompute alone, and (B) the
# same baseline recompute immediately followed by a rejected round 1.
# Since both start from a clean slate and baseline_code/records/boundary
# are identical, (B)'s post-revert counts_path must equal (A)'s
# post-baseline-recompute counts_path byte-for-byte -- confirming the
# revert discarded round 1's mutations rather than leaving them standing.
# ============================================================
counts_path_a = os.path.join(workdir, "counts_a.json")
best_counts_path_a = os.path.join(workdir, "counts_a_best.json")
counts_path_b = os.path.join(workdir, "counts_b.json")
best_counts_path_b = os.path.join(workdir, "counts_b_best.json")

result_a = tt.run_chunk_rounds(lambda rn, cbc, cbp, cbt: (_ for _ in ()).throw(AssertionError("should not be called")),
                                records50, boundary=50, counts_path=counts_path_a,
                                best_counts_path=best_counts_path_a, baseline_code=baseline_code,
                                max_rounds=0, **RUN_KWARGS)
counts_after_baseline = read_counts(counts_path_a)
assert counts_after_baseline == read_counts(best_counts_path_a)
assert result_a["current_best_code"] == baseline_code
assert result_a["rounds_run"] == 0

result_b = tt.run_chunk_rounds(lambda rn, cbc, cbp, cbt: round1_code, records50, boundary=50,
                                counts_path=counts_path_b, best_counts_path=best_counts_path_b,
                                baseline_code=baseline_code, max_rounds=1, **RUN_KWARGS)
assert result_b["log"][-1]["event"] == "reject_substitution"
assert result_b["current_best_code"] == baseline_code  # round1 rejected -> baseline still current-best
assert read_counts(counts_path_b) == read_counts(best_counts_path_b), \
    "counts_path should equal best_counts_path after a reject (reverted, not left holding round1's mutations)"
assert read_counts(counts_path_b) == counts_after_baseline, \
    "counts_path after round1's rejection should be byte-identical to the standalone baseline-recompute state"
print("Test 3 (row_failure_counts.json revert-to-baseline verified explicitly): PASS")


# ============================================================
# Test 4: zero-failure early stop.
# ============================================================
reset_counts()
records10 = make_trace(10)
perfect_code = make_candidate(range(10))  # 10/10 = 100%, zero failures
calls_es = []
def round_builder_es(round_n, current_best_code, current_best_n_pass, current_best_n_total):
    calls_es.append(round_n)
    return perfect_code

result_es = tt.run_chunk_rounds(round_builder_es, records10, boundary=10,
                                 counts_path=counts_path, best_counts_path=best_counts_path,
                                 baseline_code=None, max_rounds=3, **RUN_KWARGS)
assert calls_es == [1], f"round_builder should only be called once (round 1), got {calls_es}"
assert result_es["early_stopped"] is True
assert result_es["rounds_run"] == 1
assert result_es["rounds_skipped"] == 2
assert result_es["log"][-1] == {"event": "early_stop", "round": 1, "rounds_skipped": 2}
print("Test 4 (zero-failure early stop after round 1, rounds 2-3 never built): PASS")

# Mixed case: round 1 fails some rows (revision needed), round 2's revision
# happens to fix everything -> early stop fires after round 2, not round 1.
reset_counts()
imperfect_code = make_candidate(range(7))  # 7/10 = 70%, not clean
calls_es2 = []
def round_builder_es2(round_n, current_best_code, current_best_n_pass, current_best_n_total):
    calls_es2.append(round_n)
    return {1: imperfect_code, 2: perfect_code}[round_n]

result_es2 = tt.run_chunk_rounds(round_builder_es2, records10, boundary=10,
                                  counts_path=counts_path, best_counts_path=best_counts_path,
                                  baseline_code=None, max_rounds=3, **RUN_KWARGS)
assert calls_es2 == [1, 2]
assert result_es2["early_stopped"] is True
assert result_es2["rounds_run"] == 2
assert result_es2["rounds_skipped"] == 1
print("Test 4b (early stop fires after round 2 when round 1 wasn't clean): PASS")


# ============================================================
# Test 5: low-accuracy chunk / negative-epsilon-threshold edge case.
# baseline at 3%, candidate at 0% -> threshold = 0.03 - 0.05 = -0.02,
# candidate_accuracy=0.0 >= -0.02 -> accepted, no special-casing needed.
# ============================================================
reset_counts()
records100 = make_trace(100)
baseline_3pct = make_candidate(range(3))   # 3/100 = 3%
zero_pct_code = make_candidate([])         # 0/100 = 0%

result5 = tt.run_chunk_rounds(lambda rn, cbc, cbp, cbt: zero_pct_code, records100, boundary=100,
                               counts_path=counts_path, best_counts_path=best_counts_path,
                               baseline_code=baseline_3pct, max_rounds=1, **RUN_KWARGS)
assert abs(result5["log"][0]["accuracy"] - 0.03) < 1e-9  # baseline recompute
round_entry = result5["log"][1]
assert round_entry["event"] == "accept"
assert abs(round_entry["candidate_accuracy"] - 0.0) < 1e-9
assert abs(round_entry["threshold"] - (-0.02)) < 1e-9
assert result5["current_best_code"] == zero_pct_code
print("Test 5 (negative-threshold edge case: 0% candidate vs 3% baseline, accepted): PASS")

print("\nALL STEP 8 TESTS PASSED")