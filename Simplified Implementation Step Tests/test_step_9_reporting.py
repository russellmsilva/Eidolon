import importlib.util
import io
import os
import tempfile
from contextlib import redirect_stdout

spec = importlib.util.spec_from_file_location('trace_tools', 'trace_tools.py')
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)

RUN_KWARGS = dict(use_sandbox=False, cpu_seconds=10, mem_mb=512, per_call_seconds=2, overall_timeout=60)


def make_trace(n):
    return [
        {"step": i, "action": "a", "grid_before": [[i % 16]], "grid_after": [[(i + 1) % 16]]}
        for i in range(n)
    ]


def make_candidate(correct_steps):
    correct_steps_repr = repr(set(correct_steps))
    return f"""
class GameModel:
    def predict(self, grid_before, action, previous_state):
        i = previous_state.get("i", 0)
        correct_steps = {correct_steps_repr}
        if i in correct_steps:
            new_val = (grid_before[0][0] + 1) % 16
        else:
            new_val = (grid_before[0][0] + 2) % 16
        return [[new_val]], False, {{"i": i + 1}}
"""


workdir = tempfile.mkdtemp(prefix="eidolon_step9_test_")
counts_path = os.path.join(workdir, "row_failure_counts.json")
best_counts_path = os.path.join(workdir, "row_failure_counts_best.json")

records = make_trace(150)

# ---------------------------------------------------------------
# A deliberately mixed 3-chunk scenario, designed so the printed log's
# answer to Step 9's three "Done when" questions is unambiguous and
# independently checkable:
#   - chunk 1 (rows 0:50):   round 1 only, 90% -- normal accept
#   - chunk 2 (rows 50:100): baseline recompute 80%, round1 40%
#     (rejected), round2 60% (also rejected) -- chunk ends carrying the
#     baseline's 80% forward, LOWER than chunk 1's 90%, demonstrating
#     the epsilon-reset scheme visibly allowing a later chunk to finish
#     worse than an earlier one
#   - chunk 3 (rows 100:150): baseline = chunk 2's 80% code, round 1
#     hits zero failures (100%) and early-stops immediately
# Best chunk-ending accuracy across the run = chunk 3's 100% (also the
# final candidate's own accuracy) -- run ends ON its own peak, so no
# drift note is expected.
# ---------------------------------------------------------------

chunk1_code = make_candidate(range(45))  # 45/50 = 90%
chunk1_result = tt.run_chunk_rounds(
    lambda rn, cbc: chunk1_code, records, boundary=50,
    counts_path=counts_path, best_counts_path=best_counts_path,
    baseline_code=None, max_rounds=1, **RUN_KWARGS,
)

chunk2_baseline = make_candidate(range(80))  # 80/100 = 80% at boundary=100
chunk2_round1 = make_candidate(range(40))    # 40/100 = 40% -- rejected (threshold 75%)
chunk2_round2 = make_candidate(range(60))    # 60/100 = 60% -- also rejected (still compared to 80% baseline, not round1)
chunk2_result = tt.run_chunk_rounds(
    lambda rn, cbc: {1: chunk2_round1, 2: chunk2_round2}[rn], records, boundary=100,
    counts_path=counts_path, best_counts_path=best_counts_path,
    baseline_code=chunk2_baseline, max_rounds=2, **RUN_KWARGS,
)

chunk3_baseline = chunk2_result["current_best_code"]  # whatever chunk 2 actually ended with (the 80% baseline)
chunk3_perfect = make_candidate(range(150))  # boundary=150 -> call-indices 0..149, all correct = zero failures
chunk3_calls = []
def chunk3_builder(rn, cbc):
    chunk3_calls.append(rn)
    return chunk3_perfect

chunk3_result = tt.run_chunk_rounds(
    chunk3_builder, records, boundary=150,
    counts_path=counts_path, best_counts_path=best_counts_path,
    baseline_code=chunk3_baseline, max_rounds=3, **RUN_KWARGS,
)

# ---------------------------------------------------------------
# Drive it through Step 9's reporting exactly as Step 10's outer loop
# eventually will, capturing stdout so we can both print it for a human to
# read and assert on it programmatically.
# ---------------------------------------------------------------
buf = io.StringIO()
running_best = None
boundaries = [(0, 50, chunk1_result, 1), (50, 100, chunk2_result, 2), (100, 150, chunk3_result, 3)]
with redirect_stdout(buf):
    for i, (lo, hi, chunk_result, max_rounds) in enumerate(boundaries, start=1):
        tt.report_chunk(i, lo, hi, chunk_result, max_rounds)
        running_best = tt.update_running_best_accuracy(running_best, chunk_result["current_best_accuracy"])
    tt.print_run_summary(running_best, chunk3_result["current_best_accuracy"], n_chunks=3)

output = buf.getvalue()
print(output)  # for a human to actually read, per Step 9's "Done when"

# ---------------------------------------------------------------
# Programmatic checks matching Step 9's "Done when" three questions.
# ---------------------------------------------------------------

# Q1: which chunk ended with the best-performing code?
assert abs(chunk1_result["current_best_accuracy"] - 0.90) < 1e-9
assert abs(chunk2_result["current_best_accuracy"] - 0.80) < 1e-9  # baseline carried forward, both rounds rejected
assert abs(chunk3_result["current_best_accuracy"] - 1.00) < 1e-9
assert running_best == chunk3_result["current_best_accuracy"] == 1.0
assert "best chunk-ending accuracy seen during the run: 100.0%" in output
print("Q1 (chunk 3 ended with the best code, 100%): PASS")

# Q2: did the final candidate end up worse than the best chunk?
assert not any("NOTE: final candidate is" in line for line in output.splitlines()), \
    "run ends ON its own peak (chunk 3 = 100%), so no drift note should print"
print("Q2 (final candidate == best chunk-ending accuracy, no drift note): PASS")

# Q3: did any chunk stop early because it hit zero failures?
assert chunk3_calls == [1], "chunk 3's round_builder should only be called once (round 1, zero-failure early stop)"
assert chunk3_result["early_stopped"] is True
assert chunk3_result["rounds_skipped"] == 2
assert "[chunk 3] EARLY STOP at round 1: zero failing rows, skipping remaining 2 round(s)" in output
assert not any("EARLY STOP" in line for line in output.splitlines() if line.startswith("[chunk 1]"))
assert not any("EARLY STOP" in line for line in output.splitlines() if line.startswith("[chunk 2]"))
print("Q3 (chunk 3 explicitly logged an early stop; chunks 1-2 did not): PASS")

# Chunk 2's rejections are independently visible as their own lines (both
# rounds, since both were rejected in this scenario) -- and NOT logged as
# a code replacement, since nothing actually replaced chunk 2's code.
assert "[chunk 2] SUBSTITUTION at round 1:" in output
assert "[chunk 2] SUBSTITUTION at round 2:" in output
assert not any(line.startswith("[chunk 2] CODE REPLACED") for line in output.splitlines()), \
    "chunk 2 never actually replaced its code -- both rounds were rejected"
print("Test: chunk 2's rejections are independently visible as SUBSTITUTION lines, no false CODE REPLACED: PASS")

# Chunk 1's normal accept IS logged as a code replacement.
assert "[chunk 1] CODE REPLACED at round 1:" in output
print("Test: chunk 1's accept is logged as CODE REPLACED: PASS")

# summarize_scores/print_score_summary detail is present for each chunk
# (reused machinery, Step 9's explicit requirement -- no new mechanism).
for i in (1, 2, 3):
    assert f"chunk {i} final exact-match accuracy:" in output
    assert f"chunk {i} final accuracy by action:" in output
print("Test: per-chunk score detail present via reused summarize_scores/print_score_summary: PASS")

# Streak is reported per chunk.
assert "streak" in output.lower()
print("Test: streak metric present in chunk report: PASS")

# ---------------------------------------------------------------
# Isolated check of print_run_summary's drift note, in a case where the
# final candidate legitimately DOES end up below the run's peak (not
# exercised by the main scenario above, which ends on its own peak).
# ---------------------------------------------------------------
buf2 = io.StringIO()
with redirect_stdout(buf2):
    tt.print_run_summary(running_best_accuracy=0.90, final_accuracy=0.80, n_chunks=4)
drift_output = buf2.getvalue()
assert "best chunk-ending accuracy seen during the run: 90.0%" in drift_output
assert "final candidate's accuracy: 80.0%" in drift_output
assert "NOTE: final candidate is 10.0% points BELOW" in drift_output
print("Test: print_run_summary's drift note fires correctly when final < running best: PASS")

buf3 = io.StringIO()
with redirect_stdout(buf3):
    tt.print_run_summary(running_best_accuracy=0.80, final_accuracy=0.80, n_chunks=1)
no_drift_output = buf3.getvalue()
assert "NOTE: final candidate is" not in no_drift_output
print("Test: print_run_summary's drift note does NOT fire when final == running best: PASS")

print("\nALL STEP 9 TESTS PASSED")