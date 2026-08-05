"""
Step 11 -- Smoke test gate (before any GPU spend).

No real LLM anywhere in this file. Two independent things are checked:

PART A: two HAND-WRITTEN GameModel candidates (one correct, one deliberately
missing a hidden-state field) run directly through run_backtest against a
hand-written trace encoding a rule that genuinely needs memory beyond what's
visible in a single grid -- confirms the sequential runner, state threading,
and JSON per-row protocol (Step 3) are wired together correctly, and that a
buggy candidate produces REAL, correctly-attributable failures (not just
"some failures somewhere").

PART B: the full cmd_run_chunked entry point (Step 10), driven by a
scripted fake LLM (same technique as test_step10.py), against a trace
shaped exactly per Step 11's spec: max_examples=5, trace length 16 (NOT a
multiple of 5), a level completion at row 7 (NOT aligned to a multiple of
5) -- so the boundary sequence exercises an ordinary cap, an unaligned
level cutoff, another ordinary cap, and a whole-trace cutoff shorter than
the ordinary cap, all in one run. Checked across --max-rounds 1, 2, and 3.

The Step 7 tie-break fixture (6 rows @ count=5, 5 rows @ count=4, k=10) is
already covered by test_step7.py's Test 1 and is not duplicated here.
"""
import argparse
import importlib.util
import io
import json
import os
import shutil
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

spec = importlib.util.spec_from_file_location('trace_tools', 'trace_tools.py')
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)

_orig_run_candidate = tt.run_candidate
def _unsandboxed_run_candidate(*a, **kw):
    kw["use_sandbox"] = False
    return _orig_run_candidate(*a, **kw)
tt.run_candidate = _unsandboxed_run_candidate

tmpdir = tempfile.mkdtemp(prefix="eidolon_step11_test_")


# ================================================================
# PART A -- hand-written switch/door trace + two hand-written candidates.
#
# Rule: cell0 ("flash") is 1 exactly on a step where action=="press", else
# 0 -- purely transient, no memory needed. cell1 ("door") becomes 1
# permanently once the switch has been PRESSED AT LEAST TWICE (cumulative
# press count), and that count is NOT recoverable from the grid alone (only
# the transient flash is visible) -- a candidate that doesn't track a
# hidden press-count field genuinely cannot get this right, unlike a rule
# that's secretly Markovian in disguise.
# ================================================================
part_a_records = [
    {"step": 0,  "action": "wait",  "grid_before": [[0, 0]], "grid_after": [[0, 0]]},
    {"step": 1,  "action": "wait",  "grid_before": [[0, 0]], "grid_after": [[0, 0]]},
    {"step": 2,  "action": "press", "grid_before": [[0, 0]], "grid_after": [[1, 0]]},  # 1st press
    {"step": 3,  "action": "wait",  "grid_before": [[1, 0]], "grid_after": [[0, 0]]},  # only 1 press so far -> door stays closed
    {"step": 4,  "action": "wait",  "grid_before": [[0, 0]], "grid_after": [[0, 0]]},
    {"step": 5,  "action": "press", "grid_before": [[0, 0]], "grid_after": [[1, 1]]},  # 2nd press -> door opens
    {"step": 6,  "action": "wait",  "grid_before": [[1, 1]], "grid_after": [[0, 1]]},
    {"step": 7,  "action": "wait",  "grid_before": [[0, 1]], "grid_after": [[0, 1]]},
    {"step": 8,  "action": "press", "grid_before": [[0, 1]], "grid_after": [[1, 1]]},  # 3rd press, door already open
    {"step": 9,  "action": "wait",  "grid_before": [[1, 1]], "grid_after": [[0, 1]]},
    {"step": 10, "action": "wait",  "grid_before": [[0, 1]], "grid_after": [[0, 1]]},
    {"step": 11, "action": "wait",  "grid_before": [[0, 1]], "grid_after": [[0, 1]]},
    {"step": 12, "action": "wait",  "grid_before": [[0, 1]], "grid_after": [[0, 1]]},
]

CORRECT_CANDIDATE = """class GameModel:
    def predict(self, grid_before, action, previous_state):
        press_count = previous_state.get("press_count", 0)
        if action == "press":
            press_count += 1
        flash = 1 if action == "press" else 0
        door = 1 if press_count >= 2 else 0
        return [[flash, door]], False, {"press_count": press_count}
"""

# Deliberately missing the press-COUNT field -- opens the door after just
# ONE press instead of two, since it has no way to distinguish "pressed
# once" from "pressed twice" without the hidden counter.
BUGGY_CANDIDATE = """class GameModel:
    def predict(self, grid_before, action, previous_state):
        ever_pressed = previous_state.get("ever_pressed", False) or (action == "press")
        flash = 1 if action == "press" else 0
        door = 1 if ever_pressed else 0
        return [[flash, door]], False, {"ever_pressed": ever_pressed}
"""

correct_path = os.path.join(tmpdir, "correct_candidate.py")
buggy_path = os.path.join(tmpdir, "buggy_candidate.py")
Path(correct_path).write_text(CORRECT_CANDIDATE)
Path(buggy_path).write_text(BUGGY_CANDIDATE)

counts_a1 = os.path.join(tmpdir, "counts_a_correct.json")
result_correct = tt.run_backtest(correct_path, part_a_records, len(part_a_records), counts_a1, use_sandbox=False)
assert result_correct["accuracy"] == 1.0, f"correct candidate should backtest clean, got {result_correct['accuracy']}"
assert result_correct["failures"] == []
print(f"Part A.1 (correct switch/door candidate backtests fully clean, {result_correct['n_pass']}/{result_correct['n_total']}): PASS")

counts_a2 = os.path.join(tmpdir, "counts_a_buggy.json")
result_buggy = tt.run_backtest(buggy_path, part_a_records, len(part_a_records), counts_a2, use_sandbox=False)
failing_steps = sorted(s["step"] for s in result_buggy["failures"])
# The buggy candidate opens the door on the FIRST press already (step 2),
# instead of waiting for the second (step 5) -- it's wrong on every step
# where only one press has genuinely occurred (steps 2, 3, 4: door
# predicted open, ground truth still closed), and correct everywhere else,
# including step 5 onward where both rules happen to agree once a real
# second press has occurred.
assert failing_steps == [2, 3, 4], f"expected buggy candidate to fail exactly on steps [2, 3, 4], got {failing_steps}"
assert result_buggy["accuracy"] == (len(part_a_records) - 3) / len(part_a_records)
print(f"Part A.2 (buggy candidate fails EXACTLY on steps 2-4 -- the rows that need the missing press-count field, nowhere else): PASS")


# ================================================================
# PART B -- full cmd_run_chunked wiring against the Step 11 trace shape.
# max_examples=5, 16 rows (not a multiple of 5), level completion at row 7
# (not aligned to a multiple of 5) -> boundaries [0,5],[5,8]level,[8,13],
# [13,16]whole-trace-cutoff (verified against next_chunk_boundary directly
# above before writing this file).
# ================================================================

def make_class_code(correct_steps, goal_row=None):
    correct_steps_repr = repr(set(correct_steps))
    return f"""class GameModel:
    def predict(self, grid_before, action, previous_state):
        i = previous_state.get("i", 0)
        correct_steps = {correct_steps_repr}
        goal_row = {goal_row!r}
        predicted_goal = (goal_row is not None and i == goal_row)
        if i in correct_steps:
            new_val = (grid_before[0][0] + 1) % 16
        else:
            new_val = (grid_before[0][0] + 2) % 16
        return [[new_val]], predicted_goal, {{"i": i + 1}}
"""


def make_trace_b(n, level_row_idx=None):
    records = []
    for i in range(n):
        rec = {"step": i, "action": "a", "grid_before": [[i % 16]], "grid_after": [[(i + 1) % 16]]}
        if level_row_idx is not None and i == level_row_idx:
            rec["levels_completed_before"] = 0
            rec["levels_completed_after"] = 1
        records.append(rec)
    return records


def make_fake_llm(responses):
    state = {"i": 0}
    def _call(prompt_text):
        if state["i"] >= len(responses):
            raise AssertionError(f"fake LLM called more times than expected ({len(responses)})")
        code = responses[state["i"]]
        state["i"] += 1
        return {"text": code, "finish_reason": "stop", "completion_tokens": len(code) // 4}
    _call.n_calls = lambda: state["i"]
    return _call


def make_args(trace_path, workdir, max_rounds, log="chunk_log.jsonl"):
    return argparse.Namespace(
        trace=str(trace_path), workdir=str(workdir),
        backend="llama-cpp", model_path=None, n_ctx=32768, n_gpu_layers=-1, verbose_llama=False,
        api_base="http://x", model=None, temperature=0.2, repeat_penalty=1.3,
        presence_penalty=0.1, frequency_penalty=0.1, max_tokens=4096,
        max_examples=5, compact=False, max_rounds=max_rounds, k=3, log=log,
        automatic=True, cpu_seconds=10, max_procs=16, mem_mb=512,
        per_call_seconds=2, overall_timeout=60,
    )


records_b = make_trace_b(16, level_row_idx=7)
trace_b_path = os.path.join(tmpdir, "trace_b.jsonl")
with open(trace_b_path, "w") as f:
    for r in records_b:
        f.write(json.dumps(r) + "\n")

LEVEL_NOTE = "This chunk starts a NEW LEVEL."


# ---------------- --max-rounds 2 (default) ----------------
# chunk1 [0:5]: r1 60% (accept, no baseline), r2 100% (zero-failure early stop)
# chunk2 [5:8] (LEVEL CUTOFF chunk, rows 5-7 incl. the level row itself):
#   baseline 5/8=62.5%; r1 100% (zero-failure early stop) -- 1 round only
# chunk3 [8:13]: baseline 8/13~61.5%; r1 ~38.5% REJECTED; r2 100% (accept, zero-failure early stop)
# chunk4 [13:16] (WHOLE-TRACE CUTOFF chunk): baseline 13/16=81.25%; r1 100% (zero-failure early stop)
responses_default = [
    make_class_code(range(3)),               # chunk1 r1: 3/5 = 60%
    make_class_code(range(5)),                # chunk1 r2: 5/5 = 100%
    make_class_code(range(7), goal_row=7),    # chunk2 r1: 8/8 = 100%
    make_class_code(range(4), goal_row=7),    # chunk3 r1: 5/13 ~ 38.5% (REJECT vs ~61.5% baseline)
    make_class_code(range(13), goal_row=7),   # chunk3 r2: 13/13 = 100%
    make_class_code(range(16), goal_row=7),   # chunk4 r1: 16/16 = 100%
]
fake_llm_default = make_fake_llm(responses_default)
tt.build_llm_caller = lambda args: (fake_llm_default, None)
workdir_default = Path(tmpdir) / "runB_default"
args_default = make_args(trace_b_path, workdir_default, max_rounds=2)
buf = io.StringIO()
with redirect_stdout(buf):
    tt.cmd_run_chunked(args_default)
output_default = buf.getvalue()

assert fake_llm_default.n_calls() == 6, f"expected 6 LLM calls, got {fake_llm_default.n_calls()}"
print("Part B.1 (--max-rounds 2: 4 chunks incl. level cutoff + whole-trace cutoff, 6 LLM calls as planned): PASS")

# Boundary/chunk shape.
log_lines = (workdir_default / "chunk_log.jsonl").read_text().strip().splitlines()
chunk_reports = [json.loads(l) for l in log_lines]
assert [tuple(r["row_range"]) for r in chunk_reports] == [(0, 5), (5, 8), (8, 13), (13, 16)]
assert [r["n_rows"] for r in chunk_reports] == [5, 3, 5, 3]
print("Part B.2 (boundary sequence [0,5,8,13,16] incl. unaligned 3-row level-cutoff chunk and 3-row final chunk): PASS")

# Level-boundary lag, checked explicitly on the WIRED-TOGETHER loop (not
# just Step 4's isolated boundary function): chunk 2 (5:8) is the chunk
# whose OWN boundary(8) is the level cutoff -- its own prompt must NOT
# carry the note. Chunk 3 (8:13) is the chunk that STARTS right after that
# cutoff -- its round-1 prompt MUST carry the note.
chunk2_prompt = (workdir_default / "chunk2_prompt_round1.txt").read_text()
chunk3_prompt = (workdir_default / "chunk3_prompt_round1.txt").read_text()
assert LEVEL_NOTE not in chunk2_prompt, "chunk 2 (which ENDS on the level cutoff) must not get the note"
assert LEVEL_NOTE in chunk3_prompt, "chunk 3 (which STARTS right after the level cutoff) must get the note"
print("Part B.3 (level-boundary DESCRIPTION note correctly lagged onto chunk 3, not chunk 2 -- explicit wired-loop check): PASS")

# Round counts per chunk: chunk1=2 rounds, chunk2=1 (early stop), chunk3=2
# (reject then accept), chunk4=1 (early stop).
assert (workdir_default / "chunk1_prompt_round2.txt").exists()
assert not (workdir_default / "chunk2_prompt_round2.txt").exists()
assert (workdir_default / "chunk3_prompt_round2.txt").exists()
assert not (workdir_default / "chunk4_prompt_round2.txt").exists()
assert "[chunk 3] SUBSTITUTION at round 1:" in output_default
assert "[chunk 2] EARLY STOP at round 1:" in output_default
assert "[chunk 4] EARLY STOP at round 1:" in output_default
print("Part B.4 (per-chunk round counts match plan: 2,1,2,1 -- including chunk 3's reject-then-accept): PASS")

# Commit/revert, checked IMMEDIATELY after the rejection (not just at the
# very end of the whole run) -- rerun chunk 3's exact scenario in isolation
# via run_chunk_rounds directly so intermediate state is inspectable.
counts_iso = os.path.join(tmpdir, "counts_iso.json")
best_counts_iso = os.path.join(tmpdir, "counts_iso_best.json")
chunk2_final_code = make_class_code(range(7), goal_row=7)  # chunk2's actual final code from the run above

# Baseline-only call (max_rounds=0) captures the state right after the
# baseline recompute, BEFORE any round runs.
tt.run_chunk_rounds(
    lambda rn, cbc, cbp, cbt: (_ for _ in ()).throw(AssertionError("unreachable")),
    records_b, boundary=13, counts_path=counts_iso, best_counts_path=best_counts_iso,
    baseline_code=chunk2_final_code, max_rounds=0, use_sandbox=False,
)
post_baseline_counts = json.loads(Path(counts_iso).read_text())

# Now the real sequence: baseline, then round 1 (rejected).
counts_iso2 = os.path.join(tmpdir, "counts_iso2.json")
best_counts_iso2 = os.path.join(tmpdir, "counts_iso2_best.json")
result_iso = tt.run_chunk_rounds(
    lambda rn, cbc, cbp, cbt: make_class_code(range(4), goal_row=7), records_b, boundary=13,
    counts_path=counts_iso2, best_counts_path=best_counts_iso2,
    baseline_code=chunk2_final_code, max_rounds=1, use_sandbox=False,
)
assert result_iso["log"][-1]["event"] == "reject_substitution"
counts_after_reject = json.loads(Path(counts_iso2).read_text())
best_after_reject = json.loads(Path(best_counts_iso2).read_text())
assert counts_after_reject == best_after_reject, \
    "immediately after a rejection, row_failure_counts.json must equal row_failure_counts_best.json"
assert counts_after_reject == post_baseline_counts, \
    "immediately after a rejection, counts must be byte-identical to the pre-round baseline state"
print("Part B.5 (commit/revert verified IMMEDIATELY after a forced rejection, not just at run's end): PASS")

# ...then force an accepted round afterward and confirm best_counts updates
# to match the new counts.
result_iso2 = tt.run_chunk_rounds(
    lambda rn, cbc, cbp, cbt: make_class_code(range(13), goal_row=7), records_b, boundary=13,
    counts_path=counts_iso2, best_counts_path=best_counts_iso2,
    baseline_code=None, max_rounds=1, use_sandbox=False,
)
# (baseline_code=None here since we're just testing round-level accept/
# commit in isolation; current_best_accuracy starts None so this round is
# accepted unconditionally regardless of the number.)
counts_after_accept = json.loads(Path(counts_iso2).read_text())
best_after_accept = json.loads(Path(best_counts_iso2).read_text())
assert counts_after_accept == best_after_accept
assert counts_after_accept != post_baseline_counts, "the accepted round's mutations should now be committed"
print("Part B.6 (row_failure_counts_best.json updates to match the newly-accepted round's mutations): PASS")


# ---------------- --max-rounds 1 ----------------
# Every chunk gets exactly 1 round. Chunk 3's round 1 is deliberately the
# same REJECT as above -- confirm chunk 3 ends on the BASELINE (not the
# rejected candidate), with no round 2 ever built, even though round 2
# would have fixed it (as shown in the max-rounds-2 run above).
responses_r1 = [
    make_class_code(range(3)),               # chunk1 r1: 60%
    make_class_code(range(7), goal_row=7),    # chunk2 r1: 100%
    make_class_code(range(4), goal_row=7),    # chunk3 r1: ~38.5% (REJECT)
    make_class_code(range(16), goal_row=7),   # chunk4 r1: 100%
]
fake_llm_r1 = make_fake_llm(responses_r1)
tt.build_llm_caller = lambda args: (fake_llm_r1, None)
workdir_r1 = Path(tmpdir) / "runB_maxrounds1"
args_r1 = make_args(trace_b_path, workdir_r1, max_rounds=1)
buf_r1 = io.StringIO()
with redirect_stdout(buf_r1):
    tt.cmd_run_chunked(args_r1)

assert fake_llm_r1.n_calls() == 4, f"expected exactly 4 LLM calls (1 per chunk), got {fake_llm_r1.n_calls()}"
for n in range(1, 5):
    assert not (workdir_r1 / f"chunk{n}_prompt_round2.txt").exists(), f"chunk {n} got a round 2 with --max-rounds 1"
log_lines_r1 = (workdir_r1 / "chunk_log.jsonl").read_text().strip().splitlines()
chunk_reports_r1 = [json.loads(l) for l in log_lines_r1]
# chunk 3 ends on the baseline's accuracy (~61.5%), not round 1's rejected
# ~38.5%, and NOT the 100% it would have reached with a round 2.
baseline_acc_chunk3 = chunk_reports_r1[2]["baseline_accuracy"]
assert abs(chunk_reports_r1[2]["final_accuracy"] - baseline_acc_chunk3) < 1e-9
print("Part B.7 (--max-rounds 1: chunk 3 ends on the baseline after its only round is rejected, no round 2 ever attempted): PASS")


# ---------------- --max-rounds 3 ----------------
# Chunk 3 chains three real comparisons: round 1 vs baseline (reject),
# round 2 vs baseline again (carried forward -- accept, still imperfect),
# round 3 vs round 2's OWN accuracy (accept, zero-failure).
responses_r3 = [
    make_class_code(range(3)),                # chunk1 r1: 60%
    make_class_code(range(5)),                 # chunk1 r2: 100%
    make_class_code(range(7), goal_row=7),     # chunk2 r1: 100%
    make_class_code(range(4), goal_row=7),     # chunk3 r1: ~38.5% (REJECT vs ~61.5% baseline)
    make_class_code(range(10), goal_row=7),    # chunk3 r2: 11/13 ~84.6% (ACCEPT vs baseline, NOT vs round1's rejected value)
    make_class_code(range(13), goal_row=7),    # chunk3 r3: 100% (ACCEPT vs round2's ~84.6%, zero-failure early stop)
    make_class_code(range(16), goal_row=7),    # chunk4 r1: 100%
]
fake_llm_r3 = make_fake_llm(responses_r3)
tt.build_llm_caller = lambda args: (fake_llm_r3, None)
workdir_r3 = Path(tmpdir) / "runB_maxrounds3"
args_r3 = make_args(trace_b_path, workdir_r3, max_rounds=3)
buf_r3 = io.StringIO()
with redirect_stdout(buf_r3):
    tt.cmd_run_chunked(args_r3)
output_r3 = buf_r3.getvalue()

assert fake_llm_r3.n_calls() == 7, f"expected 7 LLM calls, got {fake_llm_r3.n_calls()}"
assert (workdir_r3 / "chunk3_prompt_round3.txt").exists()
assert "[chunk 3] SUBSTITUTION at round 1:" in output_r3
assert "[chunk 3] CODE REPLACED at round 2:" in output_r3
assert "[chunk 3] CODE REPLACED at round 3:" in output_r3
log_lines_r3 = (workdir_r3 / "chunk_log.jsonl").read_text().strip().splitlines()
chunk3_report_r3 = json.loads(log_lines_r3[2])
rounds = chunk3_report_r3["rounds"]
assert rounds[0]["decision"] == "reject"
assert rounds[1]["decision"] == "accept"
assert rounds[2]["decision"] == "accept"
# Round 2 compared against the BASELINE (round 1 was rejected, carried
# forward), not round 1's own rejected ~38.5% -- and round 3 compared
# against round 2's OWN accepted accuracy, not the baseline again.
assert abs(rounds[1]["prev_best_accuracy"] - chunk3_report_r3["baseline_accuracy"]) < 1e-9
assert abs(rounds[2]["prev_best_accuracy"] - rounds[1]["accuracy"]) < 1e-9
print("Part B.8 (--max-rounds 3: chunk 3 chains 3 real comparisons -- round2 vs baseline, round3 vs round2, not round1's rejected value): PASS")


tt.run_candidate = _orig_run_candidate
shutil.rmtree(tmpdir, ignore_errors=True)
print("\nALL STEP 11 SMOKE TESTS PASSED")