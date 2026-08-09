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

# Force every backtest in this test to skip bwrap (unavailable in this
# container) -- a test-only monkeypatch of the underlying run_candidate,
# not a change to cmd_run_chunked's own code path (which always leaves
# use_sandbox at its real default on JarvisLabs).
_orig_run_candidate = tt.run_candidate
def _unsandboxed_run_candidate(*a, **kw):
    kw["use_sandbox"] = False
    return _orig_run_candidate(*a, **kw)
tt.run_candidate = _unsandboxed_run_candidate


def make_class_code(correct_steps, goal_row=None):
    """
    goal_row: the call-index (== absolute trace row, since every backtest
    replays from row 0) at which this candidate should predict goal=True.
    Needed for any chunk whose row range includes the level-completion row
    (row 44 in this test's trace) if that row is meant to pass: goal
    correctness is checked BEFORE/INDEPENDENT of grid correctness (Step 6),
    so being in `correct_steps` alone can never make that specific row
    pass -- it would otherwise always register as a false-negative goal
    prediction (predicted False, actually True) regardless of grid content.
    """
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


def make_trace(n, level_row_idx=None):
    """
    n rows, single-cell grid stepping [[i%16]] -> [[(i+1)%16]]. If
    level_row_idx is given, that ONE record (0-indexed) carries
    levels_completed_before=0/after=1 -- everywhere else omits the field
    entirely (next_level_completion_row treats a missing field as "no
    signal", per its own docstring).
    """
    records = []
    for i in range(n):
        rec = {"step": i, "action": "a", "grid_before": [[i % 16]], "grid_after": [[(i + 1) % 16]]}
        if level_row_idx is not None and i == level_row_idx:
            rec["levels_completed_before"] = 0
            rec["levels_completed_after"] = 1
        records.append(rec)
    return records


def make_fake_llm(responses):
    """
    A stateful fake `llm_call(prompt_text) -> {"text", "finish_reason",
    "completion_tokens"}` that returns pre-baked GameModel class code in a
    fixed order, one per call, ignoring prompt content entirely -- valid
    because next_chunk_boundary's boundary sequence is fully deterministic
    and independent of what any round's candidate contains, so the whole
    chunk/round call ORDER can be planned in advance (see test comments).
    Raises if called more times than `responses` has entries.
    """
    state = {"i": 0}
    def _call(prompt_text):
        if state["i"] >= len(responses):
            raise AssertionError(f"fake LLM called more times than expected ({len(responses)})")
        code = responses[state["i"]]
        state["i"] += 1
        return {"text": code, "finish_reason": "stop", "completion_tokens": len(code) // 4}
    _call.n_calls = lambda: state["i"]
    return _call


def make_args(trace_path, workdir, max_examples=20, max_rounds=2, automatic=True, k=10, log="chunk_log.jsonl"):
    return argparse.Namespace(
        trace=str(trace_path), workdir=str(workdir),
        backend="llama-cpp", model_path=None, n_ctx=32768, n_gpu_layers=-1, verbose_llama=False,
        api_base="http://x", model=None, temperature=0.2, repeat_penalty=1.3,
        presence_penalty=0.1, frequency_penalty=0.1, max_tokens=4096,
        max_examples=max_examples, compact=False, max_rounds=max_rounds, k=k, log=log,
        automatic=automatic, cpu_seconds=10, max_procs=16, mem_mb=512,
        per_call_seconds=2, overall_timeout=60,
    )


def write_trace(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ================================================================
# Scenario 1: design doc's own worked example -- max_examples=20, a level
# completes at (0-indexed) row 44, trace length 80 -> boundaries
# 20, 40, 45, 65, 80, with is_level_boundary True ONLY for the chunk
# 40->45 call, meaning chunk 4 (45:65) -- not chunk 3 (40:45) -- is the one
# whose EXTEND_TEMPLATE should carry the level-boundary note.
# ================================================================
tmpdir = tempfile.mkdtemp(prefix="nosumina_step10_test_")
trace_path = os.path.join(tmpdir, "trace.jsonl")
records = make_trace(80, level_row_idx=44)
write_trace(records, trace_path)

# Hand-planned candidate sequence: 8 LLM calls total across 5 chunks with
# --max-rounds 2 (see accompanying comments for the accuracy arithmetic).
responses = [
    make_class_code(range(18)),               # 1. chunk1 r1: 18/20 = 90%  (accept, no baseline)
    make_class_code(range(20)),                # 2. chunk1 r2: 20/20 = 100% (zero-failure early stop)
    make_class_code(range(32)),                # 3. chunk2 r1: 32/40 = 80%  (accept vs 50% baseline)
    make_class_code(range(26)),                # 4. chunk2 r2: 26/40 = 65%  (reject, threshold 75%)
    make_class_code(range(44), goal_row=44),    # 5. chunk3 r1: 45/45 = 100% (zero-failure early stop; row 44 IS the level-completion row)
    make_class_code(range(60), goal_row=44),    # 6. chunk4 r1: 60/65 ~92.3% (accept vs ~69.2% baseline)
    make_class_code(range(65), goal_row=44),    # 7. chunk4 r2: 65/65 = 100% (zero-failure early stop)
    make_class_code(range(80), goal_row=44),    # 8. chunk5 r1: 80/80 = 100% (zero-failure early stop)
]
fake_llm = make_fake_llm(responses)
tt.build_llm_caller = lambda args: (fake_llm, None)

workdir = Path(tmpdir) / "run1"
args = make_args(trace_path, workdir, max_examples=20, max_rounds=2, automatic=True)

buf = io.StringIO()
with redirect_stdout(buf):
    tt.cmd_run_chunked(args)
output = buf.getvalue()
print(output)

assert fake_llm.n_calls() == 8, f"expected exactly 8 LLM calls, got {fake_llm.n_calls()}"
print("Test 1a (exactly 8 LLM calls across 5 chunks, matching the hand-planned round counts): PASS")

expected_prompt_files = [
    "chunk1_prompt_round1.txt", "chunk1_prompt_round2.txt",
    "chunk2_prompt_round1.txt", "chunk2_prompt_round2.txt",
    "chunk3_prompt_round1.txt",
    "chunk4_prompt_round1.txt", "chunk4_prompt_round2.txt",
    "chunk5_prompt_round1.txt",
]
for fname in expected_prompt_files:
    assert (workdir / fname).exists(), f"missing expected prompt file {fname}"
assert not (workdir / "chunk2_prompt_round3.txt").exists()
assert not (workdir / "chunk3_prompt_round2.txt").exists()  # chunk 3 early-stopped at round 1
print("Test 1b (correct prompt files per chunk, matching planned round counts incl. early stops): PASS")

# Level-boundary attribution: chunk 3 (the chunk THAT ENDS on the level
# cutoff, 40->45) must NOT show the note; chunk 4 (the chunk that STARTS
# right after it, 45->65) must.
chunk3_prompt = (workdir / "chunk3_prompt_round1.txt").read_text()
chunk4_prompt = (workdir / "chunk4_prompt_round1.txt").read_text()
assert "This chunk starts a NEW LEVEL." not in chunk3_prompt
assert "This chunk starts a NEW LEVEL." in chunk4_prompt
print("Test 1c (level-boundary DESCRIPTION note attributed to chunk 4, not chunk 3 -- the one-chunk lag): PASS")

# Chunk 2/4/5 round 1 uses EXTEND_TEMPLATE language, never REVISE_TEMPLATE's
# counterexample framing; chunk 1 round 1 uses neither.
chunk1_prompt = (workdir / "chunk1_prompt_round1.txt").read_text()
assert "revise your gamemodel class" not in chunk1_prompt.lower()
assert "### counterexample" not in chunk1_prompt.lower()
for fname in ("chunk2_prompt_round1.txt", "chunk4_prompt_round1.txt", "chunk5_prompt_round1.txt"):
    text = (workdir / fname).read_text()
    assert "extend" in text.lower(), f"{fname} doesn't look like an EXTEND_TEMPLATE prompt"
    assert "### counterexample" not in text.lower(), f"{fname} incorrectly looks like a REVISE_TEMPLATE prompt"
print("Test 1d (chunk 1 uses the seed prompt; chunks 2/4/5 round 1 use EXTEND_TEMPLATE): PASS")

for fname in ("chunk1_prompt_round2.txt", "chunk2_prompt_round2.txt", "chunk4_prompt_round2.txt"):
    text = (workdir / fname).read_text()
    assert "### counterexample" in text.lower(), f"{fname} doesn't look like a REVISE_TEMPLATE prompt"
print("Test 1e (round 2 of chunks 1/2/4 uses REVISE_TEMPLATE): PASS")

assert "best chunk-ending accuracy seen during the run: 100.0%" in output
assert "final candidate's accuracy: 100.0%" in output
assert "NOTE: final candidate is" not in output  # run ends on its own peak
assert "[chunk 2] SUBSTITUTION at round 2:" in output  # chunk2's round2 (65%) was rejected
assert "[chunk 3] EARLY STOP at round 1:" in output
assert "[chunk 4] EARLY STOP at round 2:" in output
assert "[chunk 5] EARLY STOP at round 1:" in output
print("Test 1f (end-of-run report is legible and matches the planned scenario exactly): PASS")

for n in range(1, 6):
    assert (workdir / f"chunk{n}_final.py").exists()
counts = json.loads((workdir / "row_failure_counts.json").read_text())
best_counts = json.loads((workdir / "row_failure_counts_best.json").read_text())
assert counts == best_counts
print("Test 1g (chunk{N}_final.py persisted for every chunk; counts files in lockstep): PASS")

log_lines = (workdir / "chunk_log.jsonl").read_text().strip().splitlines()
assert len(log_lines) == 5
chunk_reports = [json.loads(l) for l in log_lines]
assert [r["chunk"] for r in chunk_reports] == [1, 2, 3, 4, 5]
assert [tuple(r["row_range"]) for r in chunk_reports] == [(0, 20), (20, 40), (40, 45), (45, 65), (65, 80)]
assert abs(chunk_reports[1]["final_accuracy"] - 0.80) < 1e-9  # chunk 2 ends at round1's 80% (round2 rejected)
print("Test 1h (chunk_log.jsonl has one correctly-shaped entry per chunk, including the short level-cutoff chunk): PASS")


# ================================================================
# Scenario 2: --max-rounds 1 -- confirm every chunk stops after round 1
# regardless of whether it was clean, and NO revision prompt is ever built.
# ================================================================
responses2 = [
    make_class_code(range(18)),               # chunk1 r1: 90%
    make_class_code(range(32)),                # chunk2 r1: 32/40 = 80% vs 45%(18/40) baseline -> accept
    make_class_code(range(44), goal_row=44),    # chunk3 r1: 100%
    make_class_code(range(60), goal_row=44),    # chunk4 r1: 60/65 ~92.3% vs baseline
    make_class_code(range(80), goal_row=44),    # chunk5 r1: 100%
]
fake_llm2 = make_fake_llm(responses2)
tt.build_llm_caller = lambda args: (fake_llm2, None)
workdir2 = Path(tmpdir) / "run2"
args2 = make_args(trace_path, workdir2, max_examples=20, max_rounds=1, automatic=True)

buf2 = io.StringIO()
with redirect_stdout(buf2):
    tt.cmd_run_chunked(args2)

assert fake_llm2.n_calls() == 5, f"expected exactly 5 LLM calls (1 per chunk) with --max-rounds 1, got {fake_llm2.n_calls()}"
for n in range(1, 6):
    assert (workdir2 / f"chunk{n}_prompt_round1.txt").exists()
    assert not (workdir2 / f"chunk{n}_prompt_round2.txt").exists(), \
        f"chunk {n} should never get a round 2 with --max-rounds 1"
print("Test 2a (--max-rounds 1: exactly 1 round per chunk, no revision prompt ever built, even for chunks that were clean): PASS")

log_lines2 = (workdir2 / "chunk_log.jsonl").read_text().strip().splitlines()
chunk_reports2 = [json.loads(l) for l in log_lines2]
# chunk 4 with max_rounds=1 ends at round1's ~92.3%, NOT the 100% it reached
# with a round 2 in Scenario 1 -- confirms max_rounds genuinely caps rounds.
assert abs(chunk_reports2[3]["final_accuracy"] - (60 / 65)) < 1e-9
print("Test 2b (chunk 4 ends at round-1-only accuracy ~92.3%, not the 100% reachable with a round 2): PASS")


# ================================================================
# Scenario 3: pause_for_confirmation IS called once per round when
# --automatic is NOT passed, and is NOT called when it is.
# ================================================================
small_trace_path = os.path.join(tmpdir, "small_trace.jsonl")
write_trace(make_trace(10), small_trace_path)

pause_calls = []
orig_pause = tt.pause_for_confirmation
tt.pause_for_confirmation = lambda label, path: pause_calls.append(label)
fake_llm3 = make_fake_llm([make_class_code(range(10))])  # zero-failure at round 1 -> only 1 round/pause expected
tt.build_llm_caller = lambda args: (fake_llm3, None)
workdir3 = Path(tmpdir) / "run3"
args3 = make_args(small_trace_path, workdir3, max_examples=10, max_rounds=2, automatic=False)
buf3 = io.StringIO()
with redirect_stdout(buf3):
    tt.cmd_run_chunked(args3)
assert len(pause_calls) == 1, f"expected exactly 1 pause (round 1, zero-failure early stop skips round 2), got {len(pause_calls)}"
tt.pause_for_confirmation = orig_pause
print("Test 3a (pause_for_confirmation called once per round when --automatic is absent): PASS")

pause_calls2 = []
tt.pause_for_confirmation = lambda label, path: pause_calls2.append(label)
fake_llm3b = make_fake_llm([make_class_code(range(10))])
tt.build_llm_caller = lambda args: (fake_llm3b, None)
workdir3b = Path(tmpdir) / "run3b"
args3b = make_args(small_trace_path, workdir3b, max_examples=10, max_rounds=2, automatic=True)
buf3b = io.StringIO()
with redirect_stdout(buf3b):
    tt.cmd_run_chunked(args3b)
assert len(pause_calls2) == 0, "no pauses expected with --automatic"
tt.pause_for_confirmation = orig_pause
print("Test 3b (no pauses at all with --automatic): PASS")


# ================================================================
# Scenario 4: malformed LLM output (disallowed import) doesn't crash the
# whole run -- gets substituted with the safe fallback stub instead.
# ================================================================
bad_response = ("import os\nclass GameModel:\n"
                 "    def predict(self, grid_before, action, previous_state):\n"
                 "        return grid_before, False, previous_state\n")
fake_llm4 = make_fake_llm([bad_response])
tt.build_llm_caller = lambda args: (fake_llm4, None)
workdir4 = Path(tmpdir) / "run4"
args4 = make_args(small_trace_path, workdir4, max_examples=10, max_rounds=1, automatic=True)
buf4 = io.StringIO()
with redirect_stdout(buf4):
    tt.cmd_run_chunked(args4)  # must not raise
output4 = buf4.getvalue()
assert "substituting a safe no-op stub" in output4
candidate_text = (workdir4 / "chunk1_candidate_round1.py").read_text()
assert "Safe no-op stub" in candidate_text
print("Test 4 (disallowed import in LLM output substituted with safe fallback stub, run doesn't crash): PASS")

tt.run_candidate = _orig_run_candidate
shutil.rmtree(tmpdir, ignore_errors=True)
print("\nALL STEP 10 TESTS PASSED")