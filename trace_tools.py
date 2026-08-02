#!/usr/bin/env python3
"""
trace_tools.py — Phase 4 prep for Eidolon's predict_next_state validation.

Pipeline:
  1. inspect        Look at trace.jsonl structure without dumping raw grid values.
                     Specifically checks whether the frame field is a single grid
                     per record or an accumulating list-of-frames.
  2. preprocess     Collapse each record down to {step, action, grid} — one
                     current grid per observation, dropping any redundant
                     history the raw FrameData carries.
  3. split          Chronological 70/30 split of the cleaned trace into
                     history (shown to the model) vs. held-out (ground truth
                     to check predictions against).
  4. prompt         Build the round-1 synthesis prompt from a spread-out
                     subsample of history, using a compact grid encoding to
                     keep token count down. Two grid encodings are available:
                     one-hex-char-per-cell (default) or --compact (run-length
                     encoded rows, e.g. "0*7 3*2"), which cuts tokens further
                     for large/sparse grids. NOTE: grid content tokenizes far
                     worse than English prose (close to 1 token/char, not the
                     usual ~4 chars/token), so the printed token estimate for
                     `prompt`/`revise-prompt` uses a 1:1 chars-to-tokens
                     fallback rather than chars/4 — it's still a rough
                     estimate since no real tokenizer is loaded on that path.
  5. score          Sandboxed test of a candidate predict_next_state.py against
                     any cleaned dataset (history or held-out). No LLM calls —
                     this is plain Python execution, so it's fast and has no
                     context-window limit no matter how much history you have.
  6. evaluate       The stop condition for the verify-and-revise (CEGIS) loop:
                     scores the candidate against held-out only and logs the
                     round. Exit codes drive an orchestrating shell loop:
                       0 = STOP, held-out accuracy met --threshold
                       1 = CONTINUE, threshold not yet met, rounds remain —
                           writes fresh counterexamples from HISTORY (never
                           held-out) to --counterexamples-out for the next
                           revision prompt, if --history is also given
                       2 = STOP, --max-rounds reached without meeting threshold
  7. revise-prompt  Build a round-2+ prompt from a candidate's current code
                     plus the counterexamples `evaluate` just wrote — shows
                     the model what it predicted vs. what actually happened,
                     and asks for a fix rather than a rewrite from scratch.
  8. run-loop       Fully automated version of prompt -> LLM -> evaluate ->
                     revise-prompt -> LLM -> ..., calling the model directly
                     each round instead of manual copy-paste. Supports two
                     backends: `llama-cpp` (llama-cpp-python loading a local
                     GGUF in-process — the default, and what you'd use for a
                     quantized model on Jarvislabs) or `openai` (an
                     OpenAI-compatible HTTP server, e.g. vLLM or llama-server,
                     if you go that route instead). Also supports --compact
                     (see `prompt` above). Before each round's LLM call, if a
                     local tokenizer is available (--backend llama-cpp), the
                     actual prompt is tokenized via the loaded model and
                     checked against --n-ctx minus --max-tokens, failing fast
                     with a clear message instead of letting llama-cpp raise
                     a raw "Requested tokens exceed context window" error
                     mid-generation. --backend openai has no local tokenizer,
                     so that preflight check is skipped and only a char count
                     is printed. --verbose-llama surfaces llama.cpp's own
                     load-time log (actual GPU-offloaded layer count — GPU
                     memory usage alone doesn't confirm this) and per-call
                     prompt-eval/generation tokens-per-second, useful for
                     diagnosing an unexpectedly slow round.

Usage:
  python trace_tools.py inspect trace.jsonl --frame-key frame
  python trace_tools.py preprocess trace.jsonl clean.jsonl --frame-key frame --action-key action
  python trace_tools.py split clean.jsonl history.jsonl heldout.jsonl --history-frac 0.7
  python trace_tools.py prompt history.jsonl prompt.txt --max-examples 25
  python trace_tools.py score candidate.py history.jsonl --out results.jsonl
  python trace_tools.py evaluate candidate.py heldout.jsonl --round 1 --threshold 0.95 \\
      --max-rounds 10 --log rounds.jsonl --history history.jsonl \\
      --counterexamples-out next_counterexamples.jsonl
  python trace_tools.py revise-prompt candidate.py next_counterexamples.jsonl revision_prompt.txt
  python trace_tools.py run-loop history.jsonl heldout.jsonl \\
      --backend llama-cpp --model-path /path/to/model.Q4_K_M.gguf \\
      --n-gpu-layers -1 --n-ctx 32768 --threshold 0.95 --max-rounds 10 \\
      --compact --repeat-penalty 1.3 --frequency-penalty 0.1 --presence-penalty 0.1
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import shutil
import signal
from datetime import datetime, timezone
from pathlib import Path


# ---------- shared helpers ----------

def get_nested(obj, dotted_key):
    """Fetch obj['a']['b'] via dotted_key='a.b'. Returns None if missing."""
    cur = obj
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def extract_current_grid(frames):
    """
    frames may be:
      - a single grid: List[rows][cols]              -> return as-is
      - a list of grids: List[frames][rows][cols]     -> return last element
    Disambiguate by checking nesting depth of the first element.
    """
    if not isinstance(frames, list) or not frames:
        return frames
    first = frames[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        return frames[-1]  # list-of-grids -> take current (last) one
    return frames  # already a single grid


def grid_to_compact(grid):
    """
    Render a grid as one row per line, one character per cell.
    ARC-AGI-3 grids use color indices beyond single digits (values up to at
    least 14 seen in real traces) — plain str(c) concatenation is ambiguous
    for any cell >= 10 (e.g. row [3, 11, 3] -> "3113", indistinguishable from
    [3, 1, 1, 3]). Cells are mapped to single hex characters (0-9, a-f)
    instead, keeping the one-char-per-cell density but staying unambiguous
    for values up to 15. Raises loudly rather than silently mis-encoding if
    a value outside that range ever shows up.
    """
    HEX_DIGITS = "0123456789abcdef"

    def cell_char(c):
        if not (0 <= c < len(HEX_DIGITS)):
            raise ValueError(
                f"grid cell value {c} is outside the expected 0-15 palette. "
                f"grid_to_compact's single-char encoding needs to be widened "
                f"(e.g. space-separated ints) if the game uses more colors."
            )
        return HEX_DIGITS[c]

    return "\n".join("".join(cell_char(c) for c in row) for row in grid)


def grid_to_rle(grid):
    """
    Render a grid as one row per line, run-length encoded: each row becomes
    space-separated "<hexchar>*<count>" runs (e.g. row [0,0,0,0,0,0,0,3,3] ->
    "0*7 3*2"). Same 0-15 hex-digit cell mapping as grid_to_compact. This is
    strictly a prompt-side display shorthand to cut tokens for large/sparse
    grids with long runs of a repeated value (e.g. background) — the model
    never has to decode it, since predict_next_state always takes/returns
    plain Python int grids regardless of how they were shown in the prompt.
    """
    HEX_DIGITS = "0123456789abcdef"

    def cell_char(c):
        if not (0 <= c < len(HEX_DIGITS)):
            raise ValueError(
                f"grid cell value {c} is outside the expected 0-15 palette. "
                f"grid_to_rle's single-char encoding needs to be widened "
                f"(e.g. space-separated ints) if the game uses more colors."
            )
        return HEX_DIGITS[c]

    lines = []
    for row in grid:
        runs = []
        i = 0
        while i < len(row):
            j = i
            while j < len(row) and row[j] == row[i]:
                j += 1
            runs.append(f"{cell_char(row[i])}*{j - i}")
            i = j
        lines.append(" ".join(runs))
    return "\n".join(lines)


def encode_grid(grid, encoding="hex"):
    """Dispatch to the requested grid display encoding ('hex' or 'rle')."""
    if encoding == "rle":
        return grid_to_rle(grid)
    return grid_to_compact(grid)


# Above roughly this many changed cells, listing every one individually stops
# helping and starts costing tokens for little benefit (a genuinely global
# transformation — e.g. every cell shifts — produces a diff as big as the
# grid itself). Past that point we just report the count and let the model
# fall back to the full before/after grids shown alongside the diff.
DIFF_MAX_CELLS = 40


def diff_grid(grid_before, grid_after):
    """
    Returns a list of (row, col, before_val, after_val) for every cell that
    changed between grid_before and grid_after. This exists so the model
    doesn't have to manually eyeball-compare two long RLE/hex-encoded rows
    itself to figure out what changed — that comparison is exactly the kind
    of mechanical, error-prone-by-eye work an LLM burns huge amounts of
    reasoning tokens on (and often still gets wrong), while a few lines of
    Python do it perfectly and instantly. Values are real Python ints, same
    as the grid itself — no display-encoding involved.
    """
    changes = []
    n_rows = min(len(grid_before), len(grid_after))
    for r in range(n_rows):
        row_before, row_after = grid_before[r], grid_after[r]
        n_cols = min(len(row_before), len(row_after))
        for c in range(n_cols):
            if row_before[c] != row_after[c]:
                changes.append((r, c, row_before[c], row_after[c]))
    return changes


def format_diff(changes, max_cells=DIFF_MAX_CELLS):
    """Render a diff_grid() result as a compact 'row,col: before -> after' list."""
    if not changes:
        return "no cells changed (identity transformation for this example)"
    if len(changes) > max_cells:
        return (f"{len(changes)} cells changed — too many to list individually here; "
                f"compare the full before/after grids above instead")
    return "\n".join(f"(row {r}, col {c}): {b} -> {a}" for r, c, b, a in changes)


def build_bwrap_command(python_exe, runner_path_in_sandbox, staging_dir):
    """
    Wrap the runner invocation in bubblewrap: no network, no filesystem
    access outside the staging dir, minimal binds just to boot the
    interpreter. This is the kernel-enforced layer — it doesn't trust
    check_ast_imports at all, since that check can be bypassed via
    __import__() or bare open() (both builtins, no `import` statement).
    """
    # run_candidate execs this command with an env that's deliberately wiped
    # of secrets (just a couple of BLAS thread-count pins added on top — see
    # run_candidate) — but that also strips PATH for the bwrap invocation
    # itself, so a bare "bwrap" would only resolve against the POSIX
    # fallback path (~/bin:/usr/bin), not wherever it's actually installed
    # (e.g. a conda env's bin/, common when installed via
    # `conda install -c conda-forge bubblewrap`). Resolve the absolute path
    # now, while the *current* process's real PATH is still intact, so no
    # PATH lookup is needed at exec time.
    bwrap_path = shutil.which("bwrap")
    if bwrap_path is None:
        raise RuntimeError(
            "bwrap not found on PATH. Install it, e.g. `sudo apt-get install bubblewrap` "
            "or `conda install -c conda-forge bubblewrap`, then make sure it's on PATH "
            "in the environment you're running trace_tools.py from."
        )

    # sys.executable lives inside the conda env; bind that tree at its
    # real path so the interpreter resolves its own stdlib/site-packages
    # normally, with no path remapping needed for Python itself.
    conda_prefix = os.path.dirname(os.path.dirname(python_exe))  # .../envs/eidolon

    cmd = [
        bwrap_path,
        "--unshare-all", "--unshare-net",
        "--die-with-parent", "--new-session",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
    ]
    if os.path.exists("/lib64"):
        cmd += ["--ro-bind", "/lib64", "/lib64"]
    cmd += [
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", conda_prefix, conda_prefix,
        "--ro-bind", staging_dir, "/work",
        "--chdir", "/work",
        "--tmpfs", "/tmp",
        "--proc", "/proc",
        "--dev", "/dev",
        "--cap-drop", "ALL",
        "--hostname", "sandboxed",
        "--",
        python_exe, runner_path_in_sandbox,
    ]
    return cmd

# ---------- describe (recursive shape printer, no raw leaf dumps for big arrays) ----------

def describe(obj, path="root", max_depth=6, depth=0):
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{path}: <max depth reached>")
        return
    if isinstance(obj, dict):
        print(f"{indent}{path}: dict, keys={list(obj.keys())}")
        for k, v in obj.items():
            describe(v, f"{path}.{k}", max_depth, depth + 1)
    elif isinstance(obj, list):
        n = len(obj)
        if n == 0:
            print(f"{indent}{path}: empty list")
            return
        print(f"{indent}{path}: list[{n}] of {type(obj[0]).__name__}")
        describe(obj[0], f"{path}[0]", max_depth, depth + 1)
    else:
        print(f"{indent}{path}: {type(obj).__name__}")


# ---------- sandboxed candidate execution ----------
# A candidate predict_next_state.py is untrusted LLM output. We isolate it in
# a subprocess with CPU/memory/process-count limits, a wiped environment, and
# an AST-level import whitelist (subprocess isolation + resource.setrlimit +
# AST import checking + env wipe). This is a defensive guard against a
# buggy/runaway candidate (infinite loop, huge allocation, fork bomb), not a
# hardened sandbox against a sophisticated adversary.
#
# Deliberately NOT included: non-root user switching, --cap-drop, --read-only
# root filesystem, --tmpfs scratch dirs, container-level --memory/--pids-limit.
# Those need Docker/root and aren't available in a Kaggle notebook kernel
# either — this harness is meant to run identically there, so it's scoped to
# exactly what's achievable from plain Python without a container boundary:
# subprocess + resource.setrlimit + env wiping + a timeout. If this ever runs
# somewhere with real container control, those would be worth adding on top.

ALLOWED_IMPORTS = {"copy", "itertools", "math", "collections", "functools", "numpy"}


def check_ast_imports(source):
    """Raise ValueError if the candidate imports anything outside the whitelist."""
    import ast
    tree = ast.parse(source)
    bad = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    bad.add(root)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                bad.add(root)
    if bad:
        raise ValueError(f"candidate imports disallowed module(s): {sorted(bad)}. "
                          f"Allowed: {sorted(ALLOWED_IMPORTS)}")


RUNNER_TEMPLATE = """
import sys, json, resource, signal

resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
resource.setrlimit(resource.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))
try:
    # Caps total processes/threads for this real UID — this stops a fork bomb
    # or runaway thread spawn from a candidate. Note it's scoped per-UID, not
    # per-process-tree, so it's a shared budget with whatever else is running
    # as the same user; set generously enough not to starve the orchestrator.
    resource.setrlimit(resource.RLIMIT_NPROC, ({max_procs}, {max_procs}))
except (AttributeError, ValueError):
    pass  # not available on this platform; CPU/memory limits above still apply

def _timeout_handler(signum, frame):
    raise TimeoutError("candidate exceeded per-call time budget")
signal.signal(signal.SIGALRM, _timeout_handler)

candidate_globals = {{"__name__": "candidate", "__builtins__": __builtins__}}
with open({candidate_path!r}) as f:
    source = f.read()
exec(compile(source, {candidate_path!r}, "exec"), candidate_globals)

GameModelClass = candidate_globals.get({class_name!r})
if GameModelClass is None:
    print(json.dumps({{"error": {class_name!r} + " not defined in candidate"}}))
    sys.exit(1)

try:
    model = GameModelClass()
except Exception as e:
    print(json.dumps({{"error": f"failed to instantiate {class_name!r}: {{type(e).__name__}}: {{e}}"}}))
    sys.exit(1)

# `state` starts as an empty dict for the first row of this sequential pass —
# there is no prior call to have produced one yet. From then on it is always
# exactly what the candidate's own predict() returned on the previous row;
# the harness never reconstructs or corrects it. grid_before, in contrast, is
# always the true grid from the trace for every row (teacher-forced at the
# grid level, per design doc §1) — a wrong prediction never contaminates the
# next row's grid input, only `state` can drift.
state = {{}}

# Feeding the whole row range for this call into stdin upfront (see
# run_candidate) does NOT give any row's prediction access to a later row's
# true grid — each loop iteration only ever reads THIS row's grid_before/
# action off stdin, so a given row's prediction depends only on rows already
# processed earlier in this same sequential pass. That's enforced by the
# loop's order below, not by anything about how much data physically arrived
# in the input buffer.
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    signal.alarm({per_call_seconds})
    try:
        predicted_grid_after, goal, state = model.predict(rec["grid_before"], rec["action"], state)
    except Exception as e:
        # Abort-on-crash (design doc §1): don't try to carry the last-good
        # state forward and keep going — emit this row as an error and stop
        # the whole sequential pass right here. Every row after this one is
        # simply never emitted; the parent process's scoring treats "no
        # result for this step" as incorrect/missing.
        signal.alarm(0)
        print(json.dumps({{"step": rec["step"], "error": f"{{type(e).__name__}}: {{e}}"}}))
        break
    signal.alarm(0)
    print(json.dumps({{"step": rec["step"], "prediction": predicted_grid_after, "goal": bool(goal)}}))
"""


def run_candidate(candidate_path, records, cpu_seconds=10, mem_mb=512,
                   per_call_seconds=2, overall_timeout=60, max_procs=16,
                   use_sandbox=True, class_name="GameModel"):
    """
    Run a candidate's `class GameModel: def predict(self, grid_before, action,
    previous_state) -> (predicted_grid_after, goal, state)` from
    candidate_path, in a subprocess with CPU/memory/process-count limits and
    a per-row alarm timeout (so one pathological row can't hang the whole
    batch). The subprocess's environment is wiped (no inherited API keys,
    tokens, etc. — only a couple of BLAS thread-count pins, see below) since
    the candidate has no legitimate need for any of it.

    This is one sequential, stateful rollout over `records` in trace order:
    the candidate's class is instantiated once, and `state` — whatever the
    candidate's own predict() returned — is threaded from each row into the
    next row's call. `grid_before` fed into each call always comes straight
    from `records`, never from a prior prediction, so a wrong grid
    prediction never compounds forward; only candidate-tracked `state` can
    drift on its own. If a row's predict() call raises (including a
    SIGALRM-driven timeout), the whole rollout aborts at that point — no
    attempt is made to keep going with stale state (design doc §1) — so
    every row from the failure point onward is simply absent from the
    returned dict, and the caller's own scoring is responsible for treating
    a missing step as incorrect.

    Returns {step: {"prediction": grid, "goal": bool} or {"error": str}}.
    """
    candidate_path = str(candidate_path)
    source = Path(candidate_path).read_text()
    check_ast_imports(source)

    if use_sandbox:
        staging_dir = tempfile.mkdtemp(prefix="eidolon_sandbox_")
    else:
        staging_dir = None

    try:
        if use_sandbox:
            # Copy just the candidate into the disposable staging dir —
            # not a bind of wherever it originally lived (loop_run/, or
            # anywhere else `score`/`evaluate` were pointed at).
            staged_candidate = os.path.join(staging_dir, "candidate.py")
            shutil.copy(candidate_path, staged_candidate)
            runner_candidate_path = "/work/candidate.py"  # path as seen INSIDE bwrap
        else:
            runner_candidate_path = candidate_path

        runner_code = RUNNER_TEMPLATE.format(
            cpu=cpu_seconds, mem_bytes=mem_mb * 1024 * 1024,
            candidate_path=runner_candidate_path,
            per_call_seconds=per_call_seconds, max_procs=max_procs,
            class_name=class_name,
        )

        if use_sandbox:
            runner_path = os.path.join(staging_dir, "runner.py")
            Path(runner_path).write_text(runner_code)
            exec_cmd = build_bwrap_command(sys.executable, "/work/runner.py", staging_dir)
        else:
            fd, runner_path = tempfile.mkstemp(suffix="_runner.py")
            with os.fdopen(fd, "w") as f:
                f.write(runner_code)
            exec_cmd = [sys.executable, runner_path]

        # The entire row range for this call is serialized to stdin upfront,
        # all at once, below. This does NOT give an earlier row's prediction
        # access to a later row's true grid — the runner (RUNNER_TEMPLATE)
        # only ever reads one row's grid_before/action off stdin per loop
        # iteration, so a given row's prediction depends only on rows it has
        # already processed in that same sequential pass. That guarantee
        # comes from the runner's loop order, not from anything about how
        # much data physically arrived in the stdin buffer — worth pinning
        # down explicitly here since it was a point of confusion earlier.
        stdin_data = "\n".join(
            json.dumps({"step": r["step"], "action": r["action"], "grid_before": r["grid_before"]})
            for r in records
        )
        # env is otherwise still fully wiped (no inherited API keys/tokens —
        # the candidate has no legitimate need for any of it) EXCEPT for
        # these thread-count pins. Without them, numpy's BLAS backend
        # (OpenBLAS on this box) auto-detects the host's core count on
        # import and pre-reserves a memory buffer per thread before the
        # candidate's code runs at all — on a many-core GPU box that alone
        # can exceed the sandbox's RLIMIT_AS cap (default 512MB), producing
        # "OpenBLAS error: Memory allocation still failed" even for a
        # trivial candidate that never does real linear algebra. Pinning to
        # 1 thread makes memory use scale with what the candidate actually
        # does, not with host core count. Covers the three common backends
        # (OpenBLAS, a generic OpenMP-linked BLAS, MKL) plus numexpr, which
        # some numpy builds pull in transitively.
        sandbox_env = {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        try:
            proc = subprocess.run(
                exec_cmd, input=stdin_data, capture_output=True, text=True,
                timeout=overall_timeout, env=sandbox_env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"candidate exceeded overall timeout of {overall_timeout}s across the whole batch")
    finally:
        if use_sandbox and staging_dir:
            shutil.rmtree(staging_dir, ignore_errors=True)
        elif not use_sandbox:
            os.unlink(runner_path)

    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(f"candidate runner failed to start: {proc.stderr.strip()[-2000:]}")

    results = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "step" in r:
            results[r["step"]] = r
    return results


def score_candidate(candidate_path, records, **run_kwargs):
    """
    Run candidate against records and compare predictions to grid_after.
    Returns (scored_records, accuracy) where each scored record gets a
    "passed" bool, and failures additionally get the wrong "prediction".
    """
    raw = run_candidate(candidate_path, records, **run_kwargs)
    scored = []
    n_pass = 0
    for rec in records:
        r = raw.get(rec["step"])
        if r is None:
            scored.append({**rec, "passed": False, "error": "no result (crashed or timed out earlier in the batch)"})
        elif "error" in r:
            scored.append({**rec, "passed": False, "error": r["error"]})
        else:
            passed = r["prediction"] == rec["grid_after"]
            entry = {**rec, "passed": passed}
            if not passed:
                entry["prediction"] = r["prediction"]
            scored.append(entry)
            if passed:
                n_pass += 1
    accuracy = n_pass / len(records) if records else 0.0
    return scored, accuracy


def summarize_scores(scored):
    """
    Richer diagnostics beyond the strict exact-whole-grid-match pass rate:
      - mean_cell_accuracy: average fraction of individual cells correct
        across all records (0.0 for crashed/errored records — no predicted
        grid exists to compare). Exact-match is all-or-nothing per record —
        a candidate that gets every cell right except one still scores an
        identical 0% to a candidate that's completely wrong, so this can't
        tell "close but not quite" apart from "way off". This metric can —
        BUT in a sparse domain (most cells never change between grid_before
        and grid_after), this number is dominated by the trivially-easy
        static majority and can look deceptively high even for a candidate
        that gets every DYNAMIC cell wrong (an identity no-op scores nearly
        as well on this metric as a genuinely correct rule would). Read
        mean_changed_cell_accuracy instead when that's the situation.
      - mean_changed_cell_accuracy: same idea, but restricted to only the
        cells that actually changed in the ground truth (grid_before ->
        grid_after) — ignoring the easy static majority entirely. This is
        the number that reflects whether the candidate is learning the real
        dynamics, not just correctly leaving the background alone. None if
        every record in the batch was a no-op transition (nothing to score).
      - by_action: per-action n/n_pass/accuracy, so a rule that correctly
        nails one action but does nothing for the others doesn't just
        collapse into one flat, confusing overall percentage.
    """
    total = len(scored)
    n_pass = sum(1 for s in scored if s["passed"])
    cell_accuracies = []
    changed_cell_accuracies = []
    by_action = {}
    for s in scored:
        bucket = by_action.setdefault(s.get("action"), {"n": 0, "n_pass": 0})
        bucket["n"] += 1

        actual_changes = diff_grid(s["grid_before"], s["grid_after"])
        pred = s.get("prediction") if not s["passed"] else s["grid_after"]

        if s["passed"]:
            bucket["n_pass"] += 1
            cell_accuracies.append(1.0)
            if actual_changes:
                changed_cell_accuracies.append(1.0)
            continue

        if pred is None:
            cell_accuracies.append(0.0)  # crashed/errored — no usable prediction to compare
            if actual_changes:
                changed_cell_accuracies.append(0.0)
            continue

        grid_after = s["grid_after"]
        total_cells = sum(len(row) for row in grid_after)
        wrong_cells = len(diff_grid(pred, grid_after))
        cell_accuracies.append(1 - wrong_cells / total_cells if total_cells else 1.0)

        if actual_changes:
            n_correct = sum(
                1 for (r, c, _, after_val) in actual_changes
                if r < len(pred) and c < len(pred[r]) and pred[r][c] == after_val
            )
            changed_cell_accuracies.append(n_correct / len(actual_changes))

    for bucket in by_action.values():
        bucket["accuracy"] = bucket["n_pass"] / bucket["n"] if bucket["n"] else 0.0
    return {
        "total": total,
        "n_pass": n_pass,
        "accuracy": n_pass / total if total else 0.0,
        "mean_cell_accuracy": sum(cell_accuracies) / len(cell_accuracies) if cell_accuracies else 0.0,
        "mean_changed_cell_accuracy": (
            sum(changed_cell_accuracies) / len(changed_cell_accuracies)
            if changed_cell_accuracies else None
        ),
        "n_dynamic_records": len(changed_cell_accuracies),
        "by_action": by_action,
    }


def print_score_summary(summary, label="held-out"):
    print(f"{label} exact-match accuracy: {summary['accuracy']:.1%} "
          f"({summary['n_pass']}/{summary['total']})")
    if summary["mean_changed_cell_accuracy"] is not None:
        print(f"{label} accuracy on cells that ACTUALLY changed: "
              f"{summary['mean_changed_cell_accuracy']:.1%} "
              f"(n={summary['n_dynamic_records']} non-identity transitions — this is the "
              f"number that matters in a sparse grid; it ignores the static majority of "
              f"cells that any candidate, including a no-op, gets right for free)")
    print(f"{label} mean per-cell accuracy (ALL cells, including the static majority): "
          f"{summary['mean_cell_accuracy']:.1%} "
          f"(inflated by trivially-correct unchanged cells in a sparse grid — a no-op "
          f"candidate can score nearly as high here as a genuinely correct one; prefer "
          f"the changed-cells number above when transitions are sparse)")
    print(f"{label} accuracy by action:")
    for action, bucket in sorted(summary["by_action"].items(), key=lambda kv: (kv[0] is None, kv[0])):
        print(f"    {action}: {bucket['accuracy']:.1%} ({bucket['n_pass']}/{bucket['n']})")


def select_counterexamples(failures, k):
    """
    Pick up to k failures spread across distinct actions (round-robin across
    action buckets) rather than just the first k or the "worst" k, so the
    revision prompt sees varied failure modes per token spent.
    """
    if len(failures) <= k:
        return failures
    buckets = {}
    for f in failures:
        buckets.setdefault(f["action"], []).append(f)
    bucket_list = list(buckets.values())
    selected = []
    i = 0
    while len(selected) < k and any(bucket_list):
        bucket = bucket_list[i % len(bucket_list)]
        if bucket:
            selected.append(bucket.pop(0))
        i += 1
    return selected[:k]


def atomic_write_text(path, content):
    """
    Write text to `path` without ever leaving a truncated/partial file on
    disk if the process is interrupted mid-write (Ctrl+C, kill, crash).
    Writes to a temp file in the SAME directory (so the final os.replace()
    is an atomic rename on the same filesystem, not a cross-filesystem copy)
    then atomically swaps it into place. Until that final replace, `path`
    still holds whatever it held before this call — either the old complete
    content or nothing at all — never a half-written new version.
    Used for files that get read back later (candidate_round{N}.py is read
    by the next round's revision prompt build; prompt_round{N}.txt is meant
    to be openable/inspectable during the --automatic-off pause) where a
    truncated read would cause a confusing downstream failure rather than
    just being obviously-missing.
    """
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent) or ".", prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Covers a forced second-Ctrl+C KeyboardInterrupt landing mid-write
        # too (BaseException, not Exception) — clean up the temp file rather
        # than leaving a stray .tmp artifact, then let the interrupt/error
        # propagate normally. `path` itself was never touched, so it's still
        # whatever it was before this call.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def pause_for_confirmation(label, path):
    """
    Block on stdin asking whether to continue, unless --automatic was passed.
    Used by cmd_run_loop to pause after each round's prompt (seed or revision)
    is written to disk but BEFORE it's sent to the LLM — gives you a chance to
    open the file and eyeball/edit it first. Any answer other than explicit
    'n'/'no' is treated as "continue" (bare Enter just continues), so pausing
    at every round is a lightweight inspection point, not a repeated hurdle.
    An EOF/interrupt on stdin (e.g. running this non-interactively without
    --automatic by mistake) also aborts, rather than hanging or looping.

    Ctrl+C here is handled directly (rather than via GracefulInterrupt below)
    because this point is already safe to stop at immediately: the prompt
    file is already fully written and closed, and no LLM call or subprocess
    is in flight yet — there's nothing to protect by deferring.
    """
    prompt = f"{label} written to {path}. Continue? [Y/n] "
    try:
        resp = input(prompt).strip().lower()
    except EOFError:
        print("\nNo input available to confirm continuation (stdin closed). "
              "Pass --automatic to run without pausing. Aborting.")
        sys.exit(130)
    except KeyboardInterrupt:
        print("\nAborted by user (Ctrl+C) at the confirmation prompt.")
        sys.exit(130)
    if resp in ("n", "no"):
        print("Aborted by user before this round's LLM call.")
        sys.exit(130)


class GracefulInterrupt:
    """
    Context manager for the risky span of a round: the LLM call, candidate
    extraction/write, sandboxed backtest, and log write — the operations
    that are either expensive to redo (an LLM generation that can take
    minutes on local GPU inference) or briefly touch disk in a way that's
    safer to let finish than to abort mid-write.

    First Ctrl+C inside the `with` block: caught here, NOT re-raised. Sets
    `.requested = True` and prints a message, then returns control to
    whatever was running (the LLM call, the subprocess wait, etc.), which
    continues uninterrupted to its natural completion. The caller is
    expected to check `.requested` right after the `with` block ends and
    stop the outer loop there — i.e. the current round is allowed to finish
    and be recorded normally; only the *next* round is skipped.

    Second Ctrl+C inside the same `with` block: treated as "no, I really
    mean now" — restores Python's default SIGINT behavior and lets it raise
    KeyboardInterrupt immediately, same as if this handler were never
    installed. This is a deliberate escape hatch for a hung/slow call the
    person doesn't want to wait out; whatever partial state that leaves is
    the tradeoff being knowingly accepted at that point (mitigated, but not
    eliminated, by the prompt/candidate files being written atomically —
    see atomic_write_text below — so a forced exit can at worst leave a
    round's *files* missing, never half-written/corrupted).

    Note this only ever affects the *parent* Python process's own signal
    handling. The sandboxed candidate subprocess (bwrap) shares this
    process's terminal foreground group and can still receive SIGINT
    directly from the terminal on the very first Ctrl+C, independent of
    this handler — if that happens mid-backtest, the subprocess call
    returns early with a nonzero/negative return code, which the existing
    exception handling in cmd_run_loop already treats as a failed/0%-scoring
    round rather than crashing, so this is a safe (if slightly surprising)
    outcome, not a new hole.
    """

    def __init__(self):
        self.requested = False
        self._first_handler_installed = False
        self._old_handler = None

    def _handle_first(self, signum, frame):
        self.requested = True
        print("\nInterrupt received — finishing the current round (LLM call/"
              "backtest/log write in progress), then stopping before the "
              "next round. Press Ctrl+C again to stop immediately instead "
              "(may leave this round's files missing, though never "
              "half-written).")
        # Escalate: a second Ctrl+C should behave like an ordinary,
        # immediate interrupt rather than being swallowed a second time.
        signal.signal(signal.SIGINT, self._old_handler)

    def __enter__(self):
        self._old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_first)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self._old_handler)
        return False  # never swallow an exception (e.g. a forced second Ctrl+C)


# ---------- chunk boundary computation ----------
# Pure functions, no LLM/sandbox involved — support for the chunked
# curriculum (design doc §2): the trace is processed in chronological
# chunks, each ending at whichever of three cutoffs comes soonest.
#
# Boundary values (`prev_boundary`, the return value, `len(records)`) are
# all in the same units: a CUMULATIVE ROW COUNT, i.e. "this many rows have
# been covered so far" — equivalently, a Python slice upper bound, so
# `records[prev_boundary:next_boundary]` is always exactly the new rows a
# chunk introduces. This is why a level-completion row at 0-indexed record
# position `idx` is reported as `idx + 1`, not `idx` — the record at index
# idx IS the chunk's last (inclusive) row, and idx + 1 rows have been
# covered through and including it.

def next_level_completion_row(records, prev_boundary):
    """
    Scan forward from `prev_boundary` (a cumulative-row-count boundary, so
    the scan starts at 0-indexed record position `prev_boundary`, the first
    row not yet covered by any prior chunk) for the first row that itself
    completes a level.

    Each cleaned record is one transition and carries BOTH endpoints of that
    transition's levels_completed value — `levels_completed_before` (from
    that row's pre_observation) and `levels_completed_after` (from its
    post_observation), per cmd_preprocess's extraction and the same
    `goal_reached = post_observation.levels_completed >
    pre_observation.levels_completed` rule used for is_goal certification.
    A level completes DURING a given row iff that row's own
    levels_completed_after > levels_completed_before — this is a property
    of a single record, never a comparison against a neighboring record.

    Returns the cumulative-row-count boundary value for that completion
    (idx + 1, so it can be used directly as a chunk boundary / slice upper
    bound — see module note above), or None if no row completes a level
    anywhere from `prev_boundary` through the end of the trace. A record
    missing either field is treated as "no signal there" and skipped,
    rather than raising — not every trace will have this field populated.
    """
    for idx in range(prev_boundary, len(records)):
        rec = records[idx]
        before = rec.get("levels_completed_before")
        after = rec.get("levels_completed_after")
        if before is None or after is None:
            continue
        if after > before:
            return idx + 1
    return None


def next_chunk_boundary(records, prev_boundary, max_examples):
    """
    Compute where the next chunk ends, per design doc §2's boundary rule.
    Three cutoffs compete; whichever is soonest (smallest) wins:
      - the normal fixed chunk-size cap, prev_boundary + max_examples
      - the whole-trace cutoff, len(records) — the final chunk may be short
      - a level cutoff, if a level completes before either of the above

    No minimum chunk size is enforced — if a level completes on the very
    next row after prev_boundary, the returned boundary can legitimately be
    prev_boundary + 1. Do not add clamping/merging logic for that case.

    Returns (next_boundary, is_level_boundary). is_level_boundary answers
    "was THIS boundary (this chunk's own end) produced by a level cutoff?" —
    it does NOT say anything about whether the chunk that starts after this
    boundary should get any special prompt treatment (e.g. an EXTEND_TEMPLATE
    DESCRIPTION note) for having just crossed a level. That's a property of
    the NEXT chunk, one call later — callers that need it must carry this
    return value forward themselves with a one-chunk lag; this function only
    ever describes its own chunk's ending.
    """
    candidates = [prev_boundary + max_examples, len(records)]
    level_row = next_level_completion_row(records, prev_boundary)
    if level_row is not None:
        candidates.append(level_row)
    next_boundary = min(candidates)
    is_level_boundary = (next_boundary == level_row)  # False if level_row is None
    return next_boundary, is_level_boundary


# ---------- backtest (chunked design): full replay, scoring, goal discounting ----------
# Step 6 of the chunked-curriculum design (design doc §6, §9). Distinct from
# score_candidate/summarize_scores above, which remain exactly as-is for the
# older flat harness (score/evaluate/run-loop) they still serve.

def atomic_write_json(path, data):
    """
    Write `data` as JSON to `path` via temp-file-then-os.replace(), so a
    process interruption mid-write can never leave `path` corrupted or
    half-written. Shared by run_backtest (row_failure_counts.json) and
    Step 8's commit/revert snapshot (row_failure_counts_best.json).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except BaseException:
        # os.replace itself is atomic, so if we get past it `path` is
        # guaranteed to hold a complete write, never a partial one -- this
        # cleanup only matters for failures BEFORE that point (e.g. disk
        # full while writing the temp file).
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_row_failure_counts(path):
    """
    Load the persistent per-row failure-tracking file (design doc §6), or
    an empty dict if it doesn't exist yet (the very first backtest call of
    a run). Keys are trace step indices stored as strings (JSON object keys
    are always strings) -- callers should key lookups with str(step).
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def actual_goal_for_record(rec):
    """
    Ground truth for goal-reached on a single row: levels_completed_after >
    levels_completed_before, both extracted per-row during preprocessing
    (Step 4/cmd_preprocess). False if either field is missing -- a trace
    that never carried levels_completed simply never certifies a goal
    transition, the same graceful "no signal" treatment as
    next_level_completion_row uses.
    """
    before = rec.get("levels_completed_before")
    after = rec.get("levels_completed_after")
    if before is None or after is None:
        return False
    return after > before


def run_backtest(candidate_path, records, boundary, counts_path, **run_kwargs):
    """
    Full teacher-forced replay of records[0:boundary] against candidate_path
    (Step 3's sequential/stateful runner), scored per design doc §9, with
    the persistent per-row row_failure_counts.json (design doc §6) read at
    the start and rewritten at the end.

    Per-row scoring:
      - A row that crashed, timed out, or was never reached (Step 3's
        abort-on-crash cut the replay short before this row) always counts
        as a failure. No grid/goal comparison is attempted for it.
      - predicted_goal == actual_goal is always required for a pass — both
        false positives and false negatives count as incorrect, no
        exemption (design doc §9).
      - GOAL-DISCOUNTING DECISION (explicit, as Step 6 asks to note in code
        rather than leave ambiguous): when predicted_goal is True, the grid
        comparison is EXCLUDED ENTIRELY from the pass/fail decision for
        that row — not scored-but-flagged. The candidate cannot know the
        next level's initial grid, so holding a wrong predicted_grid_after
        against it on a row it itself flagged as a level transition isn't
        meaningful. Concretely:
          predicted_goal == True:  passed = (predicted_goal == actual_goal)
          predicted_goal == False: passed = (predicted_goal == actual_goal)
                                            and (predicted_grid_after == actual_grid_after)

    Returns:
      {
        "accuracy": float,   # exact-match pass rate over THIS call's row range
        "n_pass": int,
        "n_total": int,
        "scored": [...],     # one entry per row, in step order
        "failures": [...],   # subset of "scored" where passed is False
        "streak": int,       # longest run of consecutive passed rows, THIS call's replay
        "counts_path": str,
      }

    Each "scored" entry is the original record plus "passed" (bool),
    "predicted_goal", "actual_goal", "goal_discounted" (bool — whether grid
    comparison was excluded for this row), "prediction" (present whenever a
    predicted grid exists at all, i.e. no error — regardless of pass/fail,
    unlike score_candidate's failed-only convention, since a revision
    prompt may still want a discounted row's prediction later), and
    "error" (if the row crashed or was never reached).

    row_failure_counts.json is read-mutated-written UNCONDITIONALLY on
    every call, including rounds a caller later rejects — this function
    does not know or care about accept/reject. Step 8 owns the
    commit/revert snapshot (row_failure_counts_best.json) that keeps this
    file consistent with whichever code is genuinely current-best after
    each round's decision; do not add that logic here.
    """
    row_range = records[:boundary]
    raw = run_candidate(candidate_path, row_range, **run_kwargs)
    counts = load_row_failure_counts(counts_path)

    scored = []
    passed_flags = []
    for rec in row_range:
        step_key = str(rec["step"])
        actual_grid = rec["grid_after"]
        actual_goal = actual_goal_for_record(rec)

        result = raw.get(rec["step"])
        if result is None:
            error = "no result (candidate crashed/timed out earlier in this replay)"
            predicted_grid, predicted_goal = None, False
        elif "error" in result:
            error = result["error"]
            predicted_grid, predicted_goal = None, False
        else:
            error = None
            predicted_grid = result["prediction"]
            predicted_goal = bool(result["goal"])

        if error is not None:
            passed = False
            goal_discounted = False
        else:
            goal_match = (predicted_goal == actual_goal)
            if predicted_goal:
                goal_discounted = True  # discounted: grid comparison excluded entirely, see docstring
                passed = goal_match
            else:
                goal_discounted = False
                passed = goal_match and (predicted_grid == actual_grid)

        entry = {
            **rec,
            "passed": passed,
            "predicted_goal": predicted_goal,
            "actual_goal": actual_goal,
            "goal_discounted": goal_discounted,
        }
        if predicted_grid is not None:
            entry["prediction"] = predicted_grid
        if error is not None:
            entry["error"] = error
        scored.append(entry)
        passed_flags.append(passed)

        if step_key not in counts:
            # First time this row has ever been part of a backtest replay
            # (every replay covers 0..boundary, so this only happens once
            # per row across the whole run) -- actual_grid/actual_goal are
            # ground truth and never change, so they're set here and never
            # touched again below.
            counts[step_key] = {
                "count": 0,
                "actual_grid": actual_grid,
                "actual_goal": actual_goal,
                "predicted_grid": None,
                "predicted_goal": None,
                "error": None,
            }
        row_counts = counts[step_key]
        row_counts["count"] = 0 if passed else row_counts["count"] + 1
        row_counts["predicted_grid"] = predicted_grid  # overwritten every run, correct or not
        row_counts["predicted_goal"] = predicted_goal
        row_counts["error"] = error

    atomic_write_json(counts_path, counts)

    n_total = len(scored)
    n_pass = sum(passed_flags)
    accuracy = n_pass / n_total if n_total else 0.0

    longest_streak = 0
    current_streak = 0
    for p in passed_flags:
        current_streak = current_streak + 1 if p else 0
        longest_streak = max(longest_streak, current_streak)

    return {
        "accuracy": accuracy,
        "n_pass": n_pass,
        "n_total": n_total,
        "scored": scored,
        "failures": [s for s in scored if not s["passed"]],
        "streak": longest_streak,
        "counts_path": str(counts_path),
    }


# ---------- subcommands ----------

def cmd_inspect(args):
    path = Path(args.trace)
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(records)} records from {path}\n")

    print("--- Structure of record 0 ---")
    describe(records[0])

    # Each row is already a full transition: pre_observation -> action -> post_observation.
    # my_agent_keyboard.py's _serialize_frame_for_trace collapses each observation's
    # `frame` down to the single settled grid before writing, so we expect
    # pre_observation.frame and post_observation.frame to each be a plain
    # list[rows][cols], not a list of grids. Verify that here rather than assuming it.
    for obs_key in (args.pre_key, args.post_key):
        shapes = set()
        for rec in records:
            val = get_nested(rec, f"{obs_key}.{args.frame_key}")
            if val is None:
                continue
            depth = 0
            v = val
            while isinstance(v, list) and v:
                depth += 1
                v = v[0]
            shapes.add(depth)
        if not shapes:
            print(f"\nNo value found at '{obs_key}.{args.frame_key}' — check the keys printed above.")
        elif shapes == {2}:
            print(f"\n'{obs_key}.{args.frame_key}': consistently a single grid (2 levels of nesting) "
                  f"across all records. Matches the collapsed format from _serialize_frame_for_trace — "
                  f"no further stripping needed.")
        else:
            print(f"\n'{obs_key}.{args.frame_key}': nesting depth varies/unexpected: {sorted(shapes)} "
                  f"(2 = single grid, 3 = list of grids). Inspect a raw record before trusting `preprocess`.")


def cmd_preprocess(args):
    """
    Each input row is already a full transition: pre_observation -> action -> post_observation.
    We just pull grid_before (pre_observation.frame), action, and grid_after
    (post_observation.frame) out of it. `extract_current_grid` is kept as a defensive
    fallback in case frame ever shows up as a list-of-grids instead of an
    already-collapsed single grid, but for traces from my_agent_keyboard.py it
    should be a no-op since _serialize_frame_for_trace collapses this at write time.

    Also opportunistically pulls levels_completed off both endpoints of each
    transition (pre_observation.levels_completed / post_observation.levels_completed,
    under whatever sub-key --levels-key names) into levels_completed_before/
    levels_completed_after on the cleaned record. A single row's own before/
    after pair is what next_level_completion_row (chunk boundary computation)
    and is_goal certification both key off of — goal_reached is exactly
    levels_completed_after > levels_completed_before for that same row, per
    design doc §1/§9. Non-fatal if a trace doesn't carry this field (same
    lenient, best-effort treatment as --score-key below) — pass --levels-key ""
    to skip the attempt entirely.
    """
    in_path = Path(args.trace)
    out_path = Path(args.out)
    n = 0
    with open(in_path) as f, open(out_path, "w") as out:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            pre_frame = get_nested(rec, f"{args.pre_key}.{args.frame_key}")
            post_frame = get_nested(rec, f"{args.post_key}.{args.frame_key}")
            if pre_frame is None or post_frame is None:
                raise KeyError(f"record {i}: missing '{args.pre_key}.{args.frame_key}' or "
                                f"'{args.post_key}.{args.frame_key}'. Run `inspect` first to confirm keys.")
            action = get_nested(rec, args.action_key)
            clean = {
                "step": i,
                "action": action,
                "grid_before": extract_current_grid(pre_frame),
                "grid_after": extract_current_grid(post_frame),
            }
            if args.score_key:
                score = get_nested(rec, args.score_key)
                if score is not None:
                    clean["score"] = score
            if args.levels_key:
                levels_before = get_nested(rec, f"{args.pre_key}.{args.levels_key}")
                levels_after = get_nested(rec, f"{args.post_key}.{args.levels_key}")
                # Only set either field if BOTH endpoints are present -- a
                # record with just one side is missing exactly the
                # information a before/after comparison needs, so leaving
                # both off (rather than one real value + one None) keeps
                # next_level_completion_row's "missing = no signal, skip"
                # handling correct instead of it silently comparing a real
                # int against None (which would raise) or treating a
                # one-sided record as a false completion/non-completion.
                if levels_before is not None and levels_after is not None:
                    clean["levels_completed_before"] = levels_before
                    clean["levels_completed_after"] = levels_after
            out.write(json.dumps(clean) + "\n")
            n = i + 1
    print(f"Wrote {n} cleaned records to {out_path}")


def cmd_split(args):
    with open(args.trace) as f:
        records = [json.loads(l) for l in f if l.strip()]
    n = len(records)
    if args.shuffle:
        import random
        rng = random.Random(args.seed)
        records = records[:]  # copy so the input list order isn't mutated in place
        rng.shuffle(records)
    split_idx = int(n * args.history_frac)
    history, held_out = records[:split_idx], records[split_idx:]
    Path(args.history_out).write_text("\n".join(json.dumps(r) for r in history) + "\n")
    Path(args.heldout_out).write_text("\n".join(json.dumps(r) for r in held_out) + "\n")
    print(f"{n} records -> history={len(history)} ({args.history_out}), "
          f"held_out={len(held_out)} ({args.heldout_out})")


ENCODING_EXPLANATION_HEX = """IMPORTANT — display format vs. real data: grids below are shown as one row per line, \
one character per cell, as a compact display shorthand only. Each character is a \
single hexadecimal digit standing in for one integer color value 0-15 (0-9 mean \
themselves; a=10, b=11, c=12, d=13, e=14, f=15). For example the row "3b0" represents \
the actual list [3, 11, 0]. The function you write never sees or returns these \
characters — it always works with real Python integers."""

ENCODING_EXPLANATION_RLE = """IMPORTANT — display format vs. real data: grids below are shown one row per \
line, run-length encoded as a compact display shorthand only. Each row is a \
space-separated sequence of "<digit>*<count>" runs, where <digit> is a single \
hexadecimal digit standing in for one integer color value 0-15 (0-9 mean themselves; \
a=10, b=11, c=12, d=13, e=14, f=15) and <count> is how many times that value repeats \
consecutively. For example the row "0*7 3*2" represents the actual list \
[0, 0, 0, 0, 0, 0, 0, 3, 3]. The function you write never sees or returns this \
run-length notation — it always works with real Python integers."""


def encoding_explanation(encoding):
    return ENCODING_EXPLANATION_RLE if encoding == "rle" else ENCODING_EXPLANATION_HEX


DESCRIPTION_INSTRUCTION_COMMON_INTRO = (
    "Before writing any code, describe the objects in the grid using less than 200 "
    "words. Note: multiple objects can be of the same type and an object can have "
    "multiple color values. Also, distinct objects tend to take up less than 20% of "
    "the grid's values but there are exceptions to this."
)

DESCRIPTION_INSTRUCTION_TRAILER = (
    'Write this as a comment/docstring in your class with label "DESCRIPTION:" so it '
    "carries forward to future revisions."
)


def build_description_instruction(is_extend=False, is_level_boundary=False):
    """
    Object-description instruction shown in PROMPT_TEMPLATE/EXTEND_TEMPLATE — never
    REVISE_TEMPLATE (design doc §4: revision is deliberately scoped tightly to
    specific failing rows; the model can update DESCRIPTION opportunistically as
    part of a normal revision without a separately mandated step there).

    PROMPT_TEMPLATE (chunk 1, is_extend=False) gets the 3-step version — there's no
    prior description to consult yet. EXTEND_TEMPLATE (every chunk after the first,
    is_extend=True) gets a 4-step version whose new first step asks the model to
    check its existing DESCRIPTION comment against the new examples before deciding
    whether it still holds.

    When this EXTEND_TEMPLATE call is for the chunk that starts right after a level
    completion (is_level_boundary=True — this is the LAGGED value from the
    PREVIOUS chunk's next_chunk_boundary call, per that function's own docstring
    and Step 10's one-chunk-lag wiring, never the current chunk's own), an extra,
    stronger sentence is prepended telling the model explicitly that a new level
    just started and its persisted DESCRIPTION may no longer apply — stronger than
    the routine "may already be sufficient, or it may need revising" step 1 language
    every other EXTEND_TEMPLATE chunk gets, since a new level can introduce a whole
    new tileset the existing description never accounted for.
    """
    if not is_extend:
        steps = (
            "There are three steps to accomplishing this task:\n"
            '1: First, look at the color values in the "Starting Grid". Notice the '
            "shapes. Notice any objects that stand out with color values different "
            "from whatever colors dominate most of the grid (background regions).\n"
            "2: Second, from the changed-cell lists above, notice any objects that "
            "have moved or changed.\n"
            "3: Third, you may guess the purpose that each object has in the grid. "
            'Examples of object purposes are "player-movable object", "goal '
            'destination for player-movable object", and "object that changes '
            "another object's shape or color\"."
        )
        return f"{DESCRIPTION_INSTRUCTION_COMMON_INTRO} {steps}\n{DESCRIPTION_INSTRUCTION_TRAILER}"

    level_boundary_note = ""
    if is_level_boundary:
        level_boundary_note = (
            "This chunk starts a NEW LEVEL. Your existing DESCRIPTION comment "
            "describes the previous level's grid and may no longer apply — check it "
            "against the new grid below rather than assuming it still holds.\n\n"
        )
    steps = (
        "There are four steps to accomplishing this task:\n"
        '1: First, look at the existing "DESCRIPTION:" comment/docstring in the '
        "provided class. The description may already be sufficient, or it may need "
        "revising given the new information above.\n"
        '2: Second, look at the color values in the "Starting Grid". Notice the '
        "shapes. Notice any objects that stand out with color values different "
        "from whatever colors dominate most of the grid (background regions).\n"
        "3: Third, from the changed-cell lists above, notice any objects that have "
        "moved or changed.\n"
        "4: Fourth, you may guess the purpose that each object has in the grid. "
        'Examples of object purposes are "player-movable object", "goal '
        'destination for player-movable object", and "object that changes another '
        "object's shape or color\"."
    )
    return f"{level_boundary_note}{DESCRIPTION_INSTRUCTION_COMMON_INTRO} {steps}\n{DESCRIPTION_INSTRUCTION_TRAILER}"


PROMPT_TEMPLATE = """You are given a sequence of observed transitions from an ARC-AGI-3 game. \
Each example shows a grid, an action taken, and the resulting grid.

{encoding_explanation}

Each example below also includes a computed list of exactly which cells changed \
between consecutive grids, in real integer (row, col): before -> after form. \
Use that list directly to figure out the rule — you do not need to manually compare \
the two grids cell-by-cell yourself.

Your task: write a single Python class with this exact shape:

    class GameModel:
        def predict(self, grid_before: list[list[int]], action: str, previous_state: dict) -> tuple:
            ...

predict must return a 3-tuple (predicted_grid_after, goal, state):
  - predicted_grid_after: list[list[int]], your prediction for the grid after taking \
the given action from grid_before.
  - goal: bool, True if you predict this action reaches the next level, False otherwise.
  - state: dict, JSON-serializable (plain dicts/lists/numbers only) — everything you \
want carried forward into your NEXT predict() call as previous_state. On the very \
first call of a fresh rollout, previous_state is an empty dict {{}}; your class should \
handle that starting condition itself (e.g. via previous_state.get(...) with sensible \
defaults), not assume any particular keys already exist.

grid_before is always a list of lists of plain integers 0-15 — never the compact \
display notation shown above, whichever form it takes — and predicted_grid_after must \
be in that same form. Infer the transformation rule(s) from the examples below by \
mentally converting each displayed row back to its real integer list first. Your class \
may define as many additional methods/fields as it needs; only predict's signature is \
fixed. Use only the Python standard library or numpy — no other third-party packages. \
Return only the class definition, no explanation, no example usage.

Define GameModel exactly ONCE, and define each of its methods exactly ONCE. Do not \
write multiple draft attempts, "actually, let me try again" rewrites, or alternate \
versions — pick your best hypothesis and write a single, complete, syntactically valid \
class for it. Every if/elif/else branch and every loop body must contain a real \
statement (return, assignment, pass, etc.) — never leave a branch with only a comment \
inside it.

State your hypothesis about the transformation rule ONCE, briefly, as a short comment. \
Do not re-examine the same examples repeatedly or restate your reasoning multiple \
times — if your first hypothesis doesn't perfectly fit every example, write your best \
guess anyway and move on. You will see new examples and counterexamples and get a \
chance to fix mistakes in later rounds, so an imperfect-but-complete class now is far \
more useful than a perfect rule you never finish writing.

{examples}

{description_instruction}

Reminder: predict() takes and returns plain Python int grids at runtime, never the \
display notation used above to show you the examples.

Write your GameModel class now.
"""

EXTEND_TEMPLATE = """You are given new, previously-unseen transitions from the SAME ARC-AGI-3 \
game your class below was already built for. These are NOT failures — your class \
simply hasn't seen these examples yet. Extend or adjust it so it also correctly \
handles them, WITHOUT rewriting it from scratch.

Here is your current class:

```python
{candidate_code}
```

{encoding_explanation}

Each example below also includes a computed list of exactly which cells changed \
between consecutive grids, in real integer (row, col): before -> after form. \
Use that list directly to figure out the rule — you do not need to manually compare \
the two grids cell-by-cell yourself.

{examples}

{description_instruction}

Your predict method must keep this exact signature:

    def predict(self, grid_before: list[list[int]], action: str, previous_state: dict) -> tuple:
        ...

returning (predicted_grid_after, goal, state) exactly as before — see your class's own \
docstring above for what each element means if you need a reminder. grid_before is \
always a list of lists of plain integers 0-15 — never the compact display notation \
shown above, whichever form it takes — and predicted_grid_after must be in that same \
form. Use only the Python standard library or numpy — no other third-party packages. \
Return only the full, updated class definition, no explanation, no example usage.

Define GameModel exactly ONCE, and define each of its methods exactly ONCE — extend or \
adjust the existing methods/fields shown above rather than writing multiple draft \
attempts, "actually, let me try again" rewrites, or alternate versions of the whole \
class. Every if/elif/else branch and every loop body must contain a real statement \
(return, assignment, pass, etc.) — never leave a branch with only a comment inside it.

State what you changed, if anything, ONCE, briefly, as a short comment. Do not \
re-examine the same examples repeatedly or restate your reasoning multiple times — \
write your best update and move on, even if it's imperfect. You'll get revision rounds \
later if something is still wrong.

Reminder: predict() takes and returns plain Python int grids at runtime, never the \
display notation used above to show you the examples.

Write your updated GameModel class now.
"""

REVISE_TEMPLATE = """Your previous candidate for predict_next_state got some transitions wrong. \
Here is the current code:

```python
{candidate_code}
```

{encoding_explanation}

Each counterexample below includes computed diffs — exactly which cells actually \
changed, and exactly which cells your prediction changed instead — in real integer \
(row, col): before -> after form. Use those directly to see where your rule diverges \
from the truth; you do not need to manually compare grids cell-by-cell yourself.

It produced the WRONG grid_after for the following examples:

{counterexamples}

Revise the function so it correctly handles these cases while continuing to handle \
the cases it already gets right. Use only the Python standard library. Return only \
the corrected function definition, no explanation.

Define predict_next_state exactly ONCE. Do not write multiple draft attempts or \
alternate versions — revise your single best hypothesis and write one complete, \
syntactically valid function. Every if/elif/else branch and loop body must contain a \
real statement — never leave a branch with only a comment inside it.

State your revised hypothesis ONCE, briefly, as a short comment. Do not re-examine \
the same counterexamples repeatedly or restate your reasoning multiple times — write \
your best fix and move on, even if it's imperfect. You'll get another revision round \
if it's still wrong.

Reminder: predict_next_state takes and returns plain Python int grids at runtime, \
never the display notation used above to show you the examples.
"""


def build_examples_block(records, encoding="hex", is_extend=False):
    """
    Render a contiguous batch of records as one continuous replay, not
    independent snapshots (design doc §4): the full encoded grid is shown
    exactly once (`Starting Grid`), and every subsequent example is shown
    only as a diff against the immediately preceding one — a full grid for
    every example is redundant, since each subsequent starting grid is just
    the previous example's resulting grid, already fully implied by the
    previous example's diff.

    `is_extend` controls only the one extra "not the game's start" sentence
    (PROMPT_TEMPLATE calls with is_extend=False, EXTEND_TEMPLATE with
    is_extend=True) — everything else is shared. This function backs the
    round-1/extend path only; build_counterexamples_block (REVISE_TEMPLATE)
    is unchanged and continues showing full predicted/actual grid pairs per
    counterexample, since a counterexample's whole purpose is the
    predicted-vs-actual comparison, not a contiguous replay.

    Precondition: `records` must be genuinely contiguous —
    records[i]["grid_before"] == records[i-1]["grid_after"] for every i > 0
    — since the diff labels ("Grid i-1 -> Grid i") assume it. True by
    construction for a chunk's round-1 batch (chronological, no shuffle);
    asserted explicitly below rather than silently trusted, since a silent
    violation here would produce a prompt that's internally inconsistent
    without ever raising an error.
    """
    assert records, "build_examples_block requires at least one record"
    for i in range(1, len(records)):
        assert records[i]["grid_before"] == records[i - 1]["grid_after"], (
            f"build_examples_block requires contiguous records: record {i - 1} "
            f"(step {records[i - 1]['step']})'s grid_after does not match record {i} "
            f"(step {records[i]['step']})'s grid_before. Pass a genuinely contiguous "
            f"chunk batch, not a sampled/shuffled subset."
        )

    lines = [
        "The examples below form one continuous sequence, not independent snapshots.",
        '"Starting Grid" is the earliest state in this batch.',
    ]
    if is_extend:
        lines.append(
            "Starting Grid here is NOT the start of the game — it is only the first "
            "state in this particular batch of new examples. Earlier trace history "
            "exists before it; rely on your class's own carried-forward state/"
            "description for anything from before this batch."
        )
    lines.append(f"\nStarting Grid:\n{encode_grid(records[0]['grid_before'], encoding)}\n")

    prev_label = "Starting Grid"
    for i, rec in enumerate(records, start=1):
        label = f"Grid {i}"
        changes = diff_grid(rec["grid_before"], rec["grid_after"])
        block = (
            f"\n### {label} (trace step {rec['step']})\n"
            f"action: {rec['action']}\n"
            f"changed cells ({prev_label} -> {label}):\n{format_diff(changes)}\n"
        )
        if len(changes) > DIFF_MAX_CELLS:
            block += (
                f"\n{label} in full (shown because too many cells changed to list "
                f"individually):\n{encode_grid(rec['grid_after'], encoding)}\n"
            )
        lines.append(block)
        prev_label = label
    return "\n".join(lines)


def spread_sample(records, max_examples):
    """
    Evenly-spaced subsample across the whole list (not just the first N).

    SUPERSEDED for round-1 prompt building under the chunked curriculum
    (design doc §2) — build_initial_prompt now takes chunk 1 as the trace's
    first max_examples rows in chronological order instead, since
    build_examples_block's contiguity precondition requires it (an
    evenly-spaced sample isn't contiguous and would trip that assertion).
    Left defined, unused by build_initial_prompt, only in case any other
    legacy caller still depends on the old flat-harness sampling behavior.
    """
    if not max_examples or len(records) <= max_examples:
        return records
    step = len(records) / max_examples
    idxs = sorted(set(int(i * step) for i in range(max_examples)))
    return [records[i] for i in idxs]


def build_initial_prompt(records, max_examples, encoding="hex"):
    """
    Chunk 1's round-1 prompt. Under the chunked curriculum (design doc §2),
    chunk 1 is simply the trace's first max_examples rows in chronological
    order — see spread_sample's docstring for why this is no longer an
    evenly-spaced subsample across the whole trace.
    """
    sampled = records[:max_examples]
    prompt = PROMPT_TEMPLATE.format(
        encoding_explanation=encoding_explanation(encoding),
        examples=build_examples_block(sampled, encoding=encoding, is_extend=False),
        description_instruction=build_description_instruction(is_extend=False),
    )
    return prompt, len(sampled)


def build_extend_prompt(candidate_code, new_records, encoding="hex", is_level_boundary=False):
    """
    Round-1 prompt for every chunk after the first (design doc §4). Shows
    the candidate's current class in full plus only this chunk's newly
    introduced rows — not the whole trace-so-far, which would defeat the
    point of a bounded per-chunk prompt.

    is_level_boundary must be the LAGGED value from the PREVIOUS chunk's
    next_chunk_boundary call (Step 4/Step 10's one-chunk lag), never this
    chunk's own not-yet-relevant is_level_boundary — see
    build_description_instruction's docstring for why.
    """
    return EXTEND_TEMPLATE.format(
        candidate_code=candidate_code.strip(),
        encoding_explanation=encoding_explanation(encoding),
        examples=build_examples_block(new_records, encoding=encoding, is_extend=True),
        description_instruction=build_description_instruction(
            is_extend=True, is_level_boundary=is_level_boundary
        ),
    )


def build_counterexamples_block(counterexamples, encoding="hex"):
    """Format failing records, showing what was predicted vs. what actually happened."""
    blocks = []
    for i, rec in enumerate(counterexamples):
        pred = rec.get("prediction")
        actual_changes = diff_grid(rec["grid_before"], rec["grid_after"])
        if pred is not None:
            pred_str = encode_grid(pred, encoding)
            pred_changes = diff_grid(rec["grid_before"], pred)
            pred_diff_str = format_diff(pred_changes)
        else:
            pred_str = f"(error: {rec.get('error')})"
            pred_diff_str = "(candidate raised an error — see above — so no predicted grid to diff)"
        blocks.append(
            f"### Counterexample {i + 1}\n"
            f"action: {rec['action']}\n"
            f"grid_before:\n{encode_grid(rec['grid_before'], encoding)}\n"
            f"your prediction:\n{pred_str}\n"
            f"correct grid_after:\n{encode_grid(rec['grid_after'], encoding)}\n"
            f"what actually changed (grid_before -> correct grid_after), computed for "
            f"you — real int values:\n{format_diff(actual_changes)}\n"
            f"what your prediction changed instead (grid_before -> your prediction):\n"
            f"{pred_diff_str}\n"
        )
    return "\n".join(blocks)


def build_revise_prompt(candidate_code, counterexamples, encoding="hex"):
    return REVISE_TEMPLATE.format(
        candidate_code=candidate_code.strip(),
        encoding_explanation=encoding_explanation(encoding),
        counterexamples=build_counterexamples_block(counterexamples, encoding=encoding),
    )


def estimate_tokens_fallback(text):
    """
    Rough token estimate for use when no real tokenizer is loaded (the manual
    `prompt`/`revise-prompt` path). The usual "~4 chars/token" rule of thumb
    is calibrated for English prose; this tool's prompts are mostly dense
    grid encodings (hex digits or RLE runs), which don't compress into BPE
    merges nearly as well and tokenize closer to 1 token/char in practice
    (confirmed against a live tokenizer: a 43,090-char grid-heavy prompt came
    out to 41,493 real tokens, not the ~10,772 the old chars//4 estimate
    gave). This is still just a ballpark — prefer a real tokenizer (see
    run-loop's preflight check) whenever one is available.
    """
    return len(text)


def extract_code(text):
    """
    Pull code out of a ```python ... ``` fence if present, else assume the
    whole reply is code. Handles a truncated response (generation cut off by
    --max-tokens or a repetition loop before the closing fence arrives) by
    matching to end-of-string in that case — otherwise the old regex simply
    failed to match, and the fallback path kept the literal opening ```python
    marker as part of "code", guaranteeing an immediate SyntaxError on line 1
    instead of surfacing the actual (truncated/incomplete) candidate body.
    """
    import re
    m = re.search(r"```(?:python)?\s*\n?(.*?)(?:```|\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def keep_first_function_def(source, class_name="GameModel"):
    """
    Generalized from an earlier single-function version to work against the
    class-based candidate contract (a top-level `class GameModel: ...` with
    a `predict` method and whatever other helper methods/fields the model
    adds — see design doc's candidate contract).

    Handles two independent degrading-redraft failure modes — the model
    writes an attempt, then "actually, let me reconsider..." and rewrites
    it again, sometimes several times, each version emptier/more hedged
    than the last (down to bare `pass` placeholders):

      1. Multiple top-level `class GameModel` redefinitions in the same
         response — keep the first complete class, drop everything from
         the start of the second redefinition onward (this also discards
         any trailing "final answer" prose after it, same as before).
      2. Multiple `def` definitions of the same method name *inside* the
         kept class's body (e.g. two `def predict(...)` in one class) —
         within the kept class, keep only the first definition of each
         duplicated method name, trimming just those extra method bodies
         out of the class while leaving every other method/field untouched.

    Python doesn't error on either kind of duplicate; each `class`/`def`
    simply rebinds the name, so whichever definition is LAST is the only
    one ever actually used at runtime — and in every case observed so far,
    the last attempt is the most degraded one, not the best one. Without
    this trim, run_candidate would silently score the model's worst
    attempt each round while a better (if still wrong) first attempt gets
    thrown away for free.

    Returns (trimmed_source, num_defs_found), where num_defs_found is the
    combined count of extra top-level class redefinitions plus extra
    in-class method redefinitions found (0 if the source was already
    clean).
    """
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, 0  # let the caller's own ast.parse (check_ast_imports) raise this normally

    class_matches = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    if not class_matches:
        return source, 0

    lines = source.splitlines(keepends=True)
    num_defs_found = len(class_matches) - 1  # extra top-level class redefinitions, dropped wholesale

    first_class = class_matches[0]
    # ast's end_lineno is 1-indexed and inclusive; slicing to it keeps
    # everything through the end of the first class's body and drops every
    # subsequent full-class redefinition (plus any trailing "final answer"
    # prose), exactly as the original single-function version did.
    trimmed_source = "".join(lines[:first_class.end_lineno])

    # Second pass: look for duplicate method defs *within* the kept class.
    # Re-parse the trimmed source so all line numbers below are relative to
    # it (the class's own lineno/end_lineno don't shift from the first
    # parse since nothing before it was touched, but re-parsing keeps this
    # robust rather than relying on that).
    trimmed_tree = ast.parse(trimmed_source)
    trimmed_class = next(
        n for n in trimmed_tree.body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )

    methods_by_name = {}
    for node in trimmed_class.body:
        if isinstance(node, ast.FunctionDef):
            methods_by_name.setdefault(node.name, []).append(node)

    duplicate_methods = {name: nodes for name, nodes in methods_by_name.items() if len(nodes) > 1}
    if not duplicate_methods:
        return trimmed_source, num_defs_found

    num_defs_found += sum(len(nodes) - 1 for nodes in duplicate_methods.values())

    # Every duplicate method definition after the first one for its name
    # gets dropped, by line range (including its own decorators, if any).
    drop_ranges = []
    for nodes in duplicate_methods.values():
        for extra in nodes[1:]:
            start = extra.lineno
            if extra.decorator_list:
                start = min(d.lineno for d in extra.decorator_list)
            drop_ranges.append((start, extra.end_lineno))

    trimmed_lines = trimmed_source.splitlines(keepends=True)
    keep_mask = [True] * len(trimmed_lines)
    for start, end in drop_ranges:
        for i in range(start - 1, end):  # convert 1-indexed inclusive to 0-indexed
            keep_mask[i] = False

    final_source = "".join(line for line, keep in zip(trimmed_lines, keep_mask) if keep)
    return final_source, num_defs_found


def call_llm_openai(api_base, model, prompt, temperature=0.2, max_tokens=4096,
                     frequency_penalty=0.0, presence_penalty=0.0, timeout=180):
    """
    POST to an OpenAI-compatible /chat/completions endpoint — used if you're
    serving the model over HTTP (vLLM, or llama-server's OpenAI-compat mode)
    instead of loading it directly via llama-cpp-python. frequency_penalty
    and presence_penalty are both standard OpenAI chat-completions fields,
    unlike llama.cpp's repeat_penalty, which has no equivalent in this API
    shape — so only frequency_penalty/presence_penalty are forwarded here.

    Returns a dict {"text", "finish_reason", "completion_tokens"} rather than
    just the text, so callers can tell "hit --max-tokens mid-thought" (a real
    unfinished response) apart from "the model chose to stop" (finish_reason
    == "stop") — the two look identical if you only look at the text itself,
    but they call for very different fixes (raise --max-tokens vs. the
    generation is being cut short by something else, e.g. penalty settings).
    """
    import urllib.request
    url = api_base.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    choice = data["choices"][0]
    return {
        "text": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": data.get("usage", {}).get("completion_tokens"),
    }


def build_llm_caller(args):
    """
    Returns (call, tokenize):
      - call: a prompt -> {"text", "finish_reason", "completion_tokens"} dict
        callable for the chosen --backend. finish_reason distinguishes a
        genuinely truncated response ("length" — hit --max-tokens mid-thought)
        from one the model chose to end on its own ("stop") — the raw text
        looks the same either way, but they call for different fixes.
      - tokenize: a prompt -> token_count callable using the model's own
        tokenizer, or None if no local tokenizer is available for this
        backend (e.g. --backend openai talking to a remote server). Callers
        should treat None as "can't verify context-window fit locally."
    For llama-cpp, the GGUF is loaded ONCE here and reused across all rounds —
    reloading a multi-GB model file every round would dominate the runtime.
    """
    if args.backend == "llama-cpp":
        if not args.model_path:
            raise SystemExit("--model-path is required for --backend llama-cpp "
                              "(path to your .gguf file)")
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise SystemExit(
                "llama-cpp-python isn't installed. Install a CUDA-enabled build, e.g.:\n"
                "  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --no-cache-dir"
            ) from e
        llm = Llama(
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            verbose=args.verbose_llama,
        )

        def _call(prompt):
            resp = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                repeat_penalty=args.repeat_penalty,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
            )
            choice = resp["choices"][0]
            return {
                "text": choice["message"]["content"],
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": resp.get("usage", {}).get("completion_tokens"),
            }

        def _tokenize(prompt):
            # Raw tokenize() undercounts vs. what create_chat_completion actually
            # sends, since the chat template adds role markers/special tokens on
            # top of the raw text — pad the estimate a bit so the preflight check
            # errs toward catching a too-tight fit rather than missing one.
            return len(llm.tokenize(prompt.encode("utf-8")))

        return _call, _tokenize

    elif args.backend == "openai":
        if not args.model:
            raise SystemExit("--model is required for --backend openai (the model name your server expects)")
        call = lambda prompt: call_llm_openai(
            args.api_base, args.model, prompt, temperature=args.temperature, max_tokens=args.max_tokens,
            frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty,
        )
        return call, None

    else:
        raise SystemExit(f"unknown --backend {args.backend!r}, expected 'llama-cpp' or 'openai'")


def cmd_prompt(args):
    with open(args.history) as f:
        records = [json.loads(l) for l in f if l.strip()]

    encoding = "rle" if args.compact else "hex"
    prompt, n_used = build_initial_prompt(records, args.max_examples, encoding=encoding)
    Path(args.out).write_text(prompt)
    est_tokens = estimate_tokens_fallback(prompt)
    print(f"Wrote prompt to {args.out}: {len(prompt)} chars, ~{est_tokens} tokens "
          f"(rough 1-token/char fallback estimate — no tokenizer loaded on this path; "
          f"grid content tokenizes far worse than chars//4 prose assumes), "
          f"{n_used} transition examples"
          + (" [RLE-compact grid encoding]" if args.compact else " [hex grid encoding]"))


def cmd_revise_prompt(args):
    candidate_code = Path(args.candidate).read_text()
    with open(args.counterexamples) as f:
        counterexamples = [json.loads(l) for l in f if l.strip()]
    encoding = "rle" if args.compact else "hex"
    prompt = build_revise_prompt(candidate_code, counterexamples, encoding=encoding)
    Path(args.out).write_text(prompt)
    est_tokens = estimate_tokens_fallback(prompt)
    print(f"Wrote revision prompt to {args.out}: {len(prompt)} chars, ~{est_tokens} tokens "
          f"(rough 1-token/char fallback estimate — no tokenizer loaded on this path), "
          f"{len(counterexamples)} counterexamples"
          + (" [RLE-compact grid encoding]" if args.compact else " [hex grid encoding]"))


def cmd_backtest(args):
    """
    Standalone CLI wrapper around run_backtest, mainly for manual testing/
    debugging outside the full chunked run-loop (Step 10 wires run_backtest
    directly into that loop without going through this command).

    Exit 0 if the replay is fully clean (zero failing rows), exit 1
    otherwise. This is a simpler convention than the older evaluate
    command's 0/1/2 — there's no fixed accuracy threshold to check here;
    per-chunk accept/reject against a baseline is Step 8's epsilon
    comparison, not this command's job.
    """
    with open(args.trace) as f:
        records = [json.loads(l) for l in f if l.strip()]
    boundary = args.boundary if args.boundary is not None else len(records)

    result = run_backtest(
        args.candidate, records, boundary, args.counts,
        cpu_seconds=args.cpu_seconds, mem_mb=args.mem_mb, max_procs=args.max_procs,
        per_call_seconds=args.per_call_seconds, overall_timeout=args.overall_timeout,
    )

    print(f"{args.candidate} vs {args.trace}[0:{boundary}]: "
          f"{result['n_pass']}/{result['n_total']} passed ({result['accuracy']:.1%}), "
          f"longest streak {result['streak']}")
    print(f"row_failure_counts written to {result['counts_path']}")

    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(s) for s in result["scored"]) + "\n")
        print(f"Wrote per-row results to {args.out}")

    if result["failures"]:
        print(f"{len(result['failures'])} failing row(s).")
        sys.exit(1)
    print("Replay fully clean (zero failing rows).")
    sys.exit(0)


def cmd_score(args):
    with open(args.dataset) as f:
        records = [json.loads(l) for l in f if l.strip()]
    scored, accuracy = score_candidate(
        args.candidate, records,
        cpu_seconds=args.cpu_seconds, mem_mb=args.mem_mb, max_procs=args.max_procs,
        per_call_seconds=args.per_call_seconds, overall_timeout=args.overall_timeout,
    )
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(s) for s in scored) + "\n")
        print(f"Wrote per-record results to {args.out}")
    n_pass = sum(1 for s in scored if s["passed"])
    print(f"{args.candidate} vs {args.dataset}: {n_pass}/{len(scored)} passed ({accuracy:.1%})")
    print_score_summary(summarize_scores(scored), label=str(args.dataset))


def cmd_evaluate(args):
    """
    The stop condition for the verify-and-revise loop. Scores the candidate
    against held-out ONLY — held-out never gets used to pick counterexamples,
    so its accuracy stays a clean signal of whether the architecture actually
    generalizes, not just fits what it's already seen. Decides:
      - exit 0: STOP, threshold met
      - exit 2: STOP, max rounds reached without meeting threshold
      - exit 1: CONTINUE — and if --history/--counterexamples-out are given,
                writes the next round's counterexamples from history failures
    """
    with open(args.heldout) as f:
        heldout_records = [json.loads(l) for l in f if l.strip()]

    scored, accuracy = score_candidate(
        args.candidate, heldout_records,
        cpu_seconds=args.cpu_seconds, mem_mb=args.mem_mb, max_procs=args.max_procs,
        per_call_seconds=args.per_call_seconds, overall_timeout=args.overall_timeout,
    )
    n_pass = sum(1 for s in scored if s["passed"])
    summary = summarize_scores(scored)

    log_entry = {
        "round": args.round,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate": str(args.candidate),
        "heldout_accuracy": accuracy,
        "n_passed": n_pass,
        "n_total": len(scored),
        "heldout_mean_cell_accuracy": summary["mean_cell_accuracy"],
        "heldout_mean_changed_cell_accuracy": summary["mean_changed_cell_accuracy"],
        "heldout_n_dynamic_records": summary["n_dynamic_records"],
        "heldout_by_action": summary["by_action"],
    }
    if args.log:
        with open(args.log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    print(f"Round {args.round}: held-out accuracy {accuracy:.1%} ({n_pass}/{len(scored)}), "
          f"threshold {args.threshold:.1%}")
    print_score_summary(summary, label="held-out")

    if accuracy >= args.threshold:
        print(f"STOP: threshold met at round {args.round}.")
        sys.exit(0)

    if args.round >= args.max_rounds:
        print(f"STOP: reached max rounds ({args.max_rounds}) without meeting threshold.")
        sys.exit(2)

    print(f"CONTINUE: threshold not met, {args.max_rounds - args.round} round(s) remain.")

    if args.history and args.counterexamples_out:
        with open(args.history) as f:
            history_records = [json.loads(l) for l in f if l.strip()]
        hist_scored, hist_accuracy = score_candidate(
            args.candidate, history_records,
            cpu_seconds=args.cpu_seconds, mem_mb=args.mem_mb, max_procs=args.max_procs,
            per_call_seconds=args.per_call_seconds, overall_timeout=args.overall_timeout,
        )
        failures = [s for s in hist_scored if not s["passed"]]
        chosen = select_counterexamples(failures, args.k)
        Path(args.counterexamples_out).write_text("\n".join(json.dumps(c) for c in chosen) + "\n")
        n_hist_pass = len(history_records) - len(failures)
        print(f"History accuracy {hist_accuracy:.1%} ({n_hist_pass}/{len(history_records)}). "
              f"Wrote {len(chosen)} counterexamples to {args.counterexamples_out} for the next revision prompt.")
    elif args.history or args.counterexamples_out:
        print("Note: both --history and --counterexamples-out are needed to write next-round "
              "counterexamples; skipping since only one was given.")

    sys.exit(1)


def cmd_run_loop(args):
    """
    Fully automated verify-and-revise loop: builds the prompt, calls the LLM
    over its OpenAI-compatible API, scores the reply, and repeats — no manual
    copy-paste. Each round's prompt and candidate are written to --workdir so
    you can inspect exactly what was sent/produced if something looks off.
    """
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    with open(args.history) as f:
        history_records = [json.loads(l) for l in f if l.strip()]
    with open(args.heldout) as f:
        heldout_records = [json.loads(l) for l in f if l.strip()]

    run_kwargs = dict(cpu_seconds=args.cpu_seconds, mem_mb=args.mem_mb, max_procs=args.max_procs,
                       per_call_seconds=args.per_call_seconds, overall_timeout=args.overall_timeout)

    llm_call, tokenize = build_llm_caller(args)  # loads the GGUF once here for --backend llama-cpp
    encoding = "rle" if args.compact else "hex"

    counterexamples = None
    for round_n in range(1, args.max_rounds + 1):
        if counterexamples is None:
            prompt_text, n_used = build_initial_prompt(history_records, args.max_examples, encoding=encoding)
            print(f"Round {round_n}: seed prompt with {n_used} examples")
            round_label = f"Round {round_n}: Prompt"
        else:
            prev_candidate_code = Path(workdir / f"candidate_round{round_n - 1}.py").read_text()
            prompt_text = build_revise_prompt(prev_candidate_code, counterexamples, encoding=encoding)
            print(f"Round {round_n}: revision prompt with {len(counterexamples)} counterexamples")
            round_label = f"Round {round_n}: Revision Prompt"

        prompt_path = workdir / f"prompt_round{round_n}.txt"
        atomic_write_text(prompt_path, prompt_text)

        # By default (no --automatic), pause here: the prompt is on disk but
        # has NOT yet been sent to the LLM, so this is the point to open
        # prompt_path and inspect/edit it before committing GPU/API time to it.
        if not args.automatic:
            pause_for_confirmation(round_label, prompt_path)

        if tokenize is not None:
            # Real tokenizer from the loaded model — this is the number that
            # actually matters, not the chars//4 (or chars//1) estimates used
            # on the manual prompt/revise-prompt path. Fail fast with a clear
            # message rather than letting llama-cpp raise mid-generation.
            n_prompt_tokens = tokenize(prompt_text)
            budget = args.n_ctx - args.max_tokens
            print(f"Round {round_n}: prompt is {n_prompt_tokens} tokens "
                  f"(budget {budget} = --n-ctx {args.n_ctx} - --max-tokens {args.max_tokens})")
            if n_prompt_tokens > budget:
                raise SystemExit(
                    f"Round {round_n}: prompt ({n_prompt_tokens} tokens) exceeds the available "
                    f"budget ({budget} tokens). Fix one of: lower --max-examples/--k, add "
                    f"--compact for RLE grid encoding, raise --n-ctx, or lower --max-tokens."
                )
        else:
            print(f"Round {round_n}: prompt is {len(prompt_text)} chars "
                  f"(no local tokenizer for --backend openai — token count not verified "
                  f"against the server's context window; watch for a context-length error)")

        # Everything from here through the log write is the "risky" span this
        # round can't cleanly redo if interrupted mid-flight (an LLM
        # generation can take minutes; a killed sandboxed backtest just looks
        # like a failed round rather than corrupting anything — see
        # GracefulInterrupt's docstring). A first Ctrl+C here is deferred
        # until this block finishes naturally; a second Ctrl+C forces an
        # immediate stop.
        with GracefulInterrupt() as interrupt:
            call_result = llm_call(prompt_text)
            response = call_result["text"]
            finish_reason = call_result.get("finish_reason")
            completion_tokens = call_result.get("completion_tokens")
            # finish_reason == "length" means generation was cut off by --max-tokens
            # mid-thought (a genuinely truncated response). finish_reason == "stop"
            # means the model emitted an end-of-sequence token on its own — the
            # response text can look identical either way, but "stop" with a short
            # completion_tokens count points at the model ending early on its own
            # (e.g. from --presence-penalty/--frequency-penalty pressure building up
            # over the response and making "just stop" look attractive), not at
            # --max-tokens being too small.
            print(f"Round {round_n}: generation finished with reason={finish_reason!r}"
                  + (f", {completion_tokens} completion tokens" if completion_tokens is not None else "")
                  + (f" (out of --max-tokens {args.max_tokens} budget)" if finish_reason == "length" else ""))
            code = extract_code(response)
            code, n_defs = keep_first_function_def(code)
            if n_defs > 1:
                print(f"Round {round_n}: candidate defined predict_next_state {n_defs} times "
                      f"(re-attempts); keeping only the first, discarding the rest")
            candidate_path = workdir / f"candidate_round{round_n}.py"
            atomic_write_text(candidate_path, code)

            try:
                scored, accuracy = score_candidate(candidate_path, heldout_records, **run_kwargs)
            except (ValueError, RuntimeError, SyntaxError, OSError) as e:
                # A candidate that fails to even run — bad import (ValueError from
                # check_ast_imports), sandbox/runner failure (RuntimeError), or
                # unparseable code (SyntaxError from ast.parse — e.g. a truncated
                # generation that got cut off by --max-tokens or a repetition loop
                # before finishing) — is scored as a zero-accuracy round rather
                # than killing the whole loop. SyntaxError is NOT a ValueError/
                # RuntimeError subclass, so it must be listed explicitly or it
                # propagates straight past this except clause and crashes cmd_run_loop.
                print(f"Round {round_n}: candidate failed to execute cleanly: "
                      f"{type(e).__name__}: {e}")
                scored, accuracy = [], 0.0

            n_pass = sum(1 for s in scored if s["passed"])
            n_total = len(heldout_records)
            summary = summarize_scores(scored) if scored else None
            log_entry = {
                "round": round_n,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "candidate": str(candidate_path),
                "heldout_accuracy": accuracy,
                "n_passed": n_pass,
                "n_total": n_total,
                "heldout_mean_cell_accuracy": summary["mean_cell_accuracy"] if summary else None,
                "heldout_mean_changed_cell_accuracy": summary["mean_changed_cell_accuracy"] if summary else None,
                "heldout_n_dynamic_records": summary["n_dynamic_records"] if summary else None,
                "heldout_by_action": summary["by_action"] if summary else None,
            }
            if args.log:
                with open(args.log, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

        print(f"Round {round_n}: held-out accuracy {accuracy:.1%} ({n_pass}/{n_total}), "
              f"threshold {args.threshold:.1%}")
        if summary:
            print_score_summary(summary, label="held-out")

        if interrupt.requested:
            print(f"STOP: interrupted after round {round_n} completed. "
                  f"Last completed candidate: {candidate_path}")
            return

        if accuracy >= args.threshold:
            print(f"STOP: threshold met at round {round_n}. Final candidate: {candidate_path}")
            return

        if round_n == args.max_rounds:
            print(f"STOP: reached max rounds ({args.max_rounds}) without meeting threshold. "
                  f"Best-effort candidate: {candidate_path}")
            return

        hist_scored, hist_accuracy = score_candidate(candidate_path, history_records, **run_kwargs)
        failures = [s for s in hist_scored if not s["passed"]]
        if not failures:
            print("CONTINUE: no history failures to learn from, but held-out threshold not met — "
                  "this usually means the rule overfits the seed examples. Resampling next round's seed.")
            counterexamples = None
            continue
        counterexamples = select_counterexamples(failures, args.k)
        print(f"CONTINUE: history accuracy {hist_accuracy:.1%}, "
              f"{len(counterexamples)} counterexamples selected for round {round_n + 1}")


# ---------- CLI wiring ----------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Print trace structure without dumping raw values")
    p_inspect.add_argument("trace")
    p_inspect.add_argument("--pre-key", default="pre_observation")
    p_inspect.add_argument("--post-key", default="post_observation")
    p_inspect.add_argument("--frame-key", default="frame",
                            help="Key for the grid field within pre/post observation dicts")
    p_inspect.set_defaults(func=cmd_inspect)

    p_pre = sub.add_parser("preprocess", help="Collapse each row to {step, action, grid_before, grid_after}")
    p_pre.add_argument("trace")
    p_pre.add_argument("out")
    p_pre.add_argument("--pre-key", default="pre_observation")
    p_pre.add_argument("--post-key", default="post_observation")
    p_pre.add_argument("--frame-key", default="frame")
    p_pre.add_argument("--action-key", default="action")
    p_pre.add_argument("--score-key", default=None)
    p_pre.add_argument("--levels-key", default="levels_completed",
                        help="Sub-key of pre/post observation dicts holding the level counter, "
                             "extracted into levels_completed_before/levels_completed_after on "
                             "each cleaned record (needed by chunk boundary computation and "
                             "is_goal certification). On by default; pass --levels-key \"\" to "
                             "skip extraction if a trace doesn't carry this field.")
    p_pre.set_defaults(func=cmd_preprocess)

    p_split = sub.add_parser("split", help="Chronological 70/30 split into history / held-out")
    p_split.add_argument("trace")
    p_split.add_argument("history_out")
    p_split.add_argument("heldout_out")
    p_split.add_argument("--history-frac", type=float, default=0.7)
    p_split.add_argument("--shuffle", action="store_true",
                          help="Shuffle before splitting (each row is a self-contained "
                               "transition, so this is safe — order isn't needed for "
                               "the predict_next_state task)")
    p_split.add_argument("--seed", type=int, default=0)
    p_split.set_defaults(func=cmd_split)

    p_prompt = sub.add_parser("prompt", help="Build the predict_next_state synthesis prompt")
    p_prompt.add_argument("history")
    p_prompt.add_argument("out")
    p_prompt.add_argument("--max-examples", type=int, default=25,
                           help="Cap examples shown, spread across the trajectory (default 25)")
    p_prompt.add_argument("--compact", action="store_true",
                           help="Run-length-encode grid rows (e.g. '0*7 3*2') instead of "
                                "one hex char per cell — cuts tokens a lot for large/sparse grids")
    p_prompt.set_defaults(func=cmd_prompt)

    p_backtest = sub.add_parser("backtest", help="Full teacher-forced replay + scoring for the "
                                                  "chunked design (Step 6), class-based candidates only")
    p_backtest.add_argument("candidate", help="Path to a .py file defining class GameModel with a predict method")
    p_backtest.add_argument("trace", help="Cleaned jsonl trace file (chronological, no shuffle)")
    p_backtest.add_argument("--boundary", type=int, default=None,
                             help="Replay records[0:boundary] (default: the whole trace)")
    p_backtest.add_argument("--counts", default="row_failure_counts.json",
                             help="Path to the persistent per-row failure-tracking file (read+rewritten)")
    p_backtest.add_argument("--out", default=None, help="Write per-row scored results here")
    p_backtest.add_argument("--cpu-seconds", type=int, default=10)
    p_backtest.add_argument("--max-procs", type=int, default=16,
                             help="RLIMIT_NPROC cap for the candidate subprocess (per-UID, not per-tree)")
    p_backtest.add_argument("--mem-mb", type=int, default=512)
    p_backtest.add_argument("--per-call-seconds", type=int, default=2)
    p_backtest.add_argument("--overall-timeout", type=int, default=60)
    p_backtest.set_defaults(func=cmd_backtest)

    p_score = sub.add_parser("score", help="Sandboxed test of a candidate against any dataset")
    p_score.add_argument("candidate", help="Path to a .py file defining predict_next_state(grid, action) -> grid")
    p_score.add_argument("dataset", help="Cleaned jsonl file (history.jsonl or heldout.jsonl)")
    p_score.add_argument("--out", default=None, help="Write per-record pass/fail results here")
    p_score.add_argument("--cpu-seconds", type=int, default=10)
    p_score.add_argument("--max-procs", type=int, default=16,
                          help="RLIMIT_NPROC cap for the candidate subprocess (per-UID, not per-tree)")
    p_score.add_argument("--mem-mb", type=int, default=512)
    p_score.add_argument("--per-call-seconds", type=int, default=2)
    p_score.add_argument("--overall-timeout", type=int, default=60)
    p_score.set_defaults(func=cmd_score)

    p_eval = sub.add_parser("evaluate", help="Stop-condition check for the verify-and-revise loop")
    p_eval.add_argument("candidate")
    p_eval.add_argument("heldout")
    p_eval.add_argument("--round", type=int, required=True, help="Current round number (1-indexed)")
    p_eval.add_argument("--threshold", type=float, default=0.95, help="Held-out accuracy to stop at")
    p_eval.add_argument("--max-rounds", type=int, default=10)
    p_eval.add_argument("--log", default=None, help="Append this round's result as a json line here")
    p_eval.add_argument("--history", default=None,
                         help="If continuing, score against this to pick next-round counterexamples")
    p_eval.add_argument("--counterexamples-out", default=None,
                         help="Where to write next-round counterexamples if continuing")
    p_eval.add_argument("--k", type=int, default=10, help="Number of counterexamples to select if continuing")
    p_eval.add_argument("--cpu-seconds", type=int, default=10)
    p_eval.add_argument("--max-procs", type=int, default=16,
                         help="RLIMIT_NPROC cap for the candidate subprocess (per-UID, not per-tree)")
    p_eval.add_argument("--mem-mb", type=int, default=512)
    p_eval.add_argument("--per-call-seconds", type=int, default=2)
    p_eval.add_argument("--overall-timeout", type=int, default=60)
    p_eval.set_defaults(func=cmd_evaluate)

    p_revise = sub.add_parser("revise-prompt", help="Build a revision prompt from a candidate + counterexamples")
    p_revise.add_argument("candidate", help="Path to the current candidate .py file")
    p_revise.add_argument("counterexamples", help="Counterexamples jsonl written by `evaluate`")
    p_revise.add_argument("out", help="Where to write the revision prompt text")
    p_revise.add_argument("--compact", action="store_true",
                           help="Run-length-encode grid rows (e.g. '0*7 3*2') instead of "
                                "one hex char per cell — cuts tokens a lot for large/sparse grids")
    p_revise.set_defaults(func=cmd_revise_prompt)

    p_loop = sub.add_parser("run-loop", help="Fully automated verify-and-revise loop against a live LLM")
    p_loop.add_argument("history")
    p_loop.add_argument("heldout")
    p_loop.add_argument("--workdir", default="loop_run", help="Where per-round prompts/candidates are written")
    p_loop.add_argument("--backend", choices=["llama-cpp", "openai"], default="llama-cpp",
                         help="llama-cpp: load a local GGUF in-process via llama-cpp-python (default). "
                              "openai: call an OpenAI-compatible HTTP server instead (vLLM, llama-server).")
    p_loop.add_argument("--model-path", default=None,
                         help="[llama-cpp] path to the .gguf file, e.g. Qwen3-Coder-Next-UD-Q4_K_XL.gguf")
    p_loop.add_argument("--n-ctx", type=int, default=32768, help="[llama-cpp] context window to allocate")
    p_loop.add_argument("--n-gpu-layers", type=int, default=-1,
                         help="[llama-cpp] layers to offload to GPU; -1 = all")
    p_loop.add_argument("--verbose-llama", action="store_true",
                         help="[llama-cpp] show llama.cpp's own load-time log (confirms how many "
                              "layers actually landed on GPU vs. CPU — nvidia-smi memory alone "
                              "doesn't tell you that) plus per-call timing stats (prompt eval "
                              "tokens/sec, generation tokens/sec) after every round. Useful for "
                              "diagnosing an unexpectedly slow round without guessing.")
    p_loop.add_argument("--api-base", default="http://localhost:8000/v1",
                         help="[openai] OpenAI-compatible base URL")
    p_loop.add_argument("--model", default=None, help="[openai] model name as the server expects it")
    p_loop.add_argument("--temperature", type=float, default=0.2)
    p_loop.add_argument("--repeat-penalty", type=float, default=1.3,
                         help="[llama-cpp only] penalizes tokens already seen in the response so "
                              "far; llama.cpp's own default (1.1) and our earlier 1.15 both proved "
                              "insufficient to stop a real repetition loop we hit (model got stuck "
                              "restating the same reasoning until --max-tokens cut it off before "
                              "reaching a return statement) — 1.3 is a stronger starting point. No "
                              "equivalent field exists in the OpenAI chat-completions schema, so "
                              "--backend openai ignores this.")
    p_loop.add_argument("--presence-penalty", type=float, default=0.1,
                         help="Flat penalty applied to any token already used at all, regardless "
                              "of how many times (unlike --frequency-penalty, which scales with "
                              "repeat count). llama-cpp and openai backends both support this. "
                              "Kept small (0.1) — a higher value (we tried 0.3) stacked with "
                              "--repeat-penalty can make 'just stop' look increasingly attractive "
                              "the longer a response runs (more of the vocabulary has been used "
                              "and is now penalized), causing the model to end generation early "
                              "(finish_reason='stop') well before finishing the function — check "
                              "the finish_reason/completion_tokens line each round printed by this "
                              "command to see if that's happening before raising this further.")
    p_loop.add_argument("--frequency-penalty", type=float, default=0.1,
                         help="Extra per-repeat-token penalty that scales with how often a token "
                              "has already appeared (llama-cpp and openai backends both support "
                              "this). Works alongside --repeat-penalty against repetition loops; "
                              "kept small (0.1) so it doesn't distort legitimate repeated content "
                              "like grid digits or RLE run markers.")
    p_loop.add_argument("--max-tokens", type=int, default=4096)
    p_loop.add_argument("--max-examples", type=int, default=25, help="Seed examples for round 1")
    p_loop.add_argument("--compact", action="store_true",
                         help="Run-length-encode grid rows (e.g. '0*7 3*2') instead of "
                              "one hex char per cell — cuts tokens a lot for large/sparse grids")
    p_loop.add_argument("--threshold", type=float, default=0.95)
    p_loop.add_argument("--max-rounds", type=int, default=10)
    p_loop.add_argument("--k", type=int, default=10, help="Counterexamples per revision round")
    p_loop.add_argument("--log", default="rounds.jsonl")
    p_loop.add_argument("--automatic", action="store_true",
                         help="Run every round back-to-back with no pauses. Default (flag "
                              "off): after each round's prompt (seed or revision) is written "
                              "to --workdir, the loop stops and asks whether to continue — "
                              "before that round's LLM call is made — so you can open and "
                              "inspect/edit prompt_round{N}.txt first. Answer 'n' to abort "
                              "the run at that point; anything else (including a bare Enter) "
                              "continues.")
    p_loop.add_argument("--cpu-seconds", type=int, default=10)
    p_loop.add_argument("--max-procs", type=int, default=16,
                         help="RLIMIT_NPROC cap for the candidate subprocess (per-UID, not per-tree)")
    p_loop.add_argument("--mem-mb", type=int, default=512)
    p_loop.add_argument("--per-call-seconds", type=int, default=2)
    p_loop.add_argument("--overall-timeout", type=int, default=60)
    p_loop.set_defaults(func=cmd_run_loop)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()