#!/usr/bin/env python3
"""
trace_tools.py — chunked-curriculum program synthesis harness for Eidolon's
GameModel candidates (ARC-AGI-3).

Pipeline:
  1. inspect        Look at trace.jsonl structure without dumping raw grid values.
                     Specifically checks whether the frame field is a single grid
                     per record or an accumulating list-of-frames.
  2. preprocess     Collapse each record down to {step, action, grid_before,
                     grid_after, levels_completed_before/after,
                     available_actions_before} — one cleaned
                     transition per observation, dropping any redundant
                     history the raw FrameData carries.
  3. prompt         Build chunk 1's seed prompt (PROMPT_TEMPLATE) from the
                     first --max-examples rows of a records file, using a
                     compact grid encoding to keep token count down. Two
                     grid encodings are available: one-hex-char-per-cell
                     (default) or --compact (run-length encoded rows, e.g.
                     "0*7 3*2"), which cuts tokens further for large/sparse
                     grids. NOTE: grid content tokenizes far worse than
                     English prose (close to 1 token/char, not the usual ~4
                     chars/token), so the printed token estimate for
                     `prompt`/`revise-prompt` uses a 1:1 chars-to-tokens
                     fallback rather than chars/4 — it's still a rough
                     estimate since no real tokenizer is loaded on that path.
  4. backtest       Full teacher-forced replay of a candidate GameModel class
                     against a trace (or a row_range[0:boundary] slice of
                     one), scoring per-row with goal discounting (a
                     candidate that predicts a level-completion transition
                     has its grid prediction excluded from scoring, since
                     it can't know the next level's starting grid) and
                     updating the persistent row_failure_counts.json that
                     `revise-prompt`/`run-chunked` read. No LLM calls —
                     plain sandboxed Python execution.
  5. revise-prompt  Build a round-2+ prompt from a candidate's current code
                     plus row_failure_counts.json (written by `backtest`) —
                     selects the --k rows with the highest failure count
                     across the whole trace-so-far, shows the model what it
                     predicted vs. what actually happened for each, and
                     asks for a fix to the GameModel class rather than a
                     rewrite from scratch.
  6. run-chunked    The actual chunked-curriculum harness this project runs.
                     Loads ONE full cleaned trace file, chronologically, no
                     split. Drives an outer loop over chunks (each chunk's
                     boundary is whichever comes soonest of: a fixed
                     --max-examples row cap, a level completion, or the end
                     of the trace), each running an inner loop of up to
                     --max-rounds rounds (chunk 1: seed PROMPT_TEMPLATE;
                     every later chunk's round 1: EXTEND_TEMPLATE against a
                     freshly-recomputed baseline; every round after: the
                     row_failure_counts.json-driven REVISE_TEMPLATE),
                     accepting or rejecting each round's candidate against
                     whichever code came before it (never regressing more
                     than a small fixed tolerance) and stopping a chunk
                     early the moment it hits zero failing rows. Reports
                     after every chunk and persists whichever code each
                     chunk ends with as the next chunk's starting point.
                     Supports two backends: `llama-cpp` (llama-cpp-python
                     loading a local GGUF in-process — the default, and
                     what you'd use for a quantized model on Jarvislabs) or `openai` (an
                     OpenAI-compatible HTTP server, e.g. vLLM or
                     llama-server, if you go that route instead). Before
                     each round's LLM call, if a local tokenizer is
                     available (--backend llama-cpp), the actual prompt is
                     tokenized via the loaded model and checked against
                     --n-ctx minus --max-tokens, failing fast with a clear
                     message instead of letting llama-cpp raise a raw
                     "Requested tokens exceed context window" error
                     mid-generation. --backend openai has no local
                     tokenizer, so that preflight check is skipped and only
                     a char count is printed. --verbose-llama surfaces
                     llama.cpp's own load-time log (actual GPU-offloaded
                     layer count — GPU memory usage alone doesn't confirm
                     this) and per-call prompt-eval/generation
                     tokens-per-second, useful for diagnosing an
                     unexpectedly slow round. By default (no --automatic),
                     pauses before every round's LLM call so the written
                     prompt can be inspected/edited first.

Usage:
  python trace_tools.py inspect trace.jsonl --frame-key frame
  python trace_tools.py preprocess trace.jsonl clean.jsonl --frame-key frame --action-key action
  python trace_tools.py prompt clean.jsonl prompt.txt --max-examples 25
  python trace_tools.py backtest candidate.py clean.jsonl --boundary 50 --counts row_failure_counts.json
  python trace_tools.py revise-prompt candidate.py row_failure_counts.json revision_prompt.txt --k 10
  python trace_tools.py run-chunked clean.jsonl \\
      --backend llama-cpp --model-path /path/to/model.Q4_K_M.gguf \\
      --n-gpu-layers -1 --n-ctx 32768 --max-examples 25 --max-rounds 2 \\
      --compact --repeat-penalty 1.3 --frequency-penalty 0.1 --presence-penalty 0.1 \\
      --workdir chunked_run --automatic
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


HEX_DIGITS = "0123456789abcdef"


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


def format_flat_diff(changes):
    """
    Render a diff_grid() result as a 'row,col: before -> after' list, one
    changed cell per line, real integer values throughout (never the
    hex/RLE display shorthand — see ENCODING_EXPLANATION_HEX/RLE, which
    only ever describes how full GRIDS are shown, not this list).
    """
    return "\n".join(f"(row {r}, col {c}): {b} -> {a}" for r, c, b, a in changes)


def format_masked_diff(changes, shape_grid):
    """
    Render a diff_grid() result as a "masked grid": one line per row of
    `shape_grid` (used only for its dimensions), with runs of identical
    consecutive cells RLE-compressed — both unchanged runs ("*N": N
    consecutive cells that didn't change) and changed-value runs
    ("before>after*N": N consecutive cells that all changed from the same
    before value to the same after value). A single, non-repeated changed
    cell is just "before>after" with no "*1" suffix — the suffix only
    appears once a run has 2+ cells, since a bare "before>after" is
    already unambiguous on its own. All values use hex digits (the same
    0-15 alphabet ENCODING_EXPLANATION_HEX already describes for grids),
    never real integers — this is fundamentally a grid-shaped format, and
    mixing real integers into a grid-shaped context would be more
    confusing than staying consistent with how every other grid in the
    prompt is shown.

    Exists alongside format_flat_diff as the second of two candidates
    format_diff picks between — see format_diff's docstring for why picking
    dynamically beats a fixed cell-count threshold. This representation
    wins specifically when a diff is sparse (few changed cells scattered
    across a mostly-unchanged grid): the RLE runs collapse long unchanged
    OR uniformly-changed stretches to a few characters each, which a flat
    per-cell list cannot do, while still preserving the 2D spatial layout
    of what changed (a flat list of (row, col) pairs doesn't show a moving
    object's shape the way a masked grid naturally does).

    Deliberately does NOT also RLE-compress runs of identical ROW STRINGS
    (e.g. many consecutive "*64" rows collapsing to one "*64 x40" line) —
    that's a real further reduction (measured independently), but adds a
    second, distinct compression axis for the model to track correctly on
    top of this one, and its benefit is shape-dependent in a way this
    per-row compression isn't (a curved/irregular moving object's own rows
    rarely repeat exactly, even though the background surrounding it
    usually does). Holding off on that specific addition until there's a
    concrete signal that prompt format complexity — as opposed to model
    capability — is actually the bottleneck worth spending it on.
    """
    changed = {(r, c): (b, a) for r, c, b, a in changes}
    lines = []
    for r, row in enumerate(shape_grid):
        # First pass: one raw token per cell (None for unchanged, else the
        # "before>after" hex pair) — kept separate from the RLE pass below
        # so the same run-collapsing logic applies uniformly to both kinds
        # of run instead of needing two different code paths.
        raw_tokens = []
        for c in range(len(row)):
            if (r, c) in changed:
                b, a = changed[(r, c)]
                raw_tokens.append(f"{HEX_DIGITS[b]}>{HEX_DIGITS[a]}")
            else:
                raw_tokens.append(None)

        tokens = []
        i = 0
        while i < len(raw_tokens):
            j = i
            while j < len(raw_tokens) and raw_tokens[j] == raw_tokens[i]:
                j += 1
            run_len = j - i
            if raw_tokens[i] is None:
                tokens.append(f"*{run_len}")
            elif run_len > 1:
                tokens.append(f"{raw_tokens[i]}*{run_len}")
            else:
                tokens.append(raw_tokens[i])
            i = j
        lines.append(" ".join(tokens))
    return "\n".join(lines)


def format_diff(changes, shape_grid):
    """
    Render a diff_grid() result using whichever of two candidate formats is
    actually shorter for THIS specific diff: format_flat_diff (a real-int
    per-cell list) or format_masked_diff (a masked grid with RLE-compressed
    unchanged runs). Returns (format_name, text) — callers embed
    format_name inline next to the diff so each block is self-describing
    regardless of which format won, rather than relying on the model to
    remember a single "sometimes it looks like X, sometimes like Y"
    explanation from earlier in a long prompt.

    Deliberately NOT a fixed cell-count threshold (the previous design,
    DIFF_MAX_CELLS): measured against a real 400-row ls20 trace, the
    crossover point between these two formats turned out to depend on
    things a fixed number can't capture — grid dimensions (format_masked_diff's
    per-row "*N" overhead scales with grid height), and how clustered vs.
    scattered the changes are (RLE compresses clustered changes far better
    than scattered ones). Comparing actual rendered length per diff is
    cheap (plain string-length comparisons, not model calls) and is
    correct by construction for any game's grid size and change pattern,
    with no per-game tuning required.

    shape_grid must be a grid the same shape as the one the diff was
    computed against (grid_after for a normal before/after transition
    diff; the correct/actual grid for a build_revise_row_block prediction-
    vs-correct diff) — only its dimensions are used, not its values (every
    changed cell's actual before/after values already come from `changes`
    itself).
    """
    if not changes:
        return "flat", "no cells changed (identity transformation for this example)"
    flat = format_flat_diff(changes)
    masked = format_masked_diff(changes, shape_grid)
    if len(masked) < len(flat):
        return "masked-grid", masked
    return "flat-list", flat




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
        # Deliberately no writable location anywhere in the sandbox. An
        # earlier version mounted `--tmpfs /tmp` for general scratch space,
        # but the candidate contract (predict() is a pure function: input
        # via stdin, output via stdout/return value, never touches disk)
        # has no legitimate need for one, and a writable /tmp is exactly
        # what lets an allowed-but-untrusted library like numpy write
        # arbitrarily large files via numpy.save()/ndarray.tofile()/
        # memmap() -- calls that use numpy's OWN internal open(), not the
        # candidate's restricted builtins (see RUNNER_TEMPLATE), so no
        # Python-level restriction can catch them. RLIMIT_FSIZE would cap
        # any single write but still allow it to happen; not creating a
        # writable inode anywhere removes the capability at the mount
        # level instead, which no library-internal code path can route
        # around. --dev below does NOT implicitly create a writable
        # /dev/shm (that needs its own explicit --tmpfs), so this closes
        # the other common "second /tmp" people forget to check for.
        # Every remaining path in this sandbox is either read-only
        # (/usr, /lib, /bin, the conda env, /work) or a namespaced
        # synthetic filesystem with no general write surface (/proc, /dev).
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


MAX_CANDIDATE_OUTPUT_BYTES = 64 * 1024 * 1024  # 64MB across one whole
# sequential run-loop invocation (all rows in one `backtest`/round). Far
# more than any real ARC grid trace needs, but small enough to actually
# bound host memory pressure from a runaway or malicious candidate.


def run_with_output_cap(exec_cmd, stdin_data, env, overall_timeout,
                         max_output_bytes=MAX_CANDIDATE_OUTPUT_BYTES):
    """
    Like `subprocess.run(capture_output=True, timeout=overall_timeout)`,
    but also kills the child and stops reading once total stdout+stderr
    crosses max_output_bytes.

    `capture_output=True` alone buffers output with no size limit at all.
    That's fine for a well-behaved candidate, but run_candidate replays
    every row of a chunk in ONE subprocess invocation (see the runner's
    per-row loop in RUNNER_TEMPLATE) — a candidate that prints a
    chunky-but-legal JSON line every row can accumulate far more total
    output than its own RLIMIT_AS over the course of a long replay, since
    each row's data is written to the pipe and freed before the next row
    is produced; the process's resident memory never has to hold more
    than one row's worth at a time even as cumulative output climbs into
    the hundreds of MB. RLIMIT_AS bounds a snapshot, not a running total,
    so it does not catch this. This does, by tracking bytes actually read
    off the pipe and killing the child immediately once the cap is
    crossed, from two reader threads (stdout, stderr) that run
    concurrently with waiting on the process so a candidate blocked on a
    full pipe buffer can't stall out the overall_timeout enforcement below.

    Returns an object with .returncode, .stdout, .stderr — the subset of
    subprocess.run's result run_candidate actually uses. Raises
    subprocess.TimeoutExpired on timeout, matching subprocess.run's own
    behavior, so callers don't need to change their except clause.
    """
    import threading

    proc = subprocess.Popen(
        exec_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=env,
    )

    chunks = {"stdout": [], "stderr": []}
    total_bytes = 0
    killed_for_size = False
    lock = threading.Lock()

    def _reader(stream, key):
        nonlocal total_bytes, killed_for_size
        try:
            for chunk in iter(lambda: stream.read(65536), ""):
                if not chunk:
                    break
                with lock:
                    chunks[key].append(chunk)
                    total_bytes += len(chunk.encode("utf-8", errors="ignore"))
                    over_cap = total_bytes > max_output_bytes
                if over_cap and not killed_for_size:
                    killed_for_size = True
                    proc.kill()
                    break
        except (ValueError, OSError):
            pass  # stream closed under us because the process was killed

    t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout"))
    t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr"))
    t_out.start()
    t_err.start()

    try:
        if stdin_data:
            proc.stdin.write(stdin_data)
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass  # candidate process may have already exited or been killed

    try:
        proc.wait(timeout=overall_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        t_out.join(timeout=5)
        t_err.join(timeout=5)
        raise subprocess.TimeoutExpired(exec_cmd, overall_timeout)

    t_out.join(timeout=5)
    t_err.join(timeout=5)

    class _Result:
        pass

    result = _Result()
    result.returncode = proc.returncode
    result.stdout = "".join(chunks["stdout"])
    result.stderr = "".join(chunks["stderr"])
    if killed_for_size:
        result.stderr += (
            f"\n[host] candidate output exceeded {max_output_bytes} bytes "
            f"across this run; process was killed."
        )
    return result


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
import sys, json, resource, signal, builtins as _builtins_module

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

# Restrict the builtins visible inside the candidate's OWN exec()'d
# namespace. check_ast_imports only inspects `import`/`from` statements at
# the host level, so it never sees a candidate reaching for the same
# capability through a bare builtin instead: open('/etc/passwd'),
# __import__('os'), eval(...), exec(...), compile(...). Everything except
# __import__ is safe to delete outright, since nothing in ordinary
# candidate code calls them implicitly -- a candidate that never wrote
# `eval` in its source will never invoke it by accident. __import__ is
# different: CPython's IMPORT_NAME bytecode op calls builtins.__import__
# under the hood for EVERY `import` statement, including the ones on
# ALLOWED_IMPORTS the candidate is supposed to have (numpy, copy,
# itertools, ...) -- deleting it outright would break every legitimate
# import too. So it's wrapped with the same allowlist check instead of
# removed, closing the __import__('os')-style bypass while leaving
# `import numpy` working.
#
# This is a courtesy filter for a careless candidate, not the real
# security boundary -- bwrap's namespace/capability isolation is (see
# build_bwrap_command). A determined adversary already inside this
# process can still reach live module objects through class-introspection
# gadgets that never touch __builtins__ at all (e.g. walking
# ().__class__.__mro__[-1].__subclasses__() to find an object whose type
# already has `os` bound in its closure, since numpy's own import graph
# pulls `os` into this process regardless of what candidate code does).
# Nothing at the Python level can close that; only the sandbox can.
_ALLOWED_IMPORTS_RUNTIME = set({allowed_imports!r})
_real_import = _builtins_module.__import__

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in _ALLOWED_IMPORTS_RUNTIME:
        raise ImportError(f"import of {{name!r}} is not allowed in a candidate")
    return _real_import(name, globals, locals, fromlist, level)

_safe_builtins = dict(vars(_builtins_module))
_safe_builtins["__import__"] = _guarded_import
for _name in ("open", "eval", "exec", "compile", "input", "breakpoint", "exit", "quit"):
    _safe_builtins.pop(_name, None)

candidate_globals = {{"__name__": "candidate", "__builtins__": _safe_builtins}}
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
# grid level, so a candidate never has to compound its own past mistakes) —
# a wrong prediction never contaminates the next row's grid input, only
# `state` can drift.
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
        # Deliberately abort the whole sequential pass on any error rather
        # than carrying the last-good state forward and continuing — a
        # crash means `state` is now unknown/untrustworthy, and silently
        # continuing with stale state would make every later row's result
        # look like a legitimate prediction instead of what it actually is
        # (built on data the candidate never really produced). Emit this
        # row as an error and stop right here instead. Every row after this
        # one is simply never emitted; the parent process's scoring treats
        # "no result for this step" as incorrect/missing.
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
    attempt is made to keep going with stale state, since a crash means
    `state` can no longer be trusted — so every row from the failure point
    onward is simply absent from the
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
            class_name=class_name, allowed_imports=sorted(ALLOWED_IMPORTS),
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
            proc = run_with_output_cap(
                exec_cmd, stdin_data, sandbox_env, overall_timeout,
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
        elif "error" in r:
            # RUNNER_TEMPLATE prints exactly this shape (no "step" key) for
            # the two failures that happen BEFORE the per-row loop even
            # starts -- class_name not defined in the candidate at all, or
            # defined but failed to instantiate -- since there's no `rec`
            # in scope yet at that point to attach a step to. Surfacing it
            # here as a real, loud RuntimeError instead of silently
            # dropping it matters: without this, every row would instead
            # report the generic, WRONG "no result (candidate crashed/timed
            # out earlier in this replay)" (see run_backtest), completely
            # hiding that the actual cause was "no usable class was ever
            # defined" -- a 0%-accuracy round with no visible explanation,
            # not a crash, so nothing would stop the run to surface it.
            raise RuntimeError(f"candidate runner failed before processing any rows: {r['error']}")
    return results


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
    Used by cmd_run_chunked to pause after each round's prompt (seed/extend
    or revision) is written to disk but BEFORE it's sent to the LLM — gives
    you a chance to open the file and eyeball/edit it first. Any answer other than explicit
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
    Context manager for the risky span of a round that's actually wrapped in
    practice: the LLM call plus candidate extraction/write (see
    make_round_builder) — the one part of a round that's genuinely expensive
    to redo (an LLM generation that can take minutes on local GPU
    inference). It does NOT cover the backtest that follows or the log
    write at the end of a chunk — both happen later, outside any `with
    GracefulInterrupt()` block, in run_chunk_rounds/cmd_run_chunked. See
    "If a Ctrl+C lands during the backtest instead" below for what that
    actually means in practice; it's a real gap, just a bounded one.

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
    returns early with a nonzero/negative return code, which run_candidate
    surfaces as a RuntimeError. cmd_run_chunked deliberately does NOT catch
    this (see _validate_candidate_code's docstring) — it propagates and
    halts the run, same as any other genuine subprocess/infra failure,
    rather than being silently absorbed into a fake failed round. Whatever
    partial chunk state that leaves is the same disk-consistency guarantee
    described above: no file involved is ever left half-written, only
    possibly missing.

    If a Ctrl+C lands during the backtest instead (i.e. after this class's
    `with` block has already exited): there's no graceful handling at all
    at that point — it's an ordinary KeyboardInterrupt, uncaught anywhere in
    the call chain, which crashes cmd_run_chunked immediately with a raw
    traceback instead of the clean "STOP: interrupted after chunk N
    completed" message a Ctrl+C during the LLM call gets. The actual
    consequences are bounded, though: row_failure_counts.json is written in
    one shot at the very end of a successful run_backtest call (see
    atomic_write_json), so an interrupt mid-backtest never corrupts it or
    leaves it half-updated — it's simply untouched this round, exactly as
    it was after the previous round/chunk. What's actually lost is the
    CURRENT chunk's in-memory progress (every round completed so far within
    that one chunk) — every chunk before it already has its
    chunk{N}_final.py and chunk_log.jsonl entry safely persisted, since
    those are only written after a chunk fully finishes. There's no
    resume/checkpoint feature in this harness regardless of where a Ctrl+C
    lands, though, so recovering from this means restarting run-chunked
    from chunk 1 either way — this gap only changes HOW abruptly that
    happens (a crash instead of a clean stop), not whether you can pick up
    partway through.
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
# Pure functions, no LLM/sandbox involved — the trace is processed in
# chronological chunks, each ending at whichever of three cutoffs comes
# soonest.
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
    Compute where the next chunk ends. Three cutoffs compete; whichever is
    soonest (smallest) wins:
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


# ---------- backtest: full replay, scoring, goal discounting ----------
# Full teacher-forced replay of a candidate against a trace, with
# persistent per-row failure tracking across rounds/chunks (see
# run_backtest and load_row_failure_counts below).

def atomic_write_json(path, data):
    """
    Write `data` as JSON to `path` via temp-file-then-os.replace(), so a
    process interruption mid-write can never leave `path` corrupted or
    half-written. Shared by run_backtest (row_failure_counts.json) and the
    per-chunk round loop's commit/revert snapshot (row_failure_counts_best.json).
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
    Load the persistent per-row failure-tracking file, or an empty dict if
    it doesn't exist yet (the very first backtest call of a run). Keys are
    trace step indices stored as strings (JSON object keys are always
    strings) -- callers should key lookups with str(step).
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def actual_goal_for_record(rec):
    """
    Ground truth for goal-reached on a single row: levels_completed_after >
    levels_completed_before, both extracted per-row by cmd_preprocess. False
    if either field is missing -- a trace that never carried
    levels_completed simply never certifies a goal transition, the same
    graceful "no signal" treatment as next_level_completion_row uses.
    """
    before = rec.get("levels_completed_before")
    after = rec.get("levels_completed_after")
    if before is None or after is None:
        return False
    return after > before


def grid_shape_error(grid):
    """
    Return a short reason string if `grid` isn't a well-formed grid (a
    non-empty list of non-empty, equal-length rows of plain ints), else
    None. A candidate's predict() has no fixed return-type enforcement --
    it could hand back a ragged list, a numpy array, a list containing
    None/str cells, etc. Without this check, that malformed value would
    ride along as "prediction" all the way to encode_grid when a later
    revision prompt gets built (see cmd_revise_prompt/build_prompt), where
    it fails as an unrelated-looking exception deep in prompt construction
    instead of a clear per-row result here. Called right after a
    prediction is pulled out of the runner's stdout, so a malformed
    prediction is scored exactly like a crashed/timed-out row -- a
    failure, not a crash of the whole backtest.
    """
    if not isinstance(grid, list) or not grid:
        return "prediction is not a non-empty list of rows"
    width = None
    for row in grid:
        if not isinstance(row, list) or not row:
            return "a row is not a non-empty list"
        for cell in row:
            if not isinstance(cell, int) or isinstance(cell, bool):
                return "a row contains a non-int cell"
        if width is None:
            width = len(row)
        elif len(row) != width:
            return "rows have inconsistent width"
    return None


def run_backtest(candidate_path, records, boundary, counts_path, **run_kwargs):
    """
    Full teacher-forced replay of records[0:boundary] against candidate_path
    (run_candidate's sequential/stateful runner), with the persistent
    per-row row_failure_counts.json read at the start and rewritten at the
    end.

    Per-row scoring:
      - A row that crashed, timed out, or was never reached (run_candidate's
        abort-on-crash cut the replay short before this row) always counts
        as a failure. No grid/goal comparison is attempted for it.
      - predicted_goal == actual_goal is always required for a pass — both
        false positives and false negatives count as incorrect, no
        exemption.
      - GOAL-DISCOUNTING DECISION (explicit, rather than left ambiguous):
        when predicted_goal is True, the grid comparison is EXCLUDED
        ENTIRELY from the pass/fail decision for that row — not
        scored-but-flagged. The candidate cannot know the next level's
        initial grid, so holding a wrong predicted_grid_after against it on
        a row it itself flagged as a level transition isn't meaningful.
        Concretely:
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
    since a revision prompt may still want a discounted row's prediction
    later), and "error" (if the row crashed or was never reached).

    row_failure_counts.json is read-mutated-written UNCONDITIONALLY on
    every call, including rounds a caller later rejects — this function
    does not know or care about accept/reject. run_chunk_rounds owns the
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
            predicted_grid = result["prediction"]
            predicted_goal = bool(result["goal"])
            shape_error = grid_shape_error(predicted_grid)
            if shape_error:
                error = f"malformed prediction: {shape_error}"
                predicted_grid = None
            else:
                error = None

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
            # per row across the whole run) -- actual_grid/actual_goal/
            # available_actions_before are ground truth and never change,
            # so they're set here and never touched again below.
            # available_actions_before uses .get() rather than direct
            # indexing since older cleaned traces (preprocessed before
            # --actions-key existed) won't carry this key at all --
            # format_row_available_actions handles a None gracefully.
            counts[step_key] = {
                "count": 0,
                "actual_grid": actual_grid,
                "actual_goal": actual_goal,
                "available_actions_before": rec.get("available_actions_before"),
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


# ---------- per-chunk epsilon-reset code selection with baseline recompute ----------
# Replaces an earlier draft's continuous best-so-far checkpoint. Two problems solved
# simultaneously: never let a single revision regress the candidate
# drastically without any check, and never let the candidate get stuck
# reverting to old code that ignores new rows it hasn't learned to handle
# yet, simply because old code's accuracy looks artificially better on a
# smaller row range than a new candidate scored on a larger one. Solved by
# only ever comparing accuracies measured on the SAME row range, and
# resetting which comparison is "live" at each chunk boundary rather than
# tracking one continuous global best.

EPSILON = 0.05  # flat constant -- every chunk, every round, never scaled by chunk size


def atomic_copy_json(src_path, dst_path):
    """
    Copy src_path's JSON content to dst_path via atomic_write_json's
    temp-file-then-os.replace() pattern -- NOT a plain in-place file copy,
    so a process interruption mid-copy can never leave dst_path corrupted
    or half-written. Used for both directions of the
    row_failure_counts.json <-> row_failure_counts_best.json commit/revert
    in run_chunk_rounds below.

    A missing src_path (e.g. reverting before any round has ever been
    accepted) is treated as an empty counts file, consistent with
    load_row_failure_counts's own missing-file handling, rather than
    raising.
    """
    src_path = Path(src_path)
    if src_path.exists():
        with open(src_path) as f:
            data = json.load(f)
    else:
        data = {}
    atomic_write_json(dst_path, data)


def _write_temp_candidate_file(code):
    """Materialize a candidate code string to a throwaway temp .py file for run_backtest, which needs a path."""
    fd, path = tempfile.mkstemp(suffix="_candidate.py", prefix="eidolon_chunk_round_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
    except BaseException:
        os.unlink(path)
        raise
    return path


def run_chunk_rounds(round_builder, records, boundary, counts_path, best_counts_path,
                      baseline_code=None, max_rounds=2, **run_kwargs):
    """
    Per-chunk epsilon-reset round loop: drives up to max_rounds rounds of
    extend/revise + backtest + accept/reject for ONE chunk, keeping
    row_failure_counts.json (`counts_path`) and row_failure_counts_best.json
    (`best_counts_path`) in lockstep with whichever candidate is genuinely
    current-best.

    round_builder(round_n, current_best_code) -> candidate_code is supplied
    by the caller (the outer per-chunk loop, cmd_run_chunked), not built in
    here, so this stays testable without a real LLM call. round_n starts at
    1: round 1 should build EXTEND_TEMPLATE (chunk 1: PROMPT_TEMPLATE) from
    current_best_code; every round after should build REVISE_TEMPLATE from
    current_best_code and the CURRENT row_failure_counts.json on disk at
    counts_path (which this function keeps in lockstep with
    current_best_code between calls -- see below). current_best_code is
    None only on chunk 1's round 1 call (no prior code to show).

    baseline_code: chunk k>1's prior current_best_code, re-backtested
    against THIS chunk's (larger) row range before round 1 -- a "baseline
    recompute", required because that code was previously only ever scored
    against chunk k-1's smaller row range, so no fair same-denominator
    comparison exists yet without it. This recompute is itself always an
    automatic "accept" (a re-measurement of already-accepted code, not a
    new candidate being proposed), so it commits immediately and becomes
    what round 1 compares against and, if round 1 is rejected, what round
    1's revert restores to. Pass None for chunk 1, which has no prior code
    and skips this step entirely -- its round 1 is simply accepted outright
    (nothing to compare against).

    Every round's backtest mutates counts_path in place as it scores each
    row (see run_backtest). A round's mutations must not survive if that
    round is rejected, or a later revision prompt in this chunk would show
    predicted_grid/predicted_goal from a discarded candidate rather than
    the code actually still in play:
      - ACCEPT (round's candidate becomes current-best, including the
        automatic chunk-1/baseline-recompute/zero-failure cases): copy
        counts_path -> best_counts_path, committing this round's mutations
        as the new baseline-for-reverting-to.
      - REJECT: copy best_counts_path -> counts_path, discarding this
        round's mutations entirely and restoring the file to exactly what
        it was after the last accepted round.
    A single "best" snapshot is sufficient (not a per-round history stack)
    because a reject only ever needs to undo the IMMEDIATELY PRECEDING
    round's mutations -- the comparison chain never skips ahead or compares
    non-adjacent rounds (each round is only ever compared against whichever
    code was current-best going into it), so counts_path is always at most
    one round's worth of mutations away from being correct.

    Zero-failure early stop: after ANY round's own backtest (never the
    baseline recompute) comes back with zero failing rows across the full
    chunk row range, that round is accepted (a zero-failure candidate can
    never be more than epsilon worse than anything, so this is never a
    special case -- it's a trivial consequence of the accept check below)
    and the loop stops immediately, regardless of how many rounds of
    max_rounds remain.

    **run_kwargs forwarded to run_backtest -> run_candidate (cpu_seconds,
    mem_mb, use_sandbox, etc).

    Returns:
      {
        "current_best_code": str,        # this chunk's final code
        "current_best_accuracy": float,
        "current_best_scored": [...],     # the winning round's/baseline recompute's own "scored" list (run_backtest) -- feed straight into summarize_scores/print_score_summary, never re-run
        "current_best_streak": int,       # the winning round's/baseline recompute's own longest-pass-streak -- reported per chunk, never fed into a prompt
        "rounds_run": int,                # how many rounds of the ROUND LOOP actually ran (excludes the baseline recompute, which isn't a "round")
        "early_stopped": bool,
        "rounds_skipped": int,            # max_rounds - rounds_run if early_stopped else 0
        "log": [...],                     # one dict per event (baseline_recompute/accept/reject/early_stop) -- raw material for reporting, not printed here
      }

    On return, counts_path is guaranteed to exactly match current_best_code's
    own backtest state -- every branch above maintains this as an invariant,
    round by round (and the baseline recompute, chunk k>1, establishes it
    before round 1 even runs).
    """
    log = []
    current_best_code = None
    current_best_accuracy = None
    current_best_scored = None  # feeds summarize_scores/print_score_summary directly, once a chunk finishes
    current_best_streak = None  # reported per chunk, never fed into a prompt

    def backtest_code(code):
        path = _write_temp_candidate_file(code)
        try:
            return run_backtest(path, records, boundary, counts_path, **run_kwargs)
        finally:
            os.unlink(path)

    if baseline_code is not None:
        result = backtest_code(baseline_code)
        current_best_code = baseline_code
        current_best_accuracy = result["accuracy"]
        current_best_scored = result["scored"]
        current_best_streak = result["streak"]
        atomic_copy_json(counts_path, best_counts_path)  # baseline recompute is always an automatic "accept"
        log.append({
            "event": "baseline_recompute",
            "accuracy": current_best_accuracy,
            "n_pass": result["n_pass"],
            "n_total": result["n_total"],
            "streak": result["streak"],
        })

    round_n = 0
    early_stopped = False
    for round_n in range(1, max_rounds + 1):
        candidate_code = round_builder(round_n, current_best_code)
        result = backtest_code(candidate_code)
        candidate_accuracy = result["accuracy"]

        prev_best_accuracy = current_best_accuracy  # snapshot before this round's decision
        # Allowed to go negative (e.g. prev_best_accuracy=0.02) and needs no
        # clamping/special-casing: candidate_accuracy (always >= 0) then
        # trivially clears the threshold, which is exactly the desired
        # behavior for an already near-zero baseline (there's essentially
        # nothing left to protect).
        threshold = None if prev_best_accuracy is None else prev_best_accuracy - EPSILON
        accept = threshold is None or candidate_accuracy >= threshold

        if accept:
            current_best_code, current_best_accuracy = candidate_code, candidate_accuracy
            current_best_scored = result["scored"]
            current_best_streak = result["streak"]
            atomic_copy_json(counts_path, best_counts_path)  # commit this round's mutations
            log.append({
                "event": "accept", "round": round_n,
                "candidate_accuracy": candidate_accuracy,
                "prev_best_accuracy": prev_best_accuracy, "threshold": threshold,
                "n_pass": result["n_pass"], "n_total": result["n_total"], "streak": result["streak"],
            })
        else:
            atomic_copy_json(best_counts_path, counts_path)  # revert: discard this round's mutations
            log.append({
                "event": "reject_substitution", "round": round_n,
                "candidate_accuracy": candidate_accuracy,
                "prev_best_accuracy": prev_best_accuracy, "threshold": threshold,
                "n_pass": result["n_pass"], "n_total": result["n_total"], "streak": result["streak"],
            })
            # The code we just reverted BACK TO might independently already
            # be zero-failure -- this can only happen via a baseline
            # recompute (the one accepted state never itself checked against
            # the early-stop condition below, since baseline recomputes are
            # deliberately exempt from it) that this round's rejected
            # revision attempt failed to improve on. It's a fact about the
            # REVERTED-TO state, not about `result` (this round's own
            # candidate, which -- since a zero-failure round is always
            # accepted, never rejected -- must have had failures of its
            # own). Checked explicitly, right here, so a next round is never
            # even attempted against a target that already has nothing left
            # to fix -- catching it here means build_revise_prompt can never
            # be asked to build a revision prompt with zero counterexamples
            # to show.
            reverted_counts = load_row_failure_counts(counts_path)
            if not any(entry.get("count", 0) > 0 for entry in reverted_counts.values()):
                early_stopped = True
                log.append({"event": "early_stop", "round": round_n, "rounds_skipped": max_rounds - round_n})
                break

        if not result["failures"]:  # zero failing rows across the FULL chunk row range this round
            early_stopped = True
            log.append({"event": "early_stop", "round": round_n, "rounds_skipped": max_rounds - round_n})
            break

    return {
        "current_best_code": current_best_code,
        "current_best_accuracy": current_best_accuracy,
        "current_best_scored": current_best_scored,
        "current_best_streak": current_best_streak,
        "rounds_run": round_n,
        "early_stopped": early_stopped,
        "rounds_skipped": (max_rounds - round_n) if early_stopped else 0,
        "log": log,
    }


# ---------- reporting ----------
# Everything here is READ-ONLY over run_chunk_rounds' already-finished
# decisions: it consumes the returned "log" list and
# "current_best_scored"/"current_best_streak", and never feeds anything
# back into code selection. No new scoring/comparison logic lives here —
# only formatting and a running-best bookkeeping value.

def _fmt_pct(x):
    """None-safe percent formatter — chunk 1's round 1 has no baseline/threshold to show."""
    return "n/a" if x is None else f"{x:.1%}"


def log_code_replacement(chunk_number, entry):
    """
    One explicit line per ACCEPT event that replaces the chunk's current-
    best code with a genuinely new candidate — makes replacement frequency
    easy to scan on its own, independent of the full per-chunk detail in
    print_chunk_report. Deliberately NOT called for a baseline-recompute
    accept (see log_chunk_events) — recomputing an old candidate's score
    against a new row range isn't "code changing", it's re-measuring code
    that was already current going into this chunk.
    """
    print(f"[chunk {chunk_number}] CODE REPLACED at round {entry['round']}: "
          f"{entry['candidate_accuracy']:.1%} accuracy "
          f"(prev best {_fmt_pct(entry['prev_best_accuracy'])}, threshold {_fmt_pct(entry['threshold'])})")


def log_substitution(chunk_number, entry):
    """
    One explicit line per REJECT event: the rejected round's candidate is
    discarded outright, and whichever code was current-best going into this
    round is carried forward unchanged — this is the log line that
    documents that carry-forward.
    """
    print(f"[chunk {chunk_number}] SUBSTITUTION at round {entry['round']}: rejected "
          f"{entry['candidate_accuracy']:.1%} accuracy candidate "
          f"(more than epsilon={EPSILON:.0%} below prev best {_fmt_pct(entry['prev_best_accuracy'])}, "
          f"threshold {_fmt_pct(entry['threshold'])}) — current-best carried forward unchanged")


def log_early_stop(chunk_number, entry):
    """
    One explicit line per zero-failure early-stop event — a "nothing left
    to fix" event, distinct from the code-replacement line above even
    though a zero-failure round is always ALSO an accept (zero failures
    trivially clears any threshold, since a threshold measures how much
    worse a candidate is allowed to be).
    """
    print(f"[chunk {chunk_number}] EARLY STOP at round {entry['round']}: zero failing rows, "
          f"skipping remaining {entry['rounds_skipped']} round(s)")


def log_chunk_events(chunk_number, chunk_result):
    """
    Stream one explicit line per event in a finished chunk's log, in the
    order they actually happened. This is what actually produces the
    explicit code-replacement and early-stop lines — replayed after the
    fact from run_chunk_rounds' already-tested structured log, rather than
    requiring run_chunk_rounds itself to print anything (kept that function
    pure/silent and unit-testable on stdout).
    """
    for entry in chunk_result["log"]:
        if entry["event"] == "baseline_recompute":
            print(f"[chunk {chunk_number}] baseline recompute: {entry['accuracy']:.1%} "
                  f"({entry['n_pass']}/{entry['n_total']}, streak {entry['streak']})")
        elif entry["event"] == "accept":
            log_code_replacement(chunk_number, entry)
        elif entry["event"] == "reject_substitution":
            log_substitution(chunk_number, entry)
        elif entry["event"] == "early_stop":
            log_early_stop(chunk_number, entry)


def build_chunk_report(chunk_number, prev_boundary, boundary, chunk_result, max_rounds):
    """
    Turn one finished chunk's run_chunk_rounds() return dict into a
    structured report entry: chunk number, row range covered, baseline
    accuracy (if applicable, chunk 1 has none), per-round accuracy/
    accept-reject/epsilon for every round that actually ran, whether/where
    the zero-failure early stop fired, and which round's code the chunk
    ends with. Purely a reshaping of chunk_result — no new computation, no
    influence on which code chunk_result picked.
    """
    log = chunk_result["log"]
    baseline_entry = next((e for e in log if e["event"] == "baseline_recompute"), None)
    round_entries = [e for e in log if e["event"] in ("accept", "reject_substitution")]
    early_stop_entry = next((e for e in log if e["event"] == "early_stop"), None)

    return {
        "chunk": chunk_number,
        "row_range": (prev_boundary, boundary),
        "n_rows": boundary - prev_boundary,
        "max_rounds": max_rounds,
        "epsilon": EPSILON,
        "baseline_accuracy": baseline_entry["accuracy"] if baseline_entry is not None else None,
        "rounds": [
            {
                "round": e["round"],
                "accuracy": e["candidate_accuracy"],
                "prev_best_accuracy": e["prev_best_accuracy"],
                "threshold": e["threshold"],
                "decision": "accept" if e["event"] == "accept" else "reject",
            }
            for e in round_entries
        ],
        "early_stop": (
            {"round": early_stop_entry["round"], "rounds_skipped": early_stop_entry["rounds_skipped"]}
            if early_stop_entry is not None else None
        ),
        "rounds_run": chunk_result["rounds_run"],
        "final_accuracy": chunk_result["current_best_accuracy"],
        "streak": chunk_result["current_best_streak"],
    }


def print_chunk_report(report):
    """Human-readable rendering of one build_chunk_report() entry."""
    lo, hi = report["row_range"]
    print(f"\n=== Chunk {report['chunk']}: rows [{lo}:{hi}) ({report['n_rows']} rows), "
          f"epsilon={report['epsilon']:.0%}, max_rounds={report['max_rounds']} ===")
    print(f"  baseline: {_fmt_pct(report['baseline_accuracy'])}"
          + ("" if report["baseline_accuracy"] is not None else " (chunk 1 — no prior code)"))
    for r in report["rounds"]:
        marker = "ACCEPT" if r["decision"] == "accept" else "REJECT"
        print(f"  round {r['round']}: {r['accuracy']:.1%} vs prev-best {_fmt_pct(r['prev_best_accuracy'])} "
              f"(threshold {_fmt_pct(r['threshold'])}) -> {marker}")
    if report["early_stop"] is not None:
        print(f"  early stop at round {report['early_stop']['round']} "
              f"({report['early_stop']['rounds_skipped']} round(s) of max_rounds skipped)")
    print(f"  chunk {report['chunk']} ends with: round {report['rounds_run']}'s decision "
          f"-> final accuracy {_fmt_pct(report['final_accuracy'])}, streak {report['streak']}")


def print_chunk_score_detail(chunk_number, chunk_result):
    """
    Per-candidate accuracy detail (exact-match, changed-cell accuracy,
    by-action breakdown) for the chunk's final code, via the EXISTING
    summarize_scores/print_score_summary (no new mechanism needed for this)
    — fed chunk_result["current_best_scored"] directly, so the candidate is
    never re-run just to produce this report.
    """
    scored = chunk_result.get("current_best_scored")
    if not scored:
        return
    print_score_summary(summarize_scores(scored), label=f"chunk {chunk_number} final")


def report_chunk(chunk_number, prev_boundary, boundary, chunk_result, max_rounds):
    """
    Convenience wrapper: everything the outer per-chunk loop
    (cmd_run_chunked) needs to call once a chunk finishes, in the right
    order — event-by-event log lines, then the structured chunk summary,
    then per-candidate score detail for the chunk's final code.
    """
    log_chunk_events(chunk_number, chunk_result)
    print_chunk_report(build_chunk_report(chunk_number, prev_boundary, boundary, chunk_result, max_rounds))
    print_chunk_score_detail(chunk_number, chunk_result)


def update_running_best_accuracy(running_best, chunk_final_accuracy):
    """
    Running best-chunk-ending-accuracy-so-far across the whole run — read-
    only bookkeeping, updated once per chunk with that chunk's OWN final
    accuracy (never a mid-chunk round's accuracy, and never a rejected
    candidate's). Never fed back into the epsilon-reset comparison in any
    way — this exists purely so end-of-run reporting can show whether the
    final candidate ended up worse than some earlier chunk's peak, which
    the epsilon-reset scheme explicitly allows to happen (a later chunk's
    code only ever has to stay within epsilon of the round immediately
    before it, not of the run's all-time best).
    """
    if running_best is None or chunk_final_accuracy > running_best:
        return chunk_final_accuracy
    return running_best


def print_run_summary(running_best_accuracy, final_accuracy, n_chunks):
    """
    End-of-run summary: the run's best chunk-ending accuracy printed
    alongside the final candidate's own accuracy, so accumulated drift from
    the epsilon-reset scheme is visible at a glance without reconstructing
    it from the full per-chunk log.
    """
    print(f"\n=== Run summary: {n_chunks} chunk(s) ===")
    print(f"  best chunk-ending accuracy seen during the run: {_fmt_pct(running_best_accuracy)}")
    print(f"  final candidate's accuracy: {_fmt_pct(final_accuracy)}")
    if (running_best_accuracy is not None and final_accuracy is not None
            and final_accuracy < running_best_accuracy - 1e-9):
        print(f"  NOTE: final candidate is {(running_best_accuracy - final_accuracy):.1%} points BELOW "
              f"the best chunk-ending accuracy seen during the run — the best-performing code seen so "
              f"far was from an earlier chunk, not the final one. This is a legitimate outcome of the "
              f"epsilon-reset scheme: a later chunk's code only ever has to stay within epsilon of the "
              f"round immediately before it, never of the run's all-time best — worth knowing about, "
              f"not necessarily a bug.")


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
    levels_completed_after > levels_completed_before for that same row.
    Non-fatal if a trace doesn't carry this field (same lenient,
    best-effort treatment as --score-key below) — pass --levels-key "" to
    skip the attempt entirely.

    Same treatment for available_actions (pre_observation.available_actions,
    under --actions-key) into available_actions_before — the raw legal-
    action-set integers reported by the game itself (e.g. [1, 2, 3, 4]) at
    the state this row's action was chosen from, used by
    format_chunk_action_vocabulary and format_row_available_actions to
    tell the model what actions are actually available rather than
    leaving it to infer the action space from which strings happen to
    appear in a given prompt's examples. Before-only, not also
    post_observation's side: nothing in the harness consults an "after"
    value (it would describe the NEXT row's decision point, not this
    row's), so there's nothing to gain from pulling and persisting it.
    Pass --actions-key "" to skip.
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
            if args.actions_key:
                actions_before = get_nested(rec, f"{args.pre_key}.{args.actions_key}")
                if actions_before is not None:
                    clean["available_actions_before"] = actions_before
            out.write(json.dumps(clean) + "\n")
            n = i + 1
    print(f"Wrote {n} cleaned records to {out_path}")


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


def format_available_actions(action_ids):
    """
    Render a set of raw available_actions integers -- as they appear
    directly in a trace row's pre_observation/post_observation (e.g.
    [1, 2, 3, 4]) -- into the same "GameAction.ACTIONN" string format the
    cleaned record's own "action" field already uses (e.g.
    "GameAction.ACTION1"), so the two read as the same vocabulary in a
    prompt instead of two superficially unrelated ones.

    This assumes the trace's integer IDs line up 1:1 with ACTION{n} (true
    for every ls20 row inspected so far). An ID that doesn't fit that
    scheme (0 is RESET, not a mid-episode action, and wouldn't appear in a
    mid-trace row's available_actions anyway) is rendered as a raw literal
    rather than guessed at, so a genuine surprise in the data shows up as
    an odd-looking token instead of being silently mapped to the wrong
    action.
    """
    rendered = []
    for a in sorted(set(action_ids)):
        if isinstance(a, int) and a >= 1:
            rendered.append(f"GameAction.ACTION{a}")
        else:
            rendered.append(str(a))
    return ", ".join(rendered)


def format_chunk_action_vocabulary(records):
    """
    Action-vocabulary note for PROMPT_TEMPLATE/EXTEND_TEMPLATE, built from
    each row's available_actions_before -- the game's own reported legal-
    action set at the state the row's action was chosen FROM, i.e. ground
    truth for what this row's own decision point looked like -- rather
    than inferred from which "action" strings happen to occur in a sampled
    batch. Scoped to the CURRENT CHUNK's records only (never the whole
    trace) -- see build_initial_prompt/build_extend_prompt's call sites.

    Deliberately available_actions_before only, not also _after: _after is
    already the NEXT row's decision point (or, on a chunk's last row, past
    the transition entirely -- see next_chunk_boundary), so folding it in
    here was answering a question this note was never meant to answer.
    Dropping it also removes the one guaranteed source of disagreement a
    chunk could show (a level-completing last row's _after already
    reflecting the next level's action set) -- see format_row_available_actions
    for where a row's transition-relative available actions still matter.

    Two output shapes, chosen by comparing each row's available_actions_before
    against every other row's in the chunk:
      - Uniform (expected to be the near-universal case now): every row
        agrees, so the note states the action set plainly, with no hedging.
      - Non-uniform (should be rare-to-nonexistent for a well-formed
        trace): rows disagree, so the note names the superset and warns
        the model not to assume every row shares it. Per
        next_chunk_boundary/next_level_completion_row, a chunk never
        knowingly spans two levels -- but that guarantee depends on
        levels_completed_before/after being populated; a trace with gaps
        in that field (next_level_completion_row silently skips rows it's
        missing on) could let an undetected level transition land inside
        one chunk, or a game could simply change its own available
        actions mid-level. This branch exists to surface that rather than
        silently assert a uniformity that didn't hold.

    Falls back to a plain notice when the cleaned records don't carry
    available_actions_before at all (e.g. a trace preprocessed before
    --actions-key existed, or preprocessed with --actions-key "").
    """
    row_sets = []
    for rec in records:
        before = rec.get("available_actions_before")
        if before is None:
            continue
        row_sets.append(set(before))

    if not row_sets:
        return (
            "(Available-actions data not present for this chunk -- the cleaned "
            "trace doesn't carry available_actions_before. Infer the action "
            "vocabulary from whatever action values appear in the examples below.)"
        )

    ids = set()
    for row_ids in row_sets:
        ids |= row_ids

    uniform = all(row_ids == row_sets[0] for row_ids in row_sets)
    if uniform:
        return f"The actions available for this chunk of examples are: {format_available_actions(ids)}."

    return (
        f"The superset of actions available for this chunk of examples is "
        f"{format_available_actions(ids)}, but some examples do not have all these "
        f"actions available to them. Infer the action vocabulary from whatever "
        f"action values appear in the examples below."
    )


def format_row_available_actions(available_actions_before):
    """
    Per-counterexample "available actions for this row" line for
    REVISE_TEMPLATE, sourced from that row's own available_actions_before
    (row_failure_counts.json entries carry this once it's persisted at
    first-scoring time -- see run_backtest) rather than from a single
    aggregated note, since counterexamples are typically drawn from all
    across the trace-so-far and different rows can legitimately have
    different legal action sets (e.g. rows from different levels).

    Falls back to a plain notice for counts files persisted before this
    field existed, rather than a broken/missing line.
    """
    if available_actions_before is None:
        return "(available actions not recorded for this row)"
    return f"available actions for this row: {format_available_actions(available_actions_before)}"


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
    Object-description instruction shown in PROMPT_TEMPLATE/EXTEND_TEMPLATE
    — never REVISE_TEMPLATE (revision is deliberately scoped tightly to
    specific failing rows; the model can update DESCRIPTION opportunistically
    as part of a normal revision without a separately mandated step there).

    PROMPT_TEMPLATE (chunk 1, is_extend=False) gets the 3-step version — there's no
    prior description to consult yet. EXTEND_TEMPLATE (every chunk after the first,
    is_extend=True) gets a 4-step version whose new first step asks the model to
    check its existing DESCRIPTION comment against the new examples before deciding
    whether it still holds.

    When this EXTEND_TEMPLATE call is for the chunk that starts right after a level
    completion (is_level_boundary=True — this is the LAGGED value from the
    PREVIOUS chunk's next_chunk_boundary call, per that function's own docstring;
    the caller carries it forward one chunk, never using the current chunk's own
    value), an extra, stronger sentence is prepended telling the model explicitly
    that a new level just started and its persisted DESCRIPTION may no longer
    apply — stronger than the routine "may already be sufficient, or it may need
    revising" step 1 language every other EXTEND_TEMPLATE chunk gets, since a new
    level can introduce a whole new tileset the existing description never
    accounted for.
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
            "another object's shape or color\" (these are just examples — there may "
            "be other kinds of objects/purposes)."
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
        "object's shape or color\" (these are just examples — there may be other "
        "kinds of objects/purposes)."
    )
    return f"{level_boundary_note}{DESCRIPTION_INSTRUCTION_COMMON_INTRO} {steps}\n{DESCRIPTION_INSTRUCTION_TRAILER}"


PROMPT_TEMPLATE = """You are given a sequence of observed transitions from an ARC-AGI-3 game. \
Each example shows a grid, an action taken, and the resulting grid.

{encoding_explanation}

{action_vocabulary}

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
first call of a fresh rollout, previous_state is an empty dict {{}}. Your class must \
not raise a KeyError on that call — use previous_state.get("your_key", <default>) \
instead of previous_state["your_key"]. Store whatever state you choose to track under \
keys of your own choosing, but don't assume any of them already exist the first time \
predict() runs.

grid_before is always a list of lists of plain integers 0-15 — never the compact \
display notation shown above, whichever form it takes — and predicted_grid_after must \
be in that same form. Infer the transformation rule(s) from the examples below by \
mentally converting each displayed row back to its real integer list first. Your class \
may define as many additional methods/fields as it needs; only predict's signature is \
fixed. You may only import: copy, itertools, math, collections, functools, and numpy — no other modules, standard library or third-party. \
Return only the class definition, no explanation, no example usage.

Define GameModel exactly ONCE, and define each of its methods exactly ONCE. Do not \
write multiple draft attempts, "actually, let me try again" rewrites, or alternate \
versions — pick your best hypothesis and write a single, complete, syntactically valid \
class for it. Every if/elif/else branch and every loop body must contain a real \
statement (return, assignment, pass, etc.) — never leave a branch with only a comment \
inside it.

Do not re-examine the same examples repeatedly or restate your reasoning multiple \
times — if your first hypothesis doesn't perfectly fit every example, write your best \
guess anyway and move on. You will see new examples and counterexamples and get a \
chance to fix mistakes in later rounds, so an imperfect-but-complete class now is far \
more useful than a perfect rule you never finish writing.

Following the "Starting Grid", each example's changed cells are shown in whichever \
of three formats is most compact for that specific example — every example names \
its own format inline, so read that name and apply the matching rule below rather \
than assuming all examples use the same one:
  - "flat-list" format: one changed cell per line, "(row, col): before -> after", \
real integer values 0-15 — NOT hex digits, even though grids elsewhere use hex.
  - "masked-grid" format: one line per grid row, space-separated tokens. "*N" means \
the next N cells in that row are unchanged from the previous grid. "b>c" is a single \
changed cell, hex digit before -> after (same 0-15 hex alphabet the color legend \
above uses for grids — NOT real integers here, opposite of flat-list format). \
"b>c*N" means the next N cells all made that SAME before->after change — do not \
read it as N separate different changes, it is N cells that all changed identically.
  - "in full": the whole resulting grid, same compact display format as the Starting \
Grid above (hex/RLE per the encoding note) — used when showing the entire grid is \
itself the most compact option, e.g. most of the grid changed at once.

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

{action_vocabulary}

Following the "Starting Grid", each example's changed cells are shown in whichever \
of three formats is most compact for that specific example — every example names \
its own format inline, so read that name and apply the matching rule below rather \
than assuming all examples use the same one:
  - "flat-list" format: one changed cell per line, "(row, col): before -> after", \
real integer values 0-15 — NOT hex digits, even though grids elsewhere use hex.
  - "masked-grid" format: one line per grid row, space-separated tokens. "*N" means \
the next N cells in that row are unchanged from the previous grid. "b>c" is a single \
changed cell, hex digit before -> after (same 0-15 hex alphabet the color legend \
above uses for grids — NOT real integers here, opposite of flat-list format). \
"b>c*N" means the next N cells all made that SAME before->after change — do not \
read it as N separate different changes, it is N cells that all changed identically.
  - "in full": the whole resulting grid, same compact display format as the Starting \
Grid above (hex/RLE per the encoding note) — used when showing the entire grid is \
itself the most compact option, e.g. most of the grid changed at once.

{examples}

{description_instruction}

Your predict method must keep this exact signature:

    def predict(self, grid_before: list[list[int]], action: str, previous_state: dict) -> tuple:
        ...

returning (predicted_grid_after, goal, state) exactly as before — see your class's own \
docstring above for what each element means if you need a reminder. grid_before is \
always a list of lists of plain integers 0-15 — never the compact display notation \
shown above, whichever form it takes — and predicted_grid_after must be in that same \
form. Reminder: backtest always replays your class from row 0, so previous_state \
still starts as an empty dict {{}} on that very first call regardless of which rows \
you're looking at here — don't let a change here reintroduce a KeyError on a key \
that isn't in previous_state yet (use previous_state.get("your_key", <default>) \
rather than previous_state["your_key"]). You may only import: copy, itertools, math, collections, functools, and numpy — no other modules, standard library or third-party. \
Return only the full, updated class definition, no explanation, no example usage.

Define GameModel exactly ONCE, and define each of its methods exactly ONCE — extend or \
adjust the existing methods/fields shown above rather than writing multiple draft \
attempts, "actually, let me try again" rewrites, or alternate versions of the whole \
class. Every if/elif/else branch and every loop body must contain a real statement \
(return, assignment, pass, etc.) — never leave a branch with only a comment inside it.

Do not \
re-examine the same examples repeatedly or restate your reasoning multiple times — \
write your best update and move on, even if it's imperfect. You'll get revision rounds \
later if something is still wrong.

Reminder: predict() takes and returns plain Python int grids at runtime, never the \
display notation used above to show you the examples.

Write your updated GameModel class now.
"""

REVISE_TEMPLATE = """Your current GameModel class got some transitions wrong. Here is the \
current class:

```python
{candidate_code}
```

{encoding_explanation}

{status_notes}
Each counterexample below shows a row your class has been failing on: your predicted \
grid_after in full (when your class produced one — see the diff itself for what to do \
when it didn't), plus a computed diff against the correct grid — exactly which cells \
differ, and what the correct value is at each one, in whichever of two formats was \
more compact for that specific counterexample (each one names its own format inline, \
so read that name rather than assuming they're all the same). The correct grid_after \
itself is NOT shown separately — your predicted grid is identical to it at every cell \
NOT listed in the diff, and the diff gives you the correct value at every cell that \
IS listed, so nothing is missing by only showing one grid:
  - "flat-list" format: one differing cell per line, "(row, col): your prediction -> \
correct", real integer values 0-15 — NOT hex digits.
  - "masked-grid" format: one line per grid row, space-separated tokens. "*N" means \
the next N cells in that row match between your prediction and the correct grid. \
"b>c" is a single differing cell, hex digit your-prediction -> correct (same 0-15 \
hex alphabet the color legend above uses for grids — NOT real integers here). \
"b>c*N" means the next N cells all had that SAME your-prediction -> correct mismatch \
— do not read it as N separate different mismatches, it is N cells that are all \
wrong in the identical way.
Use that directly to see where your rule diverges from the truth; you do not need to \
manually compare grids cell-by-cell yourself. Each counterexample also states \
its own available actions — the legal action set can differ row to row (e.g. across a \
level boundary). These rows are selected because your class has failed on them most \
persistently across the WHOLE trace so far, not just the newest rows it's seen:

{counterexamples}

Revise your GameModel class so it correctly handles these cases while continuing to \
handle the cases it already gets right, WITHOUT rewriting it from scratch. Your \
predict method must keep this exact signature:

    def predict(self, grid_before: list[list[int]], action: str, previous_state: dict) -> tuple:
        ...

returning (predicted_grid_after, goal, state) exactly as before — see your class's own \
docstring above for what each element means if you need a reminder. grid_before is \
always a list of lists of plain integers 0-15 — never the compact display notation \
shown above, whichever form it takes — and predicted_grid_after must be in that same \
form. Reminder: backtest always replays your class from row 0, so previous_state \
still starts as an empty dict {{}} on that very first call regardless of which rows \
you're looking at here — don't let a change here reintroduce a KeyError on a key \
that isn't in previous_state yet (use previous_state.get("your_key", <default>) \
rather than previous_state["your_key"]). You may only import: copy, itertools, math, collections, functools, and numpy — no other modules, standard library or third-party. \
Return only the full, updated class definition, no explanation, no example usage.

Define GameModel exactly ONCE, and define each of its methods exactly ONCE — revise or \
adjust the existing methods/fields shown above rather than writing multiple draft \
attempts, "actually, let me try again" rewrites, or alternate versions of the whole \
class. Every if/elif/else branch and every loop body must contain a real statement \
(return, assignment, pass, etc.) — never leave a branch with only a comment inside it.

State what you changed in this revision in under 100 words, as a comment/docstring in \
your class labeled "REVISION:" — overwrite any previous "REVISION:" block already \
there rather than appending to it, so only your latest revision notes remain. Do not \
re-examine the same counterexamples repeatedly or restate your reasoning multiple \
times — write your best fix and move on, even if it's imperfect. You'll get another \
revision round later if something is still wrong.

Reminder: predict() takes and returns plain Python int grids at runtime, never the \
display notation used above to show you the examples.

Write your updated GameModel class now.
"""


def build_examples_block(records, encoding="hex", is_extend=False):
    """
    Render a contiguous batch of records as one continuous replay, not
    independent snapshots: the full encoded grid is shown exactly once
    (`Starting Grid`), and every subsequent example is shown only as a diff
    against the immediately preceding one — a full grid for every example
    is redundant, since each subsequent starting grid is just the previous
    example's resulting grid, already fully implied by the previous
    example's diff.

    Each per-example diff independently picks whichever of three
    representations renders shortest for THAT specific diff -- format_diff's
    own flat-list-vs-masked-grid choice, further compared here against a
    plain full-grid dump. No fixed cell-count threshold: see format_diff's
    docstring for why a threshold can't generalize across games the way a
    direct cost comparison does. Every block states inline which format it
    used, since different examples in the same prompt can legitimately use
    different ones.

    `is_extend` controls only the one extra "not the game's start" sentence
    (PROMPT_TEMPLATE calls with is_extend=False, EXTEND_TEMPLATE with
    is_extend=True) — everything else is shared. This function backs the
    round-1/extend path only; REVISE_TEMPLATE's counterexample rendering
    (build_row_counterexamples_block) is unrelated and continues showing
    full predicted/actual grid pairs per counterexample, since a
    counterexample's whole purpose is the predicted-vs-actual comparison,
    not a contiguous replay.

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
        diff_format, diff_text = format_diff(changes, rec["grid_after"])
        full_dump = encode_grid(rec["grid_after"], encoding)

        header = f"\n### {label} (trace step {rec['step']})\naction: {rec['action']}\n"
        if changes and len(full_dump) < len(diff_text):
            # For this specific diff, showing the whole resulting grid is
            # actually cheaper than any diff representation (this happens
            # at very high change density, e.g. a level transition where
            # most of the grid changed at once — masking barely helps when
            # there's little left to mask, and even a flat list of
            # thousands of changed cells loses to just showing the grid).
            block = header + (
                f"{label} in full ({len(changes)} of {sum(len(r) for r in rec['grid_after'])} "
                f"cells changed — showing the whole grid was cheaper than any diff "
                f"representation for this one) — same compact display format as the "
                f"Starting Grid above, NOT real integers or the masked-grid notation "
                f"used elsewhere:\n{full_dump}\n"
            )
        else:
            block = header + (
                f"changed cells ({prev_label} -> {label}), {diff_format} format "
                f"(see the format note above for how to read this):\n{diff_text}\n"
            )
        lines.append(block)
        prev_label = label
    return "\n".join(lines)


def build_initial_prompt(records, max_examples, encoding="hex"):
    """
    Chunk 1's round-1 prompt. Under the chunked curriculum, chunk 1 is
    simply the trace's first max_examples rows in chronological order —
    contiguous, not an evenly-spaced subsample across the whole trace,
    since build_examples_block's contiguity precondition requires it.
    """
    sampled = records[:max_examples]
    prompt = PROMPT_TEMPLATE.format(
        encoding_explanation=encoding_explanation(encoding),
        action_vocabulary=format_chunk_action_vocabulary(sampled),
        examples=build_examples_block(sampled, encoding=encoding, is_extend=False),
        description_instruction=build_description_instruction(is_extend=False),
    )
    return prompt, len(sampled)


def build_extend_prompt(candidate_code, new_records, encoding="hex", is_level_boundary=False):
    """
    Round-1 prompt for every chunk after the first. Shows the candidate's
    current class in full plus only this chunk's newly introduced rows —
    not the whole trace-so-far, which would defeat the point of a bounded
    per-chunk prompt.

    is_level_boundary must be the LAGGED value from the PREVIOUS chunk's
    next_chunk_boundary call (the caller carries it forward one chunk),
    never this chunk's own not-yet-relevant is_level_boundary — see
    build_description_instruction's docstring for why.

    action_vocabulary is deliberately derived from new_records alone (this
    chunk only), not the whole trace — see format_chunk_action_vocabulary's
    docstring for why that's the intended scope, not a limitation.
    """
    return EXTEND_TEMPLATE.format(
        candidate_code=candidate_code.strip(),
        encoding_explanation=encoding_explanation(encoding),
        action_vocabulary=format_chunk_action_vocabulary(new_records),
        examples=build_examples_block(new_records, encoding=encoding, is_extend=True),
        description_instruction=build_description_instruction(
            is_extend=True, is_level_boundary=is_level_boundary
        ),
    )


def select_top_k_failures(row_failure_counts, k):
    """
    Plain top-k selection by failure count from the persistent
    row_failure_counts.json record — not a bounded-diversity round-robin-
    by-action selection. Selection here always draws from the full
    trace-so-far, not just the current chunk's newly introduced rows.

    Only rows with count > 0 are eligible — a row at count 0 is currently
    passing (or has never been scored as failing) and is never shown in a
    revision prompt.

    Tie-break, precisely: row_failure_counts.json's key order is the order
    rows were FIRST scored, which is always ascending trace/step order
    (every backtest replay covers 0..boundary in order, and boundary only
    ever grows chunk to chunk, so a row's insertion position never changes
    once set). When the failure-count group straddling the k-th slot has
    more members than the remaining slots, members are kept in that
    ascending file order and dropped from the END of the tied group until
    it fits exactly. Worked example: k=10, 6 rows at count=5, 5 rows at
    count=4 -> all 6 count-5 rows, plus the first 4 (file order) of the 5
    count-4 rows; the 5th (last in file order) count-4 row is dropped.

    Returns a list of (step_key, entry) tuples, longest-failing first, at
    most k long.
    """
    order_index = {step_key: i for i, step_key in enumerate(row_failure_counts.keys())}
    eligible = [
        (step_key, entry) for step_key, entry in row_failure_counts.items()
        if entry.get("count", 0) > 0
    ]
    eligible.sort(key=lambda kv: (-kv[1]["count"], order_index[kv[0]]))
    return eligible[:k]


def build_revise_status_notes(most_recent_passed):
    """
    The "state of play" sentence REVISE_TEMPLATE wants up front, before the
    counterexamples themselves: whether the single most recent (highest-
    step) row of the current row range was predicted correctly or not.

    Note: an earlier version of this also surfaced a note when the epsilon
    check rejected the immediately preceding round's candidate ("this
    round is built from an earlier round's code, not the one you just
    wrote"). That's been dropped — each revision call is a fresh, stateless
    LLM call with no memory of prior rounds, so telling the model "the
    previous round was rejected" gives it nothing actionable; it has no
    visibility into what that rejected candidate looked like or why it
    differs from the code shown here. The rejection itself is still logged
    (log_substitution), just not surfaced inside the prompt.

    Returns "" when most_recent_passed is None (row_failure_counts empty).
    """
    if most_recent_passed is None:
        return ""
    return (
        "The single most recent row of the current row range was predicted "
        + ("CORRECTLY." if most_recent_passed else "INCORRECTLY.")
        + "\n"
    )


def format_prediction_diff_section(predicted_grid, actual_grid):
    """
    Render the "cells where your prediction differs..." section (intro
    line + diff text) for one row_failure_counts entry, used by both
    branches of build_revise_row_block that show a predicted-vs-correct
    diff.

    predicted_grid is None means the candidate's predict() call produced
    no usable grid at all for this row (crashed, or genuinely returned
    None) — this is NOT the same thing as an empty diff_grid() result,
    which means the prediction was a PERFECT match. Piping predicted_grid
    is None straight through diff_grid()+format_diff() previously produced
    an empty changes list, which format_diff renders as "no cells changed
    (identity transformation for this example)" — falsely claiming a
    flawless prediction that never actually happened. This function exists
    specifically to keep those two very different situations from
    rendering as the same sentence.

    Can a genuinely non-None predicted_grid still equal actual_grid
    exactly here, producing a TRUE "no cells changed" result? Depends
    entirely on which of build_revise_row_block's branches called this:
      - Ordinary wrong-transition branch (predicted_goal == actual_goal,
        both False): PROVABLY IMPOSSIBLE for any row actually selected as
        a counterexample. run_backtest computes
        passed = goal_match and (predicted_grid == actual_grid) for this
        exact case, so an exact grid match together with a goal match
        would have made the row PASS (count reset to 0) rather than fail
        — a row only reaches counterexample selection (count > 0) here by
        having predicted_grid != actual_grid.
      - False-positive-goal branch (predicted_goal=True, actual_goal=
        False): genuinely reachable, and meaningful when it happens.
        goal_discounted=True means passed = goal_match ONLY — the grid
        comparison is excluded from pass/fail entirely, so a row can fail
        purely on the goal mismatch while predicted_grid still happens to
        equal actual_grid exactly. When that happens, "no cells changed"
        is both true and the single most useful thing this section can
        say: the grid transformation logic is correct, only the goal-
        completion heuristic needs fixing.
      - Missed-level-completion branch never calls this function at all
        (build_revise_row_block skips the grid diff there entirely).
    """
    if predicted_grid is None:
        return (
            "no prediction available for this row — predicted_grid was None, so "
            "there is nothing to diff against the correct grid. This does NOT mean "
            "your prediction matched; it means predict() didn't return a usable grid "
            "for this row at all — check for a crash/timeout or an early return "
            "somewhere in your class.\n"
        )
    diff = diff_grid(predicted_grid, actual_grid)
    diff_format, diff_text = format_diff(diff, actual_grid)
    return (
        f"cells where your prediction differs from the correct grid, {diff_format} "
        f"format (your prediction -> correct):\n{diff_text}\n"
    )


def format_grid_and_diff_section(predicted_grid, actual_grid, encoding):
    """
    Render the grid(s) + diff section for one build_revise_row_block
    counterexample. Shows ONLY the predicted grid in full, not both
    predicted and correct — the diff already carries "before(predicted)
    -> after(correct)" at every cell that differs, and is identical to
    the correct grid at every cell that doesn't, so predicted grid + diff
    reconstructs the correct grid exactly. Showing both grids in full was
    pure redundancy that cost a full extra grid dump per counterexample
    regardless of how small the actual diff was — for k counterexamples
    that's k full grid dumps saved.

    Falls back to showing the correct grid alone when predicted_grid is
    None (nothing else to show; format_prediction_diff_section's own "no
    prediction available" message explains why there's no predicted grid,
    but the model still needs SOME concrete grid to look at).
    """
    if predicted_grid is not None:
        pred_str = encode_grid(predicted_grid, encoding)
        return f"your predicted grid_after:\n{pred_str}\n" + format_prediction_diff_section(predicted_grid, actual_grid)
    return f"correct grid_after:\n{encode_grid(actual_grid, encoding)}\n" + format_prediction_diff_section(predicted_grid, actual_grid)


def build_revise_row_block(i, step_key, entry, encoding="hex"):
    """
    Render one selected row_failure_counts.json entry as a counterexample.
    Built entirely from the entry's own fields — count, actual_grid/
    actual_goal (ground truth), predicted_grid/predicted_goal/error (most
    recent attempt), available_actions_before (ground truth, see
    run_backtest) — no grid_before/action are available in this schema,
    so the diff shown here is between the predicted grid and the correct
    grid directly, not a before/after transition diff.

    Three distinct cases:
      - False-positive goal (predicted_goal=True, actual_goal=False):
        actual_grid is a normal same-level continuation; predicted grid +
        diff shown as usual (see format_grid_and_diff_section), plus an
        explicit note that the goal prediction was wrong.
      - False-negative goal (predicted_goal=False, actual_goal=True): the
        grid diff is omitted entirely — actual_grid here is the NEXT
        level's starting grid, not a comparable continuation — and the row
        is labeled as a missed level-completion transition rather than an
        ordinary wrong-transition row.
      - Ordinary wrong-transition row (goal correctly predicted False, grid
        wrong): predicted vs. correct grid, with a computed diff.
    A row that crashed/timed out (error is not None) is called out
    separately regardless of which of the above it would otherwise be.
    """
    count = entry.get("count", 0)
    actual_grid = entry["actual_grid"]
    actual_goal = entry["actual_goal"]
    predicted_grid = entry.get("predicted_grid")
    predicted_goal = entry.get("predicted_goal")
    error = entry.get("error")

    header = (
        f"### Counterexample {i} (trace step {step_key}, failed {count}x so far)\n"
        f"{format_row_available_actions(entry.get('available_actions_before'))}\n"
    )

    if error is not None:
        return (
            header
            + f"Your class raised an error or timed out on this row instead of "
              f"returning a prediction: {error}\n"
              f"correct grid_after:\n{encode_grid(actual_grid, encoding)}\n"
        )

    if predicted_goal and not actual_goal:
        return (
            header
            + "FALSE-POSITIVE goal prediction: you predicted goal=True (level complete) "
              "on this row, but the level did NOT actually complete here.\n"
            + format_grid_and_diff_section(predicted_grid, actual_grid, encoding)
        )

    if (not predicted_goal) and actual_goal:
        return (
            header
            + "MISSED LEVEL-COMPLETION TRANSITION: you predicted goal=False on this "
              "row, but the level actually completed here. The correct grid_after for "
              "this row is the NEXT level's starting grid, not a continuation of the "
              "current one, so no cell-by-cell grid diff is shown — focus on detecting "
              "that this action completes the level, not on matching the resulting "
              "grid.\n"
        )

    # Ordinary wrong-transition row: goal correctly predicted False, grid wrong.
    return (
        header
        + format_grid_and_diff_section(predicted_grid, actual_grid, encoding)
    )


def build_row_counterexamples_block(selected, encoding="hex"):
    """Render a top-k selection (select_top_k_failures) as REVISE_TEMPLATE's {counterexamples} block."""
    return "\n".join(
        build_revise_row_block(i, step_key, entry, encoding=encoding)
        for i, (step_key, entry) in enumerate(selected, start=1)
    )


def build_revise_prompt(candidate_code, row_failure_counts, k=10, encoding="hex"):
    """
    Revision-round prompt builder for the chunked design — takes a
    persistent row_failure_counts.json record rather than a pre-selected
    counterexamples list keyed on grid_before/action. Counterexamples are
    selected via plain top-k failure count directly from that record,
    covering the entire trace-so-far, not just the current chunk's newly
    introduced rows — see select_top_k_failures for the exact tie-break
    rule.

    row_failure_counts must already reflect whichever code is genuinely
    current-best for this chunk — i.e. the per-chunk round loop's
    commit/revert has already run for any prior round in this chunk. This
    function trusts the dict handed to it and does no commit/revert
    bookkeeping of its own.

    Two distinct "not enough rows" situations, handled differently on
    purpose:
      - k > len(row_failure_counts) (asking for more counterexamples than
        rows have EVER been scored, period) is a caller misconfiguration —
        there is no way this could ever be satisfiable at this point in the
        run, so this raises immediately rather than silently proceeding
        with a much smaller k than intended.
      - Fewer than k rows are CURRENTLY FAILING (e.g. k=10 but only 6 rows
        have count > 0) is completely normal and expected as a candidate
        improves — select_top_k_failures simply returns however many
        failing rows exist (via a plain list slice, which is a no-op if
        the list is already shorter than k). This must NOT raise.
    """
    total_rows = len(row_failure_counts)
    if k > total_rows:
        raise ValueError(
            f"k={k} exceeds the total number of rows currently tracked in "
            f"row_failure_counts ({total_rows}) — there can never be more than "
            f"{total_rows} counterexamples to select from at this point in the run. "
            f"This usually means --k was set larger than intended for the current "
            f"chunk/trace size; lower k, or wait until more rows have been backtested."
        )

    selected = select_top_k_failures(row_failure_counts, k)
    if not selected:
        # The zero-failure early stop in the per-chunk round loop is
        # specifically designed to make this unreachable — a chunk with
        # zero failing rows should never reach a revision round at all.
        # Fail loudly rather than silently building a prompt with no
        # counterexamples. Distinct from the k > total_rows case above:
        # this can fire even when k is perfectly reasonable, if every
        # tracked row currently has count 0.
        raise ValueError(
            "build_revise_prompt called with no failing rows in row_failure_counts — "
            "this should be unreachable; the zero-failure early stop in the per-chunk "
            "round loop is supposed to prevent a revision round from ever being built here."
        )

    most_recent_step = next(reversed(row_failure_counts), None)
    most_recent_passed = (
        row_failure_counts[most_recent_step]["count"] == 0
        if most_recent_step is not None else None
    )

    return REVISE_TEMPLATE.format(
        candidate_code=candidate_code.strip(),
        encoding_explanation=encoding_explanation(encoding),
        status_notes=build_revise_status_notes(most_recent_passed),
        counterexamples=build_row_counterexamples_block(selected, encoding=encoding),
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
    run-chunked's preflight check) whenever one is available.
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


def keep_first_class_def(source, class_name="GameModel"):
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
            # top of the raw text — pad the estimate so the preflight check
            # errs toward catching a too-tight fit rather than missing one.
            # CHAT_TEMPLATE_TOKEN_PADDING is a generous fixed guess at that
            # overhead (role delimiters, BOS/EOS, etc.), not measured against
            # this specific model's real chat template — cheap insurance
            # given the alternative is a raw mid-generation context-overflow
            # error instead of this preflight's own clear message.
            CHAT_TEMPLATE_TOKEN_PADDING = 64
            return len(llm.tokenize(prompt.encode("utf-8"))) + CHAT_TEMPLATE_TOKEN_PADDING

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
    """
    Manual test/inspection entry point for build_revise_prompt. Reads
    row_failure_counts.json (as written by `backtest`, or a hand-
    constructed fixture for testing) rather than a counterexamples jsonl —
    see build_revise_prompt's docstring for why the older counterexamples-
    list contract no longer applies here.
    """
    candidate_code = Path(args.candidate).read_text()
    row_failure_counts = load_row_failure_counts(args.counts)
    encoding = "rle" if args.compact else "hex"
    try:
        prompt = build_revise_prompt(candidate_code, row_failure_counts, k=args.k, encoding=encoding)
    except ValueError as e:
        # Covers both of build_revise_prompt's validation failures: --k
        # larger than the total number of rows ever tracked (misconfigured
        # --k for this trace/chunk size), and zero currently-failing rows
        # (should be unreachable once the per-chunk round loop's zero-
        # failure early stop is wired in, but easy to hit here when
        # hand-testing against a clean fixture).
        print(f"Could not build revision prompt: {e}")
        sys.exit(1)
    Path(args.out).write_text(prompt)
    est_tokens = estimate_tokens_fallback(prompt)
    n_selected = len(select_top_k_failures(row_failure_counts, args.k))
    print(f"Wrote revision prompt to {args.out}: {len(prompt)} chars, ~{est_tokens} tokens "
          f"(rough 1-token/char fallback estimate — no tokenizer loaded on this path), "
          f"{n_selected} counterexamples (top-{args.k} by failure count)"
          + (" [RLE-compact grid encoding]" if args.compact else " [hex grid encoding]"))


def cmd_backtest(args):
    """
    Standalone CLI wrapper around run_backtest, mainly for manual testing/
    debugging outside the full run-chunked pipeline (cmd_run_chunked wires
    run_backtest directly into that loop without going through this
    command).

    Exit 0 if the replay is fully clean (zero failing rows), exit 1
    otherwise. There's no fixed accuracy threshold to check here; per-chunk
    accept/reject against a baseline is the per-chunk round loop's epsilon
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


# ---------- run-chunked ----------
# Loads the full trace once, chronologically, and drives the chunked-
# curriculum outer loop (boundaries computed by next_chunk_boundary) with
# an inner extend/revise round loop per chunk (run_chunk_rounds' epsilon-
# reset selection).

FALLBACK_CANDIDATE_CODE = '''class GameModel:
    """Safe no-op stub substituted in when this round's LLM output could not
    be validated as safe code (SyntaxError or a disallowed import) -- see
    _validate_candidate_code. Always predicts no change and no goal, so it
    scores low but never crashes the sandboxed backtest."""
    def predict(self, grid_before, action, previous_state):
        return grid_before, False, previous_state
'''


def _defines_class_method(tree, class_name, method_name):
    """
    True iff `tree` (an already-parsed ast.Module) contains a class named
    class_name with a method named method_name anywhere in its body,
    scanning the WHOLE tree (ast.walk) rather than just the top level --
    consistent with check_ast_imports' own whole-tree scanning style. Not a
    check on the method's runtime behavior (a class whose __init__ crashes,
    or whose predict() has the wrong signature, still passes this) -- just
    the static shape every template explicitly asks for ("Define GameModel
    exactly ONCE" with a `predict` method).
    """
    import ast
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return True
    return False


def _validate_candidate_code(code, round_label, class_name="GameModel", method_name="predict"):
    """
    Best-effort preflight validation of this round's extracted candidate
    code, run BEFORE it ever reaches run_chunk_rounds -> run_backtest ->
    run_candidate -- none of which have (or need) their own defensive
    handling for genuinely malformed code, since up to now every caller
    controlled its own test fixtures. Real LLM output is exactly the case
    that needs this: a truncated generation (SyntaxError), a disallowed
    import (ValueError from check_ast_imports), or code that's syntactically
    fine but never actually defines a usable class_name/method_name (e.g.
    the extracted text turned out to be prose or an unrelated snippet, not
    the model's real answer -- see extract_code's docstring for how that can
    happen) would otherwise either crash the run or silently run a
    guaranteed-0%-accuracy round with a misleading "no result (crashed/timed
    out)" error on every single row (see run_candidate's RUNNER_TEMPLATE
    parsing), never revealing the real cause. All three are substituted
    with FALLBACK_CANDIDATE_CODE instead -- a real, always-parseable,
    always-safe class that just scores low rather than crashing anything or
    hiding what actually went wrong.

    This only checks STATIC shape (does a class_name class with a
    method_name method exist anywhere in the tree), not runtime behavior --
    a class whose __init__ raises, or whose predict() takes the wrong
    arguments, passes this check and still surfaces later, as a loud
    RuntimeError out of run_candidate (deliberately not caught, see below).

    Deliberately does NOT catch RuntimeError/OSError here (an actual
    subprocess/sandbox infrastructure failure, as opposed to a bad
    candidate) -- those still propagate and halt the run, since silently
    absorbing an infra problem into a fake low-accuracy round would hide
    something that actually needs attention during a real GPU run.
    """
    import ast
    try:
        tree = ast.parse(code)
        check_ast_imports(code)
        if not _defines_class_method(tree, class_name, method_name):
            raise ValueError(f"no class {class_name!r} with a {method_name!r} method found in the extracted code")
        return code
    except (SyntaxError, ValueError) as e:
        print(f"{round_label}: candidate failed validation ({type(e).__name__}: {e}); "
              f"substituting a safe no-op stub for this round so the run doesn't crash")
        return FALLBACK_CANDIDATE_CODE


def make_round_builder(chunk_number, prev_boundary, boundary, lagged_level_boundary,
                        records, counts_path, workdir, llm_call, tokenize, encoding, args, run_state):
    """
    Builds the round_builder(round_n, current_best_code) -> candidate_code
    callback run_chunk_rounds expects, closing over everything one chunk's
    worth of rounds needs: which template to build (PROMPT_TEMPLATE for
    chunk 1's round 1, EXTEND_TEMPLATE for round 1 of every later chunk
    using the LAGGED is_level_boundary, REVISE_TEMPLATE — reading
    row_failure_counts.json fresh each time, since run_chunk_rounds already
    keeps counts_path in lockstep with current_best_code between rounds —
    for every round after), the live LLM call, code extraction/trimming,
    and per-round prompt/candidate files under --workdir for inspection.

    run_state is a small shared mutable dict — currently just
    {"interrupted": bool} — used to carry a Ctrl+C signal out to the outer
    per-chunk loop. NOTE on interrupt granularity: this stops the run
    before the NEXT CHUNK starts, not before the next round within the
    SAME chunk — a deliberate simplification, since run_chunk_rounds has no
    hook for signaling early termination mid-round-loop without modifying
    that already-tested function. In practice this means, worst case,
    up to (max_rounds - 1) more LLM calls complete before the run actually
    stops after a Ctrl+C.
    """
    def round_builder(round_n, current_best_code):
        if round_n == 1:
            if chunk_number == 1:
                prompt_text, _ = build_initial_prompt(records, boundary, encoding=encoding)
                round_label = f"Chunk {chunk_number} Round {round_n}: Seed Prompt"
            else:
                new_records = records[prev_boundary:boundary]
                prompt_text = build_extend_prompt(
                    current_best_code, new_records, encoding=encoding,
                    is_level_boundary=lagged_level_boundary,
                )
                round_label = f"Chunk {chunk_number} Round {round_n}: Extend Prompt"
        else:
            row_failure_counts = load_row_failure_counts(counts_path)
            # Clamp against TOTAL rows tracked (not just currently-failing
            # ones) -- select_top_k_failures already handles "k exceeds the
            # number of currently-failing rows" gracefully on its own (a
            # plain list slice, never raises), so this clamp exists solely
            # to keep --k from tripping build_revise_prompt's separate
            # misconfiguration guard (k > total rows ever tracked), which a
            # genuinely small early chunk (fewer rows than --k) can hit
            # under perfectly ordinary conditions -- not a sign anything's
            # actually wrong.
            effective_k = min(args.k, len(row_failure_counts))
            prompt_text = build_revise_prompt(current_best_code, row_failure_counts, k=effective_k, encoding=encoding)
            round_label = f"Chunk {chunk_number} Round {round_n}: Revision Prompt"

        prompt_path = workdir / f"chunk{chunk_number}_prompt_round{round_n}.txt"
        atomic_write_text(prompt_path, prompt_text)

        # Pause before this round's LLM call unless --automatic is set —
        # firing at every per-chunk prompt-write point.
        if not args.automatic:
            pause_for_confirmation(round_label, prompt_path)

        if tokenize is not None:
            n_prompt_tokens = tokenize(prompt_text)
            budget = args.n_ctx - args.max_tokens
            print(f"{round_label}: prompt is {n_prompt_tokens} tokens "
                  f"(budget {budget} = --n-ctx {args.n_ctx} - --max-tokens {args.max_tokens})")
            if n_prompt_tokens > budget:
                raise SystemExit(
                    f"{round_label}: prompt ({n_prompt_tokens} tokens) exceeds the available "
                    f"budget ({budget} tokens). Fix one of: lower --max-examples/--k, add "
                    f"--compact for RLE grid encoding, raise --n-ctx, or lower --max-tokens."
                )
        else:
            print(f"{round_label}: prompt is {len(prompt_text)} chars "
                  f"(no local tokenizer for --backend openai — token count not verified "
                  f"against the server's context window; watch for a context-length error)")

        with GracefulInterrupt() as interrupt:
            call_result = llm_call(prompt_text)
            response = call_result["text"]
            finish_reason = call_result.get("finish_reason")
            completion_tokens = call_result.get("completion_tokens")
            print(f"{round_label}: generation finished with reason={finish_reason!r}"
                  + (f", {completion_tokens} completion tokens" if completion_tokens is not None else "")
                  + (f" (out of --max-tokens {args.max_tokens} budget)" if finish_reason == "length" else ""))
            code = extract_code(response)
            code, n_defs = keep_first_class_def(code)
            if n_defs > 0:
                print(f"{round_label}: candidate's GameModel class/methods redefined {n_defs} extra "
                      f"time(s) (re-attempts); keeping only the first, discarding the rest")
            code = _validate_candidate_code(code, round_label)
            candidate_path = workdir / f"chunk{chunk_number}_candidate_round{round_n}.py"
            atomic_write_text(candidate_path, code)

        if interrupt.requested:
            run_state["interrupted"] = True

        return code

    return round_builder


def validate_cleaned_trace(records, trace_path):
    """
    Fail fast if `records` still looks like a RAW trace (pre_observation ->
    action -> post_observation rows straight from the game harness) rather
    than the output of `preprocess` (flat {step, action, grid_before,
    grid_after, ...} rows).

    run-chunked is the expensive, unattended entry point -- it loads a GGUF
    (or connects to an API backend) and can run for a long time -- so this
    exists to catch the "forgot to preprocess" mistake before any of that
    expensive setup happens, rather than partway into chunk 1 when
    build_examples_block's first KeyError on "grid_before" would otherwise
    be the only symptom.

    Checks only the first record: it's the trace's FORMAT that's at issue
    here, not a per-row content problem, and every record in a single trace
    file comes from the same pipeline stage. Distinguishes "definitely raw"
    (has pre_observation/post_observation) from "just not cleaned-shaped"
    (neither raw nor cleaned markers present, e.g. a hand-built or corrupted
    file) so the error message can point at the actual fix in the common
    case rather than a generic "something's wrong" message.
    """
    first = records[0]
    if "grid_before" in first and "grid_after" in first:
        return

    if "pre_observation" in first or "post_observation" in first:
        raise SystemExit(
            f"{trace_path} looks like a RAW trace (its first record has "
            f"'pre_observation'/'post_observation' keys), not a preprocessed one. "
            f"run-chunked needs the flat {{step, action, grid_before, grid_after, "
            f"...}} format that `preprocess` produces, not the raw pre_observation -> "
            f"action -> post_observation rows the game harness writes.\n\n"
            f"Fix: preprocess this file first, e.g.:\n"
            f"  python trace_tools.py preprocess {trace_path} <cleaned_output.jsonl>\n"
            f"then pass <cleaned_output.jsonl> to run-chunked instead."
        )

    raise SystemExit(
        f"{trace_path} doesn't look like a preprocessed trace — its first record "
        f"is missing 'grid_before'/'grid_after'. run-chunked needs the flat "
        f"{{step, action, grid_before, grid_after, ...}} format that `preprocess` "
        f"produces; run `preprocess` on the raw trace first, then pass its output "
        f"to run-chunked."
    )


def cmd_run_chunked(args):
    """
    Fully automated chunked-curriculum loop: loads the whole cleaned trace
    once, chronologically, and drives an outer loop over chunks (boundaries
    computed by next_chunk_boundary) each running an inner loop of up to
    --max-rounds rounds (extend/seed then revise, using run_chunk_rounds'
    epsilon-reset selection with baseline recompute), reporting after every
    chunk, and persisting whichever code each chunk ends with as the next
    chunk's starting point.
    """
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    counts_path = workdir / "row_failure_counts.json"
    best_counts_path = workdir / "row_failure_counts_best.json"
    log_path = (workdir / args.log if args.log and not os.path.isabs(args.log) else Path(args.log)) if args.log else None

    if args.max_examples < 1:
        raise SystemExit("--max-examples must be at least 1 (used as the chunk-size cutoff; "
                          "0 or negative would never advance the chunk boundary and loop forever)")
    if args.max_rounds < 1:
        raise SystemExit(f"--max-rounds must be at least 1 (got {args.max_rounds}) -- chunk 1 with 0 "
                          f"rounds would never produce any code at all, since even the seed prompt is "
                          f"round 1.")

    # Refuse a dirty --workdir rather than silently mixing this run's
    # results with a previous run's leftovers: row_failure_counts.json
    # would get read-mutated-written on top of stale counts from whatever
    # ran here before, and the log would end up with duplicate/overlapping
    # chunk numbers from two unrelated runs concatenated together -- neither
    # failure is a crash, both are silently wrong-looking results, which is
    # worse than refusing to start.
    preexisting = [p for p in (counts_path, best_counts_path, log_path) if p is not None and p.exists()]
    if preexisting:
        preexisting_list = "\n  ".join(str(p) for p in preexisting)
        raise SystemExit(
            f"Refusing to start: --workdir '{workdir}' already contains file(s) from a previous run:\n"
            f"  {preexisting_list}\n\n"
            f"Starting into a dirty workdir would silently mix this run's results with the old one's.\n\n"
            f"To fix: either pass a fresh --workdir for this run, or if you really do want to overwrite "
            f"the previous run, delete/move the file(s) listed above first."
        )

    with open(args.trace) as f:
        records = [json.loads(l) for l in f if l.strip()]
    if not records:
        raise SystemExit(f"{args.trace} contains no records")
    validate_cleaned_trace(records, args.trace)

    run_kwargs = dict(cpu_seconds=args.cpu_seconds, mem_mb=args.mem_mb, max_procs=args.max_procs,
                       per_call_seconds=args.per_call_seconds, overall_timeout=args.overall_timeout)
    llm_call, tokenize = build_llm_caller(args)  # loads the GGUF once here for --backend llama-cpp
    encoding = "rle" if args.compact else "hex"
    run_state = {"interrupted": False}

    prev_boundary = 0
    pending_level_boundary = False  # chunk 1 has no predecessor and never consults this (PROMPT_TEMPLATE, not EXTEND_TEMPLATE)
    current_best_code = None
    running_best_accuracy = None
    chunk_number = 0
    final_chunk_result = None

    while prev_boundary < len(records):
        chunk_number += 1
        boundary, is_level_boundary = next_chunk_boundary(records, prev_boundary, args.max_examples)
        lagged_level_boundary = pending_level_boundary  # what THIS chunk's EXTEND_TEMPLATE should use
        pending_level_boundary = is_level_boundary       # for the NEXT chunk (one-chunk lag)

        baseline_code = current_best_code if chunk_number > 1 else None

        round_builder = make_round_builder(
            chunk_number, prev_boundary, boundary, lagged_level_boundary,
            records, counts_path, workdir, llm_call, tokenize, encoding, args, run_state,
        )

        chunk_result = run_chunk_rounds(
            round_builder, records, boundary, counts_path, best_counts_path,
            baseline_code=baseline_code, max_rounds=args.max_rounds, **run_kwargs,
        )

        report_chunk(chunk_number, prev_boundary, boundary, chunk_result, args.max_rounds)
        if log_path is not None:
            report = build_chunk_report(chunk_number, prev_boundary, boundary, chunk_result, args.max_rounds)
            with open(log_path, "a") as f:
                f.write(json.dumps(report) + "\n")
                f.flush()
                os.fsync(f.fileno())

        running_best_accuracy = update_running_best_accuracy(running_best_accuracy, chunk_result["current_best_accuracy"])
        current_best_code = chunk_result["current_best_code"]
        final_chunk_result = chunk_result

        chunk_final_path = workdir / f"chunk{chunk_number}_final.py"
        atomic_write_text(chunk_final_path, current_best_code)
        print(f"[chunk {chunk_number}] final code persisted to {chunk_final_path}")

        prev_boundary = boundary

        if run_state["interrupted"]:
            print(f"STOP: interrupted after chunk {chunk_number} completed. "
                  f"Last completed candidate: {chunk_final_path}")
            break

    print_run_summary(running_best_accuracy, final_chunk_result["current_best_accuracy"], n_chunks=chunk_number)
    print(f"Final candidate: {workdir / f'chunk{chunk_number}_final.py'}")


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
    p_pre.add_argument("--actions-key", default="available_actions",
                        help="Sub-key of pre_observation holding the raw legal-action-set list "
                             "(e.g. [1, 2, 3, 4]), extracted into available_actions_before on "
                             "each cleaned record (used by the prompt templates' "
                             "action_vocabulary note and per-counterexample available-actions "
                             "line). On by default; pass --actions-key \"\" to skip extraction "
                             "if a trace doesn't carry this field.")
    p_pre.set_defaults(func=cmd_preprocess)

    p_prompt = sub.add_parser("prompt", help="Build the chunk-1 seed prompt from a records file")
    p_prompt.add_argument("history")
    p_prompt.add_argument("out")
    p_prompt.add_argument("--max-examples", type=int, default=25,
                           help="Rows to include, taken from the start of the file in order (default 25)")
    p_prompt.add_argument("--compact", action="store_true",
                           help="Run-length-encode grid rows (e.g. '0*7 3*2') instead of "
                                "one hex char per cell — cuts tokens a lot for large/sparse grids")
    p_prompt.set_defaults(func=cmd_prompt)

    p_backtest = sub.add_parser("backtest", help="Full teacher-forced replay + scoring for the "
                                                  "chunked design, class-based candidates only")
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

    p_revise = sub.add_parser("revise-prompt", help="Build a revision prompt from a candidate + row_failure_counts.json")
    p_revise.add_argument("candidate", help="Path to the current candidate .py file (a GameModel class)")
    p_revise.add_argument("counts", help="Path to row_failure_counts.json (written by `backtest`)")
    p_revise.add_argument("out", help="Where to write the revision prompt text")
    p_revise.add_argument("--k", type=int, default=10,
                           help="Number of top-failing rows (by failure count) to include as counterexamples")
    p_revise.add_argument("--compact", action="store_true",
                           help="Run-length-encode grid rows (e.g. '0*7 3*2') instead of "
                                "one hex char per cell — cuts tokens a lot for large/sparse grids")
    p_revise.set_defaults(func=cmd_revise_prompt)

    p_chunked = sub.add_parser("run-chunked", help="Fully automated CHUNKED-CURRICULUM run against a live LLM")
    p_chunked.add_argument("trace", help="The FULL cleaned trace, one file, chronological (no history/heldout split)")
    p_chunked.add_argument("--workdir", default="chunked_run", help="Where per-round prompts/candidates/counts are written")
    p_chunked.add_argument("--backend", choices=["llama-cpp", "openai"], default="llama-cpp",
                            help="llama-cpp: load a local GGUF in-process via llama-cpp-python (default). "
                                 "openai: call an OpenAI-compatible HTTP server instead (vLLM, llama-server).")
    p_chunked.add_argument("--model-path", default=None,
                            help="[llama-cpp] path to the .gguf file, e.g. Qwen3-Coder-Next-UD-Q4_K_XL.gguf")
    p_chunked.add_argument("--n-ctx", type=int, default=32768, help="[llama-cpp] context window to allocate")
    p_chunked.add_argument("--n-gpu-layers", type=int, default=-1,
                            help="[llama-cpp] layers to offload to GPU; -1 = all")
    p_chunked.add_argument("--verbose-llama", action="store_true",
                            help="[llama-cpp] show llama.cpp's own load-time log plus per-call timing "
                                 "stats after every round.")
    p_chunked.add_argument("--api-base", default="http://localhost:8000/v1",
                            help="[openai] OpenAI-compatible base URL")
    p_chunked.add_argument("--model", default=None, help="[openai] model name as the server expects it")
    p_chunked.add_argument("--temperature", type=float, default=0.2)
    p_chunked.add_argument("--repeat-penalty", type=float, default=1.3,
                            help="[llama-cpp only] penalizes tokens already seen in the response so far; "
                                 "llama.cpp's own default (1.1) proved insufficient to stop a real "
                                 "repetition loop hit during testing (model got stuck restating the same "
                                 "reasoning until --max-tokens cut it off before finishing) — 1.3 is a "
                                 "stronger starting point.")
    p_chunked.add_argument("--presence-penalty", type=float, default=0.1)
    p_chunked.add_argument("--frequency-penalty", type=float, default=0.1)
    p_chunked.add_argument("--max-tokens", type=int, default=4096)
    p_chunked.add_argument("--max-examples", type=int, default=25,
                            help="Rows per ORDINARY chunk (the boundary rule may end a chunk "
                                 "earlier than this, on a level cutoff or the end of the trace)")
    p_chunked.add_argument("--compact", action="store_true",
                            help="Run-length-encode grid rows (e.g. '0*7 3*2') instead of "
                                 "one hex char per cell — cuts tokens a lot for large/sparse grids")
    p_chunked.add_argument("--max-rounds", type=int, default=2,
                            help="Rounds PER CHUNK (extend/seed, then up to this many revisions) — "
                                 "chunked design default is 2; a chunk can still end sooner via the "
                                 "zero-failure early stop")
    p_chunked.add_argument("--k", type=int, default=10, help="Top-failing counterexamples per revision round")
    p_chunked.add_argument("--log", default="chunk_log.jsonl",
                            help="Append each chunk's structured report (build_chunk_report) here as "
                                 "json lines, relative to --workdir unless given as an absolute path. "
                                 "Empty string disables this.")
    p_chunked.add_argument("--automatic", action="store_true",
                            help="Run every round of every chunk back-to-back with no pauses. Default "
                                 "(flag off): after each round's prompt is written to --workdir, the loop "
                                 "stops and asks whether to continue, before that round's LLM call is made.")
    p_chunked.add_argument("--cpu-seconds", type=int, default=10)
    p_chunked.add_argument("--max-procs", type=int, default=16,
                            help="RLIMIT_NPROC cap for the candidate subprocess (per-UID, not per-tree)")
    p_chunked.add_argument("--mem-mb", type=int, default=512)
    p_chunked.add_argument("--per-call-seconds", type=int, default=2)
    p_chunked.add_argument("--overall-timeout", type=int, default=60)
    p_chunked.set_defaults(func=cmd_run_chunked)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()