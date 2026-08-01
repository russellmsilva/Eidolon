# Eidolon — Simplified Capability-Diagnostic: Implementation Plan

Companion to `Eidolon_Simplified_Diagnostic_Design.md`. That document is the reference for _what_ and _why_; this document is the ordered checklist for _building it_ in `trace_tools.py`. Each step names the functions it touches, what "done" looks like, and where to stop and test before moving on. Build in this order — later steps assume earlier ones are working, and the smoke test (Step 11) is a deliberate gate before any real GPU time is spent.

**Revision note:** this version reflects the chunked-curriculum design (design doc §2–§8), replacing an earlier draft of this plan built around a single full-trace pass per round. Steps below that changed substantially from that earlier draft say so explicitly.

**Revision note 2:** Steps renumbered from a prior version of this plan to insert a new Step 4 (chunk boundary computation with level/whole-trace cutoffs, design doc §2) — everything that was Step 4 onward shifted up by one (old Step 13 is now Step 14). Also reflects: the per-round pause/`--automatic` behavior now spelled out explicitly in Step 10 (design doc §3a); and Step 5's grid-formatting section, which fully implements what this plan previously tracked as a "deferred follow-up," is no longer deferred.

**Revision note 3:** two corrections after review, both propagated through Steps 7, 8, 9, 10, and 11: (1) Step 8's epsilon-reset scheme is now a generalized round loop over `range(1, args.max_rounds + 1)` rather than hardcoded to exactly round 1 + round 2 — `--max-rounds` is a real CLI value, default **2**; (2) Step 7's counterexample selection is plain top-k by failure count (no `select_counterexamples` round-robin reuse), with an explicit deterministic tie-break rule for the boundary case where a failure-count group straddles the `k`-th slot.

**Revision note 4:** added a zero-failure early stop to Step 8's round loop (`break` as soon as a round's backtest returns zero failing rows), propagated into Step 6's return-value contract, Step 9's logging, Step 10's wiring, and Step 11's smoke test. Previously a chunk that was already fully solved would still run every remaining round up to `--max-rounds`, and a revision round could have been built against an empty counterexample list.

**Revision note 5:** three fixes after a full read-through, propagated through Steps 4, 5, 6, 7, 8, 9, 10, and 11: (1) `is_level_boundary` was being wired into the same chunk's `EXTEND_TEMPLATE` that computed it, rather than the _next_ chunk's — Step 10 now carries it forward with an explicit one-chunk lag; (2) epsilon is now a flat, additive `0.05` constant compared directly against accuracy fractions (`candidate_accuracy >= current_best_accuracy - 0.05`), replacing the unit-mismatched and, once corrected, non-scaling "10% of rows-so-far" formula — the right-hand side is explicitly allowed to go negative with no special-casing; (3) `row_failure_counts.json` now carries a richer per-row schema (ground truth plus the most recent prediction, Step 6) and a commit/revert snapshot against `row_failure_counts_best.json` (Step 8) that keeps the file always consistent with whichever code is genuinely current-best, closing a gap where a rejected round's results could otherwise leak into a later revision prompt within the same chunk.

---

## Step 1 — Generalize `keep_first_function_def` for a class contract

Current `keep_first_function_def(source, func_name="predict_next_state")` scans top-level `ast.FunctionDef` nodes matching one name and slices to the first complete one. This needs to work against a class body instead.

- Change target detection to look for `ast.ClassDef` nodes (not `ast.FunctionDef`) at the top level, matching whatever class name convention you settle on (e.g. `GameModel`).
- Within the matched class, handle two possible degrading-redraft patterns:
  1. **Multiple top-level class redefinitions** — keep the first, drop everything from the start of the second onward.
  2. **Multiple `def predict` (or other method) definitions inside one class body** — within the kept class's body, walk its `ast.FunctionDef` nodes and, per method name, keep only the first if there are duplicates, trimming the class body accordingly.
- Return signature: keep returning `(trimmed_source, num_defs_found)` — decide whether `num_defs_found` reports class-level duplicates, method-level duplicates, or both, so the loop's log line stays meaningful.

**Test before moving on:** hand-write 2–3 small synthetic Python strings covering: (a) one clean class, (b) two full class redefinitions, (c) one class with two `predict` methods inside it. Confirm the function trims each correctly with a throwaway script.

**Done when:** all three synthetic cases trim to exactly the first complete definition.

---

## Step 2 — Add `numpy` to `ALLOWED_IMPORTS` and verify it inside the sandbox

- Add `"numpy"` to the `ALLOWED_IMPORTS` set.
- Verify importability inside bwrap specifically — write a one-line throwaway candidate (`import numpy; ...`) and run it through the existing `run_candidate` sandboxed path.

**Done when:** a trivial candidate that does `import numpy` runs successfully through `run_candidate` with `use_sandbox=True`, with no `bwrap` bind errors. If it fails, confirm where numpy actually lives in the `eidolon` conda env (`python -c "import numpy; print(numpy.__file__)"`) and add its path to the `--ro-bind` list in `build_bwrap_command` if `conda_prefix`'s existing bind doesn't already cover it.

---

## Step 3 — Rewrite the sandboxed runner: state threading and abort-on-crash

- The existing `run_candidate`/`RUNNER_TEMPLATE` is already a single subprocess for the _entire_ input batch, looping over stdin lines with a per-row `SIGALRM` reset and an overall `subprocess.run(..., timeout=overall_timeout)`. **None of that needs to change.** The only real change here is threading state across the loop: instantiate the candidate's class once at the top of the runner, before the loop, then on each iteration call `predict(grid_before, action, previous_state)` and feed its returned `state` into the next iteration's `previous_state` argument. `grid_before` for each row always comes from the trace, never from a prior prediction.
- **Abort-on-crash:** if a `predict()` call raises an exception or times out, do not attempt to carry the last-good `state` forward and continue. Emit an error result for that row, then `break` out of the loop — every row from that point through the end of the current call's row range is implicitly scored as incorrect/missing by whatever calls this runner. This is a deliberate simplification (design doc §1); do not build recovery logic for this here.
- Per-row, the runner emits `{"step", "prediction", "goal"}` per line to stdout as it goes (extends the existing per-line JSON output with the new `"goal"` field) — scoring against ground truth happens in the parent process, using the trace records already loaded there, exactly as today.
- **Confirm explicitly (this was a point of confusion earlier — pin it down in a code comment so it isn't re-litigated later):** feeding the entire row range for the current call into the subprocess's stdin upfront, and only ever passing one row's `grid_before`/`action` into `predict()` per loop iteration, already guarantees a given row's prediction depends only on rows processed earlier in that same sequential pass — this is enforced by the loop's order, not by controlling how much data physically arrives in the input buffer.

**Done when:** a hand-written trivial `GameModel` (e.g., always returns the input grid unchanged, `goal=False`, empty state) runs end-to-end inside bwrap against a short synthetic trace, returning correct per-row results; and a second hand-written candidate that deliberately raises on some row correctly aborts and reports the abort point.

---

## Step 4 — Chunk boundary computation with level/whole-trace cutoffs

Small, self-contained piece of logic that Step 10's outer loop calls at the start of every chunk to decide how many new rows this chunk covers. Implement and unit-test it in isolation before wiring it into the loop — it's a pure function over already-loaded trace data (no LLM, no sandbox), so there's no reason to only discover a bug here once a real run is underway.

- Signature roughly: `next_chunk_boundary(records, prev_boundary, max_examples) -> int`, where `records` is the full cleaned trace (each record already carries whatever `levels_completed`/goal ground-truth field `preprocess`/`--score-key` extracted — see `Eidolon_Redesign.md` §2 and design doc §9, which already assume this field is available for goal scoring; boundary computation reuses the same field, no new extraction needed).
- Formula (design doc §2):

  ```python
  candidates = [prev_boundary + max_examples, len(records)]
  level_row = next_level_completion_row(records, prev_boundary)
  if level_row is not None:
      candidates.append(level_row)
  next_boundary = min(candidates)
  is_level_boundary = (next_boundary == level_row)  # False if level_row is None
  ```

  `next_level_completion_row(records, prev_boundary)` scans forward from `prev_boundary` for the first row where `levels_completed` increases relative to the previous row, and returns that row's index (inclusive — the level-completion row is the _last_ row of the chunk that ends there), or `None` if no such row exists before the end of the trace. (`None` can't go directly into `min()` alongside integers — hence filtering it out above rather than the simpler-looking but broken `min(a, b, None)`.)

- **No minimum chunk size** — if a level completes on the very next row after `prev_boundary`, `next_boundary` can be `prev_boundary + 1`. Do not add clamping/merging logic for this case.
- **No epsilon interaction** — epsilon (Step 8) is now a flat constant (`0.05`), not derived from rows-so-far at all, so there is nothing for this function to coordinate with on that front; the earlier "epsilon needs no special-casing for small chunks" comment is now moot rather than merely true, since epsilon no longer depends on chunk size in any way.
- **Return `is_level_boundary`, and be precise about what it describes:** this boolean answers "was _this_ boundary (the one `next_chunk_boundary` just computed, i.e. this chunk's own end) produced by a level cutoff?" — it does **not** describe whether the _current_ chunk's `EXTEND_TEMPLATE` should get the level-boundary `DESCRIPTION` note. That's a property of the chunk that starts _after_ this boundary, i.e. the next call. **Do not wire this return value directly into the same chunk's `EXTEND_TEMPLATE` call** — see Step 10, which is responsible for carrying `is_level_boundary` forward one chunk (the boundary computed while processing chunk _k_ is what chunk _k+1_'s `EXTEND_TEMPLATE` needs, not chunk _k_'s own). Getting this backwards is an easy, quiet bug: it doesn't crash, it just fires the `DESCRIPTION` note on the wrong chunk (one chunk too early) and never fires it on the chunk that actually needed it.

**Test before moving on:** hand-construct a small fake `records` list mirroring the design doc's worked example (`max_examples=20`, a level completes at row 45, trace length 80) and confirm `next_chunk_boundary` produces the boundary sequence `20, 40, 45, 65, 80` when called repeatedly with `prev_boundary` set to each previous result — and confirm the level-cutoff boolean is `True` only for the _call_ whose own boundary is 45 (i.e. the chunk 40→45), not for the call whose boundary is 65 (chunk 45→65) — that distinction is exactly the lag Step 10 has to account for.

**Done when:** the unit test above passes, including the boundary-boolean check attributed to the correct call.

---

## Step 5 — Build `EXTEND_TEMPLATE`, and update `PROMPT_TEMPLATE`/`build_examples_block`

New template and builder function for the extend round, plus shared changes to the example-rendering logic used by both `PROMPT_TEMPLATE` and `EXTEND_TEMPLATE`. Per design doc §4:

- Signature roughly: `build_extend_prompt(candidate_code, new_records, encoding="hex", is_level_boundary=False) -> str`, where `is_level_boundary` is supplied by Step 10's outer loop — specifically, the `is_level_boundary` value from the _previous_ chunk's `next_chunk_boundary` call (see Step 4's note on this), not from computing the current chunk's own boundary.
- Shows the candidate's current class source code in full, framed as "don't rewrite from scratch, extend or adjust this."
- Shows only the chunk's newly-introduced rows, using the (now-updated, see below) `build_examples_block` formatting — reuse that helper rather than duplicating it.
- Explicit language distinguishing this from the revision prompt: these are new, previously-unseen examples, not failures.
- Same "define each name exactly once, no multiple draft attempts" guardrail language as the existing templates, phrased for a class with potentially multiple methods.

**`build_examples_block` changes (shared by `PROMPT_TEMPLATE` and `EXTEND_TEMPLATE`, not `build_counterexamples_block`) — supersedes the earlier step-index/adjacency-sentence plan and fully implements what was previously tracked as a deferred follow-up:**

Showing a full encoded grid for every example in a contiguous window is redundant beyond the first — each subsequent starting grid is just the previous example's resulting grid, already fully implied by the previous example's diff. New signature: `build_examples_block(records, encoding="hex", is_extend=False) -> str`, where `is_extend` controls only the one extra "not the game's start" sentence below (`PROMPT_TEMPLATE` calls with `is_extend=False`, `EXTEND_TEMPLATE` with `is_extend=True`) — everything else is shared:

```python
def build_examples_block(records, encoding="hex", is_extend=False):
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
```

- **Precondition:** `records` must be genuinely contiguous — `records[i]["grid_before"] == records[i-1]["grid_after"]` for every `i > 0` — since the diff labels (`Grid i-1 -> Grid i`) assume it. True by construction for a chunk's round-1 batch (chronological, no shuffle); assert it explicitly in code rather than silently trusting it, since a silent violation here would produce a prompt that's internally inconsistent without erroring.
- `Grid i`'s overflow fallback (`Grid i in full`) reuses the same `Grid i` label as its header — not a separately-named "Resulting Grid" — so naming stays consistent whether the shown grid is `Starting Grid`, an ordinary `Grid i`, or an overflow fallback.
- This change applies only to `build_examples_block` — `build_counterexamples_block` (used by `REVISE_TEMPLATE`) is unchanged, since counterexamples need the full predicted-vs-actual grid comparison regardless of overflow, not a contiguous replay.

**Level-boundary `DESCRIPTION` flag (`EXTEND_TEMPLATE` only, uses Step 4's `is_level_boundary`):** when the chunk this `EXTEND_TEMPLATE` prompt is being built for starts immediately after a level-completion cutoff, insert an additional sentence into the object-description instruction below (before its normal step 1) explicitly telling the model this chunk starts a new level and the persisted `DESCRIPTION` should be checked against the new grid rather than assumed to still apply. Non-boundary chunks get the object-description instruction exactly as written below, unchanged.

**Object-description instruction (added to `PROMPT_TEMPLATE` and `EXTEND_TEMPLATE` only, not `REVISE_TEMPLATE`):** insert the following before the "write predict_next_state now" closing line. `PROMPT_TEMPLATE` version (three steps):

```
Before writing any code, describe the objects in the grid using less than 200 words. Note: multiple objects can be of the same type and an object can have multiple color values. Also, distinct objects tend to take up less than 20% of the grid's values but there are exceptions to this. There are three steps to accomplishing this task:
1: First, look at the color values in the "Starting Grid". Notice the shapes. Notice any objects that stand out with color values different from whatever colors dominate most of the grid (background regions).
2: Second, from the changed-cell lists above, notice any objects that have moved or changed.
3: Third, you may guess the purpose that each object has in the grid. Examples of object purposes are "player-movable object", "goal destination for player-movable object", and "object that changes another object's shape or color".
Write this as a comment/docstring in your class with label "DESCRIPTION:" so it carries forward to future revisions.
```

`EXTEND_TEMPLATE` version (four steps, first step consults the existing description):

```
Before writing any code, describe the objects in the grid using less than 200 words. Note: multiple objects can be of the same type and an object can have multiple color values. Also, distinct objects tend to take up less than 20% of the grid's values but there are exceptions to this. There are four steps to accomplishing this task:
1: First, look at the existing "DESCRIPTION:" comment/docstring in the provided class. The description may already be sufficient, or it may need revising given the new information above.
2: Second, look at the color values in the "Starting Grid". Notice the shapes. Notice any objects that stand out with color values different from whatever colors dominate most of the grid (background regions).
3: Third, from the changed-cell lists above, notice any objects that have moved or changed.
4: Fourth, you may guess the purpose that each object has in the grid. Examples of object purposes are "player-movable object", "goal destination for player-movable object", and "object that changes another object's shape or color".
Write this as a comment/docstring in your class with label "DESCRIPTION:" so it carries forward to future revisions.
```

Since the description is written directly into the class's own docstring/comment (labeled `DESCRIPTION:`), it persists automatically as part of "here is your current code" in every subsequent chunk's prompt — no separate storage or retrieval mechanism needed.

**Done when:** running the builder against a hand-written class string plus 2–3 fake contiguous fake records produces a prompt that reads correctly — exactly one `Starting Grid:`, each subsequent example correctly labeled `Grid i (trace step N)` with a diff keyed to the correct previous-grid label, the overflow fallback triggering and correctly labeled in a manufactured high-change-count case, the `is_extend` sentence present only when `is_extend=True`, and the level-boundary `DESCRIPTION` sentence present only when `is_level_boundary=True`.

---

## Step 6 — `run_backtest`: full replay, scoring, goal discounting

Core scoring function (and CLI subcommand), replacing `score`/`evaluate` for this design. Consumes the Step 3 runner's per-row output. Called potentially several times per chunk (baseline recompute, plus once per round — `--max-rounds` of them, see Step 8), always against the row range `0` through the current chunk's boundary.

- **Scoring per row** (design doc §9):
  - If `predicted_goal` is `True`: discount the grid comparison for this row. Decide explicitly whether "discounted" means excluded from grid-accuracy metrics entirely or scored-but-flagged separately, and note the choice in a code comment so it isn't ambiguous later.
  - Otherwise: score `predicted_grid_after` against `true_grid_after` via the existing `diff_grid`-based comparison logic (reuse `score_candidate`'s exact-match / changed-cell-accuracy machinery where it still applies).
  - Compare `predicted_goal` against ground truth (`levels_completed` increased). Both false positives and false negatives count as incorrect for counter purposes (design doc §9) — no exemption.
- **Persistent per-row record, richer than a bare counter** (design doc §6): stored durably across the whole run — not per chunk — as a JSON file in `--workdir` (`row_failure_counts.json`), read at the start of each backtest call and rewritten at the end. Per-row schema:

  ```python
  {
      "<step_index>": {
          "count": 0,                 # increment on incorrect, reset to 0 on correct — unchanged behavior
          "actual_grid": [...],        # ground truth grid_after; set ONCE (first time this row is scored), never overwritten again
          "actual_goal": False,        # ground truth goal-reached; set ONCE, never overwritten again
          "predicted_grid": [...],     # this row's most recent prediction; OVERWRITTEN every backtest run, correct or incorrect
          "predicted_goal": False,     # this row's most recent goal prediction; OVERWRITTEN every backtest run
          "error": None                # this row's most recent crash message if predict() raised on it, else None; OVERWRITTEN every backtest run
      }
  }
  ```

  `actual_grid`/`actual_goal` duplicate what's already in the cleaned trace — that's intentional, not an oversight: it makes this one file self-sufficient for Step 7 to build a complete counterexample from, without a separate lookup into the trace file. `predicted_*`/`error` always get overwritten on every run regardless of correct/incorrect, so a row that happened to be scored correctly on its most recent run simply has `predicted_grid == actual_grid` sitting there — harmless, since a correct row's `count` is 0 and it won't be selected by Step 7's top-k anyway.

- **Consistency with current-best code (design doc §7):** this file's `predicted_*`/`error` fields get overwritten by _every_ round's backtest, including rounds that later get rejected by Step 8's epsilon check. Step 6 itself does not handle this — Step 8 owns the commit/revert logic (a `row_failure_counts_best.json` snapshot) that keeps the file consistent with whichever code is actually current-best after each round's accept/reject decision. Do not add commit/revert logic here; `run_backtest` just reads-mutates-writes unconditionally on every call, as before.
- **Atomic writes:** write both `row_failure_counts.json` and (Step 8's) `row_failure_counts_best.json` via a temp-file-then-`os.replace()` pattern rather than an in-place write, so a process interruption mid-write can't leave either file corrupted/half-written. This is a small, mechanical addition — a shared helper (`atomic_write_json(path, data)`) used by both this step and Step 8 is the natural place for it.
- **Streak metric:** after each backtest call, compute the longest run of consecutive correctly-predicted rows across the row range just replayed. Report in `run_backtest`'s output and whatever gets logged — never fed into either prompt template.
- **Return value:** needs to report enough for Step 8's epsilon comparison and zero-failure early-stop check, and Step 9's reporting — at minimum, overall pass rate for this call's row range, the per-row pass/fail list (for updating counters and for Step 8 to detect "zero failing rows" — an empty failures list — directly, without recomputing it from the counters file), and the streak value.

**Done when:** `run_backtest` run against the Step 3 trivial candidate on a short synthetic trace produces sensible output — correct per-row pass/fail, an updated `row_failure_counts.json` with the full enriched schema (not just counts) correctly populated, a streak number, and a pass rate that matches hand-checking the synthetic trace. **Additionally, as a direct correctness check on the counters themselves:** independently recompute, from the raw trace file, how many times each row _should_ have failed by this point in a hand-traced sequence of rounds, and confirm the `count` values in `row_failure_counts.json` match exactly, row for row — this catches off-by-one or double-increment bugs that "the file has some content" checks alone would miss.

---

## Step 7 — Revision prompt builder (every round after round 1, in every chunk)

Rewrite `build_revise_prompt`/`REVISE_TEMPLATE` for the new contract. Per design doc §3, §6, §9:

- **Counterexample selection: plain top-k by failure count, no round-robin/diversity logic.** Pull from the **persistent per-row record across the entire trace-so-far** (Step 6's `row_failure_counts.json`), not restricted to the current chunk's new rows. Sort rows by `count` descending and take the top `--k`. Do **not** reuse `select_counterexamples`'s bounded-diversity round-robin-by-action logic here — that function stays as-is for the flat harness it already serves, but this design's selection is a plain top-k, full stop.
  - **Tie-break rule, precisely:** if the failure-count group straddling the `k`-th position has more members than the remaining slots, keep members in the order they appear in `row_failure_counts.json` (ascending step/trace order — the counter file is written in that order) and drop from the end of that tied group until it fits exactly.
  - **Worked example to test against:** `--k 10`, 6 rows at failure-count 5, 5 rows at failure-count 4. Expected selection: all 6 count-5 rows, plus the first 4 (in file order) of the 5 count-4 rows — the 5th count-4 row (last in file order among that tied group) is dropped, for exactly 10 total.
- **Rendering each selected row: read directly from its `row_failure_counts.json` entry, nothing else needed.** Now that the schema (Step 6) carries `actual_grid`/`actual_goal`/`predicted_grid`/`predicted_goal`/`error` alongside `count`, every selected row's counterexample can be built from that one file entry — no separate lookup into the trace file for ground truth, and no need to reach into whichever round's `run_backtest` call happened to run most recently for the prediction. This also means the row detail is guaranteed consistent with whichever code is genuinely current-best, _provided_ Step 8's commit/revert has already run for any prior round in this chunk — this function should never be called before that.
- **Superseded idea — do not implement:** an earlier draft considered biasing selection toward the earliest chronologically-failing row specifically. The chunked curriculum (Steps 3–5, including the new boundary-cutoff logic) already enforces this ordering structurally; plain top-failure-count selection (with the tie-break above) is sufficient and is what should be implemented.
- State explicitly whether the single most recent row of the current row range passed or failed.
- For each shown row: **false-positive goal prediction** (`predicted_goal=True`, `actual_goal=False`) → include `actual_grid` as normal. **False-negative goal prediction** (`predicted_goal=False`, `actual_goal=True`) → omit the grid diff (the row's `actual_grid` in this case is the _next level's_ initial grid per design doc §9, not a comparable continuation), explicitly label the row as a missed level-completion transition rather than an ordinary wrong-transition row.
- If Step 8's epsilon check rejected the immediately preceding round's candidate (i.e. this round is being built from an earlier round's — or the baseline's — code, not the immediately preceding round's output), include an explicit note to that effect, framed per design doc §7's substitution language.
- Do **not** include the streak metric anywhere in this prompt.

**Done when:** running the builder against hand-constructed fake `row_failure_counts.json` entries (using the full enriched schema, including a case that exercises the tie-break rule exactly as in the worked example above) produces a correct prompt for all three row cases (ordinary wrong-transition, false-positive goal, false-negative goal) built entirely from those entries — no test fixture should need to separately supply trace records or a round's raw backtest output alongside the counters file — plus a correctly tie-broken top-k selection, and correctly includes the substitution note when triggered.

---

## Step 8 — Per-chunk epsilon-reset code selection with baseline recompute

Per design doc §7. This is the step that changed most from the earlier draft's continuous best-so-far checkpoint — implement this version, not that one.

- **Epsilon: a flat constant, `0.05`, not derived from chunk size at all.** The earlier "10% of rows-so-far" formulation is dropped entirely — it had a units bug (it computed a row count, and the pseudocode below subtracted it directly from an accuracy fraction without normalizing) and, once normalized, reduced to a constant fraction regardless of N anyway, so there was nothing left to "recompute per chunk." Comparison is `candidate_accuracy >= current_best_accuracy - epsilon`, both sides plain 0–1 fractions, epsilon the same `0.05` everywhere, every chunk, every round — nothing chunk-dependent to compute here at all. **The right-hand side is allowed to go negative** (e.g. `current_best_accuracy=0.02`, `epsilon=0.05` → `-0.03`) — do not clamp it to 0 or special-case it; `candidate_accuracy` (always ≥ 0) trivially clears a negative threshold, which is the correct behavior for an already near-zero baseline.
- **Chunk 1:** no prior code exists, so there's nothing to recompute a baseline against. Round 1's candidate is simply backtested and becomes the starting point (`current_best`) going into round 2.
- **Chunk _k_ > 1, before round 1:** take the code chunk _k-1_ ended with, and run it through `run_backtest` (Step 6) against chunk _k_'s (larger) row range. This produces the fair, same-denominator baseline accuracy for chunk _k_, and becomes the initial `current_best` for the comparison chain below. This recompute — not epsilon — is what actually keeps every within-chunk comparison measured over the same "rows-so-far" denominator; epsilon itself has no denominator awareness and needs none.
- **Generalized round loop (not hardcoded to exactly two rounds — `--max-rounds` may be any positive integer, default 2), with a zero-failure early stop and a `row_failure_counts.json` commit/revert kept in lockstep with the accept/reject decision:**

  ```python
  # current_best_code/current_best_accuracy start as (baseline_code, baseline_accuracy)
  # for chunk k > 1, or (None, None) for chunk 1 (nothing to compare round 1 against).
  # Baseline recompute (chunk k > 1) is itself always an automatic "accept" — commit its
  # row_failure_counts.json state to row_failure_counts_best.json before this loop starts.
  EPSILON = 0.05
  for round_n in range(1, max_rounds + 1):
      candidate_code = run_round(round_n, current_best_code)  # extend for round 1, revise otherwise
      candidate_accuracy, failures = run_backtest(candidate_code, row_range=chunk_row_range)  # mutates row_failure_counts.json in place as a side effect
      if current_best_accuracy is None or candidate_accuracy >= current_best_accuracy - EPSILON:
          current_best_code, current_best_accuracy = candidate_code, candidate_accuracy  # accept
          atomic_copy("row_failure_counts.json", "row_failure_counts_best.json")  # commit this round's counter mutations
      else:
          log_substitution(round_n, rejected=candidate_code)  # reject, current_best unchanged (Step 9)
          atomic_copy("row_failure_counts_best.json", "row_failure_counts.json")  # revert: discard this round's counter mutations
      if not failures:  # zero failing rows across the FULL chunk row range this round
          log_early_stop(round_n, rounds_skipped=max_rounds - round_n)  # Step 9
          break  # no further rounds this chunk, regardless of remaining max_rounds budget
  # chunk k ends with current_best_code; row_failure_counts.json now matches it exactly
  ```

  Each round is only ever compared against the immediately preceding round's outcome (or the baseline, for round 1 when `k > 1`) — never against an earlier, non-adjacent round, and never re-comparing against the original baseline once round 2 has already been evaluated. With the default `--max-rounds 2` and no early stop, this reduces exactly to the original two-step comparison (round 1 vs. baseline, round 2 vs. round 1's outcome); higher values simply extend the same chain one more comparison per additional round.
  - **Zero-failure early stop, why it's placed after (not before) the accept/reject check:** a round with zero failing rows has the maximum possible `candidate_accuracy`, so it is mathematically guaranteed to satisfy `candidate_accuracy >= current_best_accuracy - EPSILON` and always gets accepted — there's no ordering hazard from checking `failures` after the accept/reject branch rather than before it. The `break` then prevents Step 7 from ever being asked to build a revision prompt with an empty counterexample list (see design doc §6's note that this should be treated as a bug if it ever happens, precisely because this early stop is supposed to make it unreachable).
  - **Why a single "best" snapshot is enough, not a per-round history stack:** a reject only ever needs to undo the _immediately preceding_ round's mutations (the comparison chain never skips ahead or compares non-adjacent rounds — see above), so `row_failure_counts.json` is always at most one round's worth of mutations away from being correct, and one snapshot file is always sufficient to restore it.
  - **Both `atomic_copy` calls should go through `atomic_write_json`'s temp-file-then-`os.replace()` pattern** (Step 6), not a plain in-place file copy, so an interruption mid-copy can't leave either file half-written.

- Persist whatever chunk _k_ ends with (`current_best_code`) as the code fed into chunk _k+1_'s `EXTEND_TEMPLATE` (Step 5).

**Done when:** a manually-simulated 3-chunk sequence (e.g. chunk 1 ends at 90%, chunk 2's baseline-recompute scores 80% on the new range, round 1 scores 60% (rejected — more than 5 percentage points below the 80% baseline), round 2 scores 78% (accepted — within 5 points of the 80% baseline)) — trace through this by hand or with a small unit test and confirm the logic picks the right candidate at each stage, **and** that `row_failure_counts.json` after round 1's rejection has been reverted to exactly what it was after the baseline recompute (not left holding round 1's mutations). **Additionally**, re-run the same test with `--max-rounds 3` (adding a round 3 that scores, say, 76% — within 5 points of round 2's 78% — and confirm it's accepted as chunk 2's final code, with `row_failure_counts.json` now committed to round 3's mutations) and with `--max-rounds 1` (confirm chunk 2 ends with round 1's extend-only result, compared only against the baseline, with no revision round at all) to confirm the loop generalizes correctly rather than only working for the default case. **Also test the early stop specifically:** simulate a chunk where round 1 (or, separately, an interior revision round) returns zero failing rows with `--max-rounds` set to 3 or more, and confirm the loop breaks immediately — no further rounds run, `log_early_stop` fires with the correct `rounds_skipped` count, and the zero-failure round's code becomes `current_best_code`. **Also test a low-accuracy chunk specifically** (e.g. baseline at 3%, epsilon=0.05 giving a negative threshold) and confirm the comparison still behaves correctly (accepts) rather than raising or needing a special case.

---

## Step 9 — Reporting

Per design doc §8. Implement alongside Steps 6–8, not as an afterthought:

- Per-chunk log entry: chunk number, row range covered, baseline accuracy (if applicable), per-round accuracy for every round that actually ran (round 1 through `--max-rounds`, or fewer if the zero-failure early stop fired), epsilon used (now a flat `0.05` constant for every chunk — still worth logging per-round for legibility even though the value never changes), per-round accept/reject, whether/where the zero-failure early stop fired, and which round's code the chunk ends with.
- A separate, explicit log line every time the "current-best" code actually changes within a chunk (not just embedded in the per-chunk entry) — makes replacement frequency easy to scan independently. **No separate logging needed for the `row_failure_counts.json` commit/revert (Step 8)** — a commit always coincides exactly with an accept, and a revert always coincides exactly with a reject/`log_substitution`, so the existing accept/reject log line already tells you which happened; a duplicate log channel would just repeat the same information.
- A separate, explicit log line every time the zero-failure early stop fires (`log_early_stop` from Step 8's loop) — chunk number, round it fired at, `rounds_skipped` — distinct from the code-replacement log line above, since this is a "nothing left to fix" event, not a "code got replaced" event.
- A running best-chunk-ending-accuracy-so-far across the whole run, tracked read-only (never used to select code), printed alongside the final candidate's accuracy at the end of the run.
- Reuse existing `summarize_scores`/`print_score_summary` for per-candidate accuracy detail (exact-match, changed-cell accuracy, by-action) — no new mechanism needed for this part.
- Streak metric printed per chunk, not embedded in any prompt.

**Done when:** a full run of the Steps 3–8 pipeline against a small synthetic multi-chunk trace produces a log that lets you answer, just by reading it, "which chunk ended with the best-performing code, did the final candidate end up worse than that, and did any chunk stop early because it hit zero failures."

---

## Step 10 — Wire it all into the chunked run-loop command

Adapt `cmd_run_loop` (or write a new command) to drive the chunked curriculum:

- Load the full cleaned trace once, chronologically, no split, no shuffle.
- Outer loop over chunks, each boundary computed by Step 4's `next_chunk_boundary` (not a plain `prev_boundary + max_examples` — that's now only one of three cutoffs it considers), until the boundary reaches `len(records)`. **Carry `is_level_boundary` forward with a one-chunk lag** (Step 4's note): before computing chunk _k_'s own boundary, `pending_level_boundary` already holds whatever `is_level_boundary` chunk _k-1_'s `next_chunk_boundary` call returned — that's the value chunk _k_'s `EXTEND_TEMPLATE` should use. Only _after_ computing chunk _k_'s own boundary does `pending_level_boundary` get overwritten with chunk _k_'s own `is_level_boundary`, ready for chunk _k+1_. (Chunk 1 has no predecessor, so `pending_level_boundary` starts `False` and chunk 1 never consults it anyway, since it uses `PROMPT_TEMPLATE` not `EXTEND_TEMPLATE`.) Within each chunk, an **inner loop over `range(1, args.max_rounds + 1)`** (Step 8's comparison chain, including its zero-failure `break`), not a hardcoded pair of rounds and not guaranteed to run all `args.max_rounds` iterations:
  - **Round 1 of any chunk:** chunk 1 builds `PROMPT_TEMPLATE` (Step 5's `EXTEND_TEMPLATE` builder is not used here); chunk _k_ > 1 recomputes the baseline first (Step 8), then builds `EXTEND_TEMPLATE` (Step 5) from the current-best code — passing the **lagged** `pending_level_boundary` value described above (not chunk _k_'s own, not-yet-computed-relevant `is_level_boundary`) so the level-boundary `DESCRIPTION` note fires on exactly the chunk that starts right after a level cutoff. Either way: call LLM, extract/trim code (Step 1), backtest (Step 6), apply epsilon selection (Step 8) — against the baseline for chunk _k_>1, or accepted outright for chunk 1 (no baseline to compare against). If this round's backtest returns zero failing rows, Step 8's loop breaks here and the chunk ends after round 1 — no revision round runs.
  - **Every round after round 1, up to `args.max_rounds`, unless the previous round already triggered the zero-failure break:** build the revision prompt (Step 7) from whichever code is current-best after the previous round, call LLM, extract/trim, backtest, apply epsilon check against the previous round's outcome (Step 8), and check for the zero-failure break again.
  - After the inner loop finishes (whether by reaching `args.max_rounds` or by the zero-failure break): update `row_failure_counts.json`/`row_failure_counts_best.json` (Step 8's commit/revert already keeps these correct round-by-round; nothing extra needed here beyond persisting them to disk if not already durable), log everything including any early-stop event (Step 9), persist the chunk's final code (`current_best_code` from Step 8) as input to the next chunk.
- `args.max_rounds` defaults to **2** (design doc §3) but is a real CLI value, not hardcoded — the existing `--max-rounds` flag is reused as-is, just with its default changed from the flat harness's `10` to `2` for this chunked command. The loop's real outer bound is still exhausting the trace's chunks, not rounds-to-convergence within one; `--max-rounds` sets a _ceiling_ on how many rounds happen inside each chunk, not a guarantee that they all run — the zero-failure early stop (Step 8) can end a chunk sooner.
- **Per-round pause / `--automatic` flag (design doc §3a):** by default, after each round's prompt (`PROMPT_TEMPLATE`/`EXTEND_TEMPLATE` for round 1, `REVISE_TEMPLATE` for every round after) is written to `--workdir`, pause and ask whether to continue before making that round's LLM call — same behavior already present in the existing flat `run-loop` command (`pause_for_confirmation`), just now firing at every per-chunk prompt-write point that actually runs (up to `args.max_rounds` of them per chunk, fewer if the zero-failure early stop cuts a chunk short) instead of the two a flat run had. `--automatic` skips every pause and runs unattended; reuse the existing helper rather than writing a new one.

**Done when:** the full loop runs end-to-end against a small synthetic multi-chunk trace (see Step 11) without crashing, correctly computing chunk boundaries (including at least one level cutoff and the final whole-trace cutoff), correctly attributing the level-boundary `DESCRIPTION` note to the chunk _after_ a level cutoff (not the chunk the cutoff ended), correctly running `--max-rounds` rounds per chunk (test with the default `2` and at least one other value — see Step 11), correctly stopping a chunk early on zero failing rows without attempting to build a revision prompt for it, correctly updating all persisted state each chunk (including the commit/revert files staying consistent with current-best code), pausing appropriately between rounds unless `--automatic` is passed, and producing a legible end-of-run report.

---

## Step 11 — Smoke test gate (before any GPU spend)

- Hand-write a synthetic trace long enough to span **at least 2 ordinary chunks plus one level cutoff plus the whole-trace cutoff** — mirror the design doc's worked example shape: e.g. `--max-examples 5`, a trace of length ~16, with a `levels_completed` increment placed partway through what would otherwise be the second or third chunk (not aligned to a `max_examples` multiple), and a trace length that also isn't a multiple of 5, so both cutoff types in Step 4 actually fire, not just the ordinary `max_examples` cap.
- Hand-write two candidates: one correct `GameModel`, one deliberately missing the switch-tracking field.
- Run the full Step 10 loop with `--automatic` at the **default `--max-rounds 2`** (no LLM involved yet — feed the hand-written candidates directly into the backtest/epsilon-selection machinery, bypassing the LLM-call step, to isolate whether the harness plumbing itself is correct; `--automatic` is used here specifically so this scripted test doesn't block on stdin), then **repeat with `--max-rounds 1`** (confirm every chunk ends after its extend round only, comparing solely against the baseline — no revision round runs at all) and **`--max-rounds 3`** (confirm a chunk correctly chains three comparisons: round 1 vs. baseline, round 2 vs. round 1, round 3 vs. round 2) — this is what actually exercises Step 8's generalized round loop rather than just its default-case behavior.
- Hand-construct a `row_failure_counts.json` fixture matching the tie-break worked example from Step 7 (6 rows at count 5, 5 rows at count 4, `--k 10`) and confirm the revision prompt builder selects exactly the expected 10 rows, dropping the correct one.
- **Level-boundary lag, tested explicitly (not just implicitly via the existing DESCRIPTION-flag check below):** using the trace's level-cutoff chunk boundary, confirm the `DESCRIPTION` note fires on the `EXTEND_TEMPLATE` prompt for the chunk that _starts_ right after the cutoff, and does **not** fire on the chunk that _ends_ at the cutoff (that chunk's own round 1, if any exists before it, should not have gotten the note either). This is specifically checking Step 10's one-chunk lag, not just Step 4's unit-level `is_level_boundary` computation (already covered in Step 4's own test) — the bug this catches wouldn't show up in Step 4's isolated test at all, only in the wired-together loop.
- **`row_failure_counts.json` commit/revert, tested explicitly:** force a scenario where a round gets rejected (e.g. feed a candidate that regresses more than 5 points against its baseline) and confirm that, immediately after the rejection, `row_failure_counts.json` is byte-for-byte identical to `row_failure_counts_best.json` from before that round ran — not left holding the rejected round's mutated predictions. Then force an accepted round afterward and confirm `row_failure_counts_best.json` updates to match the new `row_failure_counts.json`. Also confirm a subsequent revision prompt built after the reject-then-accept sequence renders `predicted_grid`/`predicted_goal` matching the _accepted_ code's actual behavior, not the rejected round's.
- **Negative-epsilon edge case, tested explicitly:** construct a chunk where the baseline accuracy is very low (e.g. under 5%) and confirm the epsilon check (`candidate_accuracy >= current_best_accuracy - 0.05`) still accepts a candidate without erroring or needing a special case, even though the right-hand side is negative.
- **Zero-failure early stop, tested explicitly:** run the correct `GameModel` (which should backtest clean, i.e. zero failing rows, on every chunk) with `--max-rounds 3`. Confirm every chunk ends after round 1 — round 1's own backtest already has zero failures, so `log_early_stop` fires with `rounds_skipped=2`, and Step 7's revision-prompt builder is never invoked for that chunk at all (no `prompt_round2.txt`/`prompt_round3.txt` should be written for that chunk). Separately, hand-construct a mixed case: a candidate that fails on chunk 1 (so round 1 → round 2 revision runs normally) but where round 2's revision happens to fix every remaining failure — confirm the early stop fires _after_ round 2 instead of round 1 in that case, and that chunk 1 still only ran 2 of the available 3 rounds.
- Confirm: Step 4's boundary sequence matches hand-calculation exactly, including which boundaries are level-cutoffs vs. the whole-trace cutoff; the level-boundary chunk correctly triggers the `DESCRIPTION` flag in the following `EXTEND_TEMPLATE` prompt (per the lag test above); the correct candidate backtests clean across every chunk (high pass rate, long streak, no persistent failing rows) _and_ correctly triggers the early stop as just described; the broken candidate produces a real, legible failure pattern; each chunk's baseline recompute correctly re-scores the prior chunk's candidate against the expanded (possibly cutoff-shortened) row range; the epsilon comparison behaves correctly at every boundary, including the small level-cutoff chunk and the negative-threshold case, across all three `--max-rounds` values tested.

**Done when:** all of the above behave as expected, and running without `--automatic` correctly pauses at each per-chunk prompt-write point (confirm this once manually at the default `--max-rounds 2`, separately from the scripted `--automatic` runs above). Do not proceed to Step 12 until this passes.

---

## Step 12 — Real run against `ls20`

- Point the Step 10 loop at the actual cleaned `ls20` trace, with the local model backend (`llama-cpp`, existing Qwen3-Coder-Next-UD-Q4_K_XL setup) and a real `--max-examples` value.
- Expect roughly `(trace_length / max_examples) * max_rounds` total LLM calls (`--max-rounds` rounds per chunk; `2` at the default) plus one local (non-LLM) baseline-recompute replay per chunk after the first — budget GPU time accordingly.
- Watch each chunk's printed/logged accuracy, streak, and any current-best-substitution notes as it runs.

**Done when:** the loop completes (covers the whole trace, or you make a judgment call to stop early if results are clearly not improving), and you have a final candidate plus a full per-chunk log.

---

## Step 13 — Manual overfit check (required before trusting the result)

Apply the procedure from design doc §14 to the final surviving candidate — inspect for hardcoded literal grid/coordinate matches tracing back to specific counterexample rows, confirm branching reads as conditions on `action`/`state`/structural grid properties, and cross-reference which rows were ever shown as counterexamples against which rows the final candidate gets right. Given this design's whole-trace-scoped counterexample selection (Step 7), this check matters more here than it would under a narrower-scoped design — don't skip it.

**Done when:** you've formed a clear judgment — genuine rule capture, memorization, or somewhere in between — and written that judgment down alongside the final accuracy number.

---

## Step 14 — Decide next step based on the result

- **Clear signal (well above ~6.25% baseline, real streaks, passes the manual overfit check):** local model has real capability under this harness — proceed toward the fuller Tier 1 build or a minimal planner/policy layer with more confidence.
- **Near-random or memorization-dominated result:** re-run this exact chunked design with `--backend openai` against a frontier model (design doc §16) before concluding anything about the local model specifically.
- **Either way:** the per-chunk log, the manual overfit judgment, and this document trail are themselves a documented, honest capability finding — worth keeping intact and legible regardless of which branch you end up on.
