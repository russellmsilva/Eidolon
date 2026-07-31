# Eidolon — Simplified Capability-Diagnostic: Implementation Plan

Companion to `Eidolon_Simplified_Diagnostic_Design.md`. That document is the reference for _what_ and _why_; this document is the ordered checklist for _building it_ in `trace_tools.py`. Each step names the functions it touches, what "done" looks like, and where to stop and test before moving on. Build in this order — later steps assume earlier ones are working, and the smoke test (Step 10) is a deliberate gate before any real GPU time is spent.

**Revision note:** this version reflects the chunked-curriculum design (design doc §2–§8), replacing an earlier draft of this plan built around a single full-trace pass per round. Steps below that changed substantially from that earlier draft say so explicitly.

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

## Step 4 — Build `EXTEND_TEMPLATE`, and update `PROMPT_TEMPLATE`/`build_examples_block`

New template and builder function for the extend round, plus shared changes to the example-rendering logic used by both `PROMPT_TEMPLATE` and `EXTEND_TEMPLATE`. Per design doc §4:

- Signature roughly: `build_extend_prompt(candidate_code, new_records, encoding="hex") -> str`.
- Shows the candidate's current class source code in full, framed as "don't rewrite from scratch, extend or adjust this."
- Shows only the chunk's newly-introduced rows, using the (now-updated, see below) `build_examples_block` formatting — reuse that helper rather than duplicating it.
- Explicit language distinguishing this from the revision prompt: these are new, previously-unseen examples, not failures.
- Same "define each name exactly once, no multiple draft attempts" guardrail language as the existing templates, phrased for a class with potentially multiple methods.

**`build_examples_block` changes (shared by `PROMPT_TEMPLATE` and `EXTEND_TEMPLATE`, not `build_counterexamples_block`):**

- **Step-index / adjacency labeling:** replace the bare `### Example {i+1}` counter with the record's real trace `step_index`, and add one explicit sentence stating that consecutive examples are adjacent in the trace (example _i_'s `grid_after` is the same grid as example _i+1_'s `grid_before`) — this is what actually lets the model treat a contiguous window as one continuous sequence rather than _N_ disconnected snapshots.
- **Single-grid display with overflow fallback:** show only `Starting Grid:` per example (drop `grid_after` — the changed-cell diff already encapsulates it), _except_ when `len(changes) > DIFF_MAX_CELLS`, in which case also show the full resulting grid, labeled to explain why it's shown in this case. This matters because `format_diff`'s existing overflow message ("compare the full before/after grids above instead") would be wrong if the resulting grid were unconditionally dropped. Concretely:

```python
def build_examples_block(records, encoding="hex"):
    blocks = []
    for i, rec in enumerate(records):
        changes = diff_grid(rec["grid_before"], rec["grid_after"])
        block = (
            f"### Example (trace step {rec['step']})\n"
            f"action: {rec['action']}\n"
            f"\nStarting Grid:\n{encode_grid(rec['grid_before'], encoding)}\n\n"
            f"changed cells (grid_before -> grid_after), computed for you — real int "
            f"values, not display notation:\n{format_diff(changes)}\n"
        )
        if len(changes) > DIFF_MAX_CELLS:
            block += (
                f"\nResulting Grid (shown because too many cells changed to list "
                f"individually):\n{encode_grid(rec['grid_after'], encoding)}\n"
            )
        blocks.append(block)
    return "\n".join(blocks)
```

Add one line near the top of the rendered examples block (once per prompt, not per example) stating that consecutive examples are adjacent in the trace, per the step-index point above.

- This change applies only to `build_examples_block` — `build_counterexamples_block` (used by `REVISE_TEMPLATE`) is unchanged, since counterexamples need the predicted-vs-actual comparison regardless of overflow.

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

**Done when:** running the builder against a hand-written class string plus 2–3 fake new records produces a prompt that reads correctly, correctly labels examples by real step index with the adjacency sentence present, correctly omits `grid_after` in the normal case and includes it in a manufactured overflow case, and clearly distinguishes itself from what the revision prompt would say about the same rows.

**Deferred follow-up, do not implement now:** since the window is contiguous, showing a full `Starting Grid:` for every example is still redundant from example 2 onward even with the single-grid change above — each subsequent starting grid is just the previous example's resulting grid, already implied by the previous example's diff. The eventual fix is to show `Starting Grid:` only for the first example in the window, then represent each subsequent example purely via action + changed-cell diff (still falling back to a full grid in the `DIFF_MAX_CELLS` overflow case). This requires a real formatting pass in `build_examples_block`, not a small tweak, so it's being tracked here for a later implementation pass rather than folded into this step now.

---

## Step 5 — `run_backtest`: full replay, scoring, goal discounting

Core scoring function (and CLI subcommand), replacing `score`/`evaluate` for this design. Consumes the Step 3 runner's per-row output. Called potentially several times per chunk (baseline recompute, round 1, round 2 — see Step 7), always against the row range `0` through the current chunk's boundary.

- **Scoring per row** (design doc §9):
  - If `predicted_goal` is `True`: discount the grid comparison for this row. Decide explicitly whether "discounted" means excluded from grid-accuracy metrics entirely or scored-but-flagged separately, and note the choice in a code comment so it isn't ambiguous later.
  - Otherwise: score `predicted_grid_after` against `true_grid_after` via the existing `diff_grid`-based comparison logic (reuse `score_candidate`'s exact-match / changed-cell-accuracy machinery where it still applies).
  - Compare `predicted_goal` against ground truth (`levels_completed` increased). Both false positives and false negatives count as incorrect for counter purposes (design doc §9) — no exemption.
- **Persistent per-row incorrect counters** (design doc §6): stored durably across the whole run — not per chunk — e.g. a JSON file in `--workdir` (`row_failure_counts.json`, `{step_index: count}`), read at the start of each backtest call and rewritten at the end. Increment on incorrect, reset to 0 on correct.
- **Streak metric:** after each backtest call, compute the longest run of consecutive correctly-predicted rows across the row range just replayed. Report in `run_backtest`'s output and whatever gets logged — never fed into either prompt template.
- **Return value:** needs to report enough for Step 7's epsilon comparison and Step 8's reporting — at minimum, overall pass rate for this call's row range, the per-row pass/fail list (for updating counters and for counterexample selection), and the streak value.

**Done when:** `run_backtest` run against the Step 3 trivial candidate on a short synthetic trace produces sensible output — correct per-row pass/fail, an updated `row_failure_counts.json`, a streak number, and a pass rate that matches hand-checking the synthetic trace.

---

## Step 6 — Revision prompt builder (round 2 of every chunk)

Rewrite `build_revise_prompt`/`REVISE_TEMPLATE` for the new contract. Per design doc §3, §6, §9:

- Counterexample selection pulls from the **persistent per-row counter across the entire trace-so-far** (Step 5's `row_failure_counts.json`), not restricted to the current chunk's new rows — reuse `select_counterexamples`'s bounded-diversity round-robin-by-action logic for _which_ rows make the cut when there are more failures than fit the token budget.
- **Superseded idea — do not implement:** an earlier draft considered biasing selection toward the earliest chronologically-failing row specifically. The chunked curriculum (Steps 3–4) already enforces this ordering structurally; plain top-failure-count selection is sufficient and is what should be implemented.
- State explicitly whether the single most recent row of the current row range passed or failed.
- For each shown row: **false-positive goal prediction** → include the true `grid_after` as normal. **False-negative goal prediction** → omit the grid diff, explicitly label the row as a missed level-completion transition rather than an ordinary wrong-transition row.
- If Step 7's epsilon check rejected round 1's candidate (i.e. round 2 is being built from the baseline/prior-chunk code, not round 1's output), include an explicit note to that effect, framed per design doc §7's substitution language.
- Do **not** include the streak metric anywhere in this prompt.

**Done when:** running the builder against hand-constructed fake `row_failure_counts` plus a few fake row records produces a correct prompt for all three cases: an ordinary wrong-transition row, a false-positive goal row, and a false-negative goal row — and correctly includes the substitution note when triggered.

---

## Step 7 — Per-chunk epsilon-reset code selection with baseline recompute

Per design doc §7. This is the step that changed most from the earlier draft's continuous best-so-far checkpoint — implement this version, not that one.

- **Epsilon:** 10% of rows-so-far at the _current_ chunk's boundary (recompute this value fresh each chunk — it is not a fixed number and not based on total final trace length).
- **Chunk 1:** no prior code exists, so there's nothing to recompute a baseline against. Round 1's candidate is simply backtested and becomes the starting point for round 2's comparison.
- **Chunk _k_ > 1, before round 1:** take the code chunk _k-1_ ended with, and run it through `run_backtest` (Step 5) against chunk _k_'s (larger) row range. This produces the fair, same-denominator baseline accuracy for chunk _k_.
- **After round 1:** compare round 1's candidate's accuracy (on chunk _k_'s row range) against the baseline's accuracy (also on chunk _k_'s row range — both already same-denominator, no further recompute needed). If round 1 is not more than epsilon worse, it becomes chunk _k_'s current-best; otherwise the baseline remains current-best going into round 2.
- **After round 2:** compare round 2's candidate against whichever is current-best after round 1 (again same row range both times, no recompute). If round 2 is not more than epsilon worse, it becomes chunk _k_'s final code; otherwise chunk _k_'s current-best from the round-1 comparison is carried forward instead, and this substitution is logged explicitly (Step 8).
- Persist whatever chunk _k_ ends with as the code fed into chunk _k+1_'s `EXTEND_TEMPLATE` (Step 4).

**Done when:** a manually-simulated 3-chunk sequence (e.g. chunk 1 ends at 90%, chunk 2's baseline-recompute scores 80% on the new range, round 1 scores 60% (rejected, more than 10%-of-rows-so-far worse), round 2 scores 78% (accepted, within epsilon of the 80% baseline) — trace through this by hand or with a small unit test and confirm the logic picks the right candidate at each stage.

---

## Step 8 — Reporting

Per design doc §8. Implement alongside Steps 5–7, not as an afterthought:

- Per-chunk log entry: chunk number, row range covered, baseline accuracy (if applicable), round 1 accuracy, round 2 accuracy, epsilon value used, round 1 accept/reject, round 2 accept/reject, and which code the chunk ends with.
- A separate, explicit log line every time the "current-best" code actually changes within a chunk (not just embedded in the per-chunk entry) — makes replacement frequency easy to scan independently.
- A running best-chunk-ending-accuracy-so-far across the whole run, tracked read-only (never used to select code), printed alongside the final candidate's accuracy at the end of the run.
- Reuse existing `summarize_scores`/`print_score_summary` for per-candidate accuracy detail (exact-match, changed-cell accuracy, by-action) — no new mechanism needed for this part.
- Streak metric printed per chunk, not embedded in any prompt.

**Done when:** a full run of the Steps 3–7 pipeline against a small synthetic multi-chunk trace produces a log that lets you answer, just by reading it, "which chunk ended with the best-performing code, and did the final candidate end up worse than that."

---

## Step 9 — Wire it all into the chunked run-loop command

Adapt `cmd_run_loop` (or write a new command) to drive the chunked curriculum:

- Load the full cleaned trace once, chronologically, no split, no shuffle.
- Outer loop over chunks (chunk size = `--max-examples`) until the whole trace is covered:
  - Chunk 1: build `PROMPT_TEMPLATE` prompt (Step 4's builder is not used here), call LLM, extract/trim code (Step 1), backtest (Step 5), that's round 1; then build revision prompt (Step 6), call LLM, extract/trim, backtest again, that's round 2; apply epsilon selection (Step 7) with no baseline (chunk 1 special case).
  - Chunk _k_ > 1: recompute baseline (Step 7), build `EXTEND_TEMPLATE` prompt (Step 4) from the current-best code, call LLM, extract/trim, backtest — round 1; apply epsilon check against baseline; build revision prompt (Step 6) from whichever is current-best, call LLM, extract/trim, backtest — round 2; apply epsilon check against round 1's outcome.
  - After each chunk: update `row_failure_counts.json`, log everything (Step 8), persist the chunk's final code as input to the next chunk.
- `--max-rounds` is fixed at 2 per chunk in this design (design doc §3) — the loop's real bound is exhausting the trace's chunks, not a rounds-to-convergence threshold.

**Done when:** the full loop runs end-to-end against a small synthetic multi-chunk trace (see Step 10) without crashing, correctly updating all persisted state each chunk, and producing a legible end-of-run report.

---

## Step 10 — Smoke test gate (before any GPU spend)

- Hand-write a synthetic trace long enough to span **at least 2 chunks** — e.g. set a small `--max-examples` (like 5) for this test specifically, and write ~10–15 rows encoding a trivial rule (a switch-flip unlocks a door two steps later), including at least one `levels_completed` increment to exercise goal handling.
- Hand-write two candidates: one correct `GameModel`, one deliberately missing the switch-tracking field.
- Run both through the full Step 9 loop — **no LLM involved yet** (feed the hand-written candidates directly into the backtest/epsilon-selection machinery, bypassing the LLM-call step, to isolate whether the harness plumbing itself is correct).
- Confirm: the correct candidate backtests clean across both chunks (high pass rate, long streak, no persistent failing rows); the broken candidate produces a real, legible failure pattern; the chunk-2 baseline recompute correctly re-scores chunk 1's candidate against the expanded row range; the epsilon comparison behaves correctly at the chunk boundary.

**Done when:** all of the above behave as expected. Do not proceed to Step 11 until this passes.

---

## Step 11 — Real run against `ls20`

- Point the Step 9 loop at the actual cleaned `ls20` trace, with the local model backend (`llama-cpp`, existing Qwen3-Coder-Next-UD-Q4_K_XL setup) and a real `--max-examples` value.
- Expect roughly `(trace_length / max_examples) * 2` total LLM calls (two rounds per chunk) plus one local (non-LLM) baseline-recompute replay per chunk after the first — budget GPU time accordingly.
- Watch each chunk's printed/logged accuracy, streak, and any current-best-substitution notes as it runs.

**Done when:** the loop completes (covers the whole trace, or you make a judgment call to stop early if results are clearly not improving), and you have a final candidate plus a full per-chunk log.

---

## Step 12 — Manual overfit check (required before trusting the result)

Apply the procedure from design doc §14 to the final surviving candidate — inspect for hardcoded literal grid/coordinate matches tracing back to specific counterexample rows, confirm branching reads as conditions on `action`/`state`/structural grid properties, and cross-reference which rows were ever shown as counterexamples against which rows the final candidate gets right. Given this design's whole-trace-scoped counterexample selection (Step 6), this check matters more here than it would under a narrower-scoped design — don't skip it.

**Done when:** you've formed a clear judgment — genuine rule capture, memorization, or somewhere in between — and written that judgment down alongside the final accuracy number.

---

## Step 13 — Decide next step based on the result

- **Clear signal (well above ~6.25% baseline, real streaks, passes the manual overfit check):** local model has real capability under this harness — proceed toward the fuller Tier 1 build or a minimal planner/policy layer with more confidence.
- **Near-random or memorization-dominated result:** re-run this exact chunked design with `--backend openai` against a frontier model (design doc §16) before concluding anything about the local model specifically.
- **Either way:** the per-chunk log, the manual overfit judgment, and this document trail are themselves a documented, honest capability finding — worth keeping intact and legible regardless of which branch you end up on.
