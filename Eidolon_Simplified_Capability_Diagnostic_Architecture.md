# Eidolon — Simplified Capability-Diagnostic Architecture

Purpose of this document: a single reference collating the _simplified_ harness redesign discussed as a faster, cheaper alternative to the full Tier 1 redesign in `Eidolon_Redesign.md`, scoped specifically to answer one question — does the local model (Qwen3-Coder-Next-UD-Q4_K_XL) have any real capability to extract and encode an ARC-AGI-3 game's rules as an executable program — before investing further engineering time in either direction.

**Revision note:** this version supersedes an earlier draft of this same document that ran the entire trace through a single backtest pass per round, with a continuous best-so-far checkpoint and no explicit curriculum ordering. That approach is replaced here by a **chunked curriculum** (§2–§7) after concluding that a single-pass design risked forcing the model to reconcile late-game and early-game mechanics simultaneously in one prompt, with no way to confirm it had solidified earlier mechanics before being shown later ones.

---

## 0. Why this design exists, and what it deliberately is not

The original empirical result — best candidate under the old stateless, threshold-sampled harness scored 0.8% exact-match / 5.5% changed-cell accuracy on held-out `ls20` data, statistically near the ~6.25% random baseline — left two live explanations: the local model has no real signal-extraction capability, or the old harness's structural blind spots (no persistent state, no exhaustive certification, single-function contract) were suppressing capability that's actually there.

The textbook way to resolve that ambiguity cleanly is a frontier-model tie-breaker on a fixed harness (change one variable at a time). That step is being **deliberately skipped** here for cost and setup-time reasons. The tradeoff being knowingly accepted: if this design's local-model run comes back near-random, it will not be fully possible to distinguish "the local model has no capability" from "the harness still isn't right" — a weaker result than the clean two-outcome test would have given. If this design's local-model run is inconclusive or negative, **re-running this same design against a frontier-model backend is the documented fallback** (see §15).

This design is explicitly scoped as **cheaper than full Tier 1**, not as Tier 1 itself. Several Tier 1 components are dropped for simplicity even though they are known to add real value (§11), and several Tier 2 components are confirmed out of scope entirely unless this diagnostic shows real signal (§12).

### 0a. Deliberate divergence from Schema: offline batch-fitting, not live online refinement

Schema's actual mechanism (as understood from prior discussion) calls the model at each live timestep as an agent plays, refining its world model through passive correction on moves it would take anyway. This design does not do that. The entire `ls20` trace was already recorded, offline, before this test begins — there is no live agent taking actions during this test, and no timestep the harness "hasn't reached yet" in the sense a live agent would experience. This design is closer to fitting a model to an already-completed experiment log than to an agent learning in real time.

This matters for two reasons, both worth stating plainly rather than letting them surface as later confusion:

- It is **why** running the full trace-so-far through a backtest at every chunk boundary is not information leakage — the candidate's code never sees future rows _as input_ at any point it's making a prediction; only `state`, built strictly causally row-by-row within a single sequential pass, ever carries forward. The trace's later rows sitting on disk (or in a subprocess's stdin buffer) at the time an earlier row is scored is a data-transfer detail, not a causal one.
- It is **not** a substitute for testing how the candidate would perform if deployed live, choosing its own actions with only whatever it's seen so far. That is a genuinely different, harder problem (live/active exploration, explicitly out of scope — see §12) and this design does not answer it. A strong result here says the model can encode rules given a complete, already-observed record of the game; it says nothing about how well it would do driving the game itself.

---

## 1. Candidate contract: a class, not a single function

The old single-function `predict_next_state(grid, action) -> grid` contract is replaced with a class. A class with multiple methods and instance fields is expected to hold a game's logic far more naturally than one free function — it gives the model a coherent place to define and update hidden state alongside the transition logic that depends on it, rather than threading a state dict through a single call signature.

Required shape (exact method/field names to be finalized at implementation time, semantics fixed here):

```python
class GameModel:
    def predict(self, grid_before: list[list[int]], action: str, previous_state: dict) -> tuple:
        """
        Returns (predicted_grid_after, goal, state):
          - predicted_grid_after: list[list[int]], the model's prediction for this step
          - goal: bool, True if the model predicts this action reaches the next level
          - state: dict, JSON-serializable — all fields the candidate wants carried
                   into the next call as previous_state
        """
        ...
```

- `previous_state` is always the state dict the candidate itself returned on the prior call — never something the harness reconstructs or corrects. `grid_before` fed into `predict` is always the true grid from the trace (teacher-forced at the grid level), so a wrong prediction never contaminates the next step's grid input.
- **Known and accepted gap:** nothing corrects `state` against ground truth the way `grid_before` is teacher-forced. If the candidate's own state-tracking logic drifts, that error carries forward through every subsequent `previous_state` with no external signal telling it otherwise. This is treated as part of what's being tested, not a harness artifact to patch — see §6's streak metric for how this gets surfaced rather than hidden inside one flat accuracy number.
- **Abort-on-crash:** if a `predict()` call raises an exception or times out mid-replay, the entire replay for that call is aborted at the point of failure — no attempt is made to carry the last-good `state` forward and continue scoring subsequent rows. Every row from the failure point to the end of the current replay's row range is scored as incorrect. This is a deliberate simplification, not an oversight: a more resilient recovery path (continuing with stale-but-valid state) is a real improvement worth making in a later architecture pass, but is not required to answer this diagnostic's core question.
- The model may define as many additional methods/fields on the class as it needs; only `predict`'s signature is fixed by the contract.

---

## 2. Chunked curriculum: processing the trace in order, not all at once

Rather than building one round-1 prompt from a sample of the whole trace and then only ever revising against the full trace thereafter, the trace is processed in **chronological chunks**, each introducing a fixed number of new rows:

- Chunk size = `--max-examples` (the same knob that previously controlled round-1 sample size). Chunk 1 covers rows `0` to `max_examples`; chunk 2 covers rows `0` to `2 * max_examples`; chunk _k_ covers rows `0` to `k * max_examples` (each chunk's _new_ rows are the ones between the previous chunk's boundary and this one's). The final chunk may cover fewer than `max_examples` new rows if the trace length isn't an exact multiple.
- No `--shuffle`, still fully chronological — order is load-bearing.
- No history/held-out file split, and no automatic reserved slice withheld from counterexample selection (unchanged reasoning from the prior draft — see §13 for the manual check that substitutes for this).
- No code-file split of `trace_tools.py` into separate modules for this iteration (unchanged).
- **Superseded idea, noted for the record:** an earlier point in design discussion considered a fixed 40%-earliest/60%-most-recent row split specifically for the round-1 prompt, to avoid unbounded growth from a naive "show everything from the start" approach. The chunked curriculum makes this moot — chunk 1 already covers the earliest rows, and each subsequent chunk naturally adds the next recent rows in order, so no separate windowing decision is needed.

---

## 3. Two rounds per chunk: extend, then revise

`--max-rounds` is fixed at exactly **2 per chunk** in this design (not a tunable convergence threshold within a chunk) — the outer loop's real bound is the number of chunks needed to cover the whole trace, not rounds-to-convergence within one.

- **Round 1 — "extend" round.**
  - **Chunk 1** has no prior code, so round 1 is identical to the original from-scratch initial-synthesis prompt (`PROMPT_TEMPLATE`), built from chunk 1's row range.
  - **Chunk _k_ > 1** uses a new prompt, `EXTEND_TEMPLATE` (§4): shows the candidate its current class code plus only the _newly introduced_ rows for this chunk, and asks it to extend/adjust the class to handle them while preserving behavior that already works — framed like the revision prompt's "don't rewrite from scratch," but presenting new unseen examples rather than failures.
  - The resulting candidate is backtested (full replay, §5) against the _entire_ row range covered so far (rows `0` to this chunk's boundary), not just the new rows.
- **Round 2 — revision round.** Standard `REVISE_TEMPLATE`, built from the top currently-failing rows selected from the **full replay across the entire trace-so-far** (rows `0` to the current chunk's boundary), using the persistent per-row failure counter (§6) — not restricted to this chunk's new rows. Reasoning: game content builds on itself; a candidate that hasn't solidified early-game mechanics won't get late-game mechanics right either, so counterexample selection should always be free to reach back into earlier rows, not just the newest ones.
  - The resulting candidate is backtested (full replay) against the same row range again.
- Per-chunk code selection between round 1's and round 2's candidates is governed by the epsilon-reset scheme in §7, not by simply always keeping round 2's output.

---

## 4. `EXTEND_TEMPLATE`, and updates to `PROMPT_TEMPLATE`/`build_examples_block`

New prompt template, distinct from both `PROMPT_TEMPLATE` and `REVISE_TEMPLATE`, plus two changes shared by both `PROMPT_TEMPLATE` and `EXTEND_TEMPLATE`.

**`EXTEND_TEMPLATE` itself:**

- Shows the candidate's current class source code in full (same "don't rewrite from scratch, extend/fix what's here" framing as the revision prompt).
- Shows only the rows newly introduced by the current chunk (not the whole trace-so-far — that would defeat the point of a bounded per-chunk prompt).
- Explicitly instructs the model that these are **new, previously-unseen examples**, not failures — the framing is "extend or adjust your existing class so it also correctly handles these," distinct from the revision prompt's "here's what you got wrong, fix it."
- Same encoding options (`hex`/`rle`), same "define each name exactly once" / no-multiple-draft-attempts language as the existing templates, updated to talk about extending a class's methods rather than writing a function from scratch.

**Step-index / adjacency labeling (`build_examples_block`, used by both `PROMPT_TEMPLATE` and `EXTEND_TEMPLATE`):** since the window is now contiguous (§2), each example should carry its real trace `step_index` rather than a bare `### Example i` counter, and the block should state explicitly that consecutive examples are adjacent in the trace — i.e. example _i_'s `grid_after` is the same grid as example _i+1_'s `grid_before`. Without this, nothing in the rendered prompt tells the model the window is one continuous sequence rather than _N_ independent snapshots, which is the whole point of making it contiguous in the first place.

**Grid display change (`build_examples_block`):** show only a single `Starting Grid:` per example (not both before/after grids) — the changed-cell diff list already encapsulates the resulting grid for the normal case. **Exception:** if the number of changed cells exceeds `DIFF_MAX_CELLS` (the existing overflow threshold `format_diff` already uses), also show the full resulting grid for that example, labeled to explain why — `format_diff`'s own overflow message says "compare the full before/after grids above instead," which would be actively wrong if the resulting grid were unconditionally omitted. This change applies only to `build_examples_block` (round-1/extend path) — `build_counterexamples_block` (used by `REVISE_TEMPLATE`) is unchanged and continues showing both grids, since a counterexample's whole purpose is the predicted-vs-actual comparison.

**Object-description instruction, added to both `PROMPT_TEMPLATE` and `EXTEND_TEMPLATE` (not `REVISE_TEMPLATE`):** before writing code, the model is asked to describe the objects it can identify across the shown examples, in under 200 words combined, and to persist that description as a `DESCRIPTION:`-labeled comment/docstring in the class it submits — so it carries forward and gets shown back to the model (as part of "here is your current code") on every subsequent chunk, rather than being regenerated from scratch or discarded. This description is not restricted to `REVISE_TEMPLATE` because revision is deliberately scoped tightly to specific failing rows; if the persisted description needs updating in light of a failure, the model can do so opportunistically as part of a normal revision without a separately mandated step.

`PROMPT_TEMPLATE` version of the instruction (three steps, no prior description to consult):

> Before writing any code, describe the objects in the grid using less than 200 words. Note: multiple objects can be of the same type and an object can have multiple color values. Also, distinct objects tend to take up less than 20% of the grid's values but there are exceptions to this. There are three steps to accomplishing this task:
> 1: First, look at the color values in the "Starting Grid". Notice the shapes. Notice any objects that stand out with color values different from whatever colors dominate most of the grid (background regions).
> 2: Second, from the changed-cell lists above, notice any objects that have moved or changed.
> 3: Third, you may guess the purpose that each object has in the grid. Examples of object purposes are "player-movable object", "goal destination for player-movable object", and "object that changes another object's shape or color".
> Write this as a comment/docstring in your class with label "DESCRIPTION:" so it carries forward to future revisions.

`EXTEND_TEMPLATE` version (four steps, first step consults the existing description):

> Before writing any code, describe the objects in the grid using less than 200 words. Note: multiple objects can be of the same type and an object can have multiple color values. Also, distinct objects tend to take up less than 20% of the grid's values but there are exceptions to this. There are four steps to accomplishing this task:
> 1: First, look at the existing "DESCRIPTION:" comment/docstring in the provided class. The description may already be sufficient, or it may need revising given the new information above.
> 2: Second, look at the color values in the "Starting Grid". Notice the shapes. Notice any objects that stand out with color values different from whatever colors dominate most of the grid (background regions).
> 3: Third, from the changed-cell lists above, notice any objects that have moved or changed.
> 4: Fourth, you may guess the purpose that each object has in the grid. Examples of object purposes are "player-movable object", "goal destination for player-movable object", and "object that changes another object's shape or color".
> Write this as a comment/docstring in your class with label "DESCRIPTION:" so it carries forward to future revisions.

**Known soft limitation:** the "less than 20% of the grid's values" and "different color from the background" heuristics are reasonable-sounding guesses, not values measured against real `ls20` frames — worth tuning later if the description quality in practice suggests they're off. Relatedly, a stationary object that happens to share its color with the background cannot be detected by the background-contrast heuristic in step 1/2 — a real residual gap, in the same spirit as §15's distant-context limitation.

---

### 4a. Deferred follow-up: reformat contiguous example display to avoid redundant grids

Not implemented yet — noted here for a later pass. Even with the single-`Starting Grid:`-per-example change above, showing a full starting grid for _every_ example in a contiguous window is still redundant from example 2 onward, since each subsequent "Starting Grid" is just the previous example's resulting grid, already fully implied by the previous example's diff. The intended eventual fix: show `Starting Grid:` only once, for the first example in the window, then let each subsequent example be represented purely by its action + changed-cell diff (still falling back to a full grid only in the `DIFF_MAX_CELLS` overflow case). This needs a real formatting pass in `build_examples_block` to render correctly — not a one-line change — so it is being deferred rather than implemented alongside the rest of §4's changes.

---

## 5. Sequential backtest runner with full replay

- One subprocess call per (chunk, round) pair — and, per §7, an additional call at the start of each chunk _k_ > 1 to recompute the incoming baseline candidate's accuracy on the new row range.
- Each call replays the **entire row range from row 0 through the current chunk's boundary**, sequentially, in one subprocess, threading `state` between iterations exactly as described in §1. The candidate class is instantiated fresh at the start of each such call — `state` never persists _across_ separate subprocess calls, only within one sequential replay.
- Resource limits (`RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_NPROC`), the existing per-row `SIGALRM` reset, and the existing overall `subprocess.run(..., timeout=overall_timeout)` wrapper are unchanged from the current implementation — the only change from the pre-existing runner is threading `state` across loop iterations (§1) and aborting the full replay on any per-row exception/timeout (§1), rather than treating rows as independent.
- **On apparent cost:** replaying from row 0 at every chunk means total row-level executions across a full run grow roughly with the square of the number of chunks — but this cost is entirely local, sandboxed Python execution (list/grid comparisons), not LLM generation, which remains linear in the number of chunks (two LLM calls per chunk, plus the chunk-boundary baseline recompute, which is also local Python, not an LLM call). This is a deliberate trade of cheap local compute for correctness and fairness of comparison (§7), and does not meaningfully increase the actual bottleneck cost (LLM inference time) of running this design.

---

## 6. Per-row failure tracking and streak metric

- A persistent per-trace-row incorrect-prediction counter is maintained across **every** chunk and round for the whole run (not reset at chunk boundaries) — incremented when a row is predicted incorrectly, reset to zero when predicted correctly. This is what backs the whole-replay counterexample selection in §3.
- The revision prompt (round 2 of each chunk) shows the top incorrectly-predicted rows by this counter, bounded by token budget, reusing `select_counterexamples`'s bounded-diversity round-robin-by-action logic to choose which rows make the cut when there are more failures than fit.
- **Superseded idea, noted for the record:** an earlier point in design discussion considered biasing counterexample selection toward the earliest chronologically-failing row specifically, as a way to force the model to solidify early-game mechanics before being shown later ones. The chunked curriculum (§2–§3) now provides that ordering structurally — each chunk only ever introduces new rows after earlier ones are already established — so this extra selection bias is no longer needed; plain top-failure-count selection (as originally implemented) is sufficient.
- The revision prompt also states whether the single most recent (chronologically last) row of the current row range was predicted correctly.
- **Longest correct-prediction streak** (max run-length of consecutive correctly-predicted rows within a chunk's full replay) is tracked and reported per chunk (§8), but is **not** included in either prompt template. It exists to distinguish two failure modes that look identical in a flat accuracy number: near-random accuracy _with_ a long streak somewhere means the candidate can genuinely follow the rule for a while and something (a state-drift bug, a rare trigger) broke it partway through; near-random _with no streak ever exceeding a couple of rows_ is a much cleaner signal that no real rule-following is happening at all.
- **Known limitation, unchanged from prior discussion:** a counterexample may need context from an earlier trace row to be understandable, and this design does not surface any such earlier context automatically beyond what the failing row's own diff shows (see §14 for the full statement of this limitation).

---

## 7. Per-chunk epsilon-reset code selection with recomputed baseline

Replaces the earlier draft's continuous best-so-far checkpoint. Two problems needed solving simultaneously: (a) never letting a single revision regress the candidate drastically without any check, and (b) never letting the candidate get permanently stuck reverting to old code that ignores new rows it hasn't yet learned to handle, simply because old code's accuracy looks artificially better on a smaller row range than the new candidate scored on a larger one. This scheme addresses both by **only ever comparing accuracies measured on the same row range**, and resetting which comparisons are "live" at each chunk boundary rather than tracking one continuous global best.

- **Baseline recompute, chunk _k_ > 1 only:** before round 1 begins, re-run the code that chunk _k-1_ ended with through a full replay against chunk _k_'s (larger) row range. This produces a same-denominator "baseline" accuracy for chunk _k_ — the prior code was only ever previously scored against chunk _k-1_'s smaller row range, so this recompute is required before any fair comparison can happen. Chunk 1 has no prior code, so no baseline recompute occurs there.
- **Epsilon = 10% of rows-so-far at the current chunk's boundary** (not total final trace length, not a fixed row count) — this scales the tolerance proportionally to how much trace exists at each point, avoiding both over-triggering on small early denominators and being meaninglessly loose once the trace is long.
- **Within chunk _k_:**
  - Round 1 ("extend") candidate is compared against the recomputed baseline. If round 1's accuracy is not more than epsilon worse than the baseline, round 1's candidate becomes chunk _k_'s current best; otherwise the baseline (unextended prior code) remains current best going into round 2.
  - Round 2 ("revise") candidate is compared against whichever candidate is currently best after round 1 — both already scored on the identical row range, so no further recompute is needed here. If round 2 does not regress by more than epsilon, it becomes chunk _k_'s final code; otherwise chunk _k_'s current best (whichever of baseline/round-1 that was) is carried forward instead, and the substitution is logged explicitly (§8).
- **This "resets" every chunk** in the sense that no accuracy comparison ever crosses a chunk boundary directly without an explicit recompute — sidestepping the denominator-mismatch problem for within-chunk comparisons, while paying the recompute cost exactly (and only) where the denominator actually changes.
- **Accepted tradeoff:** this deliberately allows small chunk-to-chunk accuracy drifts to accumulate over many chunks with no global floor preventing it — chosen over the alternative failure mode of a candidate becoming permanently unable to progress past code that already ignores whichever new rows it's currently struggling with. Accumulating drift, if it happens, is made visible via the reporting in §8 rather than prevented structurally.

---

## 8. Reporting

- Per chunk, log: chunk number, row range covered, baseline accuracy (if recomputed), round 1 accuracy, round 2 accuracy, the epsilon value used this chunk, whether round 1 was accepted or rejected against the baseline, whether round 2 was accepted or rejected against round 1's outcome, and which code (baseline / round 1 / round 2) the chunk ends with.
- Log every code-replacement event as its own explicit line (not just embedded in the general per-chunk log), so how often the "current best" code actually changes is easy to scan independent of the full chunk-by-chunk detail.
- Track and report a running **best-chunk-ending-accuracy-so-far across the whole run** — read-only, purely for reporting, never used to select code — printed alongside the final candidate's accuracy when the run finishes, so any accumulated drift (§7) from the reset scheme is immediately visible without needing to reconstruct it from the full per-chunk log.
- Per-candidate backtest accuracy (exact-match, changed-cell accuracy, by-action breakdown) reuses the existing `summarize_scores`/`print_score_summary` machinery already present in `trace_tools.py` — no new mechanism needed for this part.
- The streak metric (§6) is reported per chunk's final backtest, never fed into either prompt template.

---

## 9. Goal prediction and discounting

- `predict` returns a `goal: bool` alongside the predicted grid. Ground truth for goal-reached: `levels_completed` increases between `pre_observation` and `post_observation`.
- If the candidate predicts `goal=True` for a step, the predicted grid is **discounted** in scoring rather than compared directly — the candidate cannot know the next level's grid, so grid comparison for a goal-transition step isn't meaningful in the same way as an ordinary step.
- Both **false positives** (predicted goal, wasn't actually reached) and **false negatives** (didn't predict goal, but it was actually reached) count as incorrect predictions for the per-row counter (§6) — no exemption.
- They are **not treated identically in what the revision prompt shows**:
  - **False positive:** the true `grid_after` (same-level continuation) exists and is shown as normal.
  - **False negative:** the true `grid_after` is the next level's initial grid, not a continuation of the same screen — there is nothing comparable to diff against, so the grid comparison is omitted. The row is still explicitly labeled in the revision prompt as a missed goal-transition, distinct from an ordinary silently-wrong row, so the model can learn "you're under-detecting goal conditions" as a distinct failure category.

---

## 10. Sandbox update

Add `numpy` to `ALLOWED_IMPORTS`. Matrix representations meaningfully simplify how a candidate can encode certain object/event tracking within `state`, and this pattern has been observed in comparable designs. Confirm it's importable inside the bwrap jail before relying on it — bwrap only binds what's explicitly listed, and the bound `conda_prefix` should cover it, but this needs verification, not assumption.

---

## 11. What stays unchanged from the existing harness

- Grid encoding (`grid_to_compact` / `grid_to_rle`, hex 0–15 palette), `diff_grid`/`format_diff`, the `--compact` RLE option, and their token-estimate caveats are unchanged.
- The sandboxed subprocess execution model (bubblewrap, `check_ast_imports`, resource limits, env wipe) is unchanged in spirit — it just now wraps a sequential rollout replaying from row 0 through the current chunk boundary, potentially called multiple times per chunk (baseline recompute + round 1 + round 2).
- Three prompt templates now exist rather than two: `PROMPT_TEMPLATE` (chunk 1, round 1 only), `EXTEND_TEMPLATE` (round 1 of every chunk after the first — new, §4), and `REVISE_TEMPLATE` (round 2 of every chunk, unchanged in purpose from the original design).
- `keep_first_function_def`'s underlying problem (Python silently keeps the _last_ redefinition of a name, and multi-draft LLM output consistently degrades with each redraft) still applies, but now needs to generalize to a **class** and its methods rather than a single top-level function.
- `build_llm_caller`'s two backends (`llama-cpp`, `openai`) are unchanged; the `openai` backend path is what would be used for the frontier-model contingency in §15, requiring no new plumbing.

---

## 12. Explicitly dropped from Tier 1 for this iteration (not because they lack value)

- **Collision detection** — a real, cheap, proof-grade check (identical `(state, action)` disagreeing on outcome proves a missing state variable) — dropped for architectural simplicity, not because it lacks diagnostic value. Worth adding back if this design's results are ambiguous and finer-grained diagnosis of _why_ is needed.
- **Round-over-round certified-step-index stagnation tracking** (the original Tier 1 doc's "stop if no improvement for 3 consecutive rounds") — not implemented in this design; the fixed 2-rounds-per-chunk structure and the epsilon-reset scheme (§7) substitute for it, at the cost of not detecting stagnation as an explicit early-stop condition.
- **Code-file split** of `trace_tools.py` into separate modules — deferred, not abandoned.
- **History/held-out split and any automatic reserved-slice mechanism** — replaced with the manual overfit check in §13.

## 13. Explicitly out of scope regardless of this diagnostic's outcome (Tier 2, unchanged from original doc)

Object segmentation & ID tracking, attribute-based type-grouping, coordinate/attribute correlation ranking, and the LLM history-query tool remain deferred: build only if this diagnostic shows a genuine plateau **and** a manual playthrough of `ls20` confirms it actually contains moving/variable-count objects or sparse hidden triggers that these components are meant to address. Also unchanged and out of scope: active/live exploration (see §0a), purely temporal hidden state with no visual correlate, MCTS, the topological/graph layer, the discrete matrix solver, and intra-task memory.

---

## 14. Manual overfit check (required step before trusting the final result)

Because no held-out slice is structurally protected from counterexample selection, and because counterexample selection is deliberately whole-replay-scoped (able to reach arbitrarily far back into the trace — §3, §6), the risk of the final candidate having simply memorized specific failing rows rather than generalizing is if anything higher in this design than in a version with narrower counterexample scope. This makes the following check more important, not less:

> Before trusting the final backtest's pass rate as evidence of real rule capture, manually inspect the surviving candidate's source for signs it's keyed to literal input values rather than to `action` and derived `state`. Concretely: look for conditionals comparing `grid_before` (or a slice/subgrid of it) against a hardcoded literal grid or coordinate list, especially ones that map directly to a specific counterexample row shown in an earlier revision prompt — that's the clearest tell the model patched a failure by memorizing it rather than generalizing. A healthy candidate's branching should read as conditions on `action`, on `state` fields the candidate itself defined and updates coherently, or on structural properties of the grid (e.g., "the cell adjacent to the player position," not "the cell at row 7, col 3"). As a sanity aid, cross-reference the list of rows that were ever selected as counterexamples across all chunks against which rows the final candidate gets right — if accuracy is high specifically on former-counterexample rows but conspicuously weaker on rows that were never shown as counterexamples, that pattern is evidence of memorization even without reading a single hardcoded literal in the code.

## 15. Known limitation: no automatic surfacing of distant related context

The revision prompt shows a failing row's own `grid_before`/`grid_after`/diff, but nothing further back in the trace that may have caused the state relevant to that row — e.g., a platform stepped on early in the trace that only matters when the player reaches a distant part of the map many steps later. If the candidate's `state` dict doesn't already track the relevant fact internally, the harness gives it no way to go looking for it; the model would have to have gotten it right from the `state`-carrying logic alone. This is not believed to be a live problem for `ls20` specifically (confirmed by direct familiarity with the game, not verified structurally), but is a real limitation of this design's counterexample presentation for any game with genuine long-range dependencies, and is the same underlying gap the Tier 2 LLM history-query tool (§13) was scoped to eventually address. Not a blocker for this diagnostic; worth revisiting if this design is later pointed at a different game.

## 16. Contingency: frontier-model rerun of this same design

If the local-model run under this design is near-random or otherwise inconclusive, the documented next step is to re-run this same chunked-curriculum design (unchanged — same chunking, same two-rounds-per-chunk structure, same epsilon-reset scheme) with a frontier-model backend (`--backend openai`, already supported by `build_llm_caller`), on the reasoning that this design's curriculum ordering, persistent whole-trace failure tracking, and fair chunk-boundary comparisons give a frontier model materially better odds of converging than either the old stateless harness or a naive single-pass version would. Two outcomes remain informative: near-random results even with a frontier model would point at a deeper representation/task-framing issue beyond what this design fixes; a frontier model converging where the local model didn't would confirm the local model's capability (not the harness) as the actual gate.

---

## 17. Summary of the question this design answers

Does the local model, given a harness that (a) introduces the trace to it in chronological chunks rather than all at once, (b) always shows it its persistently-worst-performing rows across the entire trace-so-far rather than only the newest chunk's failures, (c) never lets a within-chunk regression silently replace better-performing code without a fair, same-denominator check, and (d) is checked at the end for literal memorization rather than genuine rule capture — produce a `GameModel` whose backtest accuracy across the full `ls20` trace is meaningfully above the ~6.25% random baseline? A clearly negative result here, combined with the frontier-rerun contingency in §16, is what determines whether further investment goes into Tier 1/Tier 2 harness sophistication, into reconsidering the local model itself, or into documenting this as a real (if negative) capability finding.
