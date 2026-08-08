# Eidolon — `analyze()` Preprocessing Pass: Implementation Plan

This document proposes adding a preprocessing step before the existing code-generation call that writes `GameModel` (the class whose `predict()` method forecasts the next grid state from the current one). Right now, that call is handed only the raw grid data — the starting grid plus a run-list diff of what changed on each recorded transition — and is asked to both notice what's structurally going on across many examples _and_ write a correct transformation rule from it, in one pass. This plan adds a second, smaller code-generation call, `analyze()`, that runs once per example, before `GameModel` is written, and computes structured facts about a single transition (what shape changed, whether it translated, whether it was recolored, and so on — see the Contract section below) directly from the raw grid data using a fixed, harness-provided shape-extraction primitive (Step 0). Its output gets fed into the `GameModel`-writing prompt _alongside_ the raw diff it's describing, never in place of it, so the model synthesizing `GameModel` sees both the original grid data and this second layer of computed structure on top of it.

The reason to expect this addresses the architecture's specific failure mode — characterizing noise, never forming a real concept, observed when the model was asked to hold cross-example comparisons in its head from raw diffs alone — is that this offloads exactly the part of that failure onto deterministic code instead of prose reasoning: figuring out "is this the same shape as before, just moved" is a mechanical comparison, not something that needs to be re-derived by the model from scratch, in text, under time and token pressure, for every one of the examples in a chunk.

**Contract, locked in:**

```python
def analyze(grid_before: list[list[int]], action: str, grid_after: list[list[int]], components: dict) -> str:
    ...
```

- Returns a plain string, not a dict. Nothing downstream ever parses this output programmatically — it only ever gets dropped straight into a prompt as text — so a string is simpler to render, simpler to cap (Step 3), and avoids JSON-serialization pitfalls entirely, at no cost to precision: the string is still built by deterministic code from computed values each call, not free-form model narration, so it carries the same "this is a computed fact, not a guess" guarantee a dict would have.
- **`components` is a fixed harness primitive (Step 0), not something `analyze()` has to derive itself.** It's the pre-extracted, shape-agnostic component list — `{"before": [...], "after": [...]}`, each entry an exact pixel footprint, not a bounding box or an assumed-rectangular block. This exists because a game's objects aren't always rectangles (circles, diagonal lines, arbitrary shapes all need to work), and because a bounding box alone doesn't carry enough information for `analyze()` to reason about what actually changed shape-wise. See Step 0 for exactly what each component carries.
- **Report structural category + relative motion, not restated raw coordinates.** `analyze()` only ever sees one transition, never the rest of the chunk, so it cannot know that a component is "the player" or "a switch" — that requires noticing patterns _across_ examples (moves whenever an action fires; changes only when another component overlaps it), which is squarely the synthesis-stage model's job once it's looking at the whole chunk together, not something a single-transition function can determine alone. What _is_ computable from one transition alone, now working from Step 0's exact-footprint components instead of assumed rectangles:
  - **translating** — a `before`-component's exact footprint matches an `after`-component's footprint at a different position. Checking this takes two separate steps, worth keeping distinct: **(1) how to compare two footprints regardless of position** — shift each footprint so its own bounding-box top-left corner sits at (0,0), then compare the two resulting coordinate sets for exact equality; this is just a way to put both shapes in the same frame of reference before comparing them, nothing more. **(2) a shortcut to avoid** — it's tempting to compare bounding box _dimensions_ (width × height) instead, since that's cheaper than comparing every cell, but this is wrong: two completely different, unrelated shapes can share the same bounding box while having different cell patterns inside it — a filled 5x5 square (25 occupied cells) and a hollow 5x5 ring (16 occupied cells, empty in the middle) both have a "5x5" bounding box, so a dimensions-only check would wrongly call them a match. This is a general warning about the comparison method, not a claim that any specific object turns from a square into a ring — the exact footprint (the full cell set), not the bounding box, is what must be compared. Once two footprints are confirmed to match, report the position delta relative to the action taken, e.g. "+5 cols in the action's direction," not as an absolute before/after coordinate pair — this now correctly handles a ring, a diagonal line, or any shape, not just rectangles.
  - **stationary, recolored** — a component occupies the _same_ position and the _same_ exact footprint in both grids, but its color differs.
  - **stationary, shape changed** — a component occupies overlapping/the same general position in both grids, but its exact footprint (the set of occupied cells) differs between before and after — cells added, removed, or rearranged, as distinct from simple recoloring (identical footprint) or translation (identical footprint, different position). This is the category a fixed-position indicator whose lit pattern rearranges (rather than just recoloring in place) falls under — worth calling out explicitly since "contents changed" undersells that it's a genuine shape change, not just a color flip.
  - **appeared**/**disappeared** — a component has no match at all in the other grid.
  - **ambiguous match** — two or more `before`-components have the same shape and color, and more than one `after`-component is a similarly good candidate match for each. When several pairings are close in quality (e.g. total displacement across candidate pairings is tied or nearly tied), report that the correspondence is ambiguous rather than silently picking one and stating it as fact. See the Known Limitations section for why this only catches ambiguity the matching heuristic itself can detect, not every wrong pairing.

  Reporting raw absolute coordinates that already exist in the adjacent run-list diff adds no information and directly invites the synthesis-stage model to overfit on specific positions — a bad output looks like `"12/9-colored 5x10 region moved rows 40-44,cols 34-38 -> rows 40-44,cols 39-43"` (just the diff restated in English); a decent one, now shape-agnostic, looks like `"A 42-cell ring-shaped component (color 9) in grid_before has no exact-footprint match at its own position in grid_after; a same-shaped, same-sized ring appears 5 columns over instead — translation, direction matches action."` — category and relative motion first, coordinates only as secondary detail.

- Matching components across `before`/`after` (which one corresponds to which, under what transformation) is entirely model-authored — Step 0 only extracts exact shapes, it does not attempt correspondence. See the Known Limitations section at the end of this document for why this remains a real, unsolved gap rather than something this design claims to have closed.
- Ground-truth `grid_after` only (never a `predict()` prediction).
- Synthesis-time only. Never called from `GameModel.predict()` — `analyze()` requires `grid_after`, which doesn't exist yet at inference time, so this isn't a scope restriction, it's what the contract itself already rules out.
- Regenerated (or extended) once per chunk, at round 1 only, fed that chunk's `max_examples` examples. Revision rounds within a chunk never regenerate it.
- Persists across chunks like `GameModel` does — round 1 of chunk N gets an EXTEND-style prompt carrying forward chunk N-1's version, not a blank slate.
- Callable afterward against any row from anywhere in the trace (current chunk, or a `REVISE` counterexample pulled from `row_failure_counts.json`), since it's a pure function of one recorded transition.

---

## Step 0 — Shape-agnostic component extraction (fixed harness primitive, prerequisite to `analyze()`)

Game objects in ARC-AGI-3 aren't always rectangles — circles, diagonal lines, and arbitrary shapes all need to work. A grouping scheme based on axis-aligned row/col ranges (the kind `format_run_list_diff` already does for display purposes) would fragment anything non-rectangular into a mess of nonsensical pieces, so this needs to be a genuinely separate, shape-agnostic primitive — not a reuse of that existing rectangular grouping.

- New function, e.g. `diff_components(grid_before, grid_after) -> dict`, returning `{"before": [...], "after": [...]}`. Each list contains **every** connected same-color component in the corresponding full grid — not just components that touch a changed cell. This matters because a wall, an untouched switch, or a goal marker that never changes on a given transition would otherwise never appear as a component at all, even though its position can be exactly the causal context that explains why something else did or didn't happen (e.g. why a moving object stopped where it did). Extracting from the full grid costs nothing in prompt tokens — `components` is a function argument passed into the sandbox, not something injected into the prompt directly; only `analyze()`'s own capped string output (Step 3) ever reaches the prompt — so the only cost is a bit more sandbox compute, which is cheap.
- Each component entry carries:
  - color (the single value shared by every cell in the component)
  - **exact pixel footprint** — the full set of `(row, col)` coordinates belonging to it, not just a bounding box. A bounding box alone can't distinguish a filled square from a hollow ring from a diagonal line of the same extent, and the model genuinely needs the real shape to reason about what changed, not a rectangle standing in for it.
  - bounding box and cell count, as cheap derived convenience fields alongside the exact footprint
- Extraction is pure adjacency-based grouping (flood-fill/BFS, 4-connectivity) — group cells that are adjacent AND share the same color into one component, done separately for `grid_before` and `grid_after`, each considered as a whole grid. This makes no assumption about geometry (rectangles, rings, diagonals, arbitrary blobs all fall out correctly as long as they're internally one color) and no assumption about what counts as "background" — a large uniform background region just extracts as one (typically large) component like anything else, with no special-casing; deciding which components are "interesting" versus "just background" is left to model-authored `analyze()` logic, not decided here.
- **A real limitation, worth stating here rather than glossing over it:** a single logical game object drawn with more than one color (like ls20's box sprite, which was two adjacent same-shape regions of different colors) extracts as two separate components under this scheme. Reuniting components into one logical object, if that's even useful for a given game, is left entirely to model-authored `analyze()` logic — this primitive only guarantees "same color, same connected region," nothing about game-level object identity. This and other limitations of the approach are collected in the Known Limitations section at the end of this document.
- This is a fixed, harness-provided utility, not model-authored — the same category of thing as `diff_grid()`, `format_run_list_diff()`, or having `numpy` available: assumption-free geometry, not a decision about what counts as meaningful game structure. `analyze()` calls it, but doesn't write it, and doesn't need to reinvent it under the same prompt discipline that already produced empty loop bodies once.

**Test before moving on:** synthetic before/after grid pairs covering (a) a solid rectangle translating (sanity check against the simple case), (b) a ring/circle-like shape translating, (c) a diagonal line, (d) two same-shaped, same-colored components present simultaneously (confirm both extracted separately, not merged into one), (e) a multi-color sprite (confirm it extracts as separate same-color components, per the limitation noted above, not silently merged or dropped), (f) a component that never changes between `grid_before` and `grid_after` (confirm it's still extracted from both, even though it never appears in `diff_grid()`'s output). Confirm exact footprints match hand-computed expectations for all six — bounding box alone is not sufficient to pass this test.

**Done when:** all six synthetic cases extract correct exact-footprint components, including the always-static one, with no rectangle-only assumption anywhere in the extraction logic.

---

## Step 1 — Function-based code extraction for `analyze()`

`extract_code` was recently fixed to scan every fenced block in a response and prefer the one containing `class {class_name}`, rather than blindly taking the first fence — this is what caught chunk 1's illustrative-snippet bug. `analyze()` is a bare function, not a class, so it needs the equivalent marker check.

- Generalize `extract_code`'s marker check from a hardcoded `class {class_name}` regex to an injectable pattern, or add a thin sibling that does the same multi-fence scan but matches `def analyze(` instead. Don't duplicate the whole scanning loop — the only thing that differs is the regex.
- Separately: `analyze()` needs its own "keep first complete definition, discard degrading redraft attempts" trimming, same purpose as `keep_first_class_def`. This is a function, not a class, which is exactly the shape the _original_ `keep_first_function_def(source, func_name="predict_next_state")` handled before it was generalized to a class contract — reuse or adapt that original logic rather than writing a new trimmer from scratch.

**Test before moving on:** hand-write 3 synthetic LLM-response strings: (a) one clean `def analyze(...)` fence, (b) a response with an earlier illustrative one-line fence followed by the real `def analyze(...)` fence (mirroring chunk 1's actual failure shape), (c) two full redefinitions of `analyze` in one response (degrading redraft). Confirm extraction picks the real one in (a) and (b), and trimming keeps only the first in (c).

**Done when:** all three synthetic cases resolve to exactly the first complete, correctly-identified `analyze` definition.

---

## Step 2 — Validation, NOOP fallback, and exception logging

Mirrors `_validate_candidate_code`'s role for `GameModel`, adapted for a function contract with no persistent state and a hard size cap.

- **Syntax/contract validation:** must parse, must define exactly one top-level `analyze(grid_before, action, grid_after, components)`, must only use `ALLOWED_IMPORTS`. Same rejection path as `GameModel` candidates on failure.
- **Per-example runtime fallback:** if `analyze()` raises or times out on a _specific_ example (as opposed to failing validation outright), that one example's slot in the prompt gets a fixed sentinel string — `"[preprocessing unavailable for this example]"` — not a blank, not the raw exception text, so the synthesis-stage model can't mistake a crash for a real "nothing changed" finding. Do not let one example's failure abort building the rest of the prompt.
- **Exception logging (explicitly requested):** every per-example failure — sentinel substitution, validation rejection, timeout — gets written to a persistent log (`analyze_errors.jsonl` or similar, one line per failure: chunk, round, example/row identifier, exception type + message, truncated traceback) alongside the existing per-chunk logging. This is for offline debugging, never shown to the model. Without this, per-example fallbacks are silent, and there would be no way to tell "the model wrote a solid `analyze()` that legitimately found nothing interesting in these examples" apart from "it's crashing on every single row and that failure is otherwise invisible."
- **Output contract:** must be a plain `str` (see the contract note above for why this replaced a `dict` return). Reject and fall back to the sentinel if the return value isn't a string.

**Test before moving on:** hand-write candidates covering (a) a clean function that returns a valid string, (b) a function that raises on a specific input, (c) a function returning something that isn't a string (e.g. a numpy value left unconverted, or a dict), (d) a function with a disallowed import. Confirm (a) passes through unchanged, (b)/(c)/(d) each produce the sentinel string for the affected example(s) and a corresponding line in the exception log — and confirm a batch containing one failing example among several passing ones still returns correct output for all the others.

**Done when:** all four cases behave as described, and the exception log is non-empty and readable exactly when a fallback fired.

---

## Step 3 — Hard output-size cap

Feeding an unbounded "computed facts" block back into the prompt risks recreating exactly the token-budget exhaustion that caused chunk 1's original failure — this cap exists specifically to prevent that.

- Enforce a hard cap on `analyze()`'s returned string length per example — pick a concrete number now rather than leaving it open-ended (e.g. start at 150 words / ~800 characters, matching the order of magnitude of the existing 200-word `DESCRIPTION` cap; treat as a tunable constant, not a fixed law).
- **This cap being enforced harness-side doesn't mean the generated code will actually respect it on its own** — nothing about being told "under 150 words" makes a model reliably write self-limiting code. Two layers, both required: (1) `ANALYZE_PROMPT_TEMPLATE` (Step 5) gives an explicit, copy-pasteable pattern for staying under budget — e.g. "collect description fragments in a list, join them, and truncate the final string with `textwrap.shorten()` before returning" — so there's a concrete recipe to follow instead of an abstract instruction to invent length management from scratch; (2) the harness-side cap here is the actual guarantee, independent of whether the generated code followed that pattern correctly.
- On overflow: treat it the same as a runtime failure — substitute the sentinel string, log it (Step 2's log, with a distinct reason code like `"output_too_large"` so it's distinguishable from a crash), do not silently truncate the string (an unexpectedly truncated fact can read as a complete-but-wrong one, which is worse than an honest sentinel).

**Test before moving on:** a candidate that returns a deliberately oversized string (e.g. dumps a large chunk of the grid verbatim into the text). Confirm it's caught and replaced with the sentinel, logged with the correct reason code, distinct from a crash-caused sentinel.

**Done when:** oversized output never reaches the prompt, and the log entry is distinguishable from other failure types.

---

## Step 4 — Sandboxed batch execution, reused for both generation-time and revision-time calls

Same `bwrap`/restricted-builtins infrastructure as `GameModel` candidates, but batched to avoid per-example sandbox spin-up cost, since this may be invoked many more times per chunk than `GameModel` is (once per example at round 1, plus again for every `REVISE` counterexample drawn from anywhere in the trace).

- **Components are computed before the sandbox call, not inside it.** `diff_components()` (Step 0) is fixed, trusted harness code — run it in the parent process for each row first, then pass `(grid_before, action, grid_after, components)` together into the sandboxed `analyze()` call. Only model-authored code runs inside `bwrap`; the shape-extraction primitive doesn't need sandboxing since it isn't model-generated.
- One sandboxed process call takes a _list_ of `(grid_before, action, grid_after, components)` rows and returns a list of results (string or sentinel, per Step 2), rather than spinning up `bwrap` once per row.
- Two call sites reuse this one runner:
  1. **Round 1 of each chunk:** run against that chunk's `max_examples` rows.
  2. **REVISE prompt construction:** run on-demand against whichever rows `select_top_k_failures` selected from `row_failure_counts.json` — these can be from anywhere in the trace, always evaluated with the _current_ persisted `analyze()`, not whatever version existed when that row was first seen.
- No new persistent state threading needed here (unlike `GameModel`'s `predict()` loop) — each call to `analyze()` is independent, so the batch runner doesn't need Step 3 of the diagnostic plan's state-threading logic, just a simple per-row loop with the same abort-on-crash-for-that-row (not the whole batch) behavior as Step 2 above.

**Test before moving on:** a small synthetic batch of 3-4 rows, one of which deliberately fails, run through the sandboxed batch runner. Confirm all rows are attempted (the failing row doesn't abort the batch), correct results come back for the passing rows, and the failing row's result is the sentinel with a corresponding log entry.

**Done when:** the batch runner works correctly inside `bwrap` for both a full-chunk-sized batch and a small on-demand batch (mirroring the REVISE use case).

---

## Step 5 — `ANALYZE_PROMPT_TEMPLATE` and `ANALYZE_EXTEND_TEMPLATE`

Two templates, mirroring `PROMPT_TEMPLATE`/`EXTEND_TEMPLATE`'s own split — chunk 1 has nothing to build on, chunk 2+ extends what's already there.

- **Initial (chunk 1):** given this chunk's `max_examples` examples (same run-list-diff formatting already used elsewhere) plus, for each, its Step 0 `components` dict, write `analyze(grid_before, action, grid_after, components)`. The template must explain what a component carries (color, exact footprint as a set of coordinates, bounding box, cell count) and how to compare two footprints for a shape match under translation: shift each footprint so its own top-left corner sits at (0,0), then compare the two coordinate sets for exact equality — and explicitly warn against the tempting shortcut of comparing bounding box dimensions alone, since two unrelated shapes (e.g. a filled square and a hollow ring) can share a bounding box while having completely different cell patterns inside it. This is the concrete recipe that replaces reinventing shape-matching from raw pixels. Explicitly _not_ told what to look for in game-specific vocabulary ("player," "switch," "hazard") — instead, the template names the small set of structural categories that are actually computable from a single transition (translating, stationary/recolored, stationary/shape changed, appeared, disappeared, ambiguous match) as illustrative anchors, and instructs it to report relative motion (delta + direction relative to the action taken) rather than restating absolute coordinates already present in the adjacent run-list diff — see the contract note above for why. For the ambiguous-match case specifically, give the model a concrete recipe rather than leaving "detect ambiguity" abstract: compute a matching cost (e.g. total displacement) for the best candidate pairing among same-shaped, same-colored components, and for the next-best alternative pairing; if the two costs are within some small tolerance of each other, report the correspondence as ambiguous instead of asserting one. Make clear these categories are a starting vocabulary, not an exhaustive or mandatory one; a game whose mechanic doesn't fit any of them should get a description in the model's own words rather than a forced-fit label.
- **Explicit numeric cap, stated in the template text itself, not left implicit.** Interpolate Step 3's actual cap value (characters and/or words) directly into both templates — e.g. "Your function's return value must never exceed {cap_chars} characters for any single example. If you're describing more than one blob or change, budget roughly {cap_chars} divided by however many you're describing, and truncate your own output to stay under this — do not rely on the harness to shorten it for you. Exceeding this limit discards your entire output for that example, which is worse than a shorter but complete one." Pair this with the concrete self-limiting code pattern from Step 3 (build a list of fragments, join, then `textwrap.shorten()` the result) so the model has both the number and a recipe for hitting it.
- **Extend (chunk 2+):** shows the current persisted `analyze()` source, shows this chunk's new examples, and asks it to extend/generalize rather than rewrite from scratch — or explicitly permits scrapping and rewriting if the current approach isn't working (unlike `GameModel`'s EXTEND prompt, there's no accuracy number to weigh this decision against, since there's no ground truth for "is this a good decomposition" — say so plainly in the prompt rather than implying a metric exists). Same numeric cap statement as the initial template, restated in full rather than assumed carried over.
- **Anti-memorization instruction, required in both:** the same warning already present in `PROMPT_TEMPLATE`/`EXTEND_TEMPLATE`/`REVISE_TEMPLATE` against hardcoding row/column/color values tied to the specific batch shown — an `analyze()` that hardcodes "the object is always at columns 34-38" is actively harmful once reused against other chunks, more so than the equivalent `GameModel` failure, since it silently poisons every downstream prompt rather than just failing backtest once.
- Same structural guardrails already proven necessary for `GameModel`: exactly one definition, no draft/rewrite attempts, every branch and loop body must contain a real statement, output must be under Step 3's cap, imports limited to `ALLOWED_IMPORTS`.

**Test before moving on:** render both templates against hand-constructed fake examples (initial) and a fake persisted `analyze()` plus new examples (extend). Confirm the anti-memorization and structural-guardrail language is present in both, the extend template correctly shows the prior version's source rather than omitting it, the structural-category vocabulary (translating/stationary-recolored/stationary-shape-changed/appeared/disappeared/ambiguous-match) and relative-motion instruction appear in both, the ambiguous-match tolerance-comparison recipe is stated concretely rather than left abstract, and the interpolated numeric cap matches Step 3's actual configured value (not a hardcoded placeholder that could drift out of sync if the cap constant changes later).

**Done when:** both templates render correctly and read coherently end to end.

---

## Step 6 — Inject `analyze()` output into `PROMPT_TEMPLATE`/`EXTEND_TEMPLATE`'s examples block

- One line above the whole `{examples}` block (not repeated per example) noting that preprocessing output may be stale or wrong relative to the model's current best `GameModel` — a single sentence is enough; do not spend tokens repeating this per example.
- Within each individual example's existing block, one additional line after its run-list diff: the corresponding `analyze()` output for that example (or the Step 2 sentinel, if it failed).
- **The Starting Grid line gets no `analyze()` call and no injected line at all.** `analyze()` requires a real `(grid_before, action, grid_after)` transition — the Starting Grid is the base reference state the first diff is computed against, not itself a transition, so there's nothing to call it with there. Only Grid1 through GridN (the actual per-example diff blocks) get an injected line.
- Never replaces the raw run-list diff — always additive, so a useless or wrong `analyze()` degrades gracefully back to exactly today's prompt, not to something worse.

**Test before moving on:** render `build_examples_block` (or whatever it's renamed to once this lands) against 2-3 fake examples with a mix of real `analyze()` output and sentinel fallbacks. Confirm the disclaimer line appears exactly once, each example shows its own computed line (or sentinel) directly after its diff, and the raw diff is never dropped or altered.

**Done when:** rendered output matches this shape exactly, for both `PROMPT_TEMPLATE` and `EXTEND_TEMPLATE`.

---

## Step 7 — Inject `analyze()` output into `build_revise_row_block`

- For each selected counterexample row, run the _current_ persisted `analyze()` (via Step 4's batch runner, called with just the selected rows) against `(grid_before, action, ground_truth_grid_after)` — never against the model's own incorrect prediction.
- Append the result (or sentinel) into that counterexample's block, same disclaimer-once-above-all-counterexamples pattern as Step 6, not repeated per row.

**Test before moving on:** hand-construct fake `row_failure_counts.json` entries spanning rows from different (fake) chunks, plus a fake current `analyze()`. Confirm `build_revise_row_block` correctly runs it against each selected row's ground truth (not prediction) and renders the result inline.

**Done when:** REVISE prompts built from counterexamples anywhere in the trace correctly carry the current `analyze()`'s output, computed fresh at build time.

---

## Step 8 — Wire into the per-chunk round loop

- Round 1 of each chunk: generate/extend `analyze()` (Step 5's templates) _before_ building that round's `GameModel` prompt, run it (Step 4) against the chunk's examples, feed its output into `PROMPT_TEMPLATE`/`EXTEND_TEMPLATE` (Step 6).
- Rounds 2+ within the same chunk: reuse that round's `analyze()` unchanged — no regeneration — but still re-run it (cheap, no LLM call) against whatever counterexamples `REVISE_TEMPLATE` selects (Step 7).
- Persist the current `analyze()` source alongside `GameModel`'s own persisted state between chunks (same commit/revert-adjacent bookkeeping pattern already used for `row_failure_counts.json`, though note Step 9 below on why there's no epsilon-reset-style accept/reject for this specific piece).
- **No epsilon-reset equivalent for `analyze()` itself** — there's no ground truth to score a decomposition against, so don't try to build a parallel accept/reject lifecycle. `GameModel`'s own accept/reject is the only judge of whether a given `analyze()` was actually useful, indirectly, round to round.

**No JarvisLabs needed here.** This step tests wiring _logic_ — when `analyze()` regenerates vs. persists, whether REVISE correctly re-runs it against arbitrary rows — none of which depends on code actually executing inside a sandbox. Stub Step 4's "run in sandbox" call with a direct Python function call (no `bwrap`) for this test; the sandbox mechanism itself is Step 4's job to verify, once, independently.

**Test before moving on:** a synthetic 2-chunk sequence, fake LLM responses, stubbed (non-sandboxed) execution. Confirm chunk 1 round 1 generates `analyze()` fresh; chunk 1 rounds 2+ reuse it unchanged in the `GameModel` prompt but re-run it against that round's counterexamples; chunk 2 round 1 gets the EXTEND-style `analyze()` prompt carrying chunk 1's version forward.

**Done when:** the full sequence behaves as described without regenerating `analyze()` outside of each chunk's round 1 — runnable locally, no GPU or JarvisLabs required.

---

## Step 9 — CI additions

Add to the existing `"Simplified Implementation Step Tests"/` directory (plain asserts/print-statements, no pytest/unittest, matching project convention) and wire into `tests.yml`'s existing per-file pass/fail loop:

- Step 1's extraction/trimming cases (clean function, illustrative-snippet-before-real-definition, degrading redraft).
- Step 2's validation/fallback/logging cases (clean, crash, non-serializable, disallowed import) — assert the exception log has exactly the expected number of entries with the expected reason codes.
- Step 3's oversized-output case.
- Step 6/Step 7's rendering tests (disclaimer-once, per-example inline placement, sentinel substitution visible in rendered text).
- **Do not** add a CI test that requires the actual sandboxed `bwrap` execution path or a real LLM call (Step 4's sandbox test, Step 8's full-loop test) — these need the JarvisLabs environment specifically and stay manual, same reasoning as the existing `llama-cpp-python`-excluded-from-CI convention. Pure-Python logic (extraction, validation, template rendering) is what's CI-eligible here.

**Done when:** all new pure-Python tests pass locally and are wired into `tests.yml`'s per-file reporting loop alongside the existing suite.

---

## Step 10 — Smoke test gate (before any GPU spend)

Mirrors the existing diagnostic plan's Step 11 gate — do not proceed to Step 11 below until this passes.

**No JarvisLabs needed here either.** Same reasoning as Step 8 — this is exercising pipeline _wiring_, not the sandbox mechanism itself, so stub the "run in sandbox" call the same way. Step 4 having already confirmed real `bwrap` execution works is what makes stubbing it safe here.

- End-to-end run of Steps 1-8, wired together, against a small synthetic multi-chunk trace (no real LLM, no real game data, no real sandbox execution — stubbed per above) with a hand-scripted fake "LLM" that returns fixed `analyze()`/`GameModel` source strings on each call, so the full pipeline (generation → stubbed execution → prompt injection → revision-time re-execution → persistence across chunks) can be exercised entirely locally.
- Confirm: `analyze()` generated once per chunk at round 1 only; persisted and correctly carried into chunk 2's EXTEND-style analyze prompt; correctly re-run (not regenerated) against REVISE counterexamples in later rounds; a deliberately-crashing fake `analyze()` produces sentinels + log entries without breaking the rest of the pipeline; the size cap correctly catches an deliberately-oversized fake output.

**Done when:** the full synthetic run completes without crashing and every behavior above is confirmed by inspecting the produced prompts and logs directly.

---

## Step 11 — Fresh run on JarvisLabs against `ls20`, via `run-chunked`

This is the actual experiment. Run it as a clean `run-chunked` invocation against `ls20` from scratch, rather than manually replaying the old frozen chunk 1 artifacts through `analyze()` in isolation — that would only test `analyze()` on its own, not the actual wired-together pipeline (generation → sandbox → prompt injection → REVISE re-execution → persistence), and a clean run exercises real integration rather than a hand-assembled approximation of it.

- `--max-rounds 2`, same as the original diagnostic run, so results stay comparable.
- If this points at the _same_ recorded trace file used before (not a freshly collected gameplay session), chunk 1's boundary and content will come out identical to the original run regardless, since `next_chunk_boundary` is a deterministic function of the trace — so the controlled comparison against the original staircase-hallucination result holds either way, and this just yields a cleaner test of the real pipeline instead of a manual approximation of it. If it's a newly collected trace instead, chunk 1's actual content could differ from the original run — not a problem, just worth knowing going in rather than assuming a like-for-like comparison.
- Capture, at minimum: the raw `analyze()` response for chunk 1 round 1 (same way `chunk1_raw_round1.txt` was captured before), the extracted/validated `analyze()` source, its per-example output across the chunk (or sentinels), the resulting `GameModel` prompt with that output injected, and the resulting `GameModel` candidate — same level of raw artifact capture as the original diagnostic run, since that's what made this whole conversation's diagnosis possible.

**Done when:** all of the above artifacts are saved for at least chunk 1, whether or not the run "worked" — a clean failure capture is exactly as valuable as a success here.

---

## Step 12 — Evaluate results against the two outcome buckets

Evaluate strictly against the two buckets already agreed on, using Step 11's captured artifacts. Don't let a partial or ambiguous result get rounded up to "real improvement" — the whole point of fixing these criteria in advance is to remove that judgment call under pressure.

**Same failure (→ this specific version of `analyze()` needs rethinking — record what didn't work in the Known Limitations section below before iterating further, rather than re-guessing from scratch):**

- `analyze()` itself rambles/truncates the same way `GameModel` did in the original run — never finishes, or produces syntactically invalid code despite the same guardrail language that (mostly) worked for `GameModel` elsewhere.
- `analyze()` runs cleanly but its output is degenerate — trivially empty, or itself hallucinated (confidently reports structure that isn't in the data, checkable by hand against the same kind of ground-truth reconstruction used earlier in this project to verify the box/switch/indicator mechanic).
- The resulting `GameModel` candidate, even with good `analyze()` output available, still produces the same kind of incoherent or overfit hypothesis as before — meaning this particular attempt didn't fix the bottleneck, not necessarily that the input-representation approach itself is wrong.

**Real improvement (→ keep this approach, continue building on it):**

- `analyze()` completes without truncation and without empty loop bodies.
- Its output, spot-checked by hand against a few examples the way the box/switch mechanic was verified earlier in this project, reflects real structure in the data — even if partial or imperfectly generalized, not necessarily correct.
- The resulting `GameModel` candidate engages with an actual pattern in the transitions rather than narrating a fictional one — this is the real bar, not "gets the mechanic fully right," since round-1 perfection was never the expectation.

Write the judgment down explicitly, the same way Step 13 of the diagnostic plan requires a written overfit judgment — a clear sentence, not just a feeling, since this is the record to point to later either way.

---

## Known Limitations and Failure Modes

This section exists because this design went through several iterations in the discussion that produced it — a relation-catalog approach, then object-identity tracking, then model-authored code with rectangular pre-extracted blocks — and each one had a real flaw surfaced by the next round of questions. There is no reason to assume this is the last flaw. Anyone picking this up, including a future collaborator, should read this section before extending the design further, and should expect to find problems not listed here.

- **Correspondence matching is not solved, and this design doesn't claim to solve it.** Step 0 supplies exact shapes; deciding which `before`-component corresponds to which `after`-component, and under what transformation (translation, rotation, deformation, partial occlusion), is entirely model-authored, per-game, ad hoc logic with no general algorithm behind it and no certification that it's correct. A model might write correspondence logic that only handles translation and silently gets rotation or deformation wrong, with no mechanism in this design to detect that.
- **No general fix for arbitrary transformation types exists here, and building one is a materially bigger effort than this plan's scope.** Translation-under-matching is one invariance; rotation, scaling, and deformation each need different matching logic, and there's no single known algorithm that handles "any transformation an ARC-3 puzzle might use." Hardcoding a growing checklist of invariances would reintroduce exactly the fixed-and-not-generalizable failure mode the model-authored-code approach was chosen to avoid.
- **Multiple simultaneous same-shaped, same-colored components have no principled disambiguation — and this is the same root gap as the long-range causality limitation below, just showing up within one transition instead of across many.** Components have no persistent identity: each call to `analyze()` sees one transition and produces shapes with no notion of "this is the same object as in the previous frame." If two identical objects both move in the same transition — say object A moves to where object B ends up, and object B moves to where object A ends up — a nearest-position matching heuristic doesn't fail or flag anything; it just as confidently computes and reports the swapped pairing, attributing each object's real motion to the other one, with nothing distinguishing that output from a correct one. This isn't a case of the matching code not being clever enough; it's a direct consequence of `analyze()` having no memory across transitions, the same fact that makes long-range causal relationships invisible to it. Whether a swapped pairing actually matters downstream depends on something `analyze()` can't know: if the two objects are genuinely interchangeable (same rule governs both), a swapped label is harmless; if they're visually identical but governed by different rules (e.g. the real player versus a decoy), a swapped pairing actively corrupts the evidence the synthesis-stage model reasons from. Step 5's ambiguous-match category (flag ties/near-ties in matching cost rather than asserting a pairing) reduces how often this happens silently, but only for cases where the matching heuristic's own uncertainty was detectable — a heuristic can still be confidently wrong (a clear, non-tied nearest-neighbor pick that's nonetheless factually incorrect), and no mechanism here catches that case at all.
- **Components are extracted from full grids now (Step 0), but occlusion still hides objects, and this isn't fixable.** Extracting from the whole grid instead of just changed cells (Step 0) recovers stationary objects like walls and untouched switches. What it can't recover is an object that's fully _covered_ by another object on a given transition — if the box's sprite completely overlaps the switch decal, the decal's cells show the box's color in that frame, not the decal's, so the decal simply isn't there to extract until the box moves off it again. This isn't a gap in the extraction logic; it's what "one thing sitting on top of another" means for a grid of colors, and no amount of better component extraction changes that a covered object's color isn't visible in the frame where it's covered.
- **Multi-color logical objects fragment into multiple components** (Step 0's stated limitation) — a sprite drawn in two colors, like ls20's box, extracts as two separate components with no built-in signal that they're one object. Whether `analyze()`'s own code reunites them correctly is untested and game-specific.
- **Long-range causal relationships across many steps remain entirely out of scope.** A switch hit at step 5 that only matters at step 25 is invisible to `analyze()` regardless of how good its per-transition component extraction is — this was already flagged and explicitly deferred in an earlier TODO, and shape-agnostic components don't change that; it's a different problem, not solved by this design at all.
- **No certification exists for `analyze()`'s own correctness**, unlike `GameModel`, which is checked against recorded ground truth via backtest. A confidently wrong or subtly hallucinated `analyze()` — reporting a translation that isn't really there, or a category that doesn't apply — has no mechanism to be caught directly; the only signal is `GameModel`'s own downstream accuracy, which is indirect and may be slow to reflect a bad `analyze()`.
- **Whether this actually reduces overfitting risk, versus just relocating it, is untested.** Structured component descriptions could still give the synthesis-stage model new specific things to overfit on (exact footprint sizes, cell counts) instead of raw coordinates — trading one kind of memorization risk for another rather than eliminating it. Step 11/12 is designed to surface this empirically; it is not something this design has established in advance.
- **Real added cost against the runtime budget:** an extra LLM call per chunk (for `analyze()` generation/extension) plus additional sandboxed execution overhead (once per example at round 1, again for every REVISE counterexample) is real time spent against the 12-hour ceiling, independent of whether the mechanism actually helps.
