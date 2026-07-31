# Eidolon Harness Redesign — Collated Design & Test Plan

Purpose of this document: a single reference collating the architecture redesign discussed for `trace_tools.py`, scoped into what to build now vs. defer, with known limitations stated explicitly. Intended to be carried into a follow-up conversation before implementation begins.

---

## 0. The question this redesign exists to answer

Original empirical result: best candidate under the _old_ stateless, threshold-sampled harness scored 0.8% exact-match / 5.5% changed-cell accuracy on held-out `ls20` data — statistically near the ~6.25% random baseline. This redesign exists to re-ask that question under a harness that doesn't have the old architecture's structural blind spots (no persistent state, no exhaustive certification, no representation-revision framing), so that a near-random result (if it recurs) is actually informative about model capability rather than confounded by harness limitations.

**Scoping decision:** build the **core loop only** first (Tier 1 below), smoke-test it, run it against real `ls20` data, and let that result determine whether Tier 2 (generalization/sparsity machinery) is worth building at all. Do not build both tiers simultaneously — if Tier 1 + Tier 2 together still produce near-random results, it will be unclear whether that reflects model capability or a bug in the added machinery.

---

## 1. File split (out of a single ~900-line `trace_tools.py`)

- `grid_codec.py` — `grid_to_compact`, `grid_to_rle`, `encode_grid`, `diff_grid`, `format_diff` (unchanged, move as-is)
- `trace_io.py` — `get_nested`, `extract_current_grid`, `preprocess`/`split` (needs real changes — see §3)
- `sandbox.py` — `build_bwrap_command`, `check_ast_imports`, `ALLOWED_IMPORTS`, runner execution (rewritten for sequential stateful rollout)
- `backtest.py` — new: teacher-forced replay driver, collision detection, regression/stagnation tracking
- `prompts.py` — `PROMPT_TEMPLATE`, branching `REVISE_TEMPLATE` (hidden-state vs. transition-rule framing), example/counterexample block builders
- `llm_backend.py` — `call_llm_openai`, `build_llm_caller`, `extract_code`, generalized `keep_first_function_def`
- `trace_tools.py` — argparse wiring only

---

## 2. Candidate contract (replaces single `predict_next_state`)

```python
def init_state() -> dict: ...
def step(state: dict, action: str) -> tuple[dict, grid]: ...
def sync_state(state: dict, action: str, true_grid_after: grid) -> dict: ...
def is_goal(state: dict) -> bool: ...
```

- `state` must be JSON-serializable (plain dicts/lists/numbers only) — required for logging and any equality/collision checks.
- `sync_state` is the teacher-forcing hook: after each step's prediction is scored against the true grid, `sync_state` resyncs internal tracking against ground truth so errors don't compound forward through the rest of the replay.
- `is_goal` ground truth: `levels_completed` field — `goal_reached = post_observation.levels_completed > pre_observation.levels_completed`. Certify `is_goal` against the **synced** (true-grid-derived) state, not whatever `step()` predicted, so `is_goal` bugs and `step()` bugs are diagnosed independently.

---

## 3. Trajectory-aware preprocessing

- Drop (or heavily restrict) the `--shuffle` flag in `split` — the current code's comment claiming shuffling is safe is now false, since state persists across steps and order is load-bearing.
- Add `levels_completed` extraction alongside the existing `--score-key` hook, to support `is_goal` certification.

---

## 4. Sequential sandboxed runner (Tier 1)

Rewrite the per-line-independent runner into **one sequential in-process rollout per candidate, per round**:

```
state = init_state()
for (action, true_grid_before, true_grid_after) in history (in order):
    state, pred = step(state, action)
    log(step_index, state, action, true_grid_after, pred)
    state = sync_state(state, action, true_grid_after)
```

- One subprocess call for the whole trajectory, not one call per record.
- Total `step`/`sync_state` calls per round = history length. Total across a run = history length × number of rounds. Nothing about the replay itself gets smarter between calls within a round — only the code being replayed changes between rounds (that's the outer loop, described in §6).

---

## 5. Collision detection (Tier 1 — free, proof-grade, but recall-limited)

- Within one round's logged replay, group steps by `(state, action)` (tolerance-aware equality for floats — exact `==` will falsely report "no collision" on floating-point noise).
- If two steps share identical `(state, action)` but disagree on `true_grid_after`: **proof** that the state is missing a variable that mattered — no possible transition formula can explain a collision. This is the minimal contradiction set (the colliding pair) for a **hidden-state bug**.
- If a step fails with no collision found anywhere in the replay: evidence (not proof) that it's a **transition-rule bug** — state was sufficient, the formula computed the wrong thing. Minimal evidence = that one step (optionally + one correct neighboring step for contrast).
- **Known limitation:** collision detection has a real recall problem in a single non-repeating playthrough. If a hidden trigger (e.g., a switch) is only ever touched once in the whole trace, there may never be a second occurrence of the same tracked state to collide against — a missing variable can go completely undetected. Absence of a detected collision is _not_ proof the state is sufficient, only that this particular test didn't catch it.
- Given the above, don't force a strict binary framing in the revision prompt. When no collision is found: present the single-step evidence, but also track whack-a-mole (formula-only patches applied over several rounds with no improvement in certified-step-count) as a softer signal that a representation gap may exist even without a detected collision, and let the prompt raise both possibilities rather than asserting one confidently.

---

## 6. Round-over-round regression & stagnation tracking (Tier 1 — your collated notes, included verbatim)

> Track the certified-step-index set round over round. If a step that passed in round N-1 fails in round N, that's a regression — flag it loudly in the log, and feed it into the next revision prompt alongside the new failures, explicitly labeled as "this used to work, your last fix broke it" so the model doesn't blindly re-break what it just repaired. Without that, you can get exactly the ping-pong you're worried about: round 2 fixes 12/breaks 45, round 3 fixes 45/breaks 12, forever, with the harness only reporting "still not certified" and no visibility into why it's stuck. Worth also tracking simple stagnation as a signal: if the certified-step count doesn't monotonically improve over a few consecutive rounds, that's itself decent evidence you're looping on a genuine representation gap, not a formula typo — same qualitative conclusion as a collision, just observed across rounds instead of within one.

> Practically: keep `--max-rounds` as a hard stop (already in the design), log certified-step-count and wall-clock time every round, and set an explicit early-stop rule now — e.g., stop if certified-step-count hasn't improved over 3 consecutive rounds — so a stalled run doesn't quietly burn through your full GPU budget before you notice it's flat. That gives you a bounded-cost experiment either way: converges (great, you have your answer and a round count), or plateaus early (also your answer, and you stopped paying for it once the trend was clear).

Implementation notes:

- Since every round replays the _entire_ trajectory fresh (§4), you have the complete pass/fail set for every round for free — regression detection is a straightforward diff between round N-1's and round N's certified-step sets.
- Log per round: certified-step-count, wall-clock time, list of regressions (if any).
- Two separate budgets to track against, not one: (a) your JarvisLabs dev-time experimentation budget, where the useful output is the _trend_ in certified-step-count (converging vs. plateauing is itself a complete answer, even short of 100%); (b) the actual Kaggle submission's runtime budget (shared single GPU, per-game time limit), which must be checked separately once you know how many rounds convergence actually takes.

---

## 7. `run_backtest` CLI command (Tier 1)

Replaces `score`/`evaluate` for this architecture: full teacher-forced replay, certified only on exact match at every step (100% is the right default bar — ARC-AGI-3 transitions are deterministic given full state; a lower threshold would just quietly tolerate a representation gap). Exit codes 0/1/2 as before. Writes the round's contradiction set (collision pair or single-step evidence) instead of `select_counterexamples`'s round-robin sampling — but reuse `select_counterexamples`'s bounded-diversity logic to pick _which_ failing steps get evidence assembled for, when there are more failures than fit the token budget.

---

## 8. Branching revision prompt (Tier 1)

Two framings, gated strictly by whether the evidence is a _proven_ collision or not:

- **Collision found → confident hidden-state framing.** Forbid patching with if/else; require rewriting `init_state`/`step`'s schema to add the missing field.
- **No collision → present single-step evidence without overclaiming.** Include the "N rounds of formula-only patches without certified-step-count improvement" signal if present, and let the prompt suggest (not assert) that a representation gap may be the real issue.

---

## 9. Generalize `keep_first_function_def` (Tier 1)

Extend to scan for and keep the first complete definition of `init_state`, `step`, `sync_state`, and `is_goal` independently — the original degrading-redraft failure mode (model writes a good version, then a worse "actually, let me reconsider" redraft later in the same response, and Python's redefinition semantics silently keep whichever is _last_, which is consistently the more degraded one) applies to each of the four names independently, not just to a single function.

---

## 10. Sandbox update (Tier 1)

Add `numpy` to `ALLOWED_IMPORTS`; confirm it's importable inside the bwrap jail (bound `conda_prefix` should cover it, but verify — bwrap only binds what's explicitly listed).

---

## 11. Smoke test (before any GPU spend)

Hand-written synthetic trace (~10-15 steps) encoding a trivial switch-unblocks-goal rule, plus two hand-written candidates: one correct, one deliberately missing the switch-tracking field. Run both through `run_backtest` directly, no LLM involved. Goal: confirm the sequential runner, JSON state logging, and collision detector are wired together correctly (subprocess protocol, serialization, equality tolerance) before spending JarvisLabs time. Expect: broken candidate produces a real logged collision; correct candidate certifies clean.

## 12. Real test

Point `run-loop` at the actual `ls20` history/held-out split with Qwen3-Coder-Next, run for a bounded number of rounds, watch whether `run_backtest`'s certified-step-count ever climbs meaningfully past the old 0.8%/5.5% baseline, and whether it converges, plateaus (with the 3-round stagnation stop), or hits `--max-rounds`. This result is the actual answer to the foundational question — model capability vs. harness architecture as the bottleneck.

---

## Tier 2 — deferred generalization machinery (build only if Tier 1's real test shows a plateau AND you confirm `ls20` actually contains moving/variable-count objects or sparse hidden triggers)

These exist to handle cases beyond fixed-position hidden triggers (switches, counters). Do not build alongside Tier 1 — confirm from your own manual `ls20` playthrough that these cases are actually present before investing here.

### 12a. Object segmentation & ID tracking (fixed infrastructure, not candidate-authored)

- **Segmentation** (per-frame): connected-component labeling on same-color adjacent cells → blobs with centroid, bounding box, size, color. Generic, game-independent, belongs in `grid_codec.py`.
- **Matching** (across consecutive frames): assignment minimizing total centroid displacement (subject to compatible color/size) → stable IDs across frames. Also generic, fixed infrastructure — not something the LLM should have to author per game.
- **Merges/splits**: detecting that one occurred is mechanical (blob count changed, overlapping regions span multiple prior IDs); _what identity to assign afterward_ (inherit one parent's ID / fresh ID / provenance-tagged) is a genuine judgment call with no single correct answer — recompute IDs fresh each round, and treat provenance choice as revisable via the same counterexample loop as everything else (does this provenance choice explain subsequent transitions better than the alternative).
- ID-persistence itself is a cheap invariant, certifiable directly (e.g., don't let an ID teleport further than the game's max per-step displacement, don't let two IDs swap).

### 12b. Attribute-based type-grouping (revisable, not assumed)

- Default heuristic: group objects by color/shape as a proxy for semantic role (reasonable specifically because ARC-style puzzle conventions tend to use color/shape to signal role).
- Must be revisable — an explicit `type`/`class` tag in the candidate's own `state` schema, correctable by the counterexample loop if pooling two objects as "same type" produces worse predictions than splitting them.
- Purpose: fights the sparsity problem — pools multiple rare single-occurrence events (e.g., two different switches touched once each) into a larger sample for correlation, where each alone would be statistically weak.

### 12c. Coordinate/attribute correlation ranking

- For a target outcome (e.g., "this barrier cell opened"), gather every occurrence across the whole replay, split into two groups (occurred / didn't), and for every other coordinate or tracked-object attribute, score how cleanly its value separates the two groups (correlation / fraction-correctly-predicted). Rank, surface top candidates.
- Requires multiple occurrences to have statistical footing — weak-to-useless on true one-off events.
- **Known limitation, explicitly identified:** univariate by design. Does not cleanly surface conjunctive/AND conditions (e.g., a gate requiring _all_ of several platforms hit) — each individual condition may look like a weak, noisy predictor alone even when the conjunction is the real rule. Falls back to either presenting several top candidates together for the model to reason about jointly, or the LLM query tool (12d).
- Does not generalize to non-spatial causes (e.g., a switch on the opposite side of the map from its effect) via _spatial proximity_ — must be purely statistical (per-coordinate/attribute correlation against the outcome label), with no distance term, or it will fail exactly this case.

### 12d. LLM history-query tool (most speculative, least load-bearing — build last, if at all)

- Instead of the harness proactively pre-selecting evidence, expose a small fixed set of read-only, sandboxed query primitives over the logged replay (e.g., `find_last_change(coord)`, `find_coincident_changes(step_index)`, `find_proximity_events(coord_a, coord_b, max_distance)`) that the model can call during a revision round to test its own hypothesis.
- Cap queries per round (e.g., 3) to bound cost.
- Advantage over harness pre-selection: avoids both omission (harness guessed the wrong evidence to include) and bloat (harness over-included defensively) — the model only sees the answer to the specific question it asked.
- **Real limitation:** only works if the model can articulate a reasonable hypothesis to query in the first place. If it can't formulate the right question, this fails in a different way than the harness's methods, but it still fails — it does not sidestep the underlying capability-gate question.
- A more general (and unbuilt, higher-risk) alternative surfaced during discussion: the Duke University "Hill-Climbing" harness reportedly lets the model execute arbitrary Python against its own action history rather than calling fixed primitives — noted as a real alternative design, not adopted here due to sandboxing/safety complexity relative to the fixed-primitive version.
- **Rejected alternative, for reference:** a fully learned, LLM-authored "similarity"/relevance-ranking function revised through the same certification loop as `step()`. Rejected because relevance has no ground truth anywhere in the trace (unlike `step()`/`is_goal()`, which are checked against recorded grid/level-completion data) — the only available feedback signal is indirect and delayed (did this round's `step()` improve), which is noisy, confounded, and adds a second nested optimization problem on top of the one already being tested.

---

## Explicitly out of scope for this test (regardless of Tier 1/2 outcome)

- **Active/live exploration** — an agent that connects to the live environment and selects actions specifically to disambiguate competing hypotheses (per Schema's actual mechanism: passive correction on the moves it would take anyway, not deliberate wasted probing — see note below). This removes the hard information ceiling of a fixed offline trace, but is a different data-collection paradigm (live environment connection, action-selection policy, real per-round environment-interaction cost) and should only be pursued if the offline core loop (Tier 1) plateaus at genuinely near-random performance — a strong signal that hidden-state disambiguation requires live exploration rather than more offline harness sophistication.
- **Purely temporal hidden state with no visual correlate** (e.g., a countdown/action-counter with no grid representation) — nothing in the object/grid-centric design (Tier 1 or 2) addresses this category; not known to apply to `ls20` specifically, not solved here.
- **MCTS, the topological/graph layer, the discrete matrix solver, intra-task memory** — per your project's own scoping, not started, and out of scope for this specific harness redesign.

---

## Key limitations to carry forward regardless of implementation quality

1. **Hard ceiling:** no verification/evidence-curation mechanism can recover information that was never exhibited in the fixed, finite, offline trace — this is a property of the data-collection method, not fixable by harness sophistication.
2. **Unmeasured variable:** everything here is verification/evidence-selection scaffolding around an unproven assumption — that the local model can generate reasonable hypotheses given good evidence. Nothing in this design manufactures that capability if it isn't there; that's the actual thing the real test (§12) measures.
3. Old 0.8%/5.5% baseline was measured under the _old_ stateless harness and is not a reliable predictor of how the redesigned loop performs — the redesign changes the task itself, not just the scoring.
