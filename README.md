# Nosumina Test PR

Nosumina is a program-synthesis harness for [ARC-AGI-3](https://arcprize.org/): a local, quantized LLM writes and iteratively revises a `GameModel` class that predicts how a game's grid changes from one step to the next, on nothing but recorded gameplay traces. It's an open-source entry for the ARC-AGI-3 Kaggle competition's local-LLM-only track (no internet access, 12-hour runtime ceiling).

**The core thesis:** how you scaffold a model matters more than how big it is. Rather than trying to out-scale frontier models, Nosumina bets that a well-designed harness — persistent state, exhaustive certification against everything observed so far, and an explicit loop for revising _how_ the model represents the game, not just its rules — can unlock genuine capability from a small local model. If that holds up, the hope is it generalizes past ARC-AGI-3, to any domain where a model needs to build and revise a working model of its environment from observation rather than being told the rules up front.

This project is a practical test of that bet, run against a real local model (Qwen3-Coder-Next, quantized) on real ARC-AGI-3 traces, with results — good or bad — treated as the actual point of the exercise.

## How it works

1. **Record a trace.** Play (or script) an ARC-AGI-3 game and record each `{pre_observation, action, post_observation}` step to a `trace.jsonl` file. This can be done with [`my_agent_keyboard.py`](./my_agent_keyboard.py)
2. **Preprocess.** Collapse the raw trace into clean `{step, action, grid_before, grid_after, ...}` transitions.
3. **Synthesize.** A local LLM is shown a chunk of the trace and writes a `GameModel` class with a single `predict()` method — it threads its own state through from call to call, so it has somewhere to put things like "was the switch already hit."
4. **Certify.** The candidate is executed, sandboxed, in a full sequential replay against every transition recorded so far (not just the chunk it was written from). Every row it gets right or wrong is logged.
5. **Revise.** Rows the candidate gets wrong are fed back as counterexamples, and the model is asked to fix its `GameModel` — not rewrite from scratch — for another round.
6. **Advance the curriculum.** Once a chunk is solved (or a round budget is exhausted), the trace window advances and the process repeats, carrying the current-best code forward as the starting point for the next chunk.

This design is inspired by [Schema (Zeng et al., 2026)](https://schema-harness.github.io), which showed that writing game mechanics as certified, executable programs — verified by replay against complete interaction history, with the state representation itself open to revision on contradiction — accounts for the bulk of a harness's performance, largely independent of the underlying model's raw size. Nosumina adapts that idea to a harder constraint Schema's own setup doesn't have to work within: no frontier model, no internet access, just a quantized model running locally end to end.

## Why the candidate carries its own state

Earlier iterations of this harness asked the model for a single stateless function, `predict_next_state(grid, action) -> grid`. That collapses under any game mechanic that depends on hidden state — a switch that's only "on" because it was stepped on three moves ago has no representation in a signature that only sees the current grid. The current, implemented contract is a single method that threads its own state through explicitly:

```python
class GameModel:
    def predict(self, grid_before: list[list[int]], action: str, previous_state: dict) -> tuple:
        ...
        return predicted_grid_after, goal, state
```

- `predicted_grid_after` — the model's prediction for the grid after `action` is taken from `grid_before`.
- `goal` — `bool`, whether this action is predicted to complete the current level.
- `state` — a JSON-serializable dict (plain dicts/lists/numbers only) the candidate builds and reads itself, carried forward into the _next_ call as `previous_state`. On the first call of a fresh replay, `previous_state` is `{}`; the class is required to use `.get()` with defaults rather than direct indexing so a fresh rollout doesn't `KeyError`.

The class may define whatever additional methods or fields it wants internally — `predict`'s signature is the only fixed part of the contract. This single-method shape is the _simplified_ diagnostic design currently implemented in `trace_tools.py`; a richer four-method contract (`init_state`/`step`/`sync_state`/`is_goal`) was designed as a fuller Tier 1 redesign (see `Nosumina_Redesign.md`) but is not what's currently running — the simplified version was built first, specifically to get a faster, cheaper read on whether the local model has any real capability here before investing in the larger design.

## Scoring

Each replayed row is teacher-forced end to end — the candidate is fed the real `grid_before` and `previous_state` at every step, never its own prior (possibly wrong) prediction, so errors don't compound across a replay. A row passes only if `predicted_goal == actual_goal`; when the candidate predicts `goal = True`, the grid comparison is excluded entirely from that row's pass/fail decision, since the candidate has no way to know what the next level's starting grid looks like. `row_failure_counts.json`, rewritten after every round, tracks a running per-row failure count across the whole trace so far — this is what `revise-prompt`/`run-chunked` use to pick the top-`k` counterexamples to show the model in its next revision round. Collision detection — a stronger, harder signal that distinguishes a hidden-state bug from a plain transition-rule bug by finding two identical `(state, action)` pairs with diverging true outcomes — was designed as part of the fuller Tier 1 redesign but is **not implemented** in the current simplified harness; see "What's deliberately out of scope right now" below.

## Sandboxing

Every candidate the model writes is executed as untrusted code, full stop — there's no attempt to distinguish an honest mistake from a hallucination from something adversarial, because there's no reliable way to tell those apart from the outside. Containment is layered:

1. A static AST-level import allowlist check before anything is staged.
2. A restricted execution namespace (stripped builtins, a wrapped `__import__`) as a fast-fail courtesy layer.
3. `bubblewrap` (`bwrap`) kernel-namespace sandboxing as the actual enforcement boundary — no network, read-only filesystem, all capabilities dropped, process isolation, synthetic `/proc` and `/dev`.
4. CPU/memory/process/wall-clock/output limits enforced from outside the sandboxed process.
5. Structural validation of whatever a candidate returns, before it's used downstream.

Full threat model, what's explicitly out of scope, and known limitations of each layer are in [`SECURITY.md`](./SECURITY.md).

## Status

Actively in development, pre-results. Recent work:

- **`analyze()` preprocessing pass** (design complete, implementation in progress): a second, smaller code-generation call that runs once per example, ahead of the `GameModel`-writing call, and reports structural facts about a single transition (translating, stationary/recolored, stationary/shape-changed, appeared, disappeared, ambiguous-match) computed from a shape-agnostic flood-fill component extractor. The goal is to offload "is this the same shape as before, just moved" onto deterministic code rather than asking the model to re-derive it from raw diffs, under token pressure, across every example, every round. Details and known limitations (correspondence ambiguity, multi-color sprite fragmentation, occlusion, no ground-truth certification for `analyze()` itself) are in [`Nosumina_Analyze_Preprocessing_Implementation_Plan.md`](./Nosumina_Analyze_Preprocessing_Implementation_Plan.md).
- **Sandbox hardening and security audit** completed, verified against a checklist on the actual GPU instance.
- **`ls20`** (the ARC-AGI-3 game currently used as the test bed) mechanics reconstructed by hand: a player-controlled sprite, a step counter, an occluded switch decal, and a 3×3 indicator panel that changes pattern on activation — used as ground truth for validating the harness itself.

An earlier, stateless harness scored 0.8% exact-match / 5.5% changed-cell accuracy on held-out ls20 data — statistically indistinguishable from the ~6.25% random baseline. That result is a large part of why the harness was redesigned around persistent state and exhaustive certification. Near-0% exact-match on predicting the next grid is currently the state of the architecture even under that redesign. The analyze() preprocessing pass below is aimed at the deeper, still-open part of the problem: the model characterizing noise rather than forming a real concept of what it's looking at. Whether that's a fixable harness gap or a real ceiling in the local model is still the open question this project is trying to answer.

## Usage

The harness is a single CLI, `trace_tools.py`, with one subcommand per pipeline stage:

```bash
# Look at a raw trace's structure without dumping grid contents
python trace_tools.py inspect trace.jsonl --frame-key frame

# Collapse it to clean, chronological transitions. Important - must be run before run-chunked command to get clean.jsonl
python trace_tools.py preprocess trace.jsonl clean.jsonl --frame-key frame --action-key action

# Build the seed prompt for the first chunk
python trace_tools.py prompt clean.jsonl prompt.txt --max-examples 25

# Replay a candidate GameModel against a trace and score it
python trace_tools.py backtest candidate.py clean.jsonl --boundary 50 --counts row_failure_counts.json

# Build a revision prompt from a candidate's failures
python trace_tools.py revise-prompt candidate.py row_failure_counts.json revision_prompt.txt --k 10

# Run the full chunked-curriculum loop against a live local model
python trace_tools.py run-chunked clean.jsonl \
    --backend llama-cpp --model-path /path/to/model.Q4_K_M.gguf \
    --n-gpu-layers -1 --n-ctx 131072 --max-examples 20 --max-rounds 2 \
    --compact --repeat-penalty 1.3 --frequency-penalty 0.1 --presence-penalty 0.1 \
    --workdir chunked_run --verbose-llama
```

`run-chunked` is the actual harness the project runs end to end: it loads one full trace, walks it chunk by chunk (a chunk boundary is whichever comes first of a row cap, a level completion, or the end of the trace), runs up to `--max-rounds` synthesis/revision rounds per chunk, accepts or rejects each round against a small regression tolerance, and stops a chunk early the moment it hits zero failing rows. It supports both `llama-cpp` (in-process GGUF inference — the default) and `openai` (any OpenAI-compatible server, e.g. vLLM or `llama-server`) as backends. By default it pauses before every LLM call so the prompt can be inspected first; pass `--automatic` for unattended runs.

**Requirements:** Python 3.12, `numpy` (the only non-stdlib dependency the harness itself needs), `llama-cpp-python` if using the local backend, and `bubblewrap` on the host for sandboxing. Candidate code is restricted to `copy`, `itertools`, `math`, `collections`, `functools`, and `numpy`.

## Hardware this was built and tested against

A JarvisLabs GPU instance with an RTX PRO 6000 (Blackwell, ~97GB VRAM) — the same accelerator available in the Kaggle scoring environment via the `"rtx6000"` string — running Qwen3-Coder-Next-UD-Q4_K_XL (an 80B-parameter MoE model at Q4_K_XL quantization).

## Repository layout

| Path                                                        | What it is                                                                                                             |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `trace_tools.py`                                            | The harness itself — all pipeline stages, single file                                                                  |
| `SECURITY.md`                                               | Threat model and sandbox defense layers                                                                                |
| `CONTRIBUTING.md`                                           | How to propose changes, and the "does this help the model reason, or reason for it" bar every change is judged against |
| `Nosumina_Redesign.md`                                      | The full Tier 1 / Tier 2 harness redesign this codebase implements, with known limitations at every layer              |
| `Nosumina_Simplified_Capability_Diagnostic_Architecture.md` | The specific, cheaper diagnostic design currently being run, and why it's scoped the way it is                         |
| `Nosumina_Analyze_Preprocessing_Implementation_Plan.md`     | Design and step-by-step implementation plan for the `analyze()` preprocessing pass                                     |
| `Worst_Case_Token_Testing.md`                               | Open task: pathological-trace token-budget testing to pick safe defaults for `--compact`/`--max-examples`/`--k`        |

## What's deliberately out of scope right now

- **Collision detection** — proving a hidden-state bug (rather than just a wrong transition rule) by catching two identical `(state, action)` pairs in a replay that disagree on the true outcome. Designed as part of the fuller Tier 1 redesign; not implemented in the current simplified harness, which relies on plain per-row failure counts instead.
- **Live/active exploration** — an agent that connects to the running game and chooses actions to disambiguate its own hypotheses. Everything here fits a fixed, already-recorded trace; this is a different, harder problem, worth revisiting only if the offline core loop plateaus at genuinely near-random performance.
- **Purely temporal hidden state with no visual correlate** (e.g., an internal counter with nothing on the grid to reveal it) — not addressed by the current object/grid-centric design.
- **The three-layer Multi-Resolution Heuristic Solver** (a Semantic/LLM layer, a Graph Topological layer via NetworkX, a Discrete Matrix layer via SciPy/NumPy) and MCTS planning on top of it — the longer-term architectural vision for this project, deliberately deferred until the core synthesis-and-certification loop is validated on its own. When this phase begins, note that a naive graph-topology bonus (simplicial complex construction, Laplacian spectral properties) was already tested as an MCTS selection signal on ARC-style grids and found to produce no improvement — see [Solution Space Topology Guides MCTS Search](https://arxiv.org/abs/2511.01701). The same paper found a more targeted structural feature, rigidity analysis identifying bottleneck cells, did help, which should be the starting point for this project's Graph Topological layer rather than generic connectivity/spectral scoring.
- **Object segmentation/ID tracking across frames, attribute-based type-grouping, correlation-ranking over sparse events, and an LLM history-query tool** — generalization machinery for games with moving/variable-count objects or sparse hidden triggers, built only if the core loop plateaus _and_ a manual playthrough confirms a given game actually needs it.

## Collaboration & Contact

Fork it, make a change, open a pull request — that's the whole process, no permission needed first. One requirement: commits need to be signed off (git commit -s) per the DCO — PRs without it won't pass the required check. Beyond that, [`CONTRIBUTING.md`](./CONTRIBUTING.md) has more on what a good PR or a bigger proposal looks like.

I review and merge every PR myself, so it might take a few days depending on what else is going on — I'll get back to you either way, even if it's just a question to understand what you're going for — rather than leaving a PR unreviewed.

If you end up contributing regularly, I'll add you as a collaborator so you can work directly against the repo instead of through a fork.

If you want to reach out directly please contact me at **russell.miguel.silva [at] gmail [dot] com**.

For security issues: please don't open a public issue — see `SECURITY.md` for responsible disclosure.

## License

[MIT](./LICENSE) © 2026 Russell Silva
