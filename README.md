# Nosumina

I want to help a post-scarcity future arrive safely. One where the work required just to survive finally fades away, and we all get our time back, for our health, or for a life we've wanted for ourselves but were never able to have. Getting there safely matters as much as getting there at all. A future where automation and artificial reasoning is cheap and abundant is also a future where that power is easy to misuse, and that's the part I'd like to help get right. Nosumina is my small, early attempt at that. If this is a future you want too, I'd like to build it with you.

What stands in this repository today is far humbler than any of that. Close to nothing works yet. What follows is simply a layout of the bet I'm making and the reasoning behind it, not a claim that the bet has paid off.

The current test of that bet is [ARC-AGI-3](https://arcprize.org/), a Kaggle competition built around games no model has seen before. It is played under strict constraints: a local-LLM-only environment with no internet access, no frontier models, and run entirely offline. Nosumina is my open-source entry in that track. Inside its architecture, a small local model quietly writes and revises a Python program — a `GameModel` — that tries to predict how a game's grid shifts from one step to the next. It must learn entirely from recorded gameplay traces, rather than being handed the rules. My bet isn't that a larger model would do this better; it's that a well-built harness wrapped around a small one can go further than model scale alone would suggest.

If this design holds up on ARC-AGI-3, it could be retrofitted toward the kind of autonomous systems that could actually usher in a post-scarcity future such as automated farms and delivery robots. But getting there requires navigating two gaps this repo hasn't crossed yet: a preprocessing layer that can adapt to entirely new kinds of data, and a way to give the system a goal at all.

That second gap is the one I take more seriously. Right now, the current architecture doesn't hand the model a goal like "keep this farm alive." A system optimizing relentlessly for a goal like that needs real alignment work first, or it will find the cheapest way to reach the goal rather than what anyone actually intended. That's a future direction for this project, not a claim about what it can do today.

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

## Quickstart

**Before you start:** even a fully correct run of this harness currently ends near 0% exact-match accuracy on the grid-prediction task. That's not a sign something went wrong — it's the current state of the architecture. This Quickstart will get the harness _running end to end_, not get you an accurate `GameModel`.

**What you need:**

- A Linux host with a CUDA-capable GPU with **at least ~48GB VRAM** (this project is developed and tested against an RTX PRO 6000 Blackwell, 96GB VRAM, running Qwen3-Coder-Next at Q4_K_XL quantization, which alone uses ~49GB)
- Python 3.12
- `llama-cpp-python` already built **with CUDA support** — this is the step most likely to silently go wrong (a failed CUDA detection during build falls back to a CPU-only build with no error). Verify before continuing:
  ```bash
  python3 -c "from llama_cpp import llama_cpp as lib; print(lib.llama_supports_gpu_offload())"
  ```
  This must print `True`.
- `bubblewrap` installed and working (`bwrap --version`) — required to sandbox candidate code execution, independent of which inference backend you use
- `numpy` (the only non-stdlib dependency the harness itself needs)

Don't have a CUDA GPU, or need help getting any of the above working? See [`FULL_ENVIRONMENT_SETUP.md`](./FULL_ENVIRONMENT_SETUP.md) for a from-scratch cloud GPU setup, including a known CUDA-build failure mode and how to catch it.

**Steps:**

1. **Get a trace.** The fastest path: use [`trace.jsonl`](./trace.jsonl), a 400-row trace included at the repo root. Optionally, you can record your own by playing a game yourself with [`my_agent_keyboard.py`](./my_agent_keyboard.py) (see the docstring at the top of that file for how to swap it into the [ARC-AGI-3-Kaggle-Starter Repo](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)). This step is optional; skip straight to step 2 if you just want to see the harness run.

2. **Get the model weights.** Download Qwen3-Coder-Next-UD-Q4_K_XL (GGUF) from Unsloth's Hugging Face repo. `/home/cloud/models/...` below is just the path used on this project's own JarvisLabs setup — swap it for wherever you want the weights to live on your machine:

   ```bash
   pip install huggingface_hub
   mkdir -p /home/cloud/models
   huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \
       --include "*UD-Q4_K_XL*" \
       --local-dir /home/cloud/models/qwen3-coder-next
   ```

3. **Preprocess the trace.**

   ```bash
   python trace_tools.py preprocess trace.jsonl clean.jsonl --frame-key frame --action-key action
   ```

   (Optionally run `python trace_tools.py inspect trace.jsonl --frame-key frame` first if you're using your own trace instead of the bundled one, to confirm its `frame` field structure. `inspect` checks the nesting depth of `pre_observation.frame` and `post_observation.frame` across every record: it wants **2 levels of nesting** — a plain `list[rows][cols]`, i.e. a single settled grid — not 3, which would mean each record still holds a list of grids (one per intermediate animation tick) rather than the final collapsed one. `my_agent_keyboard.py`'s `_serialize_frame_for_trace` already collapses to depth 2 before writing, so the bundled trace and anything recorded with that script will pass this check automatically. If `inspect` reports a mix of depths or anything other than a consistent `{2}`, don't run `preprocess` yet — the frames still need to be collapsed to their final tick first, or `preprocess` will operate on the wrong shape.)

4. **Run the harness.**
   ```bash
   python trace_tools.py run-chunked clean.jsonl \
       --backend llama-cpp --model-path /path/to/model.Q4_K_M.gguf \
       --n-gpu-layers -1 --n-ctx 131072 --max-examples 20 --max-rounds 2 \
       --compact --repeat-penalty 1.3 --frequency-penalty 0.1 --presence-penalty 0.1 \
       --workdir chunked_run --verbose-llama
   ```
   These are the recommended settings for a first run. By default this pauses before every LLM call so you can inspect the prompt — pass `--automatic` once you trust it to run unattended.

**What you'll see, and what it means.** Everything `run-chunked` produces lands in `--workdir` (`chunked_run` above) — see [Results 8_4_2026](./Results%208_4_2026) in this repo for a real example of what that directory looks like after a full run. A few files worth knowing:

- **`chunk{N}_candidate_round{M}.py`** — the actual `GameModel` class the model wrote for chunk `N`, round `M`. Open one of these directly; it's plain, readable Python, not a black box.
- **`chunk_log.jsonl`** — one JSON line per round, recording what was accepted or rejected and why, across the whole run.
- **Per-round accuracy, printed to the terminal and logged**, comes in three numbers, and they mean different things:
  - **Exact-match accuracy** — the strict pass rate: did the candidate predict the _entire_ grid correctly, cell for cell? This is the headline number, and it's the one that's currently near 0%.
  - **Accuracy on cells that actually changed** — restricted to just the cells that differ between `grid_before` and `grid_after`, ignoring the (usually large) static background. This is the number that reflects whether the model is learning real game dynamics, since a candidate that changes nothing at all can still score high on the next metric below.
  - **Mean per-cell accuracy (all cells)** — nearly every grid is mostly static from one step to the next, so this number is inflated by cells a candidate gets right just by leaving them alone. A do-nothing candidate can score deceptively well here; it's included for completeness, not as the metric to trust.

So: if your first run ends with exact-match near 0%, changed-cell accuracy low, and per-cell accuracy misleadingly high — that's not a broken run. That's this project's actual, current result, and matches what's archived in `Results_8_4_2026`.

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
| `Nosumina_Redesign.md`                                      | The full Tier 1 / Tier 2 harness redesign that was replaced by the Simplified Diagnostic Architecture                  |
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

Whether you're just curious about the project, find yourself at odds with my design choices, or wish to collaborate on something bigger than a PR, I'd like to hear from you: **russell.miguel.silva [at] gmail [dot] com**.

For security issues: please don't open a public issue — see `SECURITY.md` for responsible disclosure.

## License

[MIT](./LICENSE) © 2026 Russell Silva
