# TODO: Worst-case prompt token testing across encoding/trace settings

**Goal:** confirm the harness's token budget holds up under pathological (not just typical) trace conditions, and pick defaults for `--compact`, `--max-examples`, and `--k` that stay within `--n-ctx` even in near-worst-case real games — not just the traces already tested.

**Why this matters:** `run-loop`'s preflight tokenizer check (`llm.tokenize()` vs. `--n-ctx - --max-tokens`) already fails fast rather than crashing mid-generation, but "fails fast with a clear message" is still a wasted round if it fires in the middle of an unattended run. Better to know the actual worst-case token cost ahead of time and pick flag defaults that avoid hitting it in practice.

---

## 1. Worst-case synthetic traces to construct

- **Checkerboard / alternating-cell trace:** every cell in every row alternates color value cell-to-cell (e.g. `0,1,0,1,0,1...`), and the pattern shifts by one cell each step (so the whole grid visibly "moves" every step). This is the worst case specifically for `--compact` (RLE) — every run has length 1, so RLE output (`"0*1 1*1 0*1 1*1..."`) is _longer_ than plain hex for the same row, not shorter. Worth confirming RLE can actively make things worse under the wrong input shape, not just "less good."
- **Full-grid-change trace:** every cell changes value between every `grid_before`/`grid_after` pair (100% change rate). This guarantees `diff_grid`'s changed-cell count exceeds `DIFF_MAX_CELLS` (40) on every single example, forcing the full-grid fallback (`Grid i in full`) to fire every time instead of just a diff list — meaning every example in a chunk ends up paying for a full 64×64 grid dump, not just the batch's one `Starting Grid`.
- **Threshold-straddling trace:** deliberately change just over `DIFF_MAX_CELLS` (e.g. 41–50 cells) every step, rather than the full grid. This is worth testing _separately_ from the 100%-change case — it's a more "plausible" pathological trace (a busy but not totally chaotic game) that still reliably triggers the full-grid fallback on every example, so it isolates "fallback always firing" from "grid is also maximally dense/colorful."
- **Combined worst case:** checkerboard _and_ full-grid-change together — worst case for both RLE inflation and diff-overflow fallback at once. This is the actual "how bad can it get" number to compare against `--n-ctx`.
- _(Optional, lower priority)_ **All-16-colors trace:** every cell drawn from the full 0–15 palette with no repeats nearby, to confirm color diversity itself doesn't matter for token count (it shouldn't — hex is 1 char/cell regardless of value diversity — but worth a quick confirmation rather than an assumption).

## 2. Settings to vary per trace

- **Encoding:** hex (default) vs. `--compact` (RLE) — the core comparison.
- **`--max-examples`** (chunk size / round-1 batch size): confirm token cost scaling as this grows, especially combined with the full-grid-change trace where every example pays the fallback cost rather than just the diff cost.
- **`--k`** (counterexamples per revision prompt): `build_counterexamples_block` always shows both predicted _and_ actual grids in full per counterexample — it doesn't get the "one `Starting Grid`" optimization that round-1 prompts do. Worst case here is `--k` counterexamples × 2 full 64×64 grids each, with **no RLE benefit if the counterexample rows are checkerboard-pattern** (same RLE-inflation risk as above, applied to `REVISE_TEMPLATE` specifically). This deserves its own worst-case number, separate from round-1 prompts, since it's structurally the more expensive template per row.

## 3. What to actually measure and record

For each (trace type × encoding × `--max-examples`/`--k` combination):

- Real token count via the same tokenizer path `run-loop` already uses (`llm.tokenize()`), not the chars/4 or chars/1 fallback estimates.
- Whether it exceeds a representative `--n-ctx` budget (e.g. the 32768 default).
- For the RLE cases specifically: confirm whether RLE token count is actually _higher_ than hex for the checkerboard trace, and by roughly how much — this is the number that determines whether `--compact` needs a caveat ("don't use on high-entropy/fast-moving games") documented somewhere, or whether an adaptive per-row "pick whichever encoding is shorter" mode is worth building later.

## 4. Output of this task

A short table/log of worst-case token counts per combination, and (if any combination realistically exceeds `--n-ctx` at the current default flags) either an adjusted default for `--max-examples`/`--k`, or a documented caveat that `--compact` should be avoided for known-high-entropy games.
