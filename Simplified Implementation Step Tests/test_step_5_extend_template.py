import sys
sys.path.insert(0, ".")
from trace_tools import (
    build_examples_block,
    build_description_instruction,
    build_initial_prompt,
    build_extend_prompt,
)


def check(label, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return condition


all_ok = True


def make_records(n, start_step=21, size=20):
    """
    n contiguous records, each just incrementing one cell so diffs stay
    small. Uses a 20x20 base grid (not the original 2x2) -- under the
    current dynamic format selection (see format_diff), a diff's cost is
    compared against a full-grid dump's cost, and a full dump of a 2x2
    grid is only ~6 chars, cheaper than ANY diff notation for ANY diff
    size on a grid that small. A 2x2 grid can no longer exercise the
    "ordinary small diff stays a flat list" path at all; 20x20 (~420 chars
    to dump raw) is large enough that a 1-2 cell diff genuinely stays
    cheaper as a flat list, the way this test intends to check.
    """
    records = []
    grid = [[0] * size for _ in range(size)]
    for i in range(n):
        before = [row[:] for row in grid]
        grid[0][0] = (grid[0][0] + 1) % 16
        after = [row[:] for row in grid]
        records.append({
            "step": start_step + i,
            "action": f"ACTION_{i}",
            "grid_before": before,
            "grid_after": after,
        })
    return records


# ---------------------------------------------------------------------------
# Case 1: basic contiguous batch, is_extend=False
# ---------------------------------------------------------------------------
print("=== Case 1: build_examples_block, is_extend=False ===")
records = make_records(3)
block = build_examples_block(records, encoding="hex", is_extend=False)
print(block)

all_ok &= check("exactly one 'Starting Grid:' occurrence", block.count("Starting Grid:") == 1)
all_ok &= check("Grid 1 labeled with correct trace step", f"Grid 1 (trace step {records[0]['step']})" in block)
all_ok &= check("Grid 2 labeled with correct trace step", f"Grid 2 (trace step {records[1]['step']})" in block)
all_ok &= check("Grid 3 labeled with correct trace step", f"Grid 3 (trace step {records[2]['step']})" in block)
all_ok &= check(
    "Grid 1's diff keyed to Starting Grid -> Grid 1, as a flat-list (cheapest for a 1-cell diff on a 20x20 grid)",
    "changed cells (Starting Grid -> Grid 1), flat-list format" in block,
)
all_ok &= check(
    "Grid 2's diff keyed to Grid 1 -> Grid 2, as a flat-list",
    "changed cells (Grid 1 -> Grid 2), flat-list format" in block,
)
all_ok &= check(
    "Grid 3's diff keyed to Grid 2 -> Grid 3, as a flat-list",
    "changed cells (Grid 2 -> Grid 3), flat-list format" in block,
)
all_ok &= check("is_extend sentence absent when is_extend=False", "NOT the start of the game" not in block)
all_ok &= check("no full-grid fallback shown for ordinary (non-overflow) examples", " in full (" not in block)

# ---------------------------------------------------------------------------
# Case 2: same records, is_extend=True -- the extra sentence should appear
# ---------------------------------------------------------------------------
print("\n=== Case 2: build_examples_block, is_extend=True ===")
block_extend = build_examples_block(records, encoding="hex", is_extend=True)
all_ok &= check("is_extend sentence present when is_extend=True", "NOT the start of the game" in block_extend)
all_ok &= check(
    "everything else unaffected by is_extend (same Grid labels present)",
    "Grid 1 (trace step" in block_extend and "Grid 3 (trace step" in block_extend,
)

# ---------------------------------------------------------------------------
# Case 3: full-dump fallback -- DIFF_MAX_CELLS no longer exists; format_diff
# now dynamically picks whichever of flat-list/masked-grid/full-dump
# renders shortest for a given diff (see format_diff's docstring), so
# there's no fixed cell-count threshold to target directly. To force the
# full-dump branch specifically, the diff needs to be BOTH dense AND
# non-repeating -- a uniform "every cell becomes the same new value" change
# actually compresses so well under masked-grid's value-run RLE that it
# beats a full dump even at 100% density (verified separately), so this
# uses a checkerboard-style pattern where no two adjacent cells share a
# value, defeating both flat-list and masked-grid's RLE.
# ---------------------------------------------------------------------------
print("\n=== Case 3: full-dump fallback (dense, non-repeating diff) ===")
size = 20
dense_before = [[0] * size for _ in range(size)]
dense_after = [[(r * size + c) % 16 for c in range(size)] for r in range(size)]
overflow_records = [
    {"step": 5, "action": "ACTION_A", "grid_before": dense_before, "grid_after": dense_after},
]
overflow_block = build_examples_block(overflow_records, encoding="hex", is_extend=False)
print(overflow_block)
all_ok &= check(
    "full-dump fallback triggers for a dense, non-repeating diff",
    "Grid 1 in full (" in overflow_block and "cells changed" in overflow_block,
)
all_ok &= check(
    "full-dump fallback explicitly states it's NOT the flat-list/masked-grid notation",
    "NOT real integers or the masked-grid notation used elsewhere" in overflow_block,
)
all_ok &= check(
    "full-dump fallback uses the same 'Grid i' label, not a separate 'Resulting Grid' name",
    "Resulting Grid" not in overflow_block,
)
all_ok &= check(
    "full-dump fallback does NOT also claim a flat-list/masked-grid format name for the same block",
    "flat-list format" not in overflow_block and "masked-grid format" not in overflow_block,
)

# ---------------------------------------------------------------------------
# Case 3b: masked-grid path -- a diff dense enough that a flat per-cell list
# would be expensive, but sparse/clustered enough that masking + RLE beats
# both the flat list and a full dump. No prior test exercised this branch
# at all (it didn't exist before format_diff's dynamic selection).
# ---------------------------------------------------------------------------
print("\n=== Case 3b: masked-grid fallback (sparse cluster on a large grid) ===")
sparse_before = [[4] * size for _ in range(size)]
sparse_after = [row[:] for row in sparse_before]
for r in range(5, 10):
    for c in range(5, 8):
        sparse_after[r][c] = 9  # a small clustered block of changes
sparse_records = [
    {"step": 6, "action": "ACTION_B", "grid_before": sparse_before, "grid_after": sparse_after},
]
sparse_block = build_examples_block(sparse_records, encoding="hex", is_extend=False)
print(sparse_block)
all_ok &= check(
    "a clustered mid-size diff on a large grid picks masked-grid format",
    "changed cells (Starting Grid -> Grid 1), masked-grid format" in sparse_block,
)
all_ok &= check(
    "masked-grid output contains the expected RLE run tokens",
    "*5" in sparse_block and "4>9" in sparse_block,
)
all_ok &= check(
    "masked-grid case does NOT trigger the full-dump fallback",
    "in full (" not in sparse_block,
)

# ---------------------------------------------------------------------------
# Case 4: contiguity precondition is enforced
# ---------------------------------------------------------------------------
print("\n=== Case 4: contiguity assertion ===")
broken_records = [
    {"step": 0, "action": "A", "grid_before": [[0]], "grid_after": [[1]]},
    {"step": 1, "action": "B", "grid_before": [[9]], "grid_after": [[2]]},  # doesn't match [[1]]
]
try:
    build_examples_block(broken_records)
    all_ok &= check("non-contiguous records raise AssertionError", False)
except AssertionError:
    all_ok &= check("non-contiguous records raise AssertionError", True)

# ---------------------------------------------------------------------------
# Case 5: build_description_instruction -- level-boundary sentence only for
# is_extend=True AND is_level_boundary=True.
# ---------------------------------------------------------------------------
print("\n=== Case 5: build_description_instruction level-boundary flag ===")
instr_initial = build_description_instruction(is_extend=False)
instr_extend_no_boundary = build_description_instruction(is_extend=True, is_level_boundary=False)
instr_extend_boundary = build_description_instruction(is_extend=True, is_level_boundary=True)

all_ok &= check("PROMPT_TEMPLATE instruction has 3 steps", "There are three steps" in instr_initial)
all_ok &= check("PROMPT_TEMPLATE instruction never mentions NEW LEVEL", "NEW LEVEL" not in instr_initial)
all_ok &= check("EXTEND_TEMPLATE (no boundary) has 4 steps", "There are four steps" in instr_extend_no_boundary)
all_ok &= check(
    "EXTEND_TEMPLATE (no boundary) does NOT mention NEW LEVEL",
    "NEW LEVEL" not in instr_extend_no_boundary,
)
all_ok &= check(
    "EXTEND_TEMPLATE (is_level_boundary=True) DOES mention NEW LEVEL",
    "NEW LEVEL" in instr_extend_boundary,
)
all_ok &= check(
    "is_level_boundary=True ignored when is_extend=False (PROMPT_TEMPLATE has no boundary variant)",
    "NEW LEVEL" not in build_description_instruction(is_extend=False, is_level_boundary=True),
)

# ---------------------------------------------------------------------------
# Case 6: full prompt integration -- build_initial_prompt / build_extend_prompt
# actually format cleanly and route the right content to the right template.
# ---------------------------------------------------------------------------
print("\n=== Case 6: full prompt integration ===")
full_records = make_records(3, start_step=0)
initial_prompt, n_used = build_initial_prompt(full_records, max_examples=3, encoding="hex")
all_ok &= check("build_initial_prompt uses all 3 records (n_used == 3)", n_used == 3)
all_ok &= check("initial prompt asks for GameModel class", "class GameModel" in initial_prompt)
all_ok &= check("initial prompt has no is_extend sentence", "NOT the start of the game" not in initial_prompt)
all_ok &= check("initial prompt has the 3-step DESCRIPTION instruction", "There are three steps" in initial_prompt)

fake_candidate_code = (
    "class GameModel:\n"
    "    def predict(self, grid_before, action, previous_state):\n"
    "        return grid_before, False, previous_state\n"
)
extend_prompt_no_boundary = build_extend_prompt(
    fake_candidate_code, full_records, encoding="hex", is_level_boundary=False
)
extend_prompt_boundary = build_extend_prompt(
    fake_candidate_code, full_records, encoding="hex", is_level_boundary=True
)

all_ok &= check(
    "extend prompt shows the candidate's current class code",
    "def predict(self, grid_before" in extend_prompt_no_boundary,
)
all_ok &= check("extend prompt has the is_extend sentence", "NOT the start of the game" in extend_prompt_no_boundary)
all_ok &= check("extend prompt (no boundary) has no NEW LEVEL sentence", "NEW LEVEL" not in extend_prompt_no_boundary)
all_ok &= check("extend prompt (is_level_boundary=True) has NEW LEVEL sentence", "NEW LEVEL" in extend_prompt_boundary)
all_ok &= check(
    "extend prompt asks to keep predict's exact signature",
    "must keep this exact signature" in extend_prompt_no_boundary,
)

print()
print("ALL PASS" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)