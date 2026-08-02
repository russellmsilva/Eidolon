import sys
import json
import tempfile
from pathlib import Path
from argparse import Namespace

sys.path.insert(0, ".")
from trace_tools import cmd_preprocess, next_chunk_boundary

tmpdir = Path(tempfile.mkdtemp(prefix="preprocess_levels_check_"))
raw_path = tmpdir / "raw.jsonl"
out_path = tmpdir / "clean.jsonl"


def check(label, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return condition


all_ok = True

# ---------------------------------------------------------------------------
# Case 1: normal extraction -- levels_completed present on both sides of
# every row, ticking up on row 2 (0-indexed) from 0 -> 1.
# ---------------------------------------------------------------------------
raw_rows = []
for i in range(5):
    lvl = 0 if i <= 2 else 1
    lvl_after = 1 if i == 2 else lvl  # completes exactly during row 2
    raw_rows.append({
        "pre_observation": {"frame": [[i]], "levels_completed": lvl},
        "post_observation": {"frame": [[i + 1]], "levels_completed": lvl_after},
        "action": f"ACTION_{i}",
    })

with open(raw_path, "w") as f:
    for r in raw_rows:
        f.write(json.dumps(r) + "\n")

args = Namespace(
    trace=str(raw_path), out=str(out_path),
    pre_key="pre_observation", post_key="post_observation",
    frame_key="frame", action_key="action",
    score_key=None, levels_key="levels_completed",
)
cmd_preprocess(args)

with open(out_path) as f:
    clean_records = [json.loads(l) for l in f]

print("\ncleaned records:")
for r in clean_records:
    print(" ", r)

all_ok &= check("all 5 rows written", len(clean_records) == 5)
all_ok &= check(
    "every row has levels_completed_before/after populated",
    all("levels_completed_before" in r and "levels_completed_after" in r for r in clean_records),
)
all_ok &= check(
    "row 2 (0-indexed) shows the completion: before=0, after=1",
    clean_records[2]["levels_completed_before"] == 0 and clean_records[2]["levels_completed_after"] == 1,
)
all_ok &= check(
    "row 0, 1 show before=after=0 (no completion yet)",
    all(clean_records[i]["levels_completed_before"] == 0 and clean_records[i]["levels_completed_after"] == 0
        for i in (0, 1)),
)
all_ok &= check(
    "row 3, 4 show before=after=1 (already past the completion)",
    all(clean_records[i]["levels_completed_before"] == 1 and clean_records[i]["levels_completed_after"] == 1
        for i in (3, 4)),
)

# Feed straight into next_chunk_boundary to confirm the two pieces actually
# connect correctly end-to-end (this is the whole point of fixing both at once).
next_boundary, is_level_boundary = next_chunk_boundary(clean_records, prev_boundary=0, max_examples=10)
all_ok &= check(
    "next_chunk_boundary on the freshly-preprocessed records finds the completion at boundary 3",
    next_boundary == 3 and is_level_boundary is True,
)

# ---------------------------------------------------------------------------
# Case 2: levels_completed missing from the raw trace entirely -- extraction
# should be a graceful no-op, not a crash, and grid_before/grid_after must
# still work normally.
# ---------------------------------------------------------------------------
raw_path2 = tmpdir / "raw_no_levels.jsonl"
out_path2 = tmpdir / "clean_no_levels.jsonl"
with open(raw_path2, "w") as f:
    for i in range(3):
        f.write(json.dumps({
            "pre_observation": {"frame": [[i]]},
            "post_observation": {"frame": [[i + 1]]},
            "action": f"ACTION_{i}",
        }) + "\n")

args2 = Namespace(
    trace=str(raw_path2), out=str(out_path2),
    pre_key="pre_observation", post_key="post_observation",
    frame_key="frame", action_key="action",
    score_key=None, levels_key="levels_completed",
)
cmd_preprocess(args2)  # should not raise

with open(out_path2) as f:
    clean_records2 = [json.loads(l) for l in f]

all_ok &= check("no-levels trace still preprocesses without error", len(clean_records2) == 3)
all_ok &= check(
    "no-levels trace: levels_completed_before/after simply absent, not None-valued",
    all("levels_completed_before" not in r and "levels_completed_after" not in r for r in clean_records2),
)
nb2, ilb2 = next_chunk_boundary(clean_records2, prev_boundary=0, max_examples=10)
all_ok &= check(
    "next_chunk_boundary on a no-levels trace falls back to cap/whole-trace only",
    nb2 == 3 and ilb2 is False,
)

# ---------------------------------------------------------------------------
# Case 3: --levels-key "" disables the attempt entirely, even if the raw
# trace does carry the field.
# ---------------------------------------------------------------------------
out_path3 = tmpdir / "clean_disabled.jsonl"
args3 = Namespace(
    trace=str(raw_path), out=str(out_path3),
    pre_key="pre_observation", post_key="post_observation",
    frame_key="frame", action_key="action",
    score_key=None, levels_key="",
)
cmd_preprocess(args3)
with open(out_path3) as f:
    clean_records3 = [json.loads(l) for l in f]
all_ok &= check(
    "--levels-key '' disables extraction even though the raw trace has the field",
    all("levels_completed_before" not in r for r in clean_records3),
)

print()
print("ALL PASS" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)