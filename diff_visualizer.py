#!/usr/bin/env python3
"""
diff_visualizer.py — standalone diagnostic for scanning a RAW (unpreprocessed,
pre_observation -> action -> post_observation) ls20-style trace, one row at a
time, to check whether a row's frame diff is unexpectedly large or scattered.

Deliberately kept separate from trace_tools.py: this is an occasional visual-
inspection tool for eyeballing raw traces, not part of the prompt-generation
pipeline, and has no reason to share trace_tools.py's release cycle.

Motivating question (see the ls20-9607627b RESET row investigated earlier):
row 0 of that trace showed a 351-cell diff spread across four unrelated
regions of the grid, correlating with pre_observation.full_reset=true /
post_observation.full_reset=false — i.e. it looks like a one-time "level
settles/reveals" repaint on reset, not a normal action-driven delta. This
script exists to check, across a WHOLE trace, whether that pattern is
confined to reset-adjacent rows or shows up elsewhere too.

USAGE

  Summary table for every row (fast, always safe to run on a full trace):
      python diff_visualizer.py TRACE.jsonl

  Full visual diff grid for one specific row (after the summary table has
  pointed you at a row worth a closer look):
      python diff_visualizer.py TRACE.jsonl --grid 0

  Full visual diff grid automatically for every row whose changed-cell count
  exceeds a threshold (e.g. to dump every "big" row in one pass):
      python diff_visualizer.py TRACE.jsonl --grid-over 40

  Both flags can be combined; --grid and --grid-over don't conflict.

DIFF GRID FORMAT

  One line per grid row. Each cell renders as its hex digit (0-9, a-f, per
  the same 0-15 -> single-hex-digit convention trace_tools.py's
  ENCODING_EXPLANATION_HEX uses) followed immediately by a single character:
  '*' if that cell changed between pre_observation and post_observation,
  otherwise a space. Fixed 2-characters-per-cell width throughout, so the
  '*' column for a given grid column lines up vertically across every
  printed row -- scanning down a column of '*'s shows you a vertical
  boundary/edge that moved, scanning across a row shows a horizontal one.

  The grid printed is the POST-transition (after) frame's values, with '*'
  marking cells that differ from the pre-transition (before) frame -- i.e.
  "what it looks like now, with the changes highlighted", not a separate
  before/after pair. Use --show-before to print the pre-transition frame
  (with the same '*' overlay) instead, e.g. to see what a since-vanished
  object looked like.
"""
import argparse
import json
import sys

HEX_DIGITS = "0123456789abcdef"


def to_hex_digit(value):
    """Render a single 0-15 int cell value as one hex digit. Raises on anything else,
    rather than silently truncating or mis-rendering an out-of-range/non-int value."""
    if not isinstance(value, int) or not (0 <= value <= 15):
        raise ValueError(
            f"cell value {value!r} is not an int in 0-15 -- can't render as a single hex "
            f"digit. This usually means --frame-key is pointing at the wrong field, or the "
            f"trace uses a wider color range than this script assumes."
        )
    return HEX_DIGITS[value]


def get_nested(d, dotted_key):
    """Walk a dotted key path (e.g. 'pre_observation.frame') through nested dicts.
    Returns None on any missing key/non-dict along the way, rather than raising --
    callers decide what a missing value means for them."""
    cur = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def diff_cells(before, after):
    """
    Return the set of (row, col) where before[row][col] != after[row][col].

    Raises on a dimension mismatch rather than silently zipping to the
    shorter shape -- a shape mismatch between pre/post frames is itself
    diagnostically interesting (it would mean the grid resized between
    calls) and should be surfaced, not swallowed.
    """
    if len(before) != len(after):
        raise ValueError(
            f"grid_before has {len(before)} rows but grid_after has {len(after)} rows -- "
            f"can't diff cell-by-cell. A genuine row-count mismatch between pre/post frames "
            f"would itself be worth knowing about."
        )
    changed = set()
    for r, (before_row, after_row) in enumerate(zip(before, after)):
        if len(before_row) != len(after_row):
            raise ValueError(
                f"row {r}: grid_before has {len(before_row)} cols but grid_after has "
                f"{len(after_row)} cols -- can't diff cell-by-cell."
            )
        for c, (b, a) in enumerate(zip(before_row, after_row)):
            if b != a:
                changed.add((r, c))
    return changed


def render_diff_grid(grid, changed):
    """
    Render `grid` (a list of lists of 0-15 ints) as hex digits, one row per
    line, each cell followed by '*' if its (row, col) is in `changed`, else
    a space. See the module docstring's DIFF GRID FORMAT section for why
    this exact layout (fixed 2 chars/cell, marker AFTER the digit) was
    chosen over alternatives.
    """
    lines = []
    for r, row in enumerate(grid):
        chars = []
        for c, value in enumerate(row):
            chars.append(to_hex_digit(value))
            chars.append("*" if (r, c) in changed else " ")
        lines.append("".join(chars))
    return "\n".join(lines)


def load_rows(trace_path):
    try:
        with open(trace_path) as f:
            lines = [line for line in f if line.strip()]
    except FileNotFoundError:
        sys.exit(f"{trace_path} not found -- check the path")
    except OSError as e:
        sys.exit(f"couldn't open {trace_path}: {e}")

    rows = []
    for lineno, line in enumerate(lines, start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            sys.exit(f"{trace_path} line {lineno} isn't valid JSON: {e}")

    if not rows:
        sys.exit(f"{trace_path} contains no rows")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trace", help="raw trace .jsonl (pre_observation/action/post_observation rows)")
    parser.add_argument("--pre-key", default="pre_observation",
                         help="top-level key holding the pre-transition observation dict")
    parser.add_argument("--post-key", default="post_observation",
                         help="top-level key holding the post-transition observation dict")
    parser.add_argument("--frame-key", default="frame",
                         help="sub-key of pre/post observation holding the grid itself")
    parser.add_argument("--reset-key", default="full_reset",
                         help="sub-key of pre/post observation flagging a reset; used only "
                              "to annotate the summary table, not to filter or skip rows")
    parser.add_argument("--grid", type=int, default=None, metavar="ROW_INDEX",
                         help="also print the full visual diff grid for this one row index "
                              "(0-based, in file order)")
    parser.add_argument("--grid-over", type=int, default=None, metavar="N",
                         help="also print the full visual diff grid for every row with MORE "
                              "than N changed cells")
    parser.add_argument("--show-before", action="store_true",
                         help="print the PRE-transition frame (with the same '*' overlay) "
                              "instead of the post-transition frame in any visual diff grid")
    args = parser.parse_args()

    rows = load_rows(args.trace)
    print(f"Loaded {len(rows)} rows from {args.trace}", flush=True)

    header = f"{'idx':>5} {'step':>6} {'action':<28} {'reset_pre':>9} {'reset_post':>10} {'changed':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    grid_shown = False
    for i, row in enumerate(rows):
        pre_frame = get_nested(row, f"{args.pre_key}.{args.frame_key}")
        post_frame = get_nested(row, f"{args.post_key}.{args.frame_key}")
        if pre_frame is None or post_frame is None:
            print(f"{i:>5} {'?':>6} {'(missing frame data -- check --pre/post/frame-key)':<28}", flush=True)
            continue

        try:
            changed = diff_cells(pre_frame, post_frame)
        except ValueError as e:
            # Report inline and move on to the next row, rather than letting
            # one malformed row (e.g. a genuine dimension mismatch) abort
            # the entire scan -- the whole point of this script is to sweep
            # every row in one pass, so one bad row silently killing every
            # row after it defeats that purpose.
            print(f"{i:>5} {'?':>6} {'(diff error: ' + str(e)[:60] + '...)':<28}", flush=True)
            continue

        reset_pre = get_nested(row, f"{args.pre_key}.{args.reset_key}")
        reset_post = get_nested(row, f"{args.post_key}.{args.reset_key}")
        action = row.get("action", "?")
        step = row.get("step", i)

        print(f"{i:>5} {str(step):>6} {str(action):<28} {str(reset_pre):>9} {str(reset_post):>10} {len(changed):>8}", flush=True)

        want_grid = (args.grid is not None and i == args.grid) or \
                    (args.grid_over is not None and len(changed) > args.grid_over)
        if want_grid:
            if args.grid is not None and i == args.grid:
                grid_shown = True
            shown_frame = pre_frame if args.show_before else post_frame
            which = "pre-transition" if args.show_before else "post-transition"
            print(f"\n--- row {i} diff grid ({which} values, '*' marks a changed cell) ---", flush=True)
            print(render_diff_grid(shown_frame, changed), flush=True)
            print(flush=True)

    if args.grid is not None and not grid_shown:
        if not (0 <= args.grid < len(rows)):
            print(f"\nNote: --grid {args.grid} is out of range for this trace "
                  f"(0-{len(rows) - 1}), so no grid was printed.", flush=True)
        else:
            print(f"\nNote: row {args.grid} had missing frame data or a diff error (see its "
                  f"summary line above), so no grid could be computed for it.", flush=True)


if __name__ == "__main__":
    main()