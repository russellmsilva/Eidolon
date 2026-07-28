"""Keyboard-driven ARC-AGI-3 agent for manual trace collection.

Swap this in for `agent/my_agent.py` (temporarily -- see note at the
bottom of this docstring) when you want to *play* a game yourself and
record a {pre_observation, action, post_observation} trace, instead of
the random-policy version used for automated traces.

Terminal-rendered: each frame is printed as text (color blocks) to your
terminal after every step. Keyboard-driven: keypresses map directly to
GameAction values, read one at a time -- no need to press Enter, except
for ACTION7 which needs an (x, y) point.

Key mapping (edit KEY_MAP below once you've seen what each action does --
these are reasonable guesses, not verified against this specific game):
    w / Up arrow      -> ACTION1
    s / Down arrow    -> ACTION2
    a / Left arrow    -> ACTION3
    d / Right arrow   -> ACTION4
    space             -> ACTION5
    f                 -> ACTION6
    c                 -> ACTION7 (complex action -- prompts for x,y)
    1-7               -> ACTION1-ACTION7 (numeric alternative to the above)
    r                 -> RESET
    q                 -> quit early (trace is still dumped with whatever
                          was recorded so far, ending on a RESET step)

Same JSONL trace output as my_agent.py: one line per
{pre_observation, action, post_observation} triple, written to
data/traces/trace_<timestamp>.jsonl.

Requires an interactive terminal (this reads raw keypresses from stdin;
it won't work piped or redirected).

--------------------------------------------------------------------------
HOW TO USE THIS FILE WITHOUT DISTURBING YOUR SUBMISSION AGENT:

`agent/my_agent.py` is the file `scripts/build_notebook.py` splices into
your Kaggle submission notebook, so don't leave this keyboard version in
its place. Two equivalent ways to play manually and then restore it:

    # Option A: copy swap
    cp agent/my_agent.py agent/my_agent.py.bak
    cp agent/my_agent_keyboard.py agent/my_agent.py
    make play-local GAME=ls20
    cp agent/my_agent.py.bak agent/my_agent.py
    rm agent/my_agent.py.bak

    # Option B: symlink swap
    mv agent/my_agent.py agent/my_agent_random.py
    ln -s my_agent_keyboard.py agent/my_agent.py
    make play-local GAME=ls20
    rm agent/my_agent.py
    mv agent/my_agent_random.py agent/my_agent.py
--------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import os
import sys
import time

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

try:
    import termios
    import tty
    _HAS_TERMIOS = True
except ImportError:
    import msvcrt  # Windows fallback
    _HAS_TERMIOS = False


# Rough ANSI 256-color mapping for an ARC-style 0-9 palette. Extend if this
# game uses values outside that range -- unknown values fall back to
# printing the raw number instead of a color block.
_PALETTE = {
    0: 232,  # black
    1: 25,   # blue
    2: 196,  # red
    3: 34,   # green
    4: 226,  # yellow
    5: 244,  # grey
    6: 201,  # magenta/pink
    7: 208,  # orange
    8: 51,   # cyan
    9: 94,   # maroon/brown
    10: 1,
    11: 180,
    12: 160,
    13: 70,
    14: 120,
    15: 140,
}
_NO_COLOR = os.environ.get("ARC_NO_COLOR", "") != ""


def _color_cell(v) -> str:
    # Always print exactly 2 characters per cell so rows stay aligned
    # regardless of whether the value happens to be in our hardcoded
    # palette. Unknown values still get a (stable, derived) color instead
    # of falling back to a differently-sized plain-text format.
    if _NO_COLOR:
        return f"{str(v)[-2:]:>2}"
    color = _PALETTE.get(v) if isinstance(v, int) else None
    if color is None and isinstance(v, int):
        color = 16 + (v * 37) % 216  # stable pseudo-color for out-of-palette values
    if color is None:
        return f"{str(v)[-2:]:>2}"
    return f"\x1b[48;5;{color}m  \x1b[0m"


def _serialize_frame(frame: FrameData) -> dict:
    """Same defensive dump as the random-policy agent -- kept identical so
    traces from either agent are interchangeable."""
    if hasattr(frame, "model_dump"):
        raw = frame.model_dump()
    else:
        raw = dict(vars(frame))

    def _clean(v):
        if isinstance(v, GameState):
            return v.value if hasattr(v, "value") else str(v)
        if isinstance(v, GameAction):
            return v.name if hasattr(v, "name") else str(v)
        if isinstance(v, (list, tuple)):
            return [_clean(x) for x in v]
        if isinstance(v, dict):
            return {k: _clean(x) for k, x in v.items()}
        if not isinstance(v, (str, int, float, bool, type(None))):
            return str(v)
        return v

    return {k: _clean(v) for k, v in raw.items()}


def _serialize_frame_for_trace(frame: FrameData) -> dict:
    """Same as _serialize_frame, but additionally collapses `frame` down to
    just the settled (last) grid before it's written to the trace file.

    `frame` is always a list of grids -- length 1 for a normal step, length
    N for an action whose effect takes several engine ticks to resolve
    (e.g. a multi-tile push). Only the settled state matters for the
    predict_next_state(grid, action) -> grid task, so we keep just that
    and record how many ticks there originally were as metadata, rather
    than writing every intermediate tick to disk.

    Note: this only affects what gets *written to the trace*. Terminal
    rendering still shows every tick, since that reads live off the
    FrameData object in _render_frame, not off this serialized dict.
    """
    obs = _serialize_frame(frame)
    raw_frame = obs.get("frame")
    if raw_frame:
        obs["frame_ticks"] = len(raw_frame)
        obs["frame"] = raw_frame[-1]
    else:
        obs["frame_ticks"] = 0
    return obs


def _render_frame(latest_frame: FrameData, action_count: int) -> None:
    state = getattr(latest_frame, "state", None)
    state_str = state.value if hasattr(state, "value") else str(state)
    levels_completed = getattr(latest_frame, "levels_completed", "?")
    win_levels = getattr(latest_frame, "win_levels", "?")

    print(f"\n--- step {action_count} | state={state_str} | "
          f"levels_completed={levels_completed} win_levels={win_levels} ---")

    grids = getattr(latest_frame, "frame", None)
    if not grids:
        print("(no frame data)")
        return

    # `frame` can be a single 2D grid, or a list of 2D grids representing
    # successive animation ticks produced by one action (e.g. a multi-step
    # push/slide) -- normalize to a list either way. These are sequential
    # in time, not stacked spatial layers.
    if grids and isinstance(grids[0], list) and grids[0] and isinstance(grids[0][0], list):
        ticks = grids
    else:
        ticks = [grids]

    for i, tick in enumerate(ticks):
        if len(ticks) > 1:
            label = "final" if i == len(ticks) - 1 else f"tick {i}"
            print(f"  [{label}]")
        for row in tick:
            print("  " + "".join(_color_cell(v) for v in row))

    available = getattr(latest_frame, "available_actions", None)
    if available:
        names = [a.name if hasattr(a, "name") else str(a) for a in available]
        print(f"  available_actions: {names}")


def _read_key() -> str:
    """Read a single keypress. Returns a printable char, or 'UP'/'DOWN'/
    'LEFT'/'RIGHT' for arrow keys."""
    if _HAS_TERMIOS:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                rest = sys.stdin.read(2)
                return {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT"}.get(rest, "")
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    else:
        ch = msvcrt.getch()
        if ch in (b"\xe0", b"\x00"):
            arrow = msvcrt.getch()
            return {b"H": "UP", b"P": "DOWN", b"M": "RIGHT", b"K": "LEFT"}.get(arrow, "")
        try:
            return ch.decode()
        except UnicodeDecodeError:
            return ""


KEY_MAP = {
    "w": "ACTION1", "UP": "ACTION1", "1": "ACTION1",
    "s": "ACTION2", "DOWN": "ACTION2", "2": "ACTION2",
    "a": "ACTION3", "LEFT": "ACTION3", "3": "ACTION3",
    "d": "ACTION4", "RIGHT": "ACTION4", "4": "ACTION4",
    " ": "ACTION5", "5": "ACTION5",
    "f": "ACTION6", "6": "ACTION6",
    "c": "ACTION7", "7": "ACTION7",
    "r": "RESET",
}


def _prompt_xy() -> dict:
    # Complex actions need real Enter-based input, so this briefly steps
    # outside single-keypress mode via a plain input() call.
    while True:
        raw = input("  ACTION7 needs a point -- enter as 'x,y' (0-63 each): ").strip()
        try:
            x_str, y_str = raw.split(",")
            x, y = int(x_str), int(y_str)
            if 0 <= x <= 63 and 0 <= y <= 63:
                return {"x": x, "y": y}
        except ValueError:
            pass
        print("  Invalid input, try again (example: 32,10)")


class MyAgent(Agent):
    MAX_ACTIONS = 200  # generous manual-play cap; 'q' ends the session early

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace = []
        self._pending = None
        self._quit = False

        out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "traces")
        os.makedirs(out_dir, exist_ok=True)
        self._trace_path = os.path.join(out_dir, f"trace_{int(time.time())}.jsonl")

        print("Keyboard controls: w/a/s/d or arrows = ACTION1-4, "
              "space = ACTION5, f = ACTION6, c = ACTION7 (prompts for x,y), "
              "1-7 = same actions by number, r = RESET, q = quit")

    def is_done(self, frames: list, latest_frame: FrameData) -> bool:
        self._close_pending(latest_frame)

        done = (
            self._quit
            or latest_frame.state is GameState.WIN
            or len(self._trace) >= self.MAX_ACTIONS
        )
        if done:
            self._dump_trace()
        return done

    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        self._close_pending(latest_frame)
        _render_frame(latest_frame, len(self._trace))

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            print("  (game over/not started -- auto-RESET)")
            action = GameAction.RESET
        else:
            action = self._read_action()

        if action.is_complex():
            action.data = _prompt_xy()
        if action.is_simple():
            action.reasoning = "manual keyboard play"

        self._pending = (_serialize_frame_for_trace(latest_frame), str(action))
        return action

    def _read_action(self) -> GameAction:
        while True:
            key = _read_key()
            if key == "q":
                self._quit = True
                return GameAction.RESET
            name = KEY_MAP.get(key)
            if name is None:
                continue
            return GameAction[name]

    def _close_pending(self, latest_frame: FrameData) -> None:
        if self._pending is None:
            return
        pre_obs, action_str = self._pending
        self._trace.append(
            {
                "pre_observation": pre_obs,
                "action": action_str,
                "post_observation": _serialize_frame_for_trace(latest_frame),
            }
        )
        self._pending = None

    def _dump_trace(self) -> None:
        multi_tick = 0
        with open(self._trace_path, "w") as f:
            for triple in self._trace:
                f.write(json.dumps(triple) + "\n")
                if (triple["pre_observation"].get("frame_ticks", 1) > 1
                        or triple["post_observation"].get("frame_ticks", 1) > 1):
                    multi_tick += 1
        print(f"[MyAgent] wrote {len(self._trace)} triples to {self._trace_path} "
              f"({multi_tick} had multi-tick frames, now collapsed to the settled grid)")