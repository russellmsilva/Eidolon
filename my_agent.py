"""Your ARC-AGI-3 agent. This is the *only* file you should normally edit.

`scripts/build_notebook.py` splices the contents of this file into the
Kaggle submission notebook, so your local dev loop and your Kaggle
submission stay in lock-step:

    [edit my_agent.py] → [make play-local] → [make submit]

This program plays a game with a random (scripted) policy for MAX_ACTIONS steps and
records every (pre_observation, action, post_observation) triple to a JSONL
file under data/traces/, so it can later be split into "history" vs.
"held-out ground truth" for the program-synthesis test.
 
Run with:
    make play-local GAME=ls20
(swap ls20 for whichever public game id you want to trace)

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`.
  - Class must be named `MyAgent` (the notebook's __init__.py registers it).
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.
"""
from __future__ import annotations

import random
import time
import json
import os

from typing import Any

from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent


def _serialize_frame(frame: FrameData) -> dict:
    """Defensively dump a FrameData object to a JSON-safe dict.
 
    We don't hardcode a single grid attribute name (e.g. `.frame`) because
    the exact field can vary between package versions -- instead we dump
    the whole pydantic model (or __dict__ as a fallback) and let anything
    non-serializable (like the GameState enum) get stringified.
    """
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
        # Final catch-all: anything else non-primitive gets stringified
        # rather than risking a JSON-serialization crash mid-trace.
        if not isinstance(v, (str, int, float, bool, type(None))):
            return str(v)
        return v
 
    return {k: _clean(v) for k, v in raw.items()}
 
 
class MyAgent(Agent):
    # A few dozen steps, per the plan -- bump this if you want a longer trace.
    MAX_ACTIONS = 40
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace = []
        self._pending = None  # (pre_observation_dict, action_str) awaiting its result
 
        out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "traces")
        os.makedirs(out_dir, exist_ok=True)
        self._trace_path = os.path.join(out_dir, f"trace_{int(time.time())}.jsonl")
 
    def is_done(self, frames: list, latest_frame: FrameData) -> bool:
        self._close_pending(latest_frame)
 
        done = (
            latest_frame.state is GameState.WIN
            or len(self._trace) >= self.MAX_ACTIONS
        )
        if done:
            self._dump_trace()
        return done
 
    def choose_action(self, frames: list, latest_frame: FrameData) -> GameAction:
        self._close_pending(latest_frame)
 
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
        else:
            # Random scripted policy -- no RESET, since we only want it when forced.
            candidates = [a for a in GameAction if a is not GameAction.RESET]
            action = random.choice(candidates)
            if action.is_complex():
                action.data = {"x": random.randint(0, 63), "y": random.randint(0, 63)}
 
        if action.is_simple():
            action.reasoning = "random scripted policy"
 
        # Stash the pre-action observation; we close the triple next call,
        # once we've seen the resulting frame.
        self._pending = (_serialize_frame(latest_frame), str(action))
        return action
 
    def _close_pending(self, latest_frame: FrameData) -> None:
        if self._pending is None:
            return
        pre_obs, action_str = self._pending
        self._trace.append(
            {
                "pre_observation": pre_obs,
                "action": action_str,
                "post_observation": _serialize_frame(latest_frame),
            }
        )
        self._pending = None
 
    def _dump_trace(self) -> None:
        with open(self._trace_path, "w") as f:
            for triple in self._trace:
                f.write(json.dumps(triple) + "\n")
        print(f"[MyAgent] wrote {len(self._trace)} triples to {self._trace_path}")
 
