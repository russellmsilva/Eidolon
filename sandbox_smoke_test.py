#!/usr/bin/env python3
"""
sandbox_smoke_test.py — standalone verification of trace_tools.py's real
`bwrap` sandbox path, using synthetic records and disposable toy candidate
files. No clean.jsonl or working GameModel candidate required.

Run this on the JarvisLabs box (same conda env, `bwrap` on PATH) with:
    python sandbox_smoke_test.py

It must sit next to trace_tools.py (or edit sys.path.insert below).

Everything here uses use_sandbox=True (the default / real path), unlike
earlier testing in a container without bwrap installed, which had to use
use_sandbox=False and therefore never touched the actual mount-namespace
behavior (in particular, never verified the --tmpfs /tmp removal).

Exit code is 0 if every check passed, 1 otherwise — safe to wire into a
CI step.
"""
import sys
import os
import shutil
import tempfile
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_tools import run_candidate, run_backtest, grid_shape_error, run_with_output_cap

RECORDS = [
    {"step": 0, "action": "up", "grid_before": [[0, 0], [0, 0]], "grid_after": [[1, 0], [0, 0]],
     "levels_completed_before": 0, "levels_completed_after": 0},
    {"step": 1, "action": "down", "grid_before": [[1, 0], [0, 0]], "grid_after": [[0, 0], [0, 0]],
     "levels_completed_before": 0, "levels_completed_after": 0},
]

results = []  # (name, passed: bool, detail: str)


def check(name, passed, detail=""):
    results.append((name, passed, detail))
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))


def write_candidate(tmpdir, filename, body):
    path = os.path.join(tmpdir, filename)
    with open(path, "w") as f:
        f.write(body)
    return path


def main():
    tmpdir = tempfile.mkdtemp(prefix="eidolon_smoke_")
    try:
        # ---- 1. Baseline sanity: does bwrap even run here? ----
        try:
            candidate = write_candidate(tmpdir, "cand_legit.py", """
import copy
class GameModel:
    def predict(self, grid_before, action, state):
        return copy.deepcopy(grid_before), False, state
""")
            r = run_candidate(candidate, RECORDS, use_sandbox=True)
            ok = (r.get(0, {}).get("prediction") == RECORDS[0]["grid_before"]
                  and r.get(1, {}).get("prediction") == RECORDS[1]["grid_before"])
            check("bwrap sandbox runs at all + legit allowed-import candidate works",
                  ok, f"result={r}")
        except Exception as e:
            check("bwrap sandbox runs at all + legit allowed-import candidate works",
                  False, f"{type(e).__name__}: {e}")
            print("\nbwrap itself failed to run — stopping here, nothing below "
                  "this point can be meaningfully tested.")
            print_summary()
            sys.exit(1)

        # ---- 2. __import__('os') bypass should be blocked ----
        candidate = write_candidate(tmpdir, "cand_import_os.py", """
class GameModel:
    def predict(self, grid_before, action, state):
        os_mod = __import__("os")
        return grid_before, False, state
""")
        r = run_candidate(candidate, RECORDS, use_sandbox=True)
        err = r.get(0, {}).get("error", "")
        check("__import__('os') bypass blocked", "not allowed" in err, err)

        # ---- 3. open() bypass should be blocked ----
        candidate = write_candidate(tmpdir, "cand_open.py", """
class GameModel:
    def predict(self, grid_before, action, state):
        f = open("/etc/hostname")
        return grid_before, False, state
""")
        r = run_candidate(candidate, RECORDS, use_sandbox=True)
        err = r.get(0, {}).get("error", "")
        check("open() bypass blocked", "not defined" in err or "NameError" in err, err)

        # ---- 4. __import__('signal') to disable SIGALRM should be blocked ----
        candidate = write_candidate(tmpdir, "cand_disable_alarm.py", """
class GameModel:
    def predict(self, grid_before, action, state):
        sig = __import__("signal")
        sig.signal(sig.SIGALRM, sig.SIG_IGN)
        while True:
            pass
        return grid_before, False, state
""")
        t0 = time.time()
        r = run_candidate(candidate, RECORDS, use_sandbox=True,
                           overall_timeout=5, per_call_seconds=1, cpu_seconds=10)
        elapsed = time.time() - t0
        err = r.get(0, {}).get("error", "")
        check("SIGALRM-disable bypass blocked (fails fast, not a 5s hang)",
              "not allowed" in err and elapsed < 4, f"elapsed={elapsed:.2f}s error={err}")

        # ---- 5. malformed grid shape -> clean failure via run_backtest ----
        candidate = write_candidate(tmpdir, "cand_malformed.py", """
class GameModel:
    def predict(self, grid_before, action, state):
        return [[1, 2], [3]], False, state
""")
        counts_path = os.path.join(tmpdir, "counts.json")
        result = run_backtest(candidate, RECORDS, boundary=2, counts_path=counts_path,
                               use_sandbox=True)
        errors = [row.get("error", "") for row in result["scored"]]
        check("malformed grid shape scored as clean failure, no crash",
              all("malformed prediction" in e for e in errors), f"errors={errors}")

        # ---- 6. /tmp should not be writable at all (the actual question) ----
        candidate = write_candidate(tmpdir, "cand_tmp_open.py", """
class GameModel:
    def predict(self, grid_before, action, state):
        f = open("/tmp/x", "w")
        f.write("data")
        return grid_before, False, state
""")
        r = run_candidate(candidate, RECORDS, use_sandbox=True)
        err = r.get(0, {}).get("error", "")
        # open() itself is stripped from builtins, so this should fail with
        # NameError before ever reaching the filesystem -- but the point of
        # this test is confirming SOMETHING blocks it, whichever layer.
        check("bare open('/tmp/x','w') blocked", bool(err), err)

        # ---- 7. numpy.save() to /tmp — the case builtins-stripping CANNOT catch ----
        candidate = write_candidate(tmpdir, "cand_numpy_tmp_write.py", """
import numpy
class GameModel:
    def predict(self, grid_before, action, state):
        numpy.save("/tmp/x.npy", numpy.zeros((10, 10)))
        return grid_before, False, state
""")
        r = run_candidate(candidate, RECORDS, use_sandbox=True)
        err = r.get(0, {}).get("error", "")
        # This is the one that matters: numpy's OWN open() is unrestricted,
        # so this can only be blocked by /tmp not existing/not being
        # writable at the mount level -- confirms the --tmpfs /tmp removal
        # actually works, not just the builtins-level checks above.
        check("numpy.save() to /tmp blocked at the mount level "
              "(THE key check for this change)",
              bool(err), err or "(no error — write likely SUCCEEDED, this is the regression to watch for)")

        # ---- 8. stdout flood still gets capped under real bwrap ----
        # (Rerun of the earlier non-sandboxed test, now through bwrap, to
        # make sure the cap logic still engages once real subprocess
        # plumbing/pipes are involved, not just the Python-level Popen
        # wiring tested before.)
        flood_cmd = [sys.executable, "-c",
                     "import sys\n"
                     "while True:\n"
                     "    sys.stdout.write('x' * 1000000)\n"
                     "    sys.stdout.flush()\n"]
        t0 = time.time()
        r = run_with_output_cap(flood_cmd, "", os.environ.copy(),
                                 overall_timeout=30, max_output_bytes=2_000_000)
        elapsed = time.time() - t0
        check("stdout flood killed quickly with bounded capture",
              elapsed < 5 and len(r.stdout) < 5_000_000,
              f"elapsed={elapsed:.2f}s captured={len(r.stdout)} bytes")

        print_summary()
        sys.exit(0 if all(p for _, p, _ in results) else 1)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def print_summary():
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, p, _ in results if p)
    print(f"SUMMARY: {n_pass}/{len(results)} checks passed")
    for name, passed, _ in results:
        if not passed:
            print(f"  FAILED: {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()