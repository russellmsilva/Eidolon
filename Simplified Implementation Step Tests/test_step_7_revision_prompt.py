import importlib.util
spec = importlib.util.spec_from_file_location('trace_tools', 'trace_tools.py')
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)


def grid(fill=0, w=3, h=3):
    return [[fill] * w for _ in range(h)]


# ---------- Test 1: tie-break worked example (design doc) ----------
# k=10, 6 rows at count=5, 5 rows at count=4 -> all 6 count-5 rows (file
# order), plus first 4 (file order) of the 5 count-4 rows; last count-4
# row dropped.
counts = {}
# Insert in a deliberately mixed order to make sure sort respects INSERTION
# order (file order) for the tie-break, not step-number order coincidentally
# matching it. We still key by step so build_revise_prompt's "most recent
# row" logic (last-inserted key) is meaningful.
step = 0
count5_keys = []
count4_keys = []
for i in range(6):
    k = str(step); step += 1
    counts[k] = {"count": 5, "actual_grid": grid(1), "actual_goal": False,
                 "predicted_grid": grid(2), "predicted_goal": False, "error": None}
    count5_keys.append(k)
for i in range(5):
    k = str(step); step += 1
    counts[k] = {"count": 4, "actual_grid": grid(1), "actual_goal": False,
                 "predicted_grid": grid(2), "predicted_goal": False, "error": None}
    count4_keys.append(k)

selected = tt.select_top_k_failures(counts, 10)
selected_keys = [k for k, _ in selected]
expected = count5_keys + count4_keys[:4]
assert selected_keys == expected, f"tie-break mismatch:\n got {selected_keys}\n exp {expected}"
print("Test 1 (tie-break worked example): PASS")


# ---------- Test 2: three row-rendering cases ----------
counts2 = {}
# ordinary wrong-transition row
counts2["10"] = {
    "count": 3, "actual_grid": grid(7), "actual_goal": False,
    "predicted_grid": grid(9), "predicted_goal": False, "error": None,
}
# false-positive goal (predicted True, actually False)
counts2["11"] = {
    "count": 2, "actual_grid": grid(5), "actual_goal": False,
    "predicted_grid": grid(6), "predicted_goal": True, "error": None,
}
# false-negative goal (predicted False, actually True) -- grid diff should be omitted
counts2["12"] = {
    "count": 1, "actual_grid": grid(0), "actual_goal": True,
    "predicted_grid": grid(8), "predicted_goal": False, "error": None,
}
# error/crash row
counts2["13"] = {
    "count": 4, "actual_grid": grid(3), "actual_goal": False,
    "predicted_grid": None, "predicted_goal": False, "error": "ZeroDivisionError: boom",
}
# a currently-passing row (count 0) -- must NOT be selected
counts2["9"] = {
    "count": 0, "actual_grid": grid(1), "actual_goal": False,
    "predicted_grid": grid(1), "predicted_goal": False, "error": None,
}

prompt = tt.build_revise_prompt("class GameModel:\n    pass\n", counts2, k=5, encoding="hex")

assert "def predict(self, grid_before" in prompt
assert "You may only import: copy, itertools, math, collections, functools, and numpy" in prompt
assert "GameModel class now" in prompt
assert "predict_next_state" not in prompt  # old single-function contract must be fully gone

# Selection is sorted by count descending: step13(4), step10(3), step11(2), step12(1).
# row 13: error, highest count -> Counterexample 1
assert "raised an error or timed out" in prompt
assert "ZeroDivisionError: boom" in prompt
assert "Counterexample 1 (trace step 13, failed 4x so far)" in prompt
# row 10: ordinary wrong-transition
assert "Counterexample 2 (trace step 10, failed 3x so far)" in prompt
# row 11: false positive
assert "FALSE-POSITIVE goal prediction" in prompt
assert "Counterexample 3 (trace step 11, failed 2x so far)" in prompt
# row 12: false negative -- grid diff omitted
assert "MISSED LEVEL-COMPLETION TRANSITION" in prompt
assert "Counterexample 4 (trace step 12, failed 1x so far)" in prompt
# passing row must not appear as a counterexample header
assert "trace step 9," not in prompt

# most recent row = highest-step = "9" (last inserted key), which has count==0 -> passed
assert "predicted CORRECTLY." in prompt

print("Test 2 (three row cases + error row + most-recent-row note): PASS")


# ---------- Test 3: k-validation ----------
# k > total rows ever tracked (misconfiguration) -> must raise ValueError
try:
    tt.build_revise_prompt("class GameModel:\n    pass\n", counts2, k=100)
    raise SystemExit("expected ValueError for k > total rows, none raised")
except ValueError as e:
    assert "exceeds the total number of rows" in str(e)
    print("Test 3 (k > total rows raises ValueError): PASS")

# k larger than the number of CURRENTLY FAILING rows, but <= total rows
# tracked -> must NOT raise; just returns however many are failing.
# counts2 has 5 total rows tracked (steps 9,10,11,12,13), of which 4 are
# currently failing (9 has count=0). k=5 (<= total=5) but only 4 failing.
prompt_fewer = tt.build_revise_prompt("class GameModel:\n    pass\n", counts2, k=5, encoding="hex")
n_selected = len(tt.select_top_k_failures(counts2, 5))
assert n_selected == 4, f"expected 4 failing rows selected, got {n_selected}"
assert "Counterexample 4" in prompt_fewer and "Counterexample 5" not in prompt_fewer
print("Test 3b (k > currently-failing count, k <= total rows: no error, returns fewer): PASS")

# classic worked case from the user: k=10, only 6 failing rows, but total
# rows tracked >= 10 so the misconfiguration check doesn't fire.
counts4 = {}
for i in range(12):
    step = str(100 + i)
    counts4[step] = {
        "count": 3 if i < 6 else 0,  # first 6 failing, rest passing
        "actual_grid": grid(1), "actual_goal": False,
        "predicted_grid": grid(2), "predicted_goal": False, "error": None,
    }
selected4 = tt.select_top_k_failures(counts4, 10)
assert len(selected4) == 6, f"expected 6 (all failing rows, k=10 > 6 failing but <= 12 total), got {len(selected4)}"
prompt4 = tt.build_revise_prompt("class GameModel:\n    pass\n", counts4, k=10, encoding="hex")
assert "Counterexample 6" in prompt4 and "Counterexample 7" not in prompt4
print("Test 3c (k=10, 6 failing out of 12 total tracked: no error, 6 counterexamples): PASS")



# ---------- Test 4: zero currently-failing rows raises loudly (distinct ----------
# ---------- from the k > total-rows misconfiguration case in Test 3) -----
# Empty dict entirely: k=1 > total_rows=0 -> the k-vs-total-rows check
# fires first (a more specific, more useful error than "no failing rows").
try:
    tt.build_revise_prompt("class GameModel:\n    pass\n", {}, k=1)
    raise SystemExit("expected ValueError for k > 0 total rows, none raised")
except ValueError as e:
    assert "exceeds the total number of rows" in str(e)
    print("Test 4a (empty row_failure_counts, k > 0: k-vs-total-rows error): PASS")

# Rows exist and k is well within range, but every one of them currently
# passes (count == 0) -- this is Step 8's early-stop's job to make
# unreachable in the real pipeline, so build_revise_prompt must still fail
# loudly here rather than silently building an empty-counterexamples prompt.
all_passing = {
    "1": {"count": 0, "actual_grid": grid(1), "actual_goal": False,
          "predicted_grid": grid(1), "predicted_goal": False, "error": None},
    "2": {"count": 0, "actual_grid": grid(2), "actual_goal": False,
          "predicted_grid": grid(2), "predicted_goal": False, "error": None},
}
try:
    tt.build_revise_prompt("class GameModel:\n    pass\n", all_passing, k=1)
    raise SystemExit("expected ValueError for zero failing rows, none raised")
except ValueError as e:
    assert "unreachable" in str(e)
    print("Test 4b (all rows passing, k within total-rows range: unreachable error): PASS")


# ---------- Test 5: most recent row INCORRECT case ----------
counts3 = dict(counts2)
# reorder so the last-inserted key is a currently-failing row
counts3.pop("9")
counts3["14"] = {
    "count": 2, "actual_grid": grid(1), "actual_goal": False,
    "predicted_grid": grid(1), "predicted_goal": False, "error": None,
}
prompt3 = tt.build_revise_prompt("class GameModel:\n    pass\n", counts3, k=5, encoding="hex")
assert "predicted INCORRECTLY." in prompt3
print("Test 5 (most-recent-row incorrect case): PASS")

print("\nALL STEP 7 TESTS PASSED")