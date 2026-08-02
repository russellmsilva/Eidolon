import ast
import sys

sys.path.insert(0, ".")
from trace_tools import keep_first_class_def

PASS = "PASS"
FAIL = "FAIL"

def check(name, source, expect_defs, expect_source_check):
    trimmed, n_defs = keep_first_class_def(source)
    ok_defs = (n_defs == expect_defs)
    ok_parses = True
    try:
        ast.parse(trimmed)
    except SyntaxError as e:
        ok_parses = False
        parse_err = e
    ok_source = expect_source_check(trimmed)
    status = PASS if (ok_defs and ok_parses and ok_source) else FAIL
    print(f"[{status}] {name}  (n_defs={n_defs}, expected={expect_defs}, parses={ok_parses}, source_check={ok_source})")
    if status == FAIL:
        print("----- trimmed source -----")
        print(trimmed)
        print("---------------------------")
    return status == PASS


all_ok = True

# (a) one clean class -- nothing should be trimmed, n_defs == 0
case_a = '''
class GameModel:
    def predict(self, grid_before, action, previous_state):
        state = previous_state or {}
        return grid_before, False, state

    def helper(self):
        return 1
'''
all_ok &= check(
    "a) one clean class",
    case_a,
    expect_defs=0,
    expect_source_check=lambda s: s.strip() == case_a.strip() and s.count("def predict") == 1 and s.count("def helper") == 1,
)

# (b) two full class redefinitions -- keep only the first
case_b = '''
class GameModel:
    def predict(self, grid_before, action, previous_state):
        # first attempt, the good one
        state = previous_state or {}
        return grid_before, False, state

class GameModel:
    def predict(self, grid_before, action, previous_state):
        # actually let me reconsider, worse attempt
        return grid_before, False, {}
'''
all_ok &= check(
    "b) two full class redefinitions",
    case_b,
    expect_defs=1,
    expect_source_check=lambda s: s.count("class GameModel") == 1 and "the good one" in s and "reconsider" not in s,
)

# (c) one class with two `predict` methods inside it -- keep only the first
case_c = '''
class GameModel:
    def predict(self, grid_before, action, previous_state):
        # first predict, the good one
        state = previous_state or {}
        return grid_before, False, state

    def predict(self, grid_before, action, previous_state):
        # actually wait, worse redraft of predict
        return grid_before, False, {}

    def helper(self):
        return 1
'''
all_ok &= check(
    "c) one class, two predict methods",
    case_c,
    expect_defs=1,
    expect_source_check=lambda s: (
        s.count("class GameModel") == 1
        and s.count("def predict") == 1
        and "the good one" in s
        and "worse redraft" not in s
        and "def helper" in s  # unrelated method after the duplicate must survive
    ),
)

# Bonus (d): syntax error input should pass through unchanged with 0 defs,
# per the documented contract (caller's own ast.parse raises normally).
case_d = "class GameModel:\n    def predict(self:\n        pass\n"
trimmed_d, n_defs_d = keep_first_class_def(case_d)
ok_d = (trimmed_d == case_d and n_defs_d == 0)
print(f"[{'PASS' if ok_d else 'FAIL'}] d) syntax error passthrough  (n_defs={n_defs_d})")
all_ok &= ok_d

# Bonus (e): no GameModel class present at all -- source untouched, 0 defs.
case_e = "def some_other_function():\n    return 1\n"
trimmed_e, n_defs_e = keep_first_class_def(case_e)
ok_e = (trimmed_e == case_e and n_defs_e == 0)
print(f"[{'PASS' if ok_e else 'FAIL'}] e) no GameModel class present  (n_defs={n_defs_e})")
all_ok &= ok_e

print()
print("ALL PASS" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)