"""
Census of the test suite: what is actually asserted, and what is not.

A raw pytest count flatters a suite. It includes files that assert nothing,
tests that only check the repo's own hygiene, and parametrised cases that
inflate the number without adding coverage. This separates them so the figure
quoted in the README is one that can be defended.

  behavioural  exercises system behaviour -- the model, the gate, the API,
               the fallback, the demo
  meta         checks the repo about itself: artifact validity, suite integrity
  scaffold     a test file containing no test functions at all

Run: python -m scripts.test_census
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

TESTS = pathlib.Path("tests")

META_FILES = {"test_suite_integrity.py", "test_artifacts.py"}

GROUPS = {
    "test_split.py": "data & leakage",
    "test_preprocessing.py": "data & leakage",
    "test_lightgbm_encoding.py": "data & leakage",
    "test_calibration.py": "model & cost",
    "test_cost_curve.py": "model & cost",
    "test_disputes.py": "dispute spine",
    "test_policy.py": "policy gate",
    "test_audit.py": "policy gate",
    "test_schema.py": "LLM boundary",
    "test_failure_modes.py": "LLM boundary",
    "test_adversarial_llm.py": "adversarial",
    "test_api.py": "API",
    "test_demo.py": "demo",
}


def analyse(path: pathlib.Path) -> tuple[int, int, int]:
    """(test functions, functions with no assert, parametrised cases)."""
    tree = ast.parse(path.read_text())
    funcs = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name.startswith("test_")]
    no_assert = 0
    cases = 0
    for fn in funcs:
        # Walk the tree rather than substring-matching ast.dump(): `pytest.raises`
        # renders as attr='raises' and np.testing.assert_array_equal as
        # attr='assert_array_equal', so a naive dump search reports false
        # "asserts nothing" on tests that assert perfectly well.
        asserts = False
        for node in ast.walk(fn):
            if isinstance(node, ast.Assert):
                asserts = True
                break
            if isinstance(node, ast.Call):
                f = node.func
                name = getattr(f, "attr", None) or getattr(f, "id", None) or ""
                if name.startswith("assert") or name in {"raises", "fail",
                                                          "approx", "warns"}:
                    asserts = True
                    break
            if isinstance(node, ast.With):
                for item in node.items:
                    c = item.context_expr
                    if isinstance(c, ast.Call):
                        n = getattr(c.func, "attr", None) or \
                            getattr(c.func, "id", None) or ""
                        if n in {"raises", "warns"}:
                            asserts = True
                if asserts:
                    break
        if not asserts:
            no_assert += 1
        n = 1
        for dec in fn.decorator_list:
            src = ast.dump(dec)
            if "parametrize" in src:
                for arg in getattr(dec, "args", []):
                    if isinstance(arg, (ast.List, ast.Tuple)):
                        n = max(n, len(arg.elts))
                    elif isinstance(arg, ast.Name):
                        n = 0          # module-level list, counted from pytest
        cases += n
    return len(funcs), no_assert, cases


def main() -> int:
    files = sorted(TESTS.glob("test_*.py"))
    scaffold, meta, behavioural = [], [], []
    no_assert_total = 0

    for f in files:
        n, no_assert, _ = analyse(f)
        no_assert_total += no_assert
        if n == 0:
            scaffold.append((f.name, n))
        elif f.name in META_FILES:
            meta.append((f.name, n, no_assert))
        else:
            behavioural.append((f.name, n, no_assert))

    print(f"{'file':<30} {'group':<18} {'tests':>6} {'no-assert':>10}")
    print("-" * 68)
    for name, n, na in sorted(behavioural,
                              key=lambda r: (GROUPS.get(r[0], "zz"), r[0])):
        print(f"{name:<30} {GROUPS.get(name, '—'):<18} {n:>6} {na:>10}")
    print("-" * 68)
    for name, n, na in meta:
        print(f"{name:<30} {'meta / hygiene':<18} {n:>6} {na:>10}")
    for name, n in scaffold:
        print(f"{name:<30} {'SCAFFOLD (empty)':<18} {n:>6} {'—':>10}")

    b = sum(n for _, n, _ in behavioural)
    m = sum(n for _, n, _ in meta)
    print()
    print(f"  behavioural test functions : {b}")
    print(f"  meta / hygiene functions   : {m}")
    print(f"  empty scaffold files       : {len(scaffold)}")
    print(f"  functions with no assertion: {no_assert_total}")

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True, text=True)
    tail = [l for l in collected.stdout.strip().splitlines() if "test" in l][-1:]
    print(f"\n  pytest collects            : {tail[0] if tail else '?'}")
    print("  (higher than the function count because parametrised cases expand)")

    if scaffold:
        print(f"\n  WARNING: {len(scaffold)} test file(s) assert nothing.")
        return 1
    if no_assert_total:
        print(f"\n  WARNING: {no_assert_total} test function(s) assert nothing.")
        return 1
    print("\n  Every test file and every test function asserts something.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
