"""Guard against a green suite that asserts nothing.

tests/test_schema.py, test_policy.py and test_failure_modes.py are currently
docstring-only stubs for Day 3/4 work. pytest reports them as passing files,
which makes the suite look larger than it is. This test names them explicitly,
so the day they are written the list shrinks deliberately rather than the
scaffolding being forgotten.
"""
import pathlib
import re

TESTS = pathlib.Path(__file__).parent

# Files known to be scaffolding. Remove entries as they are implemented.
KNOWN_EMPTY = {"test_schema.py", "test_policy.py", "test_failure_modes.py"}


def _test_functions(path: pathlib.Path) -> list[str]:
    return re.findall(r"^def (test_\w+)", path.read_text(), re.M)


def test_no_unexpected_empty_test_files():
    empty = {p.name for p in TESTS.glob("test_*.py") if not _test_functions(p)}
    unexpected = empty - KNOWN_EMPTY
    assert not unexpected, (
        f"these test files assert nothing and are not declared scaffolding: "
        f"{sorted(unexpected)}"
    )


def test_known_empty_stubs_are_still_empty():
    """When a stub is implemented, delete it from KNOWN_EMPTY. This fails loudly
    rather than letting the list rot into a lie about what is covered."""
    filled = {name for name in KNOWN_EMPTY
              if _test_functions(TESTS / name)}
    assert not filled, (
        f"{sorted(filled)} now contain real tests — remove them from "
        f"KNOWN_EMPTY in {pathlib.Path(__file__).name}"
    )
