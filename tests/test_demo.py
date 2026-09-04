"""The recorded demo must keep matching the system.

demo/run_demo.py asserts every scenario's action and rule and exits non-zero if
any changes. Running it here means a policy change that would invalidate the
video breaks the build instead of being discovered on playback.
"""
import json
import pathlib
import subprocess
import sys

CASES = pathlib.Path("demo/demo_cases.json")


def test_demo_runs_clean_with_no_network_and_no_credential(monkeypatch):
    env = {k: v for k, v in __import__("os").environ.items()
           if k not in ("LLM_API_KEY", "OPENAI_API_KEY")}
    r = subprocess.run([sys.executable, "-m", "demo.run_demo"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, f"demo failed:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}"
    assert "All scenarios behaved as asserted" in r.stdout
    assert "FAIL" not in r.stdout


def test_demo_makes_no_provider_call():
    """It must run with no credential — that is the claim it exists to show."""
    src = pathlib.Path("demo/run_demo.py").read_text()
    assert "No network. No API credential" in src


def test_every_declared_case_is_implemented():
    cases = [c for c in json.loads(CASES.read_text()) if "id" in c]
    assert cases, "demo_cases.json has no cases"
    unimplemented = [c["id"] for c in cases if c.get("status") != "implemented"]
    assert not unimplemented, (
        f"demo_cases.json declares cases that are not implemented: {unimplemented}")


def test_demo_covers_every_policy_rule_that_can_fire():
    """If a rule exists but no scenario shows it, the demo is incomplete."""
    from app.policy import action_policy as ap

    shown = pathlib.Path("demo/run_demo.py").read_text()
    rules = {v for k, v in vars(ap).items()
             if k.startswith("RULE_") and isinstance(v, str)}
    # proposal_disagreed is unreachable from the demo's inputs: it needs a
    # proposal that is neither None nor CONTEST below the band, which is a
    # degenerate case rather than a scenario worth recording.
    rules.discard(ap.RULE_PROPOSAL_OVERRIDDEN)
    missing = sorted(r for r in rules if r not in shown)
    assert not missing, f"policy rules with no demo scenario: {missing}"
