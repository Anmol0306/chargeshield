"""The static build must stay in step with the page and with the engine.

docs/index.html is generated: its decisions come from the real
app.policy.action_policy.decide(). Two things can rot — the scenario list in
the page drifting from the one in the builder, and the committed build going
stale against the engine. Both fail here rather than showing a viewer an error.
"""
import json
import pathlib
import re

import pytest

from scripts.build_static import SCENARIOS, decision_payload

PAGE = pathlib.Path("frontend/index.html")
STATIC = pathlib.Path("docs/index.html")


def page_scenario_ids() -> set[str]:
    return set(re.findall(r'dispute_id:"([^"]+)"', PAGE.read_text()))


def test_every_page_scenario_has_a_precomputed_decision():
    """A page scenario with no decision shows an error to a viewer instead of
    a determination."""
    missing = page_scenario_ids() - set(d["dispute_id"] for d in SCENARIOS)
    assert not missing, f"scenarios in the page with no builder entry: {missing}"


def test_builder_has_no_scenarios_the_page_never_shows():
    orphans = set(d["dispute_id"] for d in SCENARIOS) - page_scenario_ids()
    assert not orphans, f"builder entries no page scenario uses: {orphans}"


def test_one_source_file_serves_both_modes():
    """The static page is generated FROM frontend/index.html. If someone forks
    it into a second file, the live and static versions drift."""
    src = PAGE.read_text()
    assert "window.__STATIC__" in src
    assert 'fetch("/disputes/validate"' in src, "live path must still exist"
    assert 'fetch("/metrics")' in src


def test_decisions_come_from_the_real_engine():
    """Not hand-written, and not a JavaScript reimplementation of the rules."""
    payload = decision_payload()
    assert len(payload) == len(SCENARIOS)
    for d in payload.values():
        assert d["evaluated"], "no rule chain — did not come from decide()"
        assert sum(e["outcome"] == "fired" for e in d["evaluated"]) == 1


def test_scenarios_exercise_distinct_rules():
    """A scenario set where everything fires the same rule demonstrates nothing."""
    rules = {d["rule"] for d in decision_payload().values()}
    assert len(rules) >= 4, f"only {len(rules)} distinct rules covered: {rules}"


def test_fabrication_scenario_is_blocked():
    d = decision_payload()["rec_fabricated"]
    assert d["action"] == "HUMAN_REVIEW"
    assert d["rule"] == "proposal_cited_evidence_not_on_file"
    assert d["fabricated_evidence"] == ["access_activity_log"]


# --- the committed build -------------------------------------------------

@pytest.fixture(scope="module")
def built():
    if not STATIC.exists():
        pytest.skip("run `make static` first")
    return STATIC.read_text()


def test_committed_build_matches_the_current_engine(built):
    """If the policy changed and the static page was not rebuilt, the hosted
    site would show decisions the code no longer produces."""
    m = re.search(r"window\.__STATIC__ = (\{.*?\});</script>", built, re.S)
    assert m, "no injected payload found"
    embedded = json.loads(m.group(1))["decisions"]
    current = decision_payload()
    for did, cur in current.items():
        assert embedded[did]["action"] == cur["action"], f"{did} stale — run make static"
        assert embedded[did]["rule"] == cur["rule"], f"{did} stale — run make static"


def test_static_page_carries_its_own_data(built):
    assert "window.__STATIC__" in built
    assert "policy engine" in built


def test_static_page_states_the_provenance_of_every_number(built):
    """The page must say where its figures came from. Asserted on the SUBSTANCE
    rather than on a literal phrase, so the wording can be improved without
    breaking the guarantee — an earlier version of this test pinned the string
    'Static preview' and failed the moment the copy was reworded."""
    assert "Static preview" in built, "must carry the agreed label"
    assert "precomputed by the ChargeShield Python engine" in built
    assert "makes no API calls" in built
    assert "at build time" in built, "must say the determinations were precomputed"
    assert "not written by hand" in built, "must rule out hand-authored decisions"
    assert "not reimplemented in JavaScript" in built, "must rule out a JS port"
    assert "/metrics" in built, "must name where the figures come from"


def test_static_page_never_implies_the_gate_ran_live(built):
    """A viewer must not be able to read these determinations as live."""
    banner = built[built.index('class="banner"'):built.index("</div>",
                                                             built.index('class="banner"'))]
    for claim in ("live decision", "real time", "real-time", "executed on click"):
        assert claim not in banner.lower()
    assert "make api" in banner, "must tell the reader how to run it live"
