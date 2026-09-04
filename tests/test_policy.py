"""
The gate. These are the tests you show on camera.

The claim being defended is "the LLM cannot move money", and it is defended
structurally: decide() is a pure function that receives thresholds as an
argument and has no client, no filesystem handle and no source of entropy.
test_policy_is_pure enforces that by monkeypatching the network and the
filesystem out from under it, rather than by asserting a comment.
"""
import builtins
import json
import pathlib
import socket

import pytest

from app.policy.action_policy import (
    RULE_AMOUNT_CAP,
    RULE_BELOW_ECONOMIC_FLOOR,
    RULE_COST_REVIEW_BAND,
    RULE_EVIDENCE_INSUFFICIENT,
    RULE_FABRICATED_EVIDENCE,
    RULE_HIGH_FRAUD_ACCEPT,
    decide,
)
from app.policy.evidence_policy import assess_evidence
from app.policy.thresholds import PolicyConfig, load_policy_config

CONFIG = PolicyConfig(
    representment_cost_inr=500.0,
    assumed_win_rate_if_legitimate=0.70,
    human_review_cost_inr=150.0,
    auto_action_amount_cap_inr=25_000.0,
    scenario="balanced",
    assumed_dispute_fraud_rate=0.50,
    global_threshold=0.91,
    source="test",
)

REQUIREMENTS = {
    "NON_RECEIPT": {"required": ["shipping_proof", "billing_proof"],
                    "optional": ["customer_communication"]},
    "FRAUD": {"required": ["customer_communication", "billing_proof"],
              "optional": ["access_activity_log"]},
}
COMPLETE = {"shipping_proof": ["doc_a"], "billing_proof": ["doc_b"]}

MEDIAN = 6_070.0     # p* = 0.882, band [0.847, 0.918]


def call(**kw):
    base = dict(config=CONFIG, p_fraud=0.05, amount_inr=MEDIAN,
                reason_code="NON_RECEIPT", evidence=COMPLETE,
                requirements=REQUIREMENTS, proposed_action="CONTEST")
    return decide(**{**base, **kw})


# --- the five named in the original scaffolding ---------------------------

def test_blocks_contest_when_evidence_missing():
    """LLM says CONTEST, delivery proof absent -> HUMAN_REVIEW."""
    d = call(evidence={"billing_proof": ["doc_b"]}, proposed_action="CONTEST")
    assert d.action == "HUMAN_REVIEW"
    assert d.rule == RULE_EVIDENCE_INSUFFICIENT
    assert d.proposal_rejected
    assert "shipping_proof" in d.missing_required_evidence


def test_blocks_hallucinated_evidence():
    """LLM cites evidence the set does not contain -> proposal REJECTED.

    Note the evidence set here is COMPLETE: the contest would otherwise be
    allowed. Fabrication has to be caught on its own terms, not as a side
    effect of insufficiency."""
    d = call(cited_evidence=["shipping_proof", "access_activity_log"])
    assert d.rule == RULE_FABRICATED_EVIDENCE
    assert d.proposal_rejected
    assert d.fabricated_evidence == ("access_activity_log",)
    assert d.action == "HUMAN_REVIEW"


def test_accepts_when_fraud_probability_high():
    """p above the band -> ACCEPT, not CONTEST. Contesting genuine fraud burns
    cost to lose."""
    d = call(p_fraud=0.97, proposed_action="CONTEST")
    assert d.action == "ACCEPT"
    assert d.rule == RULE_HIGH_FRAUD_ACCEPT
    assert d.proposal_rejected, "a CONTEST proposal must not survive here"


def test_amount_cap_forces_review():
    """Above the cap -> HUMAN_REVIEW regardless of everything else."""
    d = call(p_fraud=0.01, amount_inr=40_000.0, evidence=COMPLETE)
    assert d.action == "HUMAN_REVIEW"
    assert d.rule == RULE_AMOUNT_CAP


def test_policy_is_pure(monkeypatch):
    """decide() must make no network call and no file read.

    Enforced by removing both capabilities, not by inspection."""
    def no_network(*a, **k):
        raise AssertionError("policy attempted a network call")

    def no_open(*a, **k):
        raise AssertionError("policy attempted a file read")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(builtins, "open", no_open)
    monkeypatch.setattr(pathlib.Path, "read_text", no_open)
    monkeypatch.setattr(pathlib.Path, "open", no_open)

    d = call()
    assert d.action == "CONTEST"


# --- determinism and totality ---------------------------------------------

def test_decide_is_deterministic():
    assert [call().action for _ in range(50)].count("CONTEST") == 50


def test_every_decision_names_the_rule_that_fired():
    """An unexplainable decision is not auditable, and an unauditable gate is
    decoration."""
    cases = [
        dict(p_fraud=0.01), dict(p_fraud=0.88), dict(p_fraud=0.99),
        dict(amount_inr=40_000.0), dict(amount_inr=400.0),
        dict(evidence={}), dict(cited_evidence=["proof_of_service"]),
        dict(reason_code="__NOT_A_REASON__"),
    ]
    for kw in cases:
        d = call(**kw)
        assert d.rule and d.rationale, kw
        assert d.action in {"CONTEST", "ACCEPT", "HUMAN_REVIEW"}


def test_unknown_reason_code_fails_closed():
    """A gate that fails open is not a gate."""
    d = call(reason_code="__NOT_A_REASON__")
    assert d.action == "HUMAN_REVIEW"
    assert d.rule == RULE_EVIDENCE_INSUFFICIENT


def test_rejects_impossible_inputs():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            call(p_fraud=bad)
    with pytest.raises(ValueError):
        call(amount_inr=-1.0)


def test_decide_is_keyword_only():
    """Positional args would let p_fraud and amount_inr swap silently -- both
    are floats, so nothing would catch it."""
    with pytest.raises(TypeError):
        decide(CONFIG, 0.5, 1000.0, "NON_RECEIPT", {}, REQUIREMENTS)


# --- the cost-derived band ------------------------------------------------

def test_review_band_is_amount_dependent():
    """The fix for the [0.595, 1.000] global band. A band that does not move
    with the amount cannot be cost-derived."""
    small = CONFIG.review_band(2_000.0)
    large = CONFIG.review_band(20_000.0)
    assert small != large
    assert large[0] > small[0], "a larger dispute is worth contesting at higher risk"
    for band in (small, large):
        assert band[1] - band[0] < 0.25, "band should be narrow, not the old 0.405"


def test_band_is_centred_on_the_indifference_point():
    for amount in (2_000.0, 6_070.0, 20_000.0):
        low, high = CONFIG.review_band(amount)
        assert low <= CONFIG.indifference_threshold(amount) <= high


def test_wider_human_review_cost_widens_the_band():
    """The band width is derived from what a human costs. If it does not
    respond to that input, it is not derived from it."""
    cheap = PolicyConfig(**{**CONFIG.__dict__, "human_review_cost_inr": 50.0})
    dear = PolicyConfig(**{**CONFIG.__dict__, "human_review_cost_inr": 400.0})
    w_cheap = cheap.review_band(MEDIAN)[1] - cheap.review_band(MEDIAN)[0]
    w_dear = dear.review_band(MEDIAN)[1] - dear.review_band(MEDIAN)[0]
    assert w_dear > w_cheap


def test_tiny_disputes_are_always_accepted():
    """w * 400 = 280 < 500, so no score can make contesting worthwhile."""
    for p in (0.0, 0.5, 0.99):
        d = call(p_fraud=p, amount_inr=400.0, evidence={})
        assert d.action == "ACCEPT"
        assert d.rule == RULE_BELOW_ECONOMIC_FLOOR


def test_economic_floor_is_checked_before_the_evidence_gate():
    """Escalating a dispute we would accept anyway spends analyst time to learn
    nothing."""
    d = call(p_fraud=0.02, amount_inr=400.0, evidence={})
    assert d.action == "ACCEPT", "should not pay for a human on a futile dispute"


def test_fabrication_is_checked_before_sufficiency():
    """Ordering matters: with sufficiency first, a fabricated citation would
    slip through whenever the real evidence happened to be complete."""
    d = call(evidence={}, cited_evidence=["proof_of_service"])
    assert d.rule == RULE_FABRICATED_EVIDENCE


def test_amount_cap_beats_everything():
    d = call(p_fraud=0.99, amount_inr=99_000.0, evidence={},
             cited_evidence=["proof_of_service"])
    assert d.rule == RULE_AMOUNT_CAP


# --- evidence gate in isolation -------------------------------------------

def test_empty_evidence_list_counts_as_missing():
    """An empty list is not evidence. Treating it as present is exactly how an
    unsubstantiated contest gets through."""
    a = assess_evidence("NON_RECEIPT", {"shipping_proof": [], "billing_proof": ["d"]},
                        REQUIREMENTS)
    assert not a.sufficient
    assert a.missing_required == ("shipping_proof",)


def test_optional_evidence_is_not_required():
    a = assess_evidence("NON_RECEIPT", COMPLETE, REQUIREMENTS)
    assert a.sufficient


def test_citing_evidence_that_is_on_file_is_not_fabrication():
    a = assess_evidence("NON_RECEIPT", COMPLETE, REQUIREMENTS,
                        cited_evidence=["shipping_proof", "billing_proof"])
    assert not a.has_fabrication


# --- the shipped config ---------------------------------------------------

def test_shipped_config_loads_and_is_frozen():
    cfg = load_policy_config()
    assert cfg.representment_cost_inr > 0
    assert 0 < cfg.assumed_win_rate_if_legitimate <= 1
    with pytest.raises(Exception):
        cfg.representment_cost_inr = 1.0     # frozen dataclass


# --- the override chain ---------------------------------------------------

def test_every_decision_reports_the_whole_chain():
    """Returning only the rule that fired gives a verdict. The chain gives the
    reasoning, which is what an auditor needs."""
    from app.policy.action_policy import RULE_ORDER

    d = call()
    assert [e.rule for e in d.evaluated] == list(RULE_ORDER)
    assert sum(e.outcome == "fired" for e in d.evaluated) == 1


def test_chain_marks_rules_after_the_firing_one_as_not_reached():
    from app.policy.action_policy import RULE_ORDER

    d = call(amount_inr=90_000.0)          # rule 1 fires
    assert d.evaluated[0].outcome == "fired"
    assert all(e.outcome == "not_reached" for e in d.evaluated[1:])
    assert len(d.evaluated) == len(RULE_ORDER)


def test_chain_records_why_each_passed_rule_did_not_fire():
    d = call()
    for e in d.evaluated:
        if e.outcome == "passed":
            assert e.detail, f"{e.rule} passed without saying why"


def test_fired_rule_in_chain_matches_the_decision():
    for kw in (dict(), dict(amount_inr=90_000.0), dict(evidence={}),
               dict(p_fraud=0.99), dict(amount_inr=400.0), dict(p_fraud=0.88),
               dict(cited_evidence=["proof_of_service"])):
        d = call(**kw)
        fired = [e.rule for e in d.evaluated if e.outcome == "fired"]
        assert fired == [d.rule], f"{kw}: chain says {fired}, decision says {d.rule}"


def test_trace_does_not_change_the_decision():
    """The trace is recorded for its side effect only. If adding it altered
    control flow, these would differ from the documented override order."""
    assert call(amount_inr=90_000.0, evidence={},
                cited_evidence=["proof_of_service"]).rule == RULE_AMOUNT_CAP
    assert call(evidence={}, cited_evidence=["proof_of_service"]).rule == \
        RULE_FABRICATED_EVIDENCE
