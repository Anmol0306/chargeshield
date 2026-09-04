"""Audit records must be able to reconstruct a decision without the code.

"The policy engine overrode the LLM" is only defensible if you can produce the
record. These tests check the record is complete enough to be that evidence.
"""
import json

import pytest

from app.policy.action_policy import decide
from app.policy.thresholds import PolicyConfig
from app.services.audit_service import AuditLog, build_record
from app.services.batch_runner import run_batch

CONFIG = PolicyConfig(
    representment_cost_inr=500.0, assumed_win_rate_if_legitimate=0.70,
    human_review_cost_inr=150.0, auto_action_amount_cap_inr=25_000.0,
    scenario="balanced", assumed_dispute_fraud_rate=0.50,
    global_threshold=0.91, source="test",
)
REQUIREMENTS = {"NON_RECEIPT": {"required": ["shipping_proof"], "optional": []}}


def make_decision(**kw):
    base = dict(config=CONFIG, p_fraud=0.05, amount_inr=6_070.0,
                reason_code="NON_RECEIPT", evidence={"shipping_proof": ["d"]},
                requirements=REQUIREMENTS)
    return decide(**{**base, **kw})


def test_record_is_json_serialisable():
    """Tuples, dataclasses and enums must all be converted. Assert the
    round-trip rather than only the absence of an exception -- a serialiser
    that silently dropped a field would still not raise."""
    r = build_record(dispute_id="disp_x", decision=make_decision(), config=CONFIG)
    restored = json.loads(json.dumps(r))
    assert restored == r, "record did not survive a JSON round-trip unchanged"


def test_record_captures_the_bands_in_force():
    """Thresholds move when config/costs.yaml moves. A record without them
    becomes unexplainable the moment the config is edited."""
    r = build_record(dispute_id="disp_x", decision=make_decision(), config=CONFIG)
    b = r["bands_in_force"]
    assert b["review_band"] and b["indifference_threshold"] is not None
    assert b["amount_cap_inr"] == CONFIG.auto_action_amount_cap_inr


def test_record_fingerprints_the_cost_assumptions():
    r = build_record(dispute_id="disp_x", decision=make_decision(), config=CONFIG)
    c = r["cost_assumptions"]
    assert c["representment_cost_inr"] == 500.0
    assert c["assumed_win_rate_if_legitimate"] == 0.70
    assert c["scenario"] == "balanced"


def test_record_names_the_rule_and_the_reason():
    r = build_record(dispute_id="disp_x",
                     decision=make_decision(evidence={}), config=CONFIG)
    assert r["rule"] == "required_evidence_missing"
    assert r["rationale"]
    assert r["proposal_honoured"] is False


def test_log_is_append_only_and_survives_reopen(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(build_record(dispute_id="a", decision=make_decision(), config=CONFIG))
    log.append(build_record(dispute_id="b", decision=make_decision(), config=CONFIG))
    assert [r["dispute_id"] for r in AuditLog(tmp_path / "audit.jsonl").read_all()] == ["a", "b"]


def test_find_returns_the_latest_record_for_a_dispute(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(build_record(dispute_id="a", decision=make_decision(), config=CONFIG))
    log.append(build_record(dispute_id="a",
                            decision=make_decision(evidence={}), config=CONFIG))
    assert log.find("a")["rule"] == "required_evidence_missing"
    assert log.find("nope") is None


def test_every_batch_decision_produces_exactly_one_record(tmp_path):
    disputes = [{
        "id": f"disp_{i}", "reason_code": "NON_RECEIPT",
        "evidence": {"shipping_proof": ["d"]} if i % 2 else {},
        "_chargeshield": {"p_fraud_calibrated": 0.1 * (i % 9),
                          "amount_inr": 6_070.0, "anchor_is_fraud": i % 3 == 0,
                          "evidence_complete": bool(i % 2)},
    } for i in range(30)]
    log = AuditLog(tmp_path / "audit.jsonl")
    results = run_batch(disputes, CONFIG, REQUIREMENTS, log)
    assert results["n_disputes"] == 30
    assert len(log.read_all()) == 30
    assert sum(results["actions"].values()) == 30


def test_wasted_representment_uses_real_labels(tmp_path):
    """Two contested disputes, one genuinely fraudulent -> 50% wasted."""
    disputes = [{
        "id": f"disp_{i}", "reason_code": "NON_RECEIPT",
        "evidence": {"shipping_proof": ["d"]},
        "_chargeshield": {"p_fraud_calibrated": 0.01, "amount_inr": 6_070.0,
                          "anchor_is_fraud": i == 0, "evidence_complete": True},
    } for i in range(2)]
    r = run_batch(disputes, CONFIG, REQUIREMENTS)["wasted_representment"]
    assert r["contested"] == 2
    assert r["contested_that_were_real_fraud"] == 1
    assert r["wasted_rate"] == pytest.approx(0.5)


def test_batch_reports_the_evidence_complete_subset():
    """The review rate is an artefact of the evidence dial, so the model-driven
    slice must be reported separately or the headline misleads."""
    disputes = [{
        "id": f"disp_{i}", "reason_code": "NON_RECEIPT",
        "evidence": {"shipping_proof": ["d"]} if i < 3 else {},
        "_chargeshield": {"p_fraud_calibrated": 0.01, "amount_inr": 6_070.0,
                          "anchor_is_fraud": False, "evidence_complete": i < 3},
    } for i in range(10)]
    r = run_batch(disputes, CONFIG, REQUIREMENTS)
    assert r["evidence_complete_subset"]["n"] == 3
    assert r["evidence_complete_subset"]["actions"]["CONTEST"] == 3
