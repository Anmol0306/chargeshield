"""API contract tests.

The property being defended: no endpoint returns an ungated proposal. A caller
must never be able to obtain the LLM's suggestion without also obtaining the
action the policy engine reached.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import DisputeProposal


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def isolate_audit(tmp_path, monkeypatch):
    """Never append to the real audit log from a test."""
    from app.services import audit_service
    monkeypatch.setattr(audit_service, "DEFAULT_PATH", tmp_path / "audit.jsonl")


COMPLETE = {"shipping_proof": ["doc_a"], "billing_proof": ["doc_b"]}


def base_dispute(**kw):
    return {"dispute_id": "disp_test", "reason_code": "NON_RECEIPT",
            "amount_inr": 6_070.0, "p_fraud": 0.05,
            "evidence": COMPLETE, **kw}


# --- health ---------------------------------------------------------------

def test_health_reports_model_and_credential_presence(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "loaded" in body["model"]
    assert isinstance(body["llm_credential_present"], bool)


def test_health_never_reveals_any_part_of_the_credential(client, monkeypatch):
    secret = "sk-live-SHOULD-NEVER-APPEAR-9999"
    monkeypatch.setenv("LLM_API_KEY", secret)
    body = client.get("/health").text
    assert secret not in body
    for fragment in (secret[:8], secret[-8:]):
        assert fragment not in body


# --- /score ---------------------------------------------------------------

def test_score_accepts_a_partial_feature_vector(client):
    r = client.post("/score", json={"features": {"TransactionAmt": 5_000.0,
                                                 "ProductCD": "W"}})
    assert r.status_code == 200
    b = r.json()
    assert 0.0 <= b["p_fraud"] <= 1.0
    assert b["features_supplied"] == 2
    assert b["features_expected"] == 392


def test_score_reports_unrecognised_fields_rather_than_ignoring_them(client):
    b = client.post("/score", json={"features": {"TransactionAmt": 100.0,
                                                 "not_a_feature": 1}}).json()
    assert b["unrecognised_fields"] == ["not_a_feature"]


def test_score_bands_are_amount_dependent(client):
    small = client.post("/score", json={"features": {"TransactionAmt": 20.0},
                                        "amount_inr": 2_000.0}).json()
    large = client.post("/score", json={"features": {"TransactionAmt": 20.0},
                                        "amount_inr": 20_000.0}).json()
    assert small["indifference_threshold"] < large["indifference_threshold"]


def test_score_rejects_unknown_top_level_fields(client):
    assert client.post("/score", json={"features": {}, "wat": 1}).status_code == 422


# --- /disputes/analyze ----------------------------------------------------

def test_analyze_returns_a_decision_not_a_bare_proposal(client):
    b = client.post("/disputes/analyze", json=base_dispute()).json()
    assert b["action"] in {"CONTEST", "ACCEPT", "HUMAN_REVIEW"}
    assert b["rule"] and b["rationale"]
    assert b["proposal_source"] in {"llm", "template"}


def test_analyze_falls_back_to_template_without_a_credential(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    b = client.post("/disputes/analyze", json=base_dispute()).json()
    assert b["proposal_source"] == "template"
    assert b["action"] == "CONTEST"


def test_analyze_blocks_when_evidence_is_missing(client):
    b = client.post("/disputes/analyze",
                    json=base_dispute(evidence={"billing_proof": ["b"]})).json()
    assert b["action"] == "HUMAN_REVIEW"
    assert b["rule"] == "required_evidence_missing"
    assert "shipping_proof" in b["missing_required_evidence"]


def test_analyze_requires_p_fraud_or_features(client):
    payload = base_dispute()
    payload.pop("p_fraud")
    assert client.post("/disputes/analyze", json=payload).status_code == 422


def test_analyze_can_derive_p_fraud_from_features(client):
    payload = base_dispute(features={"TransactionAmt": 68.0, "ProductCD": "W"})
    payload.pop("p_fraud")
    r = client.post("/disputes/analyze", json=payload)
    assert r.status_code == 200
    assert r.json()["action"] in {"CONTEST", "ACCEPT", "HUMAN_REVIEW"}


def test_amount_cap_is_enforced_through_the_api(client):
    b = client.post("/disputes/analyze",
                    json=base_dispute(amount_inr=90_000.0)).json()
    assert b["action"] == "HUMAN_REVIEW"
    assert b["rule"] == "amount_cap_exceeded"


# --- /disputes/validate: the demo endpoint --------------------------------

def test_validate_blocks_a_fabricated_proposal(client):
    """THE demo. Evidence is complete, so the contest would otherwise be
    allowed; the proposal invents an authentication record."""
    proposal = DisputeProposal(
        decision="CONTEST", evidence_status="SUFFICIENT",
        cited_evidence=["shipping_proof", "access_activity_log"],
        reasoning_summary="3-D Secure authentication confirms the cardholder.",
        draft_representment="The transaction was authenticated via 3-D Secure.")
    b = client.post("/disputes/validate", json={
        "dispute_id": "disp_demo", "reason_code": "NON_RECEIPT",
        "amount_inr": 6_070.0, "p_fraud": 0.05, "evidence": COMPLETE,
        "proposal": proposal.model_dump()}).json()

    assert b["rule"] == "proposal_cited_evidence_not_on_file"
    assert b["proposal_honoured"] is False
    assert b["fabricated_evidence"] == ["access_activity_log"]
    assert b["action"] == "HUMAN_REVIEW"


def test_validate_rejects_a_malformed_proposal_at_the_schema(client):
    r = client.post("/disputes/validate", json={
        "dispute_id": "d", "reason_code": "NON_RECEIPT", "amount_inr": 100.0,
        "p_fraud": 0.1, "evidence": {},
        "proposal": {"decision": "WIRE_THE_MONEY", "evidence_status": "SUFFICIENT",
                     "reasoning_summary": "x"}})
    assert r.status_code == 422


def test_validate_honours_a_clean_proposal(client):
    proposal = DisputeProposal(
        decision="CONTEST", evidence_status="SUFFICIENT",
        cited_evidence=["shipping_proof", "billing_proof"],
        reasoning_summary="All required evidence is on file.")
    b = client.post("/disputes/validate", json={
        "dispute_id": "disp_ok", "reason_code": "NON_RECEIPT",
        "amount_inr": 6_070.0, "p_fraud": 0.05, "evidence": COMPLETE,
        "proposal": proposal.model_dump()}).json()
    assert b["action"] == "CONTEST" and b["proposal_honoured"] is True


# --- audit ----------------------------------------------------------------

def test_every_decision_is_auditable_through_the_api(client):
    client.post("/disputes/analyze", json=base_dispute(dispute_id="disp_audit"))
    b = client.get("/audit/disp_audit").json()
    assert b["dispute_id"] == "disp_audit"
    assert b["rule"] and b["cost_assumptions"]["scenario"]


def test_audit_404s_for_an_unknown_dispute(client):
    assert client.get("/audit/disp_does_not_exist").status_code == 404


# --- batch ----------------------------------------------------------------

def test_batch_run_prices_four_policies(client):
    r = client.post("/batch/run?limit=200")
    if r.status_code == 503:
        pytest.skip("run `make link` first")
    b = r.json()
    assert b["n_disputes"] == 200
    for k in ("defend_none", "defend_all", "static_amount_rule", "chargeshield"):
        assert k in b["policy_comparison"]


def test_batch_run_does_not_write_audit_by_default(client, tmp_path):
    from app.services.audit_service import AuditLog
    log = AuditLog(tmp_path / "audit.jsonl")
    r = client.post("/batch/run?limit=10")
    if r.status_code == 503:
        pytest.skip("run `make link` first")
    assert log.read_all() == []
