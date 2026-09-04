"""The failure scenarios recorded for the demo, plus secret handling.

Every one of these asserts that a failure DEGRADES rather than propagates. The
claim being defended is that the LLM is an enhancement to a system that already
works, not a dependency of it.
"""
import json
import logging

import pytest

from app.models.schemas import DisputeProposal
from app.policy.action_policy import (
    RULE_EVIDENCE_INSUFFICIENT,
    RULE_FABRICATED_EVIDENCE,
    decide,
)
from app.policy.thresholds import PolicyConfig
from app.services import llm_service as L
from app.services.audit_service import build_record

CONFIG = PolicyConfig(
    representment_cost_inr=500.0, assumed_win_rate_if_legitimate=0.70,
    human_review_cost_inr=150.0, auto_action_amount_cap_inr=25_000.0,
    scenario="balanced", assumed_dispute_fraud_rate=0.50,
    global_threshold=0.91, source="test")

REQUIREMENTS = {
    "NON_RECEIPT": {"required": ["shipping_proof", "billing_proof"],
                    "optional": ["customer_communication"]},
}
COMPLETE = {"shipping_proof": ["doc_a"], "billing_proof": ["doc_b"]}
FAKE_KEY = "sk-test-DO-NOT-LOG-abcdef1234567890"


@pytest.fixture
def no_key(monkeypatch):
    for var in L.API_KEY_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)


def propose(**kw):
    base = dict(reason_code="NON_RECEIPT", amount_inr=6_070.0,
                evidence=COMPLETE, requirements=REQUIREMENTS)
    return L.propose(**{**base, **kw})


# --- demo_01: missing evidence -> HUMAN_REVIEW ----------------------------

def test_missing_evidence_routes_to_human_review(no_key):
    proposal, source = propose(evidence={"billing_proof": ["doc_b"]})
    assert source == "template"
    assert proposal.decision == "REVIEW"

    d = decide(config=CONFIG, p_fraud=0.05, amount_inr=6_070.0,
               reason_code="NON_RECEIPT", evidence={"billing_proof": ["doc_b"]},
               requirements=REQUIREMENTS, proposed_action=proposal.decision,
               cited_evidence=proposal.cited_evidence)
    assert d.action == "HUMAN_REVIEW"
    assert d.rule == RULE_EVIDENCE_INSUFFICIENT


# --- demo_02: hallucinated 3DS -> proposal REJECTED -----------------------

def test_hallucinated_evidence_is_rejected_by_the_gate():
    """The evidence set is COMPLETE, so the contest would otherwise be allowed.
    The LLM invents an authentication record that was never collected."""
    hallucinated = DisputeProposal(
        decision="CONTEST", evidence_status="SUFFICIENT",
        cited_evidence=["shipping_proof", "access_activity_log"],
        reasoning_summary="3-D Secure authentication confirms the cardholder.",
        draft_representment="The transaction was authenticated via 3-D Secure.")

    d = decide(config=CONFIG, p_fraud=0.05, amount_inr=6_070.0,
               reason_code="NON_RECEIPT", evidence=COMPLETE,
               requirements=REQUIREMENTS,
               proposed_action=hallucinated.decision,
               cited_evidence=hallucinated.cited_evidence)

    assert d.rule == RULE_FABRICATED_EVIDENCE
    assert d.proposal_rejected
    assert d.fabricated_evidence == ("access_activity_log",)
    assert d.action == "HUMAN_REVIEW", "a fabricating proposal must never CONTEST"


def test_llm_service_does_not_silently_repair_fabrication(with_key):
    """If this module scrubbed citations, the gate would never fire and there
    would be no audit record showing the model tried."""
    def fabricating(prompt, settings, api_key):
        return json.dumps({
            "decision": "CONTEST", "evidence_status": "SUFFICIENT",
            "missing_evidence": [], "cited_evidence": ["access_activity_log"],
            "reasoning_summary": "Authenticated.", "draft_representment": "x"})

    proposal, source = propose(call_provider=fabricating)
    assert source == "llm"
    assert "access_activity_log" in proposal.cited_evidence


# --- demo_03: malformed JSON -> template fallback -------------------------

def test_malformed_json_falls_back_to_template(with_key):
    def garbage(prompt, settings, api_key):
        return "I'm sorry, I can't help with that."

    proposal, source = propose(call_provider=garbage)
    assert source == "template"
    assert proposal.decision == "CONTEST"


def test_schema_violation_falls_back_to_template(with_key):
    def wrong_shape(prompt, settings, api_key):
        return json.dumps({"decision": "WIRE_THE_MONEY", "evidence_status": "OK"})

    assert propose(call_provider=wrong_shape)[1] == "template"


def test_retries_once_then_gives_up(with_key):
    calls = []

    def flaky(prompt, settings, api_key):
        calls.append(1)
        if len(calls) == 1:
            return "not json"
        return json.dumps({
            "decision": "CONTEST", "evidence_status": "SUFFICIENT",
            "missing_evidence": [], "cited_evidence": ["shipping_proof"],
            "reasoning_summary": "ok", "draft_representment": ""})

    proposal, source = propose(call_provider=flaky)
    assert source == "llm" and len(calls) == 2


def test_never_exceeds_the_retry_budget(with_key):
    calls = []

    def always_bad(prompt, settings, api_key):
        calls.append(1)
        return "still not json"

    assert propose(call_provider=always_bad)[1] == "template"
    assert len(calls) == L.MAX_ATTEMPTS


# --- demo_04 (partial): provider timeout -> template, no exception --------

def test_provider_timeout_degrades_to_template(with_key):
    def timeout(prompt, settings, api_key):
        raise TimeoutError("upstream timed out")

    proposal, source = propose(call_provider=timeout)
    assert source == "template"
    assert proposal.decision == "CONTEST", "the pipeline must still produce a result"


def test_no_key_means_no_network_call(no_key):
    def must_not_be_called(prompt, settings, api_key):
        raise AssertionError("attempted a provider call with no credential")

    assert propose(call_provider=must_not_be_called)[1] == "template"
    assert L.is_enabled() is False


# --- secret handling ------------------------------------------------------

def test_key_never_appears_in_the_prompt(with_key):
    seen = {}

    def capture(prompt, settings, api_key):
        seen["prompt"] = prompt
        return "not json"

    propose(call_provider=capture)
    assert FAKE_KEY not in seen["prompt"]


def test_prompt_carries_no_document_ids_and_no_fraud_score(with_key):
    prompt = L.build_user_prompt(reason_code="NON_RECEIPT", amount_inr=6_070.0,
                                 evidence=COMPLETE, requirements=REQUIREMENTS)
    assert "doc_a" not in prompt and "doc_b" not in prompt
    assert "p_fraud" not in prompt


def test_key_never_reaches_the_logs(with_key, caplog):
    def leaky(prompt, settings, api_key):
        raise RuntimeError(f"auth failed for key {api_key}")

    with caplog.at_level(logging.DEBUG):
        propose(call_provider=leaky)
    assert FAKE_KEY not in caplog.text


def test_scrub_redacts_the_credential(with_key):
    assert FAKE_KEY not in L._scrub(f"Bearer {FAKE_KEY} rejected")
    assert "***REDACTED***" in L._scrub(f"Bearer {FAKE_KEY} rejected")


def test_key_never_reaches_the_audit_record(with_key):
    d = decide(config=CONFIG, p_fraud=0.05, amount_inr=6_070.0,
               reason_code="NON_RECEIPT", evidence=COMPLETE,
               requirements=REQUIREMENTS, proposed_action="CONTEST")
    record = build_record(dispute_id="disp_x", decision=d, config=CONFIG,
                          proposal={"reasoning_summary": "ok"},
                          proposal_source="llm")
    assert FAKE_KEY not in json.dumps(record)


def test_is_enabled_does_not_reveal_the_key(with_key):
    assert L.is_enabled() is True
    assert FAKE_KEY not in str(L.load_settings())


def test_llm_api_key_takes_precedence_over_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback")
    monkeypatch.setenv("LLM_API_KEY", "sk-primary")
    assert L._api_key() == "sk-primary"


def test_openai_api_key_is_accepted_as_a_fallback(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback")
    assert L._api_key() == "sk-fallback"


def test_blank_key_is_treated_as_absent(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "   ")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert L._api_key() is None and L.is_enabled() is False


# --- provider portability -------------------------------------------------

def test_json_mode_is_dropped_when_the_provider_rejects_it(with_key):
    """JSON mode is the one call parameter that varies across OpenAI-compatible
    providers. A 400 naming response_format means 'drop that parameter', not
    'give up' — the system prompt already demands a bare JSON object."""
    seen = []

    class Rejected(Exception):
        status_code = 400
        code = "invalid_request_error"

        def __str__(self):
            return "'response_format' of type 'json_object' is not supported"

    def picky(prompt, settings, api_key, json_mode=True):
        seen.append(json_mode)
        if json_mode:
            raise Rejected()
        return json.dumps({
            "decision": "CONTEST", "evidence_status": "SUFFICIENT",
            "missing_evidence": [], "cited_evidence": ["shipping_proof"],
            "reasoning_summary": "ok", "draft_representment": ""})

    proposal, source = propose(call_provider=picky)
    assert seen == [True, False], "should retry once without JSON mode"
    assert source == "llm"


def test_json_mode_is_kept_when_the_failure_is_unrelated(with_key):
    """A billing 429 or an auth 401 must not be mistaken for a parameter
    problem — dropping JSON mode would not help and would muddy the diagnosis."""
    seen = []

    class Quota(Exception):
        status_code = 429
        code = "credit_balance_exhausted"

    def broke(prompt, settings, api_key, json_mode=True):
        seen.append(json_mode)
        raise Quota()

    assert propose(call_provider=broke)[1] == "template"
    assert seen == [True, True], "JSON mode should not be dropped for a 429"


def test_three_argument_providers_still_work(with_key):
    """Existing call sites and tests inject a 3-argument stub. Adding a fourth
    parameter must not break them."""
    def legacy(prompt, settings, api_key):
        return json.dumps({
            "decision": "CONTEST", "evidence_status": "SUFFICIENT",
            "missing_evidence": [], "cited_evidence": ["shipping_proof"],
            "reasoning_summary": "ok", "draft_representment": ""})

    assert propose(call_provider=legacy)[1] == "llm"
