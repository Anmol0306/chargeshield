"""
POST /disputes/analyze  -> proposal + policy decision + audit record
POST /disputes/draft    -> representment draft (never submits)
POST /disputes/validate -> policy check on a supplied proposal, no LLM
GET  /audit/{id}        -> the record for a decision

Every route returns the DECISION and the RULE, never a bare proposal. An
endpoint that returned the LLM's suggestion on its own would let a caller act
on ungated output, which would defeat the entire design.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    DisputeAnalysisRequest,
    PolicyDecisionResponse,
    ValidateRequest,
)
from app.policy.action_policy import decide
from app.policy.thresholds import load_policy_config
from app.services import llm_service
from app.services.audit_service import AuditLog, build_record
from app.services.risk_service import ModelUnavailable, score
from app.services.template_response import build_template_proposal

router = APIRouter(tags=["disputes"])

REQUIREMENTS_PATH = Path("evidence/requirements.json")


@lru_cache(maxsize=1)
def requirements() -> dict:
    return {k: v for k, v in json.loads(REQUIREMENTS_PATH.read_text()).items()
            if not k.startswith("_")}


def _resolve_p_fraud(request: DisputeAnalysisRequest) -> float:
    if request.p_fraud is not None:
        return request.p_fraud
    if not request.features:
        raise HTTPException(
            status_code=422,
            detail="supply either p_fraud or features to score from")
    try:
        return score(request.features)["p_fraud"]
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _respond(dispute_id: str, decision, proposal, source: str) -> PolicyDecisionResponse:
    AuditLog().append(build_record(
        dispute_id=dispute_id, decision=decision,
        config=load_policy_config(),
        proposal=proposal.model_dump() if proposal else None,
        proposal_source=source,
    ))
    return PolicyDecisionResponse(
        dispute_id=dispute_id,
        action=decision.action,
        rule=decision.rule,
        rationale=decision.rationale,
        proposal_honoured=decision.proposal_honoured,
        proposal_source=source,
        proposed_action=decision.proposed_action,
        missing_required_evidence=list(decision.missing_required_evidence),
        fabricated_evidence=list(decision.fabricated_evidence),
        indifference_threshold=decision.indifference_threshold,
        review_band=decision.review_band,
        proposal=proposal,
    )


@router.post("/disputes/analyze", response_model=PolicyDecisionResponse)
def analyze(request: DisputeAnalysisRequest) -> PolicyDecisionResponse:
    p_fraud = _resolve_p_fraud(request)

    if request.use_llm:
        proposal, source = llm_service.propose(
            reason_code=request.reason_code, amount_inr=request.amount_inr,
            evidence=request.evidence, requirements=requirements(),
            dispute_id=request.dispute_id)
    else:
        proposal, source = build_template_proposal(
            reason_code=request.reason_code, evidence=request.evidence,
            requirements=requirements(), amount_inr=request.amount_inr,
            dispute_id=request.dispute_id), "template"

    decision = decide(
        config=load_policy_config(), p_fraud=p_fraud,
        amount_inr=request.amount_inr, reason_code=request.reason_code,
        evidence=request.evidence, requirements=requirements(),
        proposed_action=proposal.decision,
        cited_evidence=proposal.cited_evidence)

    return _respond(request.dispute_id, decision, proposal, source)


@router.post("/disputes/draft", response_model=PolicyDecisionResponse)
def draft(request: DisputeAnalysisRequest) -> PolicyDecisionResponse:
    """Draft text only. Still gated, and it never submits anything anywhere.

    The draft is returned inside the decision so a caller cannot receive
    representment text without also receiving the action the policy engine
    reached — including, where relevant, that it must not be filed.
    """
    return analyze(request)


@router.post("/disputes/validate", response_model=PolicyDecisionResponse)
def validate(request: ValidateRequest) -> PolicyDecisionResponse:
    """Gate a caller-supplied proposal. No LLM. This is the demo endpoint:
    hand it a proposal citing evidence that is not on file and watch the rule
    fire."""
    decision = decide(
        config=load_policy_config(), p_fraud=request.p_fraud,
        amount_inr=request.amount_inr, reason_code=request.reason_code,
        evidence=request.evidence, requirements=requirements(),
        proposed_action=request.proposal.decision,
        cited_evidence=request.proposal.cited_evidence)
    return _respond(request.dispute_id, decision, request.proposal, "llm")


@router.get("/audit/{dispute_id}")
def audit(dispute_id: str) -> dict:
    record = AuditLog().find(dispute_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no record for {dispute_id}")
    return record
