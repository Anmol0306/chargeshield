"""
Pydantic contracts.

DisputeProposal is the LLM's ONLY permitted output shape. It is a PROPOSAL:
the policy engine decides whether any of it is allowed to have an effect.
Nothing in this module trusts the model -- validation here is the first of two
gates, and app/policy/action_policy.py is the second.

WHY `cited_evidence` EXISTS
  The original sketch had decision / evidence_status / missing_evidence /
  reasoning_summary / draft_representment. That set cannot support the
  fabrication check: to know whether a proposal invented a 3-D Secure
  authentication record, you need to know what it CLAIMED to rely on, and
  compare that against what is on file. Without this field the gate has
  nothing to inspect and "LLM proposes CONTEST -> policy BLOCKS" is
  undemonstrable.

  It is also the field an LLM is most likely to get wrong, which is the point.

WHY `decision` IS NOT THE DECISION
  The field is named `decision` because that is what the model is asked for,
  but it reaches the policy engine as `proposed_action` and is overridden by
  every rule that fires. A proposal saying CONTEST on a dispute with missing
  evidence changes nothing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ProposedDecision = Literal["CONTEST", "ACCEPT", "REVIEW"]
EvidenceStatus = Literal["SUFFICIENT", "INSUFFICIENT"]

# Cap free text. An LLM that returns a 40MB string should fail validation, not
# fill the audit log.
MAX_SUMMARY_CHARS = 1_500
MAX_DRAFT_CHARS = 6_000


class DisputeProposal(BaseModel):
    """The only shape an LLM response may take. Extra fields are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: ProposedDecision
    evidence_status: EvidenceStatus
    missing_evidence: list[str] = Field(default_factory=list)
    cited_evidence: list[str] = Field(default_factory=list)
    reasoning_summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    draft_representment: str = Field(default="", max_length=MAX_DRAFT_CHARS)

    @field_validator("missing_evidence", "cited_evidence")
    @classmethod
    def _clean_evidence_list(cls, v: list[str]) -> list[str]:
        """Deduplicate, drop blanks, and reject anything that is not a plain
        identifier. Evidence names are compared against a fixed vocabulary, so
        a value with whitespace or punctuation is a malformed response rather
        than an unknown evidence type."""
        out = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError("evidence entries must be strings")
            s = item.strip()
            if not s:
                continue
            if not s.replace("_", "").isalnum():
                raise ValueError(f"not a valid evidence identifier: {item!r}")
            if s not in out:
                out.append(s)
        return out


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction_id: int | None = None
    amount_inr: float = Field(ge=0)
    p_fraud: float | None = Field(default=None, ge=0.0, le=1.0)


class DisputeAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dispute_id: str
    reason_code: str
    amount_inr: float = Field(ge=0)
    p_fraud: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, list[str]] = Field(default_factory=dict)


class PolicyDecisionResponse(BaseModel):
    """What the API returns. The decision, the rule, and why -- never the
    proposal alone."""

    model_config = ConfigDict(extra="forbid")

    dispute_id: str
    action: Literal["CONTEST", "ACCEPT", "HUMAN_REVIEW"]
    rule: str
    rationale: str
    proposal_honoured: bool
    proposal_source: Literal["llm", "template", "none"]
    proposed_action: str | None = None
    missing_required_evidence: list[str] = Field(default_factory=list)
    fabricated_evidence: list[str] = Field(default_factory=list)
    indifference_threshold: float
    review_band: tuple[float, float]
