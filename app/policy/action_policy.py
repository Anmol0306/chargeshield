"""
THE GATE. Pure functions only -- no I/O, no LLM calls, no clock, no randomness.

That is what makes "the LLM cannot move money" a STRUCTURAL claim rather than a
promise. `decide()` cannot reach the network because it has no client; it
cannot read a threshold from disk because thresholds arrive as an argument; it
cannot be non-deterministic because it has no source of entropy. A reviewer can
verify all of that by reading one file.

THE PROPOSAL IS UNTRUSTED INPUT
  The LLM produces a DisputeProposal. This module treats it exactly as it would
  treat a form field submitted by a stranger: as a suggestion that must survive
  every rule before it has any effect. There is no path by which a proposal
  reaches an action without passing each override below.

OVERRIDE ORDER -- and why this order
  1. amount > cap                 -> HUMAN_REVIEW
     Exposure limit first, because it is the rule that does not depend on any
     model output being correct. If the score is wrong, this still holds.

  2. proposal cites evidence not on file  -> HUMAN_REVIEW, proposal REJECTED
     Before sufficiency, because a fabricating proposal is not merely wrong
     about this dispute -- it is evidence the proposal cannot be trusted at
     all. Ordering it after sufficiency would let a fabricated citation slip
     through whenever the real evidence happened to be complete.

  3. w * amount <= representment cost -> ACCEPT
     The dispute cannot repay a representment even if we win it with certainty,
     so no score and no evidence can change the answer. Placed before the
     evidence gate deliberately: escalating a dispute we would accept anyway
     spends INR 150 of analyst time to learn nothing.

  4. required evidence missing    -> HUMAN_REVIEW
     Cannot substantiate a contest, so contesting burns cost to lose.

  5. p inside the cost review band -> HUMAN_REVIEW
     Where automating is worth less than asking. Amount-dependent -- see
     app/policy/thresholds.py.

  6. p above the band             -> ACCEPT
     Likely genuinely fraudulent. Contesting burns cost to lose.

  7. otherwise                    -> honour the proposal, bounded to CONTEST
     Even here the proposal cannot invent an action: anything other than
     CONTEST at this point is a proposal disagreeing with the cost model on the
     cost model's own terms, so the deterministic action wins and the
     disagreement is recorded.

EVERY decision records which rule fired. An unexplainable decision is not
auditable, and an unauditable gate is decoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.policy.evidence_policy import EvidenceAssessment, assess_evidence
from app.policy.thresholds import PolicyConfig

Action = Literal["CONTEST", "ACCEPT", "HUMAN_REVIEW"]

# Rule identifiers. Stable strings -- they end up in the audit log and in the
# demo assertions, so renaming one is a breaking change.
RULE_AMOUNT_CAP = "amount_cap_exceeded"
RULE_FABRICATED_EVIDENCE = "proposal_cited_evidence_not_on_file"
RULE_BELOW_ECONOMIC_FLOOR = "dispute_too_small_to_repay_representment"
RULE_EVIDENCE_INSUFFICIENT = "required_evidence_missing"
RULE_COST_REVIEW_BAND = "inside_cost_review_band"
RULE_HIGH_FRAUD_ACCEPT = "fraud_probability_above_band"
RULE_PROPOSAL_HONOURED = "proposal_honoured"
RULE_PROPOSAL_OVERRIDDEN = "proposal_disagreed_with_cost_model"


@dataclass(frozen=True)
class PolicyDecision:
    action: Action
    proposal_honoured: bool
    rule: str
    rationale: str
    p_fraud: float
    amount_inr: float
    indifference_threshold: float
    review_band: tuple[float, float]
    missing_required_evidence: tuple[str, ...] = ()
    fabricated_evidence: tuple[str, ...] = ()
    proposed_action: str | None = None

    @property
    def proposal_rejected(self) -> bool:
        return not self.proposal_honoured


def decide(
    *,
    config: PolicyConfig,
    p_fraud: float,
    amount_inr: float,
    reason_code: str,
    evidence: dict,
    requirements: dict,
    proposed_action: str | None = None,
    cited_evidence: list[str] | None = None,
) -> PolicyDecision:
    """Return the action the merchant should take. Deterministic and total.

    Keyword-only on purpose: a positional call site that silently swaps
    `p_fraud` and `amount_inr` would produce a plausible-looking decision that
    is completely wrong, and both are floats so no type checker would catch it.
    """
    if not 0.0 <= p_fraud <= 1.0:
        raise ValueError(f"p_fraud must be a probability, got {p_fraud}")
    if amount_inr < 0:
        raise ValueError(f"amount_inr must be non-negative, got {amount_inr}")

    p_star = config.indifference_threshold(amount_inr)
    band = config.review_band(amount_inr)
    ev: EvidenceAssessment = assess_evidence(
        reason_code, evidence, requirements, cited_evidence
    )

    def decision(action: Action, honoured: bool, rule: str, why: str) -> PolicyDecision:
        return PolicyDecision(
            action=action,
            proposal_honoured=honoured,
            rule=rule,
            rationale=why,
            p_fraud=p_fraud,
            amount_inr=amount_inr,
            indifference_threshold=p_star,
            review_band=band,
            missing_required_evidence=ev.missing_required,
            fabricated_evidence=ev.fabricated,
            proposed_action=proposed_action,
        )

    # 1. Exposure limit. Independent of any model output being correct.
    if amount_inr > config.auto_action_amount_cap_inr:
        return decision(
            "HUMAN_REVIEW", False, RULE_AMOUNT_CAP,
            f"amount INR {amount_inr:,.0f} exceeds the automatic-action cap of "
            f"INR {config.auto_action_amount_cap_inr:,.0f}",
        )

    # 2. Fabrication. Checked before sufficiency: a proposal citing evidence
    #    that does not exist is untrustworthy about everything, not just this.
    if ev.has_fabrication:
        return decision(
            "HUMAN_REVIEW", False, RULE_FABRICATED_EVIDENCE,
            "proposal cited evidence not present in the evidence set: "
            + ", ".join(ev.fabricated),
        )

    # 3. Economically futile regardless of score or evidence.
    max_recovery = config.assumed_win_rate_if_legitimate * amount_inr
    if max_recovery <= config.representment_cost_inr:
        return decision(
            "ACCEPT", proposed_action == "ACCEPT", RULE_BELOW_ECONOMIC_FLOOR,
            f"even a certain win recovers only INR {max_recovery:,.0f} against a "
            f"INR {config.representment_cost_inr:,.0f} representment cost, so no "
            f"score or evidence can make contesting worthwhile",
        )

    # 4. Cannot substantiate.
    if not ev.sufficient:
        return decision(
            "HUMAN_REVIEW", False, RULE_EVIDENCE_INSUFFICIENT,
            f"reason code {reason_code} requires evidence not on file: "
            + ", ".join(ev.missing_required),
        )

    # 5. Automating is worth less than asking.
    low, high = band
    if low <= p_fraud <= high:
        return decision(
            "HUMAN_REVIEW", False, RULE_COST_REVIEW_BAND,
            f"p={p_fraud:.3f} lies in the cost review band [{low:.3f}, {high:.3f}] "
            f"around the indifference point p*={p_star:.3f} for an INR "
            f"{amount_inr:,.0f} dispute; the expected-cost gap between "
            f"contesting and accepting is smaller than the INR "
            f"{config.human_review_cost_inr:,.0f} cost of a human review",
        )

    # 6. Likely genuine fraud. Contesting burns cost to lose.
    if p_fraud > high:
        return decision(
            "ACCEPT", proposed_action == "ACCEPT", RULE_HIGH_FRAUD_ACCEPT,
            f"p={p_fraud:.3f} exceeds the review band; contesting a likely-genuine "
            f"fraud costs INR {config.representment_cost_inr:,.0f} to lose",
        )

    # 7. Below the band: contesting is the cost-minimising action.
    if proposed_action in (None, "CONTEST"):
        return decision(
            "CONTEST", True, RULE_PROPOSAL_HONOURED,
            f"p={p_fraud:.3f} is below the review band and required evidence is "
            f"on file; expected recovery exceeds the representment cost",
        )

    return decision(
        "CONTEST", False, RULE_PROPOSAL_OVERRIDDEN,
        f"proposal said {proposed_action} but p={p_fraud:.3f} is below the review "
        f"band with evidence on file, so contesting is cost-minimising",
    )
