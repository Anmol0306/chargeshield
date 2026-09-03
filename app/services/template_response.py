"""
Deterministic proposal builder. Written BEFORE any LLM call, on purpose.

This is not a fallback bolted on after the fact -- it is the default path, and
the LLM is an enhancement to a system that already works. If the API key is
absent, the provider is down, the response is malformed, or the schema
validation fails twice, this produces a valid DisputeProposal and the pipeline
continues. The demo runs with no network at all.

THE SAFETY PROPERTY
  This builder CANNOT fabricate evidence, because it only ever cites keys that
  are actually present in the dispute's evidence object. That is structural,
  not a matter of care: `cited_evidence` is derived from `evidence`, so there
  is no code path that produces a citation for a document that does not exist.

  That makes it the safe floor. An LLM proposal can only ever be worse on this
  axis, which is why the policy engine checks fabrication on every proposal
  regardless of source.

NO PERSUASION, NO INVENTION
  The draft representment states facts drawn from the dispute record and lists
  the documents on file. It does not argue, speculate about the cardholder's
  intent, or assert anything not present in the evidence object. A merchant
  submitting fabricated claims to a card network has a much worse problem than
  a lost dispute.
"""

from __future__ import annotations

from app.models.schemas import DisputeProposal

# Human-readable labels for Razorpay's evidence field names, used only for
# rendering the draft. Unknown keys fall back to the raw identifier rather than
# being dropped -- silently omitting a document from a representment would be
# worse than an ugly label.
EVIDENCE_LABELS = {
    "shipping_proof": "proof of shipment",
    "billing_proof": "billing record",
    "cancellation_proof": "cancellation record",
    "customer_communication": "customer correspondence",
    "proof_of_service": "proof of service delivery",
    "explanation_letter": "explanation letter",
    "refund_confirmation": "refund confirmation",
    "access_activity_log": "account access and activity log",
    "refund_cancellation_policy": "published refund and cancellation policy",
    "term_and_conditions": "accepted terms and conditions",
    "others": "additional supporting documents",
}


def _label(field: str) -> str:
    return EVIDENCE_LABELS.get(field, field)


def build_template_proposal(
    *,
    reason_code: str,
    evidence: dict,
    requirements: dict,
    amount_inr: float | None = None,
    dispute_id: str | None = None,
) -> DisputeProposal:
    """Deterministic. Same inputs always give the same proposal.

    Keyword-only for the same reason decide() is: several of these are easy to
    transpose and nothing would catch it.
    """
    present = sorted(k for k, v in evidence.items() if v)
    spec = requirements.get(reason_code)

    if spec is None:
        # Unknown reason code: propose REVIEW and cite only what exists. The
        # policy engine will also fail this closed; agreeing with it here keeps
        # the audit trail coherent instead of showing a proposal that argued
        # for something the gate then refused.
        return DisputeProposal(
            decision="REVIEW",
            evidence_status="INSUFFICIENT",
            missing_evidence=["__UNKNOWN_REASON_CODE__"],
            cited_evidence=present,
            reasoning_summary=(
                f"Reason code {reason_code!r} is not in the evidence requirements "
                f"map, so the required document set cannot be determined. "
                f"Escalating for manual assessment."
            ),
            draft_representment="",
        )

    required = list(spec.get("required", []))
    missing = sorted(set(required) - set(present))

    if missing:
        return DisputeProposal(
            decision="REVIEW",
            evidence_status="INSUFFICIENT",
            missing_evidence=missing,
            cited_evidence=present,
            reasoning_summary=(
                f"Dispute raised under {reason_code}. The required evidence set "
                f"is incomplete: {', '.join(_label(m) for m in missing)} "
                f"{'is' if len(missing) == 1 else 'are'} not on file. A "
                f"representment cannot be substantiated without "
                f"{'it' if len(missing) == 1 else 'them'}."
            ),
            draft_representment="",
        )

    return DisputeProposal(
        decision="CONTEST",
        evidence_status="SUFFICIENT",
        missing_evidence=[],
        cited_evidence=present,
        reasoning_summary=(
            f"Dispute raised under {reason_code}. All required evidence is on "
            f"file ({', '.join(_label(r) for r in required)}), so the "
            f"transaction can be substantiated from the merchant's records."
        ),
        draft_representment=_draft(reason_code, present, amount_inr, dispute_id),
    )


def _draft(reason_code: str, present: list[str], amount_inr: float | None,
           dispute_id: str | None) -> str:
    """Factual representment. States what is on file and nothing else."""
    header = "Representment"
    if dispute_id:
        header += f" for dispute {dispute_id}"
    amount = f" for INR {amount_inr:,.2f}" if amount_inr is not None else ""

    documents = "\n".join(f"  - {_label(f)}" for f in present)
    return (
        f"{header}\n\n"
        f"The merchant contests this chargeback{amount}, raised under reason "
        f"code {reason_code}.\n\n"
        f"The following documentation is submitted in support:\n"
        f"{documents}\n\n"
        f"These records are drawn from the merchant's transaction and "
        f"fulfilment systems. The merchant requests that the chargeback be "
        f"reversed on the basis of the documentation above.\n\n"
        f"No claim is made beyond what these documents evidence."
    )
