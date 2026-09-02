"""Evidence sufficiency gate. Pure function, no I/O, no LLM.

Two separate checks that are easy to conflate and must not be:

  SUFFICIENCY  does the evidence set contain everything this reason code
               requires? If not, we cannot substantiate a contest, regardless
               of what the model or the LLM thinks.

  FABRICATION  does the proposal cite evidence that is not in the evidence set?
               This is the check that catches an LLM inventing a 3-D Secure
               authentication record that was never collected. It is not a
               sufficiency question -- a proposal can cite fabricated evidence
               for a dispute whose real evidence is perfectly complete.

The requirements map is passed IN, never read from disk here, so this stays
pure and testable without a filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    missing_required: tuple[str, ...] = ()
    fabricated: tuple[str, ...] = ()

    @property
    def has_fabrication(self) -> bool:
        return bool(self.fabricated)


def assess_evidence(
    reason_code: str,
    evidence: dict,
    requirements: dict,
    cited_evidence: list[str] | None = None,
) -> EvidenceAssessment:
    """Assess an evidence set against the requirements for its reason code.

    `evidence`        the dispute's evidence object (Razorpay shape: field ->
                      list of document ids). A field that is absent OR present
                      but empty counts as missing -- an empty list is not
                      evidence, and treating it as present is exactly the kind
                      of bug that lets an unsubstantiated contest through.
    `cited_evidence`  evidence types the (untrusted) proposal claims to rely
                      on. Anything cited that is not actually on file is
                      fabrication.

    An unknown reason_code is treated as INSUFFICIENT rather than raising or
    defaulting to sufficient: an unrecognised dispute type is precisely when a
    human should look, and a gate that fails open is not a gate.
    """
    present = {k for k, v in evidence.items() if v}

    spec = requirements.get(reason_code)
    if spec is None:
        return EvidenceAssessment(
            sufficient=False,
            missing_required=("__UNKNOWN_REASON_CODE__",),
            fabricated=tuple(sorted(set(cited_evidence or []) - present)),
        )

    missing = tuple(sorted(set(spec.get("required", [])) - present))
    fabricated = tuple(sorted(set(cited_evidence or []) - present))

    return EvidenceAssessment(
        sufficient=not missing,
        missing_required=missing,
        fabricated=fabricated,
    )
