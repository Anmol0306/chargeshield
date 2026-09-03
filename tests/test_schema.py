"""DisputeProposal rejects out-of-enum decisions and missing required fields."""
import pytest
from pydantic import ValidationError

from app.models.schemas import DisputeProposal

VALID = dict(decision="CONTEST", evidence_status="SUFFICIENT",
             reasoning_summary="All required evidence is on file.")


def test_accepts_a_well_formed_proposal():
    p = DisputeProposal(**VALID)
    assert p.decision == "CONTEST"
    assert p.missing_evidence == [] and p.cited_evidence == []


@pytest.mark.parametrize("decision", ["DELETE", "contest", "REFUND", "", "CONTEST "])
def test_rejects_out_of_enum_decision(decision):
    with pytest.raises(ValidationError):
        DisputeProposal(**{**VALID, "decision": decision})


@pytest.mark.parametrize("field", ["decision", "evidence_status", "reasoning_summary"])
def test_rejects_missing_required_field(field):
    payload = {k: v for k, v in VALID.items() if k != field}
    with pytest.raises(ValidationError):
        DisputeProposal(**payload)


def test_rejects_extra_fields():
    """An LLM inventing a field is a malformed response, not a feature."""
    with pytest.raises(ValidationError):
        DisputeProposal(**VALID, authorised_refund=True)


def test_rejects_empty_reasoning():
    with pytest.raises(ValidationError):
        DisputeProposal(**{**VALID, "reasoning_summary": ""})


def test_rejects_unbounded_free_text():
    """A 40MB string should fail validation, not fill the audit log."""
    with pytest.raises(ValidationError):
        DisputeProposal(**{**VALID, "draft_representment": "x" * 10_000})


def test_evidence_lists_are_cleaned_and_deduplicated():
    p = DisputeProposal(**VALID,
                        cited_evidence=[" shipping_proof ", "shipping_proof", ""])
    assert p.cited_evidence == ["shipping_proof"]


@pytest.mark.parametrize("bad", ["../../etc/passwd", "drop table;", "a b", "<script>"])
def test_rejects_non_identifier_evidence_names(bad):
    with pytest.raises(ValidationError):
        DisputeProposal(**VALID, cited_evidence=[bad])


def test_proposal_is_immutable():
    """The gate reads the proposal after validation; it must not change under it."""
    p = DisputeProposal(**VALID)
    with pytest.raises(ValidationError):
        p.decision = "ACCEPT"
