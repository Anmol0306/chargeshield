"""
Pydantic contracts.

DisputeProposal is the LLM's ONLY permitted output shape:
  decision: Literal["CONTEST", "ACCEPT", "REVIEW"]
  evidence_status: Literal["SUFFICIENT", "INSUFFICIENT"]
  missing_evidence: list[str]
  reasoning_summary: str
  draft_representment: str

It is a PROPOSAL. The policy engine decides whether it is allowed.
"""
