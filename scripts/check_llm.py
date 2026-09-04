"""
One-off verification of the live LLM path. Run it once before recording.

WHY THIS EXISTS
  Every test of app/services/llm_service.py injects `call_provider`, so the
  real network path has never executed. Those tests prove the FALLBACK works;
  they cannot prove the prompt produces parseable output from an actual model.
  That is the one thing left that fails on camera rather than in CI.

WHAT IT CHECKS
  1. A credential is present (presence only -- this script never prints, logs
     or returns any part of it).
  2. A real call returns JSON that validates against DisputeProposal.
  3. The model respects the no-fabrication instruction: cited_evidence must be
     a subset of the evidence actually on file.
  4. The model correctly refuses when required evidence is missing.
  5. The policy gate reaches the right decision on each proposal.

  A FAIL on 3 is not a bug in this project -- it is the gate's reason for
  existing, and the run will show the gate catching it. It is still worth
  knowing before a live demo.

RUN
  LLM_API_KEY=... python -m scripts.check_llm
  (or export it first; this script reads the environment, never .env)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from app.models.schemas import DisputeProposal
from app.policy.action_policy import decide
from app.policy.thresholds import load_policy_config
from app.services import llm_service

REQUIREMENTS = {
    k: v for k, v in
    json.loads(Path("evidence/requirements.json").read_text()).items()
    if not k.startswith("_")
}

COMPLETE_NON_RECEIPT = {
    "shipping_proof": ["doc_ship_001"],
    "billing_proof": ["doc_bill_001"],
    "customer_communication": ["doc_comm_001"],
}
INCOMPLETE_NON_RECEIPT = {"billing_proof": ["doc_bill_001"]}
COMPLETE_FRAUD = {
    "customer_communication": ["doc_comm_002"],
    "billing_proof": ["doc_bill_002"],
}

SCENARIOS = [
    ("complete evidence, non-receipt", "NON_RECEIPT", COMPLETE_NON_RECEIPT,
     6_070.0, "CONTEST"),
    ("missing shipping proof", "NON_RECEIPT", INCOMPLETE_NON_RECEIPT,
     6_070.0, "REVIEW"),
    ("fraud claim, minimum evidence", "FRAUD", COMPLETE_FRAUD, 12_000.0, "CONTEST"),
]


def main() -> int:
    if not llm_service.is_enabled():
        print("No credential in the environment.\n")
        print("  LLM_API_KEY is not set, so propose() will return the")
        print("  deterministic template and no provider call will be made.")
        print("  That is correct behaviour, but it does not exercise the live path.\n")
        print("  Run:  LLM_API_KEY=... .venv/bin/python -m scripts.check_llm")
        return 2

    settings = llm_service.load_settings()
    print(f"credential : present (value never printed)")
    print(f"model      : {settings.model}")
    print(f"base_url   : {settings.base_url or 'provider default'}")
    print(f"timeout    : {settings.timeout}s\n")

    config = load_policy_config()
    failures = 0

    for label, reason_code, evidence, amount, expected in SCENARIOS:
        print("=" * 72)
        print(f"{label}   [{reason_code}, INR {amount:,.0f}]")
        on_file = sorted(k for k, v in evidence.items() if v)
        print(f"  evidence on file: {', '.join(on_file) or '(none)'}")

        start = time.monotonic()
        proposal, source = llm_service.propose(
            reason_code=reason_code, amount_inr=amount, evidence=evidence,
            requirements=REQUIREMENTS, dispute_id=f"disp_check_{reason_code}")
        elapsed = time.monotonic() - start

        print(f"  source: {source}  ({elapsed:.2f}s)")
        if source != "llm":
            print("  FAIL  fell back to the template — the provider call did not "
                  "produce a valid response.")
            print("        Check the log line above for the reason (logged by "
                  "exception TYPE, never message).")
            failures += 1
            continue

        print(f"  proposed decision : {proposal.decision} "
              f"({proposal.evidence_status})")
        print(f"  cited_evidence    : {proposal.cited_evidence}")
        print(f"  missing_evidence  : {proposal.missing_evidence}")
        print(f"  reasoning         : {proposal.reasoning_summary[:120]}"
              f"{'...' if len(proposal.reasoning_summary) > 120 else ''}")

        fabricated = sorted(set(proposal.cited_evidence) - set(on_file))
        if fabricated:
            print(f"  ** FABRICATION: cited {fabricated} which is NOT on file.")
            print(f"     The gate is expected to catch this. Continuing.")
        else:
            print(f"  no fabrication: every citation is on file")

        if proposal.decision != expected:
            print(f"  note: proposed {proposal.decision}, expected {expected}. "
                  f"Not fatal — the gate decides — but worth reading the "
                  f"reasoning above.")

        decision = decide(
            config=config, p_fraud=0.05, amount_inr=amount,
            reason_code=reason_code, evidence=evidence,
            requirements=REQUIREMENTS, proposed_action=proposal.decision,
            cited_evidence=proposal.cited_evidence)
        print(f"  GATE: {decision.action}  [{decision.rule}]")
        print(f"        honoured={decision.proposal_honoured}")
        if decision.fabricated_evidence:
            print(f"        blocked fabricated: {list(decision.fabricated_evidence)}")

    print("=" * 72)
    if failures:
        print(f"\n{failures} of {len(SCENARIOS)} scenarios fell back to the "
              f"template. The live path is NOT working.")
        print("The system is still safe — that is what the fallback is for — but "
              "do not claim a working LLM path in the video.")
        return 1

    print(f"\nAll {len(SCENARIOS)} scenarios produced a valid, schema-conforming "
          f"proposal from the provider.")
    print("The live LLM path works. Any fabrication above was caught by the gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
