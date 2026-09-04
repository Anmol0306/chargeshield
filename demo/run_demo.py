"""
The recorded demo. Runs with NO network and NO API credential.

WHY IT NEEDS NEITHER
  That is the claim. The deterministic path -- template proposal, policy gate,
  audit record -- is the system; the LLM is an enhancement on top of it. If
  this script needed a provider to run, the claim would be false.

IT IS ALSO A TEST
  Every scenario asserts its expected action and rule, and the script exits
  non-zero if any of them changes. So the thing being recorded and the thing
  being verified are the same artifact, and a demo cannot drift away from the
  behaviour it claims to show.

RUN
  make demo          (equivalently: python -m demo.run_demo)

  Run as a MODULE, not a path. `python demo/run_demo.py` puts demo/ on
  sys.path instead of the repo root, and every `from app...` import fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.models.schemas import DisputeProposal
from app.policy.action_policy import decide
from app.policy.thresholds import load_policy_config
from app.services import llm_service
from app.services.audit_service import build_record
from app.services.template_response import build_template_proposal

REQUIREMENTS = {
    k: v for k, v in
    json.loads(Path("evidence/requirements.json").read_text()).items()
    if not k.startswith("_")
}

W = 74
COMPLETE = {"shipping_proof": ["doc_ship_001"], "billing_proof": ["doc_bill_001"]}
NO_DELIVERY_PROOF = {"billing_proof": ["doc_bill_001"]}


def rule(ch: str = "─") -> None:
    print(ch * W)


def header(n: str, title: str, claim: str) -> None:
    print()
    rule("━")
    print(f"  {n}   {title}")
    print(f"       {claim}")
    rule("━")


def show_dispute(reason: str, amount: float, evidence: dict, p: float) -> None:
    print(f"  DISPUTE   {reason}   ₹{amount:,.0f}   p(fraud) = {p:.2f}")
    on_file = sorted(k for k, v in evidence.items() if v)
    required = REQUIREMENTS[reason]["required"]
    print(f"  evidence on file : {', '.join(on_file) or '(none)'}")
    print(f"  required         : {', '.join(required)}")
    missing = sorted(set(required) - set(on_file))
    if missing:
        print(f"  MISSING          : {', '.join(missing)}")


def show_proposal(p: DisputeProposal, source: str) -> None:
    print()
    print(f"  PROPOSAL  (source: {source} — untrusted)")
    print(f"    decision       : {p.decision}")
    print(f"    cited_evidence : {p.cited_evidence}")
    print(f"    reasoning      : {p.reasoning_summary[:60]}"
          f"{'…' if len(p.reasoning_summary) > 60 else ''}")


def show_decision(d) -> None:
    print()
    print(f"  POLICY ENGINE  (deterministic — no I/O, no LLM)")
    print(f"    ACTION         : {d.action}")
    print(f"    rule           : {d.rule}")
    print(f"    honoured       : {d.proposal_honoured}")
    if d.fabricated_evidence:
        print(f"    BLOCKED        : cited {list(d.fabricated_evidence)}, not on file")
    if d.missing_required_evidence:
        print(f"    missing        : {list(d.missing_required_evidence)}")
    print(f"    band for amount: [{d.review_band[0]:.3f}, {d.review_band[1]:.3f}]"
          f"   p* = {d.indifference_threshold:.3f}")


def check(label: str, got, want) -> bool:
    ok = got == want
    print(f"    {'PASS' if ok else 'FAIL'}  {label}: {got}"
          + ("" if ok else f"   (expected {want})"))
    return ok


def main() -> int:
    cfg = load_policy_config()
    failures = 0

    print()
    rule("═")
    print("  ChargeShield — deterministic demo")
    print("  No network. No API credential. This is the system, not a fallback.")
    print(f"  LLM credential present: {llm_service.is_enabled()}")
    rule("═")

    # ---- 01 ------------------------------------------------------------
    header("01", "Missing evidence",
           "A contest that cannot be substantiated must not be filed.")
    show_dispute("NON_RECEIPT", 6_070, NO_DELIVERY_PROOF, 0.05)
    prop, src = llm_service.propose(
        reason_code="NON_RECEIPT", amount_inr=6_070, evidence=NO_DELIVERY_PROOF,
        requirements=REQUIREMENTS, dispute_id="demo_01")
    show_proposal(prop, src)
    d = decide(config=cfg, p_fraud=0.05, amount_inr=6_070,
               reason_code="NON_RECEIPT", evidence=NO_DELIVERY_PROOF,
               requirements=REQUIREMENTS, proposed_action=prop.decision,
               cited_evidence=prop.cited_evidence)
    show_decision(d)
    print()
    failures += not check("action", d.action, "HUMAN_REVIEW")
    failures += not check("rule", d.rule, "required_evidence_missing")

    # ---- 02 ------------------------------------------------------------
    header("02", "The model fabricates evidence",
           "The evidence set is COMPLETE — this contest would otherwise be allowed.")
    show_dispute("NON_RECEIPT", 6_070, COMPLETE, 0.05)
    fabricated = DisputeProposal(
        decision="CONTEST", evidence_status="SUFFICIENT",
        cited_evidence=["shipping_proof", "access_activity_log"],
        reasoning_summary="3-D Secure authentication confirms the cardholder "
                          "authorised this transaction.",
        draft_representment="The transaction was authenticated via 3-D Secure.")
    show_proposal(fabricated, "llm (constructed)")
    print("    ^ access_activity_log was never collected for this dispute.")
    d = decide(config=cfg, p_fraud=0.05, amount_inr=6_070,
               reason_code="NON_RECEIPT", evidence=COMPLETE,
               requirements=REQUIREMENTS, proposed_action=fabricated.decision,
               cited_evidence=fabricated.cited_evidence)
    show_decision(d)
    print()
    failures += not check("action", d.action, "HUMAN_REVIEW")
    failures += not check("rule", d.rule, "proposal_cited_evidence_not_on_file")
    failures += not check("proposal rejected", d.proposal_rejected, True)

    # ---- 03 ------------------------------------------------------------
    header("03", "Provider returns malformed JSON",
           "A broken model degrades to the template. It cannot break the pipeline.")

    def garbage(prompt, settings, api_key):
        return "I'm sorry, I can't help with that."

    prop, src = llm_service.propose(
        reason_code="NON_RECEIPT", amount_inr=6_070, evidence=COMPLETE,
        requirements=REQUIREMENTS, dispute_id="demo_03", call_provider=garbage)
    print(f"  provider returned : {garbage(None, None, None)!r}")
    print(f"  retries           : {llm_service.MAX_ATTEMPTS} attempts, then template")
    show_proposal(prop, src)
    d = decide(config=cfg, p_fraud=0.05, amount_inr=6_070,
               reason_code="NON_RECEIPT", evidence=COMPLETE,
               requirements=REQUIREMENTS, proposed_action=prop.decision,
               cited_evidence=prop.cited_evidence)
    show_decision(d)
    print()
    failures += not check("source", src, "template")
    failures += not check("action", d.action, "CONTEST")

    # ---- 04 ------------------------------------------------------------
    header("04", "Provider unreachable",
           "An external dependency failing must not propagate. Verified against "
           "a live 429.")

    def unreachable(prompt, settings, api_key):
        raise TimeoutError("upstream unreachable")

    prop, src = llm_service.propose(
        reason_code="NON_RECEIPT", amount_inr=6_070, evidence=COMPLETE,
        requirements=REQUIREMENTS, dispute_id="demo_04", call_provider=unreachable)
    show_proposal(prop, src)
    d = decide(config=cfg, p_fraud=0.05, amount_inr=6_070,
               reason_code="NON_RECEIPT", evidence=COMPLETE,
               requirements=REQUIREMENTS, proposed_action=prop.decision,
               cited_evidence=prop.cited_evidence)
    show_decision(d)
    print()
    failures += not check("source", src, "template")
    failures += not check("action", d.action, "CONTEST")

    # ---- 05 ------------------------------------------------------------
    header("05", "Above the exposure cap",
           "Holds even if the score and the evidence are both perfect.")
    show_dispute("NON_RECEIPT", 90_000, COMPLETE, 0.01)
    d = decide(config=cfg, p_fraud=0.01, amount_inr=90_000,
               reason_code="NON_RECEIPT", evidence=COMPLETE,
               requirements=REQUIREMENTS, proposed_action="CONTEST",
               cited_evidence=["shipping_proof", "billing_proof"])
    show_decision(d)
    print()
    failures += not check("action", d.action, "HUMAN_REVIEW")
    failures += not check("rule", d.rule, "amount_cap_exceeded")

    # ---- 06 ------------------------------------------------------------
    header("06", "Likely genuine fraud",
           "Contesting real fraud burns the representment cost to lose.")
    show_dispute("NON_RECEIPT", 6_070, COMPLETE, 0.97)
    d = decide(config=cfg, p_fraud=0.97, amount_inr=6_070,
               reason_code="NON_RECEIPT", evidence=COMPLETE,
               requirements=REQUIREMENTS, proposed_action="CONTEST",
               cited_evidence=["shipping_proof", "billing_proof"])
    show_decision(d)
    print()
    failures += not check("action", d.action, "ACCEPT")
    failures += not check("rule", d.rule, "fraud_probability_above_band")

    # ---- 07 ------------------------------------------------------------
    header("07", "Too small to be worth contesting",
           "No score and no evidence can change this. It is arithmetic.")
    show_dispute("NON_RECEIPT", 400, COMPLETE, 0.02)
    print(f"  even a certain win recovers ₹"
          f"{cfg.assumed_win_rate_if_legitimate * 400:,.0f} against a "
          f"₹{cfg.representment_cost_inr:,.0f} representment cost")
    d = decide(config=cfg, p_fraud=0.02, amount_inr=400,
               reason_code="NON_RECEIPT", evidence=COMPLETE,
               requirements=REQUIREMENTS, proposed_action="CONTEST",
               cited_evidence=["shipping_proof", "billing_proof"])
    show_decision(d)
    print()
    failures += not check("action", d.action, "ACCEPT")
    failures += not check("rule", d.rule,
                          "dispute_too_small_to_repay_representment")

    # ---- 08 ------------------------------------------------------------
    header("08", "The same p(fraud), two different answers",
           "The review band is amount-dependent. This is the cost model, visible.")
    print("  Identical score. Identical evidence. Only the amount differs.\n")
    for amount, expect_action, expect_rule in (
        (6_070, "HUMAN_REVIEW", "inside_cost_review_band"),
        (2_000, "ACCEPT", "fraud_probability_above_band"),
    ):
        d = decide(config=cfg, p_fraud=0.88, amount_inr=amount,
                   reason_code="NON_RECEIPT", evidence=COMPLETE,
                   requirements=REQUIREMENTS, proposed_action="CONTEST",
                   cited_evidence=["shipping_proof", "billing_proof"])
        lo, hi = d.review_band
        print(f"  ₹{amount:>6,}   p=0.880   band [{lo:.3f}, {hi:.3f}]   "
              f"p*={d.indifference_threshold:.3f}   ->  {d.action}")
        print(f"           {d.rule}")
        failures += not check(f"₹{amount:,} action", d.action, expect_action)
        failures += not check(f"₹{amount:,} rule", d.rule, expect_rule)
    print()
    print("  A larger dispute is worth contesting at higher fraud risk, so the")
    print("  band sits higher. At ₹6,070 p=0.88 is inside it and a human decides;")
    print("  at ₹2,000 the same score is already past it and we accept.")

    # ---- audit ----------------------------------------------------------
    header("AUDIT", "Every decision is reconstructable",
           "An unauditable gate is decoration.")
    record = build_record(dispute_id="demo_02", decision=d, config=cfg,
                          proposal=fabricated.model_dump(), proposal_source="llm")
    for key in ("action", "rule", "rationale"):
        print(f"  {key:<16}: {str(record[key])[:56]}")
    print(f"  {'bands_in_force':<16}: {record['bands_in_force']['review_band']}")
    print(f"  {'cost_assumptions':<16}: scenario="
          f"{record['cost_assumptions']['scenario']}, "
          f"c=₹{record['cost_assumptions']['representment_cost_inr']:.0f}, "
          f"w={record['cost_assumptions']['assumed_win_rate_if_legitimate']}")

    print()
    rule("═")
    if failures:
        print(f"  {failures} assertion(s) FAILED — the demo no longer matches the "
              f"system.")
        rule("═")
        return 1
    print("  All scenarios behaved as asserted. Recorded behaviour == tested "
          "behaviour.")
    rule("═")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
