"""
Run the policy engine over the whole anchored dispute queue.

This is where the two halves finally meet: real held-out transactions carrying
real isFraud labels, scored by the calibrated model, decided by the
deterministic gate, and written to an audit log.

THE ONE DISPUTE-SIDE METRIC WITH REAL GROUND TRUTH
    wasted representment effort = contested disputes where isFraud == 1

  Every dispute is anchored to a real transaction, so this is measured, not
  simulated. It is compared against contest-everything, whose wasted rate is by
  definition the queue's fraud rate.

WHAT THIS DOES *NOT* MEASURE
  Win rate, money recovered, or dispute outcomes. Those need merchant-side
  resolution labels that are not public. See ml/link_disputes.py.

READ THE ACTION MIX WITH CARE
  The share of disputes routed to HUMAN_REVIEW is dominated by
  `required_evidence_missing`, and evidence availability in the synthetic queue
  is a DIAL (p_required_evidence_present in ml/link_disputes.py), not an
  observation. So the review rate is an artefact of that parameter and must
  never be quoted as a finding about the product. The action mix restricted to
  disputes with complete evidence -- where the model actually drives the
  decision -- is reported alongside it, and that is the informative one.

OUT  evaluation/batch_results.json, evidence/audit_log.jsonl
"""

from __future__ import annotations

import collections
import json
from datetime import datetime, timezone
from pathlib import Path

from app.policy.action_policy import decide
from app.policy.thresholds import PolicyConfig, load_policy_config
from app.services.audit_service import AuditLog, build_record

DISPUTES_PATH = Path("data/processed/disputes.json")
REQUIREMENTS_PATH = Path("evidence/requirements.json")
OUT_PATH = Path("evaluation/batch_results.json")


def load_requirements(path: Path = REQUIREMENTS_PATH) -> dict:
    return {k: v for k, v in json.loads(path.read_text()).items()
            if not k.startswith("_")}


def run_batch(disputes: list[dict], config: PolicyConfig, requirements: dict,
              audit: AuditLog | None = None) -> dict:
    """Decide every dispute. Returns metrics; writes audit records if given a log."""
    actions: collections.Counter = collections.Counter()
    rules: collections.Counter = collections.Counter()
    records, decisions = [], []

    for d in disputes:
        cs = d["_chargeshield"]
        dec = decide(
            config=config,
            p_fraud=cs["p_fraud_calibrated"],
            amount_inr=cs["amount_inr"],
            reason_code=d["reason_code"],
            evidence=d["evidence"],
            requirements=requirements,
            proposed_action=None,
        )
        actions[dec.action] += 1
        rules[dec.rule] += 1
        decisions.append((d, dec))
        records.append(build_record(dispute_id=d["id"], decision=dec,
                                    config=config, proposal_source="none"))

    if audit is not None:
        audit.append_many(records)

    n = len(disputes)
    queue_frauds = sum(d["_chargeshield"]["anchor_is_fraud"] for d in disputes)
    queue_rate = queue_frauds / n

    contested = [(d, x) for d, x in decisions if x.action == "CONTEST"]
    wasted = sum(d["_chargeshield"]["anchor_is_fraud"] for d, _ in contested)
    accepted = [(d, x) for d, x in decisions if x.action == "ACCEPT"]
    forfeited = sum(1 for d, _ in accepted if not d["_chargeshield"]["anchor_is_fraud"])

    wasted_rate = wasted / len(contested) if contested else 0.0

    # The informative slice: where evidence is complete, the model -- not the
    # evidence dial -- drives the decision.
    complete = [(d, x) for d, x in decisions
                if d["_chargeshield"]["evidence_complete"]]
    complete_actions = collections.Counter(x.action for _, x in complete)

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_disputes": n,
        "queue_fraud_rate": queue_rate,
        "actions": dict(actions),
        "action_share": {k: v / n for k, v in actions.items()},
        "rules_fired": dict(rules),
        "wasted_representment": {
            "contested": len(contested),
            "contested_that_were_real_fraud": wasted,
            "wasted_rate": wasted_rate,
            "contest_all_wasted_rate": queue_rate,
            "relative_reduction_vs_contest_all": (
                (queue_rate - wasted_rate) / queue_rate if queue_rate else 0.0),
            "label_source": "real isFraud on the anchor transaction",
        },
        "forfeited_winnable": {
            "accepted": len(accepted),
            "accepted_that_were_legitimate": forfeited,
        },
        "evidence_complete_subset": {
            "n": len(complete),
            "actions": dict(complete_actions),
            "action_share": {k: v / len(complete) for k, v in complete_actions.items()}
            if complete else {},
        },
        "notes": [
            "wasted_representment is measured against REAL isFraud labels on "
            "the anchor transactions. It is the only dispute-side metric here "
            "with real ground truth.",
            "No win rate, money recovered, or dispute outcome is measured or "
            "claimable.",
            "The HUMAN_REVIEW share is dominated by required_evidence_missing, "
            "and evidence availability in the synthetic queue is a parameter "
            "(p_required_evidence_present), not an observation. Do NOT quote "
            "the review rate as a finding. Use evidence_complete_subset.",
        ],
    }


def main() -> None:
    payload = json.loads(DISPUTES_PATH.read_text())
    config = load_policy_config()
    audit = AuditLog()
    audit.reset()

    results = run_batch(payload["disputes"], config, load_requirements(), audit)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    n = results["n_disputes"]
    print(f"disputes {n:,} · queue fraud rate {results['queue_fraud_rate']:.4f}\n")
    print("actions:")
    for a, c in sorted(results["actions"].items(), key=lambda kv: -kv[1]):
        print(f"  {a:>13}: {c:>6,} ({c / n:6.1%})")
    print("\nrules fired:")
    for r, c in sorted(results["rules_fired"].items(), key=lambda kv: -kv[1]):
        print(f"  {r:>42}: {c:>6,} ({c / n:6.1%})")

    w = results["wasted_representment"]
    print(f"\nwasted representment effort (REAL labels):")
    print(f"  contested {w['contested']:,}, of which {w['contested_that_were_real_fraud']:,} "
          f"were genuine fraud = {w['wasted_rate']:.1%}")
    print(f"  contest-everything would waste {w['contest_all_wasted_rate']:.1%}")
    print(f"  relative reduction: {w['relative_reduction_vs_contest_all']:.1%}")

    e = results["evidence_complete_subset"]
    print(f"\naction mix where evidence is complete (n={e['n']:,}) — "
          f"the model-driven slice:")
    for a, s in sorted(e["action_share"].items(), key=lambda kv: -kv[1]):
        print(f"  {a:>13}: {s:6.1%}")
    print(f"\nWrote {OUT_PATH} and {audit.path} ({len(audit.read_all()):,} records)")


if __name__ == "__main__":
    main()
