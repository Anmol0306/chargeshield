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


def realised_cost(action: str, is_fraud: bool, amount_inr: float, p_fraud: float,
                  config: PolicyConfig) -> float:
    """What this action ACTUALLY cost, given what the transaction turned out to be.

    Uses the real isFraud label, not the model's belief -- these disputes are
    anchored to real held-out transactions, so realised cost is measurable
    rather than projected. `w` remains an assumption (whether a contest on a
    legitimate transaction succeeds is not in the data).

        ACCEPT              -> A                    the chargeback stands
        CONTEST, fraud      -> c + A                pay the cost, lose anyway
        CONTEST, legitimate -> c + (1 - w) * A      pay the cost, usually recover
        HUMAN_REVIEW        -> h + realised cost of the analyst's action

    The analyst is modelled as choosing the cost-minimising action given the
    SAME information the model had (p), not given the true label. Modelling an
    oracle analyst would be generous to human review -- and since ChargeShield
    is the policy that routes work to humans, that generosity would flatter
    every comparator except ours. This assumption is therefore conservative for
    the claim being made.
    """
    c = config.representment_cost_inr
    w = config.assumed_win_rate_if_legitimate
    h = config.human_review_cost_inr

    def contest_cost() -> float:
        return c + amount_inr * (1.0 if is_fraud else (1.0 - w))

    if action == "CONTEST":
        return contest_cost()
    if action == "ACCEPT":
        return amount_inr

    # HUMAN_REVIEW: analyst picks by expected cost given p, then reality lands.
    expected_contest = c + amount_inr * (1.0 - w * (1.0 - p_fraud))
    return h + (contest_cost() if expected_contest < amount_inr else amount_inr)


def price_policies(decisions: list[tuple[dict, object]], config: PolicyConfig) -> dict:
    """Four policies on the SAME dispute queue, priced by the same function.

    Unlike ml/evaluate.py's transaction-level comparison, the ChargeShield row
    here IS the shipped policy engine -- amount-dependent bands, evidence gate,
    fabrication check, amount cap and economic floor, exactly as it runs.
    """
    w = config.assumed_win_rate_if_legitimate
    c = config.representment_cost_inr
    rows: dict[str, list[float]] = {k: [] for k in
                                    ("defend_none", "defend_all",
                                     "static_amount_rule", "chargeshield")}
    for d, dec in decisions:
        cs = d["_chargeshield"]
        y, a, p = bool(cs["anchor_is_fraud"]), cs["amount_inr"], cs["p_fraud_calibrated"]
        rows["defend_none"].append(realised_cost("ACCEPT", y, a, p, config))
        rows["defend_all"].append(realised_cost("CONTEST", y, a, p, config))
        rows["static_amount_rule"].append(
            realised_cost("CONTEST" if w * a > c else "ACCEPT", y, a, p, config))
        rows["chargeshield"].append(realised_cost(dec.action, y, a, p, config))

    n = len(decisions)
    out = {k: {"total_inr": sum(v), "per_dispute_inr": sum(v) / n} for k, v in rows.items()}
    static = out["static_amount_rule"]["per_dispute_inr"]
    all_ = out["defend_all"]["per_dispute_inr"]
    for k, r in out.items():
        r["saving_vs_defend_all_inr_per_dispute"] = all_ - r["per_dispute_inr"]
        r["saving_vs_static_rule_inr_per_dispute"] = static - r["per_dispute_inr"]
    return out


def policy_comparison_by_segment(decisions: list[tuple[dict, object]],
                                 config: PolicyConfig) -> dict:
    """Where does the gate earn its keep, and where does it pay for safety?

    The headline comparison across the whole queue is close to a wash, and the
    aggregate hides two opposite effects that matter more than the net:

      evidence complete   the model has what it needs, and the gate WINS
      evidence missing    the gate pays for a human rather than filing an
                          unsubstantiated representment, and LOSES

    The second is the price of a safety property, deliberately bought. Reporting
    only the net would make a chosen trade-off look like a modelling failure.
    """
    def seg(name, subset):
        if not subset:
            return None
        priced = price_policies(subset, config)
        return {
            "n": len(subset),
            "chargeshield_inr_per_dispute": priced["chargeshield"]["per_dispute_inr"],
            "static_rule_inr_per_dispute": priced["static_amount_rule"]["per_dispute_inr"],
            "chargeshield_advantage_inr_per_dispute":
                priced["static_amount_rule"]["per_dispute_inr"]
                - priced["chargeshield"]["per_dispute_inr"],
        }

    cap = config.auto_action_amount_cap_inr
    complete = [x for x in decisions if x[0]["_chargeshield"]["evidence_complete"]]
    incomplete = [x for x in decisions if not x[0]["_chargeshield"]["evidence_complete"]]
    actionable = [x for x in complete if x[0]["_chargeshield"]["amount_inr"] <= cap]

    reviews = sum(1 for _, d in decisions if d.action == "HUMAN_REVIEW")
    return {
        "all_disputes": seg("all", decisions),
        "evidence_complete": seg("complete", complete),
        "evidence_incomplete": seg("incomplete", incomplete),
        "actionable_complete_and_under_cap": seg("actionable", actionable),
        "human_review_overhead": {
            "n_reviews": reviews,
            "cost_per_review_inr": config.human_review_cost_inr,
            "total_inr": reviews * config.human_review_cost_inr,
            "inr_per_dispute_across_queue":
                reviews * config.human_review_cost_inr / len(decisions),
        },
    }


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
        "policy_comparison": price_policies(decisions, config),
        "policy_comparison_by_segment": policy_comparison_by_segment(decisions, config),
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
            "ACROSS THE WHOLE QUEUE ChargeShield is slightly WORSE than the "
            "static amount rule. The aggregate hides two opposite effects: on "
            "actionable disputes (evidence on file, under the amount cap) it "
            "wins, and on disputes with missing evidence it loses because it "
            "pays for a human rather than filing an unsubstantiated "
            "representment. The second is a safety property bought on purpose. "
            "The share of disputes with incomplete evidence is a PARAMETER "
            "(p_required_evidence_present), not an observation, so the net "
            "figure moves with a dial rather than with the model.",
            "policy_comparison prices the REAL policy engine on the dispute "
            "queue against real isFraud labels. ml/evaluate.py's comparison is "
            "a transaction-level global-threshold rule and is NOT this.",
            "Human review is modelled as an analyst choosing the cost-minimising "
            "action given the same p the model saw -- not an oracle. That is "
            "conservative for ChargeShield, which is the policy that routes to "
            "humans.",
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

    print(f"\npolicy comparison on the dispute queue (realised cost, real labels):")
    print(f"  {'policy':>20} {'INR/dispute':>12} {'vs defend-all':>14} {'vs static':>11}")
    print("  " + "-" * 60)
    for k, r in sorted(results["policy_comparison"].items(),
                       key=lambda kv: -kv[1]["per_dispute_inr"]):
        print(f"  {k:>20} {r['per_dispute_inr']:12,.0f} "
              f"{r['saving_vs_defend_all_inr_per_dispute']:14,.0f} "
              f"{r['saving_vs_static_rule_inr_per_dispute']:11,.0f}")

    seg = results["policy_comparison_by_segment"]
    print(f"\n  segment decomposition (ChargeShield minus static rule, INR/dispute):")
    for k in ("all_disputes", "evidence_complete", "evidence_incomplete",
              "actionable_complete_and_under_cap"):
        r = seg[k]
        if r:
            print(f"    {k:>36} n={r['n']:>5,}  "
                  f"{r['chargeshield_advantage_inr_per_dispute']:+8,.0f}")
    o = seg["human_review_overhead"]
    print(f"    {'human review overhead':>36} {o['n_reviews']:>7,} reviews  "
          f"{o['inr_per_dispute_across_queue']:+8,.0f} /dispute")

    e = results["evidence_complete_subset"]
    print(f"\naction mix where evidence is complete (n={e['n']:,}) — "
          f"the model-driven slice:")
    for a, s in sorted(e["action_share"].items(), key=lambda kv: -kv[1]):
        print(f"  {a:>13}: {s:6.1%}")
    print(f"\nWrote {OUT_PATH} and {audit.path} ({len(audit.read_all()):,} records)")


if __name__ == "__main__":
    main()
