"""
THE SPINE. Anchor each synthetic dispute to a real held-out TransactionID.

Each dispute carries the transaction's real isFraud label AND the model's
calibrated fraud probability. That is what connects the two halves and puts a
real label underneath a synthetic dispute layer.

WHAT IS REAL AND WHAT IS NOT
  Real      the anchor transaction, its isFraud label, its amount, and the
            model's calibrated score. All from the held-out final 15%.
  Documented the dispute entity's field names, status/phase vocabularies, and
            evidence field names -- from Razorpay's API reference (cited in
            evidence/requirements.json).
  Constructed everything else: which transactions get disputed, the reason-code
            vocabulary, and which evidence happens to be present.

  These are kept in separate namespaces in the output. Anything this project
  invented sits under `_chargeshield`, never inside the Razorpay-shaped body,
  so no reader can mistake a constructed field for a documented one.

HOW DISPUTES ARE SELECTED -- AND WHY NOT AT RANDOM
  A random sample of held-out transactions has a 3.5% fraud rate. ml/cost_curve.py
  showed that is degenerate: at 3.5%, contesting is positive-EV almost regardless
  of score and no threshold has content. But sampling to hit a target rate of 50%
  would be circular -- constructing the population to match the assumption being
  tested.

  So the queue is generated from a process with two named parameters:

      P(dispute | fraud)       = 0.80   most fraud eventually gets charged back
      P(dispute | legitimate)  = 0.03   friendly fraud, non-receipt, confusion

  and the queue composition FALLS OUT of them rather than being dialled in.
  With a 3.5% base rate that yields roughly 49% fraud in the dispute queue.

  That it lands near config/costs.yaml's assumed_dispute_fraud_rate of 0.50 is
  BY CONSTRUCTION, not by measurement -- these parameters were chosen to be
  plausible, and a reviewer who disagrees should argue with the two numbers
  above rather than with the conclusion. Both are stated in the output.

CLAIM LIMIT -- read before quoting anything from this file
  This enables exactly one dispute-side metric measured against real ground
  truth:

      wasted representment effort = disputes contested where isFraud == 1

  It does NOT measure win rate, money recovered, or dispute outcomes. Those
  require merchant-side resolution labels that are not public and are not in
  IEEE-CIS. The synthetic layer demonstrates POLICY BEHAVIOUR, never predictive
  performance.

OUT  data/processed/disputes.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROCESSED_DIR = Path("data/processed")
PREDS_DIR = Path("evaluation/preds")
REQUIREMENTS = Path("evidence/requirements.json")
OUT_PATH = PROCESSED_DIR / "disputes.json"

RANDOM_STATE = 42

# The two generative assumptions. Named, not buried.
P_DISPUTE_GIVEN_FRAUD = 0.80
P_DISPUTE_GIVEN_LEGITIMATE = 0.03

# Probability that any single REQUIRED evidence item happens to be on file.
# Independent per item, so a dispute needing two items is complete ~49% of the
# time. This is what gives the evidence gate something real to refuse.
P_REQUIRED_EVIDENCE_PRESENT = 0.70
P_OPTIONAL_EVIDENCE_PRESENT = 0.40

RESPOND_BY_DAYS = 7

# Reason codes conditional on the truth, which is the realistic part: a
# genuinely fraudulent transaction is usually disputed AS fraud, and a
# legitimate one is usually disputed for a delivery or service reason.
# CONSTRUCTED vocabulary -- see evidence/requirements.json.
REASONS_IF_FRAUD = {
    "FRAUD": 0.85,
    "SERVICE_NOT_RENDERED": 0.08,
    "NON_RECEIPT": 0.07,
}
REASONS_IF_LEGITIMATE = {
    "NON_RECEIPT": 0.34,
    "SERVICE_NOT_RENDERED": 0.24,
    "CREDIT_NOT_PROCESSED": 0.18,
    "SUBSCRIPTION_CANCELLED": 0.14,
    "FRAUD": 0.10,
}

REASON_DESCRIPTIONS = {
    "FRAUD": "Cardholder does not recognise the transaction",
    "NON_RECEIPT": "Goods or services not received",
    "SERVICE_NOT_RENDERED": "Service was not rendered as described",
    "SUBSCRIPTION_CANCELLED": "Recurring charge after cancellation",
    "CREDIT_NOT_PROCESSED": "Expected refund was not processed",
}

_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def load_requirements() -> dict:
    raw = json.loads(REQUIREMENTS.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def razorpay_id(prefix: str, rng: np.random.Generator) -> str:
    """Razorpay-style opaque identifier: prefix + 14 alphanumerics."""
    return prefix + "".join(rng.choice(list(_ALPHABET), size=14))


def choose(options: dict[str, float], rng: np.random.Generator) -> str:
    keys = list(options)
    return str(rng.choice(keys, p=[options[k] for k in keys]))


def build_evidence(reason: str, requirements: dict,
                   rng: np.random.Generator) -> tuple[dict, list[str]]:
    """Return (evidence object, list of required types that are missing).

    Evidence values are lists of document ids, matching the documented shape --
    Razorpay's evidence sub-fields are lists populated via the Documents API.
    An absent item is simply not a key, which is how a real partially-gathered
    evidence set looks.
    """
    spec = requirements[reason]
    evidence: dict = {}
    missing: list[str] = []

    for field in spec["required"]:
        if rng.random() < P_REQUIRED_EVIDENCE_PRESENT:
            evidence[field] = [razorpay_id("doc_", rng)]
        else:
            missing.append(field)

    for field in spec["optional"]:
        if rng.random() < P_OPTIONAL_EVIDENCE_PRESENT:
            evidence[field] = [razorpay_id("doc_", rng)]

    return evidence, missing


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    requirements = load_requirements()
    cfg = yaml.safe_load(open("config/costs.yaml"))
    rate = cfg["currency"]["usd_to_inr"]

    test = pd.read_parquet(
        PROCESSED_DIR / "test.parquet",
        columns=["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "ProductCD"],
    )
    scored = pd.read_parquet(PREDS_DIR / "lightgbm_test_calibrated.parquet")
    df = test.merge(scored[["TransactionID", "p_fraud_calibrated", "p_fraud_raw"]],
                    on="TransactionID", validate="one_to_one")

    # --- who gets disputed -------------------------------------------------
    y = df["isFraud"].to_numpy().astype(bool)
    p_dispute = np.where(y, P_DISPUTE_GIVEN_FRAUD, P_DISPUTE_GIVEN_LEGITIMATE)
    disputed = rng.random(len(df)) < p_dispute
    queue = df[disputed].reset_index(drop=True)

    queue_fraud_rate = float(queue["isFraud"].mean())
    print(f"held-out transactions : {len(df):,} (fraud {df['isFraud'].mean():.4f})")
    print(f"P(dispute | fraud)    : {P_DISPUTE_GIVEN_FRAUD}")
    print(f"P(dispute | legit)    : {P_DISPUTE_GIVEN_LEGITIMATE}")
    print(f"dispute queue         : {len(queue):,} disputes "
          f"(fraud {queue_fraud_rate:.4f})")
    print(f"  config assumed_dispute_fraud_rate = "
          f"{cfg['policy']['assumed_dispute_fraud_rate']} "
          f"-- matched BY CONSTRUCTION, not measured\n")

    now = int(datetime.now(timezone.utc).timestamp())
    disputes = []
    for row in queue.itertuples(index=False):
        is_fraud = bool(row.isFraud)
        reason = choose(REASONS_IF_FRAUD if is_fraud else REASONS_IF_LEGITIMATE, rng)
        evidence, missing = build_evidence(reason, requirements, rng)
        amount_inr = float(row.TransactionAmt) * rate

        disputes.append({
            # ---- Razorpay-shaped body. Field names and vocabularies are from
            # ---- the documented entity; the VALUES are constructed.
            "id": razorpay_id("disp_", rng),
            "entity": "dispute",
            "payment_id": razorpay_id("pay_", rng),
            "amount": int(round(amount_inr * 100)),   # currency subunits (paise)
            "currency": "INR",
            "amount_deducted": 0,
            "reason_code": reason,
            "reason_description": REASON_DESCRIPTIONS[reason],
            "respond_by": now + RESPOND_BY_DAYS * 86_400,
            "status": "open",
            "phase": requirements[reason]["phase"],
            "created_at": now,
            "evidence": evidence,

            # ---- Everything this project invented or derived lives here, so
            # ---- it can never be mistaken for a documented Razorpay field.
            "_chargeshield": {
                "anchor_transaction_id": int(row.TransactionID),
                "anchor_split": "test",
                "anchor_is_fraud": int(row.isFraud),        # REAL label
                "anchor_product_cd": str(row.ProductCD),
                "amount_inr": round(amount_inr, 2),
                "p_fraud_calibrated": float(row.p_fraud_calibrated),
                "p_fraud_raw": float(row.p_fraud_raw),
                "missing_required_evidence": missing,
                "evidence_complete": len(missing) == 0,
            },
        })

    complete = sum(d["_chargeshield"]["evidence_complete"] for d in disputes)
    print(f"evidence complete     : {complete:,} / {len(disputes):,} "
          f"({complete / len(disputes):.1%})")
    by_reason = pd.Series([d["reason_code"] for d in disputes]).value_counts()
    print("reason codes:")
    for k, v in by_reason.items():
        print(f"  {k:>22}: {v:>6,}")

    payload = {
        "_meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_disputes": len(disputes),
            "anchor_population": "IEEE-CIS held-out final 15%, chronological",
            "queue_fraud_rate": queue_fraud_rate,
            "generative_assumptions": {
                "p_dispute_given_fraud": P_DISPUTE_GIVEN_FRAUD,
                "p_dispute_given_legitimate": P_DISPUTE_GIVEN_LEGITIMATE,
                "p_required_evidence_present": P_REQUIRED_EVIDENCE_PRESENT,
                "p_optional_evidence_present": P_OPTIONAL_EVIDENCE_PRESENT,
                "random_state": RANDOM_STATE,
            },
            "schema_source": "https://razorpay.com/docs/api/disputes/entity/",
            "claim_limit": (
                "Anchors are real held-out transactions carrying real isFraud "
                "labels, so 'wasted representment effort' (contested where "
                "isFraud == 1) is measured against real ground truth. NOTHING "
                "else here is. This layer demonstrates policy behaviour, never "
                "predictive performance. No win rate, no money recovered, no "
                "dispute outcome is claimed or claimable -- those need "
                "merchant-side resolution labels that are not public."
            ),
            "constructed_not_documented": [
                "which transactions become disputes",
                "the reason_code vocabulary and its mapping to required evidence",
                "which evidence items are present",
                "all identifier values (disp_/pay_/doc_)",
            ],
        },
        "disputes": disputes,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
