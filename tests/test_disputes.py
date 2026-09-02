"""Dispute anchoring integrity and Razorpay schema conformance.

Two things must hold or the dispute layer is worthless:

  1. Every dispute anchors to a REAL held-out transaction and carries that
     transaction's REAL isFraud label. Otherwise "wasted representment effort"
     is measured against nothing.
  2. Nothing this project invented sits inside the Razorpay-shaped body. The
     whole defensibility of "mirrors Razorpay's documented dispute entity"
     collapses the moment a made-up field is smuggled in next to a real one.
"""
import json
import pathlib

import pandas as pd
import pytest

DISPUTES = pathlib.Path("data/processed/disputes.json")
PROCESSED = pathlib.Path("data/processed")
PREDS = pathlib.Path("evaluation/preds")

# From https://razorpay.com/docs/api/disputes/entity/ (retrieved 2026-09-02).
# If a field appears in the output that is not here, either the docs changed or
# something was invented.
DOCUMENTED_TOP_LEVEL = {
    "id", "entity", "payment_id", "amount", "currency", "amount_deducted",
    "reason_code", "reason_description", "respond_by", "status", "phase",
    "created_at", "evidence",
}
DOCUMENTED_EVIDENCE_FIELDS = {
    "amount", "summary", "shipping_proof", "billing_proof", "cancellation_proof",
    "customer_communication", "proof_of_service", "explanation_letter",
    "refund_confirmation", "access_activity_log", "refund_cancellation_policy",
    "term_and_conditions", "others", "submitted_at",
}
DOCUMENTED_STATUS = {"open", "under_review", "won", "lost", "closed"}
DOCUMENTED_PHASE = {"fraud", "retrieval", "chargeback", "pre_arbitration", "arbitration"}

NAMESPACE = "_chargeshield"


@pytest.fixture(scope="module")
def payload():
    if not DISPUTES.exists():
        pytest.skip("run `make link` first")
    return json.loads(DISPUTES.read_text())


@pytest.fixture(scope="module")
def disputes(payload):
    return payload["disputes"]


@pytest.fixture(scope="module")
def test_split():
    p = PROCESSED / "test.parquet"
    if not p.exists():
        pytest.skip("run `make data` first")
    return pd.read_parquet(p, columns=["TransactionID", "isFraud", "TransactionAmt"])


# --- anchoring ------------------------------------------------------------

def test_every_dispute_anchors_to_a_real_held_out_transaction(disputes, test_split):
    valid = set(test_split["TransactionID"].astype(int))
    anchors = {d[NAMESPACE]["anchor_transaction_id"] for d in disputes}
    assert anchors <= valid, (
        f"{len(anchors - valid)} disputes anchor to transactions that are not "
        "in the held-out test split"
    )


def test_anchors_never_come_from_train_or_val(disputes):
    """The leakage guard. A dispute anchored to a training transaction would
    carry a label the model has already seen."""
    for split in ("train", "val"):
        p = PROCESSED / f"{split}.parquet"
        if not p.exists():
            pytest.skip("run `make data` first")
        ids = set(pd.read_parquet(p, columns=["TransactionID"])["TransactionID"].astype(int))
        anchors = {d[NAMESPACE]["anchor_transaction_id"] for d in disputes}
        assert not (anchors & ids), f"{len(anchors & ids)} disputes anchor into {split}"


def test_anchor_labels_match_the_real_labels(disputes, test_split):
    """The label carried on the dispute must BE the transaction's label, not a
    copy that drifted."""
    truth = dict(zip(test_split["TransactionID"].astype(int),
                     test_split["isFraud"].astype(int)))
    for d in disputes:
        cs = d[NAMESPACE]
        assert cs["anchor_is_fraud"] == truth[cs["anchor_transaction_id"]], (
            f"dispute {d['id']} carries isFraud={cs['anchor_is_fraud']} but "
            f"transaction {cs['anchor_transaction_id']} is "
            f"{truth[cs['anchor_transaction_id']]}"
        )


def test_anchor_scores_match_the_saved_predictions(disputes):
    p = PREDS / "lightgbm_test_calibrated.parquet"
    if not p.exists():
        pytest.skip("run `make train` and `make calibrate` first")
    preds = pd.read_parquet(p)
    truth = dict(zip(preds["TransactionID"].astype(int),
                     preds["p_fraud_calibrated"].astype(float)))
    for d in disputes[:500]:
        cs = d[NAMESPACE]
        assert cs["p_fraud_calibrated"] == pytest.approx(
            truth[cs["anchor_transaction_id"]], rel=1e-6)


def test_one_dispute_per_transaction(disputes):
    anchors = [d[NAMESPACE]["anchor_transaction_id"] for d in disputes]
    assert len(anchors) == len(set(anchors))


def test_dispute_ids_are_unique(disputes):
    ids = [d["id"] for d in disputes]
    assert len(ids) == len(set(ids))


# --- schema conformance ---------------------------------------------------

def test_no_invented_fields_in_the_razorpay_body(disputes):
    """The claim is 'mirrors the documented dispute entity'. One smuggled field
    and that claim is false."""
    for d in disputes:
        body = set(d) - {NAMESPACE}
        undocumented = body - DOCUMENTED_TOP_LEVEL
        assert not undocumented, f"undocumented top-level field(s): {undocumented}"


def test_all_constructed_data_is_namespaced(disputes):
    for d in disputes:
        assert NAMESPACE in d, "constructed data must live under a namespace"
        assert isinstance(d[NAMESPACE], dict)


def test_evidence_keys_are_all_documented_razorpay_fields(disputes):
    for d in disputes:
        undocumented = set(d["evidence"]) - DOCUMENTED_EVIDENCE_FIELDS
        assert not undocumented, f"invented evidence field(s): {undocumented}"


def test_status_and_phase_use_documented_vocabularies(disputes):
    for d in disputes:
        assert d["status"] in DOCUMENTED_STATUS
        assert d["phase"] in DOCUMENTED_PHASE
        assert d["entity"] == "dispute"
        assert d["currency"] == "INR"


def test_amount_is_in_currency_subunits(disputes):
    """Razorpay amounts are integers in subunits (paise), not rupees. Getting
    this wrong understates every figure by 100x."""
    for d in disputes:
        assert isinstance(d["amount"], int)
        assert d["amount"] == round(d[NAMESPACE]["amount_inr"] * 100)


def test_respond_by_is_after_created_at(disputes):
    for d in disputes:
        assert d["respond_by"] > d["created_at"]


# --- evidence bookkeeping -------------------------------------------------

def test_missing_evidence_list_is_consistent_with_the_evidence_object(disputes):
    """The gate acts on `missing_required_evidence`. If it disagrees with what
    is actually in `evidence`, every gate decision is built on a lie."""
    reqs = json.loads(pathlib.Path("evidence/requirements.json").read_text())
    for d in disputes:
        required = set(reqs[d["reason_code"]]["required"])
        actually_missing = required - set(d["evidence"])
        assert set(d[NAMESPACE]["missing_required_evidence"]) == actually_missing
        assert d[NAMESPACE]["evidence_complete"] == (not actually_missing)


def test_both_complete_and_incomplete_evidence_sets_exist(disputes):
    """A gate that never fires and a gate that always fires are both untested."""
    complete = sum(d[NAMESPACE]["evidence_complete"] for d in disputes)
    assert 0 < complete < len(disputes)


# --- the generative assumptions are recorded ------------------------------

def test_generative_assumptions_are_recorded_in_the_output(payload):
    """These are the numbers a reviewer should argue with. If they are not in
    the artifact, the queue composition is unfalsifiable."""
    meta = payload["_meta"]
    g = meta["generative_assumptions"]
    for k in ("p_dispute_given_fraud", "p_dispute_given_legitimate",
              "p_required_evidence_present", "random_state"):
        assert k in g
    assert "claim_limit" in meta
    assert "win rate" in meta["claim_limit"].lower()
    assert 0.0 < meta["queue_fraud_rate"] < 1.0


def test_queue_is_far_more_adverse_than_the_transaction_population(disputes, test_split):
    """The entire point of generating a queue rather than sampling at random."""
    queue_rate = sum(d[NAMESPACE]["anchor_is_fraud"] for d in disputes) / len(disputes)
    assert queue_rate > 5 * test_split["isFraud"].mean()
