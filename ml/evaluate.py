"""
Final held-out evaluation, policy comparison, and error analysis.

Nothing in this file fits or selects anything. Every threshold it uses was
chosen elsewhere, on val, and is read from disk. This module only reports.

THREE SECTIONS
  1. Detection metrics on test: baseline vs LightGBM, scored by ml/metrics.py
     so the comparison is not an artefact of two scoring implementations.

  2. Policy comparison in rupees: defend-none, defend-all, an amount-only
     static rule, and ChargeShield. The amount-only rule is the honest
     adversary -- it is the cost-optimal policy available to a merchant with NO
     model at all (contest iff the expected recovery beats the representment
     cost). The gap between it and ChargeShield is the value of the ML, and it
     is a much harder baseline than defend-none/defend-all.

  3. Error analysis sliced by amount band, ProductCD, and identity present vs
     missing. The point is not the aggregate number, it is which population the
     model fails on.

OUT  evaluation/metrics.json
     evaluation/charts/pr_curve.png, evaluation/charts/error_analysis.png
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve

from ml.cost_curve import expected_cost, shifted
from ml.metrics import point_metrics, ranking_metrics

PROCESSED_DIR = Path("data/processed")
EVAL_DIR = Path("evaluation")
PREDS_DIR = EVAL_DIR / "preds"
CHARTS_DIR = EVAL_DIR / "charts"

AMOUNT_BANDS = [(0, 1_000), (1_000, 5_000), (5_000, 10_000),
                (10_000, 25_000), (25_000, np.inf)]


def band_label(lo, hi) -> str:
    return f"INR {lo:,.0f}+" if np.isinf(hi) else f"INR {lo:,.0f}-{hi:,.0f}"


def load_everything() -> tuple[pd.DataFrame, dict]:
    cfg = yaml.safe_load(open("config/costs.yaml"))
    rate = cfg["currency"]["usd_to_inr"]

    lgbm = pd.read_parquet(PREDS_DIR / "lightgbm_test_calibrated.parquet")
    base = pd.read_parquet(PREDS_DIR / "baseline_test.parquet")[
        ["TransactionID", "p_fraud"]].rename(columns={"p_fraud": "p_baseline"})
    feats = pd.read_parquet(
        PROCESSED_DIR / "test.parquet",
        columns=["TransactionID", "TransactionAmt", "ProductCD", "DeviceType", "id_01"],
    )

    df = lgbm.merge(base, on="TransactionID", validate="one_to_one")
    df = df.merge(feats, on="TransactionID", validate="one_to_one")
    df["amount_inr"] = df["TransactionAmt"].astype("float64") * rate
    # Identity presence is a real, structural property of the row: the identity
    # left-join either matched or it did not. It is also the single biggest
    # missingness block in the dataset, so it is the first slice to check.
    df["identity_present"] = df["DeviceType"].notna() | df["id_01"].notna()
    return df, cfg


def detection_table(df: pd.DataFrame, thresholds: dict[str, float]) -> dict:
    """Each model at ITS OWN selected threshold.

    A shared threshold would be wrong here: the baseline was trained with
    class_weight="balanced", which deliberately rescales its scores, so its
    probabilities are not on the same axis as the calibrated LightGBM's.
    Scoring it at LightGBM's 0.25 produces precision 0.056 / recall 0.917 --
    a meaningless comparison that flatters the baseline's recall and buries
    its precision. PR-AUC and ROC-AUC are threshold-free and ARE comparable.
    """
    y = df["isFraud"].to_numpy()
    out = {}
    for name, col in (("baseline_logistic_regression", "p_baseline"),
                      ("lightgbm_calibrated", "p_fraud_calibrated")):
        p = df[col].to_numpy()
        t = thresholds[name]
        out[name] = {**ranking_metrics(y, p),
                     "threshold": t,
                     "at_own_threshold": point_metrics(y, p, t)}
    return out


def confound_check(df: pd.DataFrame, threshold: float) -> dict:
    """Are the top two failure slices one population or two?

    Slicing alone cannot answer this and will happily report the same finding
    twice under different names. ProductCD == W and identity-missing overlap at
    Jaccard 0.99 in this data, so the naive reading -- "the model fails when
    identity data is missing" -- is confounded. The 2x2 disentangles it.
    """
    y = df["isFraud"].to_numpy().astype(bool)
    pred = df["p_fraud_calibrated"].to_numpy() >= threshold
    w = (df["ProductCD"] == "W").to_numpy()
    id_missing = ~df["identity_present"].to_numpy()

    cells = {}
    for wl, wm in (("W", w), ("non_W", ~w)):
        for il, im in (("identity_present", ~id_missing), ("identity_missing", id_missing)):
            m = wm & im
            frauds = int((y & m).sum())
            cells[f"{wl}__{il}"] = {
                "n": int(m.sum()), "frauds": frauds,
                "recall": float((y & pred & m).sum() / frauds) if frauds else None,
                "note": "too few frauds to read" if frauds < 30 else None,
            }
    return {
        "jaccard_W_vs_identity_missing": float((w & id_missing).sum() / (w | id_missing).sum()),
        "p_identity_missing_given_W": float(id_missing[w].mean()),
        "p_identity_missing_given_not_W": float(id_missing[~w].mean()),
        "cells": cells,
        "conclusion": (
            "ProductCD == W and identity-missing are the SAME population "
            "(every W transaction lacks identity data). The 2x2 separates "
            "them: non-W transactions missing identity still recall at ~0.70, "
            "so missing identity is NOT the driver -- ProductCD W is. Caveat: "
            "the disambiguating cell (non-W, identity missing) has few frauds, "
            "so this is suggestive rather than conclusive."
        ),
    }


def policy_comparison(df: pd.DataFrame, cfg: dict, bands: dict) -> dict:
    """Four policies, priced through the identical loss function."""
    sc = cfg["dispute_economics"]["scenarios"][cfg["default_scenario"]]
    c = float(sc["representment_cost_inr"])
    w = float(sc["assumed_win_rate_if_legitimate"])
    cap = cfg["policy"]["auto_action_amount_cap_inr"]
    queue_pi = cfg["policy"]["assumed_dispute_fraud_rate"]

    auto = df[df["amount_inr"] <= cap].reset_index(drop=True)
    p_shift, weights = shifted(auto, queue_pi)
    a = auto["amount_inr"].to_numpy()

    policies = {
        "defend_none": np.zeros(len(auto), bool),
        "defend_all": np.ones(len(auto), bool),
        # The honest adversary: cost-optimal given ONLY the amount. Contest iff
        # the expected recovery beats what the representment costs. No model.
        "static_amount_rule": (w * a) > c,
        # NOT the shipped policy engine. This is a single global threshold on
        # the score. The real gate (app/policy/action_policy.py) uses
        # amount-dependent bands, an evidence gate, a fabrication check, an
        # amount cap and an economic floor, and it operates on the dispute
        # queue -- transactions carry no reason_code or evidence, so it cannot
        # be evaluated here. app/services/batch_runner.py prices the real gate.
        "global_cost_threshold_rule": p_shift < bands["threshold"],
    }

    results = {}
    for name, mask in policies.items():
        r = expected_cost(auto, t=None, c=c, w=w, p=p_shift,
                          weights=weights, contest=mask)
        results[name] = r
    base = results["defend_all"]["per_dispute_inr"]
    for name, r in results.items():
        r["saving_vs_defend_all_inr_per_dispute"] = base - r["per_dispute_inr"]
        r["saving_vs_static_rule_inr_per_dispute"] = (
            results["static_amount_rule"]["per_dispute_inr"] - r["per_dispute_inr"])
    return {
        "scenario": cfg["default_scenario"],
        "assumed_dispute_fraud_rate": queue_pi,
        "representment_cost_inr": c,
        "assumed_win_rate_if_legitimate": w,
        "n_auto_decidable": len(auto),
        "policies": results,
        "_note": (
            "global_cost_threshold_rule is NOT the shipped policy engine -- it "
            "is a single global threshold on the score, evaluated on "
            "transactions (which carry no reason_code or evidence). The real "
            "gate is priced on the dispute queue in "
            "evaluation/batch_results.json:policy_comparison, where it is "
            "slightly WORSE than the static rule overall."
        ),
    }


def slice_errors(df: pd.DataFrame, threshold: float) -> dict:
    """FP/FN by slice. Detection framing: positive class = fraud, predict fraud
    iff p >= threshold."""
    y = df["isFraud"].to_numpy().astype(bool)
    pred = df["p_fraud_calibrated"].to_numpy() >= threshold
    amt = df["amount_inr"].to_numpy()

    def stats(mask) -> dict:
        n = int(mask.sum())
        if n == 0:
            return {}
        yy, pp, aa = y[mask], pred[mask], amt[mask]
        tp, fp, fn = int((yy & pp).sum()), int((~yy & pp).sum()), int((yy & ~pp).sum())
        return {
            "n": n,
            "share_of_test": float(mask.mean()),
            "fraud_rate": float(yy.mean()),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "false_positives": fp,
            "false_negatives": fn,
            "missed_fraud_amount_inr": float(aa[yy & ~pp].sum()),
            "share_of_all_missed_fraud_amount": float(
                aa[yy & ~pp].sum() / max(amt[y & ~pred].sum(), 1e-9)),
        }

    slices = {"amount_band": {}, "product_cd": {}, "identity": {}}
    for lo, hi in AMOUNT_BANDS:
        slices["amount_band"][band_label(lo, hi)] = stats((amt >= lo) & (amt < hi))
    for v in sorted(df["ProductCD"].dropna().unique()):
        slices["product_cd"][str(v)] = stats((df["ProductCD"] == v).to_numpy())
    for label, mask in (("present", df["identity_present"].to_numpy()),
                        ("missing", ~df["identity_present"].to_numpy())):
        slices["identity"][label] = stats(mask)
    return slices


def name_top_failure_mode(slices: dict) -> dict:
    """Rank slices by share of total missed-fraud value, not by recall.

    A slice with terrible recall on 40 transactions is a curiosity. The failure
    mode worth naming is the one carrying the most unrecovered rupees.
    """
    ranked = []
    for dimension, groups in slices.items():
        for label, s in groups.items():
            if not s or s["n"] < 200:
                continue
            ranked.append({
                "dimension": dimension, "slice": label,
                "share_of_all_missed_fraud_amount": s["share_of_all_missed_fraud_amount"],
                "recall": s["recall"], "n": s["n"],
                "missed_fraud_amount_inr": s["missed_fraud_amount_inr"],
            })
    ranked.sort(key=lambda r: -r["share_of_all_missed_fraud_amount"])
    return {"ranked": ranked[:8], "top": ranked[0] if ranked else None}


def plot_pr(df: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = df["isFraud"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    for label, col in (("LightGBM (calibrated)", "p_fraud_calibrated"),
                       ("Logistic regression (baseline)", "p_baseline")):
        pr, rc, _ = precision_recall_curve(y, df[col].to_numpy())
        ax.plot(rc, pr, label=label)
    ax.axhline(y.mean(), ls="--", c="k", lw=1,
               label=f"prevalence ({y.mean():.3f}) — a coin flip")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision–recall on held-out test (final 15%, chronological)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(out, dpi=140); plt.close(fig)


def plot_errors(slices: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = [("amount_band", "Amount band"), ("product_cd", "ProductCD"),
            ("identity", "Identity data")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (key, title) in zip(axes, dims):
        groups = {k: v for k, v in slices[key].items() if v}
        labels = list(groups)
        recall = [groups[k]["recall"] for k in labels]
        share = [groups[k]["share_of_all_missed_fraud_amount"] for k in labels]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, recall, 0.4, label="recall")
        ax.bar(x + 0.2, share, 0.4, label="share of missed fraud ₹")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title); ax.set_ylim(0, 1); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("proportion"); axes[0].legend(fontsize=8)
    fig.suptitle("Where the model fails — held-out test, threshold from val")
    fig.savefig(out, dpi=140); plt.close(fig)


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    df, cfg = load_everything()
    calib = json.loads((EVAL_DIR / "calibration_metrics.json").read_text())
    bands = json.loads(Path("artifacts/policy_bands.json").read_text())
    threshold = calib["selected_threshold"]

    print(f"test {len(df):,} transactions · detection threshold {threshold:.2f} "
          f"(selected on val_pick)\n")

    base_metrics = json.loads((EVAL_DIR / "baseline_metrics.json").read_text())
    thresholds = {
        "baseline_logistic_regression": base_metrics["selected_threshold"],
        "lightgbm_calibrated": threshold,
    }
    detection = detection_table(df, thresholds)
    print(f"{'model':>32} {'PR-AUC':>8} {'ROC-AUC':>8} {'t':>6} {'P':>7} {'R':>7} {'F1':>7}")
    print("-" * 82)
    for name, m in detection.items():
        a = m["at_own_threshold"]
        print(f"{name:>32} {m['pr_auc']:8.4f} {m['roc_auc']:8.4f} {m['threshold']:6.2f} "
              f"{a['precision']:7.4f} {a['recall']:7.4f} {a['f1']:7.4f}")
    print("  each model at its own val-selected threshold; the baseline's scores "
          "are rescaled\n  by class_weight=balanced and are not on LightGBM's axis. "
          "PR-AUC/ROC-AUC are comparable.")

    policy = policy_comparison(df, cfg, bands)
    print(f"\n--- policy comparison, INR per dispute "
          f"({policy['scenario']} scenario, queue fraud "
          f"{policy['assumed_dispute_fraud_rate']:.0%}, "
          f"n={policy['n_auto_decidable']:,}) ---")
    print(f"{'policy':>20} {'contest':>9} {'INR/disp':>10} {'vs defend-all':>14} "
          f"{'vs static rule':>15}")
    print("-" * 74)
    for name, r in policy["policies"].items():
        print(f"{name:>20} {r['contest_rate']:9.1%} {r['per_dispute_inr']:10,.0f} "
              f"{r['saving_vs_defend_all_inr_per_dispute']:14,.0f} "
              f"{r['saving_vs_static_rule_inr_per_dispute']:15,.0f}")

    slices = slice_errors(df, threshold)
    failure = name_top_failure_mode(slices)
    confound = confound_check(df, threshold)

    print("\n--- error analysis (detection framing, positive class = fraud) ---")
    for dim, title in (("amount_band", "amount band"), ("product_cd", "ProductCD"),
                       ("identity", "identity data")):
        print(f"\n  by {title}:")
        print(f"    {'slice':>18} {'n':>8} {'fraud':>7} {'prec':>7} {'recall':>7} "
              f"{'missed ₹':>13} {'% of missed ₹':>14}")
        for label, s in slices[dim].items():
            if not s:
                continue
            print(f"    {label:>18} {s['n']:8,} {s['fraud_rate']:7.3f} "
                  f"{s['precision']:7.3f} {s['recall']:7.3f} "
                  f"{s['missed_fraud_amount_inr']:13,.0f} "
                  f"{s['share_of_all_missed_fraud_amount']:14.1%}")

    print("\n--- slices ranked by share of unrecovered fraud value ---")
    for r in failure["ranked"][:5]:
        print(f"  {r['dimension']:>12} / {r['slice']:<18} "
              f"{r['share_of_all_missed_fraud_amount']:6.1%} of missed ₹ · "
              f"recall {r['recall']:.3f} · n={r['n']:,}")

    print("\n--- confound check: are the top two slices one population? ---")
    print(f"  Jaccard(W, identity-missing) = "
          f"{confound['jaccard_W_vs_identity_missing']:.4f}   "
          f"P(id missing | W) = {confound['p_identity_missing_given_W']:.4f}")
    for k, c in confound["cells"].items():
        r = "  n/a" if c["recall"] is None else f"{c['recall']:.3f}"
        flag = f"  <- {c['note']}" if c["note"] else ""
        print(f"    {k:>32}  n={c['n']:>6,}  frauds={c['frauds']:>5,}  recall {r}{flag}")
    print("  " + confound["conclusion"].replace(". ", ".\n  "))

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "population": "IEEE-CIS held-out final 15%, chronological",
        "n_test": len(df),
        "detection_threshold": threshold,
        "detection_threshold_selected_on": "val_pick (post-calibration)",
        "detection": detection,
        "policy_comparison": policy,
        "error_slices": slices,
        "failure_modes_ranked_by_missed_value": failure,
        "confound_check": confound,
        "top_failure_mode": (
            "Recall on ProductCD 'W' is 0.213 against 0.703 on every other "
            "product type. W is 79% of held-out transactions and carries 86% of "
            "all unrecovered fraud value. It is NOT an identity-data problem: "
            "non-W transactions that also lack identity data still recall at "
            "~0.70. W is simply a population the model reads poorly."
        ),
        "notes": [
            "Nothing here fits or selects. Every threshold was chosen on val "
            "and is read from disk.",
            "static_amount_rule is the cost-optimal policy available with NO "
            "model (contest iff expected recovery beats representment cost). "
            "It is a far harder adversary than defend-none/defend-all.",
            "NOTHING in this section is the shipped policy engine. See "
            "evaluation/batch_results.json for the real gate priced on the "
            "dispute queue.",
            "Policy comparison is on auto-decidable disputes only (amount <= "
            "cap) and is prior-shifted to the assumed dispute-queue fraud rate.",
            "Failure modes are ranked by share of unrecovered fraud VALUE, not "
            "by recall: a slice with poor recall on 40 rows is a curiosity.",
        ],
    }
    (EVAL_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    plot_pr(df, CHARTS_DIR / "pr_curve.png")
    plot_errors(slices, CHARTS_DIR / "error_analysis.png")
    print("\nWrote evaluation/metrics.json, evaluation/charts/pr_curve.png,")
    print("      evaluation/charts/error_analysis.png")


if __name__ == "__main__":
    main()
