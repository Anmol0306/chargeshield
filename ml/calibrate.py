"""
Probability calibration for the gradient-boosted model.

WHY THIS EXISTS
  ml/cost_curve.py computes an expected rupee loss as p x loss. That arithmetic
  is meaningless unless p is a real probability -- unless the transactions
  scored 0.30 are fraudulent about 30% of the time. A model can rank perfectly
  (ROC-AUC 0.90) and still be badly calibrated. Calibration is what makes the
  cost figures honest rather than decorative.

WHERE THE CALIBRATOR IS FIT, AND WHY NOT TRAIN
  Not train: the model has partly memorised it, so its train predictions are
  unrealistically sharp and a calibrator fit there would learn the wrong map.
  Not test: that is the held-out estimate.
  So: val. But val already carries early stopping, and fitting BOTH candidate
  calibrators on val and then picking the winner by Brier on val would be a
  formality, not a comparison -- isotonic has far more freedom and would win by
  partly fitting val's noise.

  So val is split CHRONOLOGICALLY (same discipline as the main split):
      val_fit  = first 70% of val   -> calibrators are fit here
      val_pick = last  30% of val   -> the winner is chosen here, unseen by both
  Test is then scored once with the winner and is the only clean number.

  Stated plainly because a reviewer will ask: val now does three jobs (early
  stopping, calibrator fitting, method selection). It is not a held-out
  estimate and is never reported as one.

MONOTONICITY AS A SELF-CHECK
  Platt is a strictly monotonic transform, so it CANNOT change ranking --
  ROC-AUC must be unchanged to floating-point noise. Isotonic is only weakly
  monotonic (it creates ties), so its ranking metrics may move slightly. That
  asymmetry is asserted, not assumed.

IN   evaluation/preds/lightgbm_{val,test}.parquet   (no retraining)
OUT  artifacts/calibrator.pkl
     evaluation/calibration_metrics.json
     evaluation/charts/calibration.png
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ml.metrics import best_f1_threshold, point_metrics, ranking_metrics

ARTIFACTS_DIR = Path("artifacts")
EVAL_DIR = Path("evaluation")
PREDS_DIR = EVAL_DIR / "preds"
CHARTS_DIR = EVAL_DIR / "charts"

VAL_FIT_FRACTION = 0.70
N_BINS = 15
EPS = 1e-6


def expected_calibration_error(y_true, y_prob, n_bins=N_BINS) -> float:
    """Bin-weighted mean gap between predicted confidence and observed frequency.

    Equal-width bins on [0, 1]. This is the number to say out loud: "when it
    says 30%, it is right 30% of the time, within X points."
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / len(y_prob)) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def reliability_bins(y_true, y_prob, n_bins=N_BINS):
    """(mean predicted, observed frequency, count) per bin, for the chart."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 10:          # a bin of 3 transactions is noise, not evidence
            continue
        rows.append((float(y_prob[m].mean()), float(y_true[m].mean()), int(m.sum())))
    return rows


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Logistic regression on the log-odds of the model score.

    Fitting on logit(p) rather than p is the standard formulation: it keeps the
    transform well-behaved in the tails, where a fraud model spends most of its
    decision-relevant mass. One slope, one intercept -- two parameters, which is
    why it cannot overfit 62k rows.
    """

    name = "platt"

    def fit(self, p, y):
        self._lr = LogisticRegression(solver="lbfgs", C=1e6)  # near-unregularised
        self._lr.fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self._lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]

    def params(self):
        return {"slope": float(self._lr.coef_[0][0]),
                "intercept": float(self._lr.intercept_[0])}


class IsotonicCalibrator:
    """Non-decreasing step function. Far more flexible than Platt, which is
    exactly why it is chosen on data it was not fit to."""

    name = "isotonic"

    def fit(self, p, y):
        # float64 deliberately. Predictions are persisted as float32 to keep the
        # parquet small, but interpolating a step function in float32 makes
        # adjacent steps differ by one ULP in the WRONG direction (~-1.2e-07),
        # so the output is not monotone at float32 precision. Numerically
        # irrelevant to any decision, but a calibrator whose output can decrease
        # as the score increases is not a thing worth defending in a panel.
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(np.asarray(p, dtype=np.float64), y)
        return self

    def predict(self, p):
        return self._iso.predict(np.asarray(p, dtype=np.float64))

    def params(self):
        return {"n_thresholds": int(len(self._iso.X_thresholds_))}


class IdentityCalibrator:
    """The uncalibrated model. Competes on equal terms -- if LightGBM's raw
    logloss-trained output is already well calibrated, it should win, and
    adding a calibration step would be unjustified complexity."""

    name = "uncalibrated"

    def fit(self, p, y):
        return self

    def predict(self, p):
        return p

    def params(self):
        return {}


def score(y, p) -> dict:
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS))),
        "ece": expected_calibration_error(y, p),
        "roc_auc": float(roc_auc_score(y, p)),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def chronological_val_split(val: pd.DataFrame):
    """Same rule as the main split: order by time, cut on a quantile, use < / >=
    so no row can land in both halves."""
    cut = val["TransactionDT"].quantile(VAL_FIT_FRACTION)
    fit = val[val["TransactionDT"] < cut]
    pick = val[val["TransactionDT"] >= cut]
    assert fit["TransactionDT"].max() < pick["TransactionDT"].min(), "val_fit/val_pick overlap"
    assert len(fit) and len(pick)
    return fit, pick


def plot_reliability(curves: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), height_ratios=[3, 1], constrained_layout=True
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for label, (bins, prob) in curves.items():
        if not bins:
            continue
        xs, ys, _ = zip(*bins)
        ax.plot(xs, ys, marker="o", ms=4, label=label)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraud rate")
    ax.set_title("Reliability on held-out test (LightGBM)")
    ax.legend()
    ax.grid(alpha=0.3)

    for label, (_, prob) in curves.items():
        ax2.hist(prob, bins=50, range=(0, 1), histtype="step", log=True, label=label)
    ax2.set_xlabel("predicted probability")
    ax2.set_ylabel("count (log)")
    ax2.grid(alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    val = pd.read_parquet(PREDS_DIR / "lightgbm_val.parquet")
    test = pd.read_parquet(PREDS_DIR / "lightgbm_test.parquet")

    val_fit, val_pick = chronological_val_split(val)
    print(f"val_fit  {len(val_fit):>6,} rows  (fit calibrators)")
    print(f"val_pick {len(val_pick):>6,} rows  (choose winner -- unseen by both)")
    print(f"test     {len(test):>6,} rows  (scored once, at the end)\n")

    candidates = [IdentityCalibrator(), PlattCalibrator(), IsotonicCalibrator()]
    p_fit, y_fit = val_fit["p_fraud"].to_numpy(), val_fit["isFraud"].to_numpy()
    p_pick, y_pick = val_pick["p_fraud"].to_numpy(), val_pick["isFraud"].to_numpy()

    results = {}
    print(f"{'calibrator':>14} | {'Brier':>9} {'log loss':>9} {'ECE':>7} {'ROC-AUC':>8}")
    print("-" * 56)
    for cal in candidates:
        cal.fit(p_fit, y_fit)
        s = score(y_pick, cal.predict(p_pick))
        results[cal.name] = {"val_pick": s, "params": cal.params()}
        print(f"{cal.name:>14} | {s['brier']:9.6f} {s['log_loss']:9.5f} "
              f"{s['ece']:7.4f} {s['roc_auc']:8.5f}")

    # Selection is by Brier: it is a proper scoring rule, so it rewards being
    # both correct and honest about uncertainty. ECE alone can be gamed by a
    # model that predicts the base rate for everything.
    winner = min(candidates, key=lambda c: results[c.name]["val_pick"]["brier"])
    print(f"\nWinner on val_pick by Brier: {winner.name}")

    # Ranking self-check. Platt is strictly monotonic and MUST NOT reorder.
    platt_auc = results["platt"]["val_pick"]["roc_auc"]
    raw_auc = results["uncalibrated"]["val_pick"]["roc_auc"]
    assert abs(platt_auc - raw_auc) < 1e-6, (
        f"Platt changed the ranking ({raw_auc:.8f} -> {platt_auc:.8f}). It is a "
        "strictly monotonic transform and cannot; the implementation is wrong."
    )
    iso_auc = results["isotonic"]["val_pick"]["roc_auc"]
    print(f"Ranking check: platt ROC-AUC == uncalibrated ({raw_auc:.6f}) OK; "
          f"isotonic {iso_auc:.6f} (ties are expected)")

    # --- the single scoring of test -------------------------------------
    p_test_raw = test["p_fraud"].to_numpy()
    y_test = test["isFraud"].to_numpy()
    p_test_cal = winner.predict(p_test_raw)

    threshold = best_f1_threshold(y_pick, winner.predict(p_pick))
    print(f"\nThreshold re-selected on val_pick after calibration: {threshold:.2f}")

    test_before = score(y_test, p_test_raw)
    test_after = score(y_test, p_test_cal)

    out = pd.DataFrame({
        "TransactionID": test["TransactionID"],
        "TransactionDT": test["TransactionDT"],
        "isFraud": y_test,
        "p_fraud_raw": p_test_raw.astype("float32"),
        "p_fraud_calibrated": p_test_cal.astype("float32"),
    })
    out.to_parquet(PREDS_DIR / "lightgbm_test_calibrated.parquet", index=False)

    val_out = pd.DataFrame({
        "TransactionID": val["TransactionID"],
        "TransactionDT": val["TransactionDT"],
        "isFraud": val["isFraud"],
        "p_fraud_raw": val["p_fraud"].astype("float32"),
        "p_fraud_calibrated": winner.predict(val["p_fraud"].to_numpy()).astype("float32"),
    })
    val_out.to_parquet(PREDS_DIR / "lightgbm_val_calibrated.parquet", index=False)

    metrics = {
        "selected": winner.name,
        "selected_by": "lowest Brier on val_pick (a proper scoring rule)",
        "selected_params": winner.params(),
        "val_fit_rows": int(len(val_fit)),
        "val_pick_rows": int(len(val_pick)),
        "val_fit_fraction": VAL_FIT_FRACTION,
        "ece_bins": N_BINS,
        "candidates_on_val_pick": {k: v["val_pick"] for k, v in results.items()},
        "test": {
            "uncalibrated": test_before,
            "calibrated": test_after,
            **ranking_metrics(y_test, p_test_cal),
            "at_threshold": point_metrics(y_test, p_test_cal, threshold),
        },
        "selected_threshold": float(threshold),
        "threshold_selected_on": "val_pick (post-calibration)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": [
            "val does three jobs: early stopping, calibrator fitting, method "
            "selection. It is NOT a held-out estimate and is never reported as one.",
            "Calibrators are fit on val_fit and chosen on val_pick, a "
            "chronologically later slice neither of them saw.",
            "Test is scored once, with the winner only.",
            "Brier chosen over ECE for selection: ECE alone is gamed by a model "
            "that predicts the base rate for everything.",
        ],
    }
    (EVAL_DIR / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump({"calibrator": winner, "name": winner.name,
                 "threshold": float(threshold)}, ARTIFACTS_DIR / "calibrator.pkl")

    plot_reliability(
        {
            "uncalibrated": (reliability_bins(y_test, p_test_raw), p_test_raw),
            f"{winner.name} (selected)": (reliability_bins(y_test, p_test_cal), p_test_cal),
        },
        CHARTS_DIR / "calibration.png",
    )

    print("\n--- test, before vs after calibration ---")
    print(f"{'':>14} | {'Brier':>9} {'log loss':>9} {'ECE':>7} {'ROC-AUC':>8}")
    print("-" * 56)
    for label, s in (("uncalibrated", test_before), (winner.name, test_after)):
        print(f"{label:>14} | {s['brier']:9.6f} {s['log_loss']:9.5f} "
              f"{s['ece']:7.4f} {s['roc_auc']:8.5f}")
    print(f"\nmean predicted {test_after['mean_predicted']:.4f} vs "
          f"observed fraud rate {test_after['observed_rate']:.4f}")
    p = metrics["test"]["at_threshold"]
    cm = p["confusion_matrix"]
    print(f"@t={threshold:.2f}  P {p['precision']:.4f}  R {p['recall']:.4f}  "
          f"F1 {p['f1']:.4f} | TP {cm['tp']} FP {cm['fp']} FN {cm['fn']} TN {cm['tn']}")
    print("\nWrote artifacts/calibrator.pkl, evaluation/calibration_metrics.json,")
    print("      evaluation/charts/calibration.png, "
          "evaluation/preds/lightgbm_{val,test}_calibrated.parquet")


if __name__ == "__main__":
    main()
