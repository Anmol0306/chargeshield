"""
Gradient-boosted primary model.

The contrast with ml/train_baseline.py is the point. The baseline is a
deliberately constrained linear model: 46 input columns, high-cardinality
identifiers excluded, one-hot + median-imputed + scaled. This model receives
everything the baseline was denied — the 7 identifier columns AND the 339
V-columns, 392 features in total — with no one-hot and no imputation.

So the gap between the two is not "one algorithm is better". It measures what
the baseline's constraint cost, which is a defensible thing to have measured.

WHY NO IMPUTATION
  LightGBM learns a default direction for NaN at each split. Given that
  missingness here is structural (id_* missing exactly where the identity
  left-join found no row; D6-D14 missing in blocks), that is strictly better
  than replacing it with a median and pretending the value was observed.

WHY NO ONE-HOT
  Native categorical splits. See fit_categorical_dtypes for the trap this
  creates.

WHY NO scale_pos_weight
  The baseline needed class_weight="balanced" to produce non-degenerate point
  metrics at t=0.5, and paid for it with destroyed calibration. This model does
  not: the operating threshold is selected from the PR curve on val anyway, so
  re-weighting buys nothing and would damage the probability scale that
  ml/calibrate.py needs tomorrow. Natural prevalence is kept.

WHAT VAL MEANS HERE
  Early stopping reads val, so val is a SELECTION set, not a clean estimate.
  Its metrics are optimistic and are labelled as such in the output. Test is
  the number to quote.

OUT  artifacts/lightgbm.pkl
     artifacts/lightgbm_metadata.json
     evaluation/lightgbm_metrics.json
     evaluation/preds/lightgbm_{val,test}.parquet
"""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from ml.features import (
    LGBM_CATEGORICAL_COLS,
    TARGET_COL,
    V_COLS,
    lightgbm_feature_columns,
)
from ml.metrics import best_f1_threshold, point_metrics, ranking_metrics

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")
EVAL_DIR = Path("evaluation")
PREDS_DIR = EVAL_DIR / "preds"

RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 100

PARAMS = dict(
    objective="binary",
    n_estimators=3000,
    learning_rate=0.05,
    num_leaves=64,
    min_child_samples=100,   # 3.5% prevalence: don't let leaves chase 5 frauds
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,    # 392 features, many V-columns near-duplicated
    reg_lambda=1.0,
    max_bin=255,
    random_state=RANDOM_STATE,
    deterministic=True,      # reproducibility over throughput — this is a
    force_row_wise=True,     # submission, not a leaderboard run
    n_jobs=-1,
    verbose=-1,
)


def fit_categorical_dtypes(train: pd.DataFrame) -> dict[str, CategoricalDtype]:
    """Learn the category->code mapping on TRAIN ONLY.

    This is the LightGBM equivalent of Failure 01, and it is silent.

    Calling .astype("category") separately on each split makes pandas assign
    integer codes per frame, in order of appearance. "visa" could be code 3 in
    train and code 1 in val. LightGBM stores a split as "code in {1,3}", so the
    served model would be reading scrambled categories — no exception, no
    warning, just quietly degraded predictions.

    One dtype, learned on train, applied unchanged to val and test.

    Consequence worth stating: a category not present in train becomes NaN
    under this dtype, so unseen levels follow the model's learned missing
    branch. That conflates "absent" with "never seen before", which for a tree
    is an acceptable degradation and the direct analogue of the baseline's
    handle_unknown="ignore".
    """
    return {c: CategoricalDtype(categories=sorted(train[c].dropna().unique()))
            for c in LGBM_CATEGORICAL_COLS}


def apply_categorical_dtypes(
    df: pd.DataFrame, dtypes: dict[str, CategoricalDtype]
) -> pd.DataFrame:
    out = df.copy()
    for col, dtype in dtypes.items():
        out[col] = out[col].astype(dtype)
    return out


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")


def save_predictions(name: str, df: pd.DataFrame, y_prob: np.ndarray) -> Path:
    out = pd.DataFrame({
        "TransactionID": df["TransactionID"].to_numpy(),
        "TransactionDT": df["TransactionDT"].to_numpy(),
        "isFraud": df[TARGET_COL].to_numpy(),
        "p_fraud": y_prob.astype("float32"),
    })
    path = PREDS_DIR / f"lightgbm_{name}.parquet"
    out.to_parquet(path, index=False)
    return path


def main() -> None:
    for d in (ARTIFACTS_DIR, EVAL_DIR, PREDS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    feature_cols = lightgbm_feature_columns()
    train, val, test = (load_split(n) for n in ("train", "val", "test"))

    print(f"Features: {len(feature_cols)} "
          f"({len(V_COLS)} V-columns, {len(LGBM_CATEGORICAL_COLS)} categorical)")

    # The one and only fit of a learned preprocessing artifact, on train only.
    cat_dtypes = fit_categorical_dtypes(train)
    train_X = apply_categorical_dtypes(train[feature_cols], cat_dtypes)
    val_X = apply_categorical_dtypes(val[feature_cols], cat_dtypes)
    test_X = apply_categorical_dtypes(test[feature_cols], cat_dtypes)

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(
        train_X, train[TARGET_COL],
        eval_set=[(val_X, val[TARGET_COL])],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    best_iter = int(model.best_iteration_)
    print(f"Best iteration: {best_iter} / {PARAMS['n_estimators']} "
          f"(early stopping on val, patience {EARLY_STOPPING_ROUNDS})")

    probs = {"val": model.predict_proba(val_X)[:, 1],
             "test": model.predict_proba(test_X)[:, 1]}

    threshold = best_f1_threshold(val[TARGET_COL].to_numpy(), probs["val"])
    print(f"Threshold chosen on VAL (best F1): {threshold:.2f}")

    metrics = {"split": {}}
    for name, d in (("val", val), ("test", test)):
        y = d[TARGET_COL].to_numpy()
        metrics["split"][name] = {
            **ranking_metrics(y, probs[name]),
            "at_val_selected_threshold": point_metrics(y, probs[name], threshold),
            "at_threshold_0.5": point_metrics(y, probs[name], 0.5),
        }
        save_predictions(name, d, probs[name])

    metrics["model"] = "LGBMClassifier"
    metrics["params"] = {k: v for k, v in PARAMS.items()}
    metrics["best_iteration"] = best_iter
    metrics["selected_threshold"] = threshold
    metrics["threshold_selected_on"] = "val"
    metrics["notes"] = [
        "val is a SELECTION set: early stopping and the operating threshold both "
        "read it. Its metrics are optimistic. Quote test.",
        "No scale_pos_weight: natural prevalence preserved so the probability "
        "scale stays usable for ml/calibrate.py.",
        "NaN is not imputed; LightGBM learns a default direction per split.",
    ]
    (EVAL_DIR / "lightgbm_metrics.json").write_text(json.dumps(metrics, indent=2))

    gain = sorted(zip(feature_cols, model.booster_.feature_importance("gain")),
                  key=lambda kv: -kv[1])
    metadata = {
        "model": "lightgbm",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": feature_cols,
        "n_features": len(feature_cols),
        "categorical_columns": LGBM_CATEGORICAL_COLS,
        "categorical_levels_fitted_on_train": {
            c: [str(v) for v in dt.categories] for c, dt in cat_dtypes.items()
        },
        "fitted_on": "train split only",
        "best_iteration": best_iter,
        "top_30_features_by_gain": [{"feature": f, "gain": float(g)} for f, g in gain[:30]],
        "versions": {
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "random_state": RANDOM_STATE,
    }
    (ARTIFACTS_DIR / "lightgbm_metadata.json").write_text(json.dumps(metadata, indent=2))
    joblib.dump({"model": model, "categorical_dtypes": cat_dtypes,
                 "feature_columns": feature_cols}, ARTIFACTS_DIR / "lightgbm.pkl")

    print("\n--- lightgbm ---")
    for name in ("val", "test"):
        m = metrics["split"][name]
        p = m["at_val_selected_threshold"]
        cm = p["confusion_matrix"]
        label = "val*" if name == "val" else name
        print(f"{label:>5}  PR-AUC {m['pr_auc']:.4f}  ROC-AUC {m['roc_auc']:.4f}  "
              f"| @t={p['threshold']:.2f} P {p['precision']:.4f} R {p['recall']:.4f} "
              f"F1 {p['f1']:.4f} | TP {cm['tp']} FP {cm['fp']} FN {cm['fn']} TN {cm['tn']}")
    print("  * val is a selection set (early stopping + threshold). Quote test.")

    print("\nTop 10 features by gain:")
    for f, g in gain[:10]:
        print(f"  {f:>16}  {g:,.0f}")

    base_path = EVAL_DIR / "baseline_metrics.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())
        bt = base["split"]["test"]["pr_auc"]
        lt = metrics["split"]["test"]["pr_auc"]
        print(f"\nTest PR-AUC  baseline {bt:.4f} -> lightgbm {lt:.4f}  "
              f"({(lt - bt) / bt:+.1%})")


if __name__ == "__main__":
    main()
