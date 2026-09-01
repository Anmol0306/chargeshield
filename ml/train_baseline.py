"""
Logistic regression baseline. Exists for the narrative arc, not the number.

This model is a deliberate FLOOR. It uses the constrained starter feature set,
one-hot encodes the manageable low-cardinality categoricals, and excludes the
high-cardinality identifier columns entirely (see ml/features.py). It is
supposed to be beaten by the gradient-boosted model. A baseline that quietly
leaks is worse than a baseline that loses.

LEAKAGE CONTROL
  The whole preprocessor lives inside one sklearn Pipeline. `.fit` is called
  exactly once, on train. Val and test only ever reach `.predict_proba`, and a
  fitted sklearn transformer computes no cross-row statistics at transform
  time. There is no code path in this file that can fit on val or test. That
  is structural, not disciplinary — see tests/test_preprocessing.py.

THRESHOLD
  PR-AUC is the headline metric because it is threshold-free and appropriate
  at 3.5% prevalence. Point metrics (precision/recall/F1/confusion) are also
  reported at a threshold, and the threshold is CHOSEN ON VAL and applied
  unchanged to test. Choosing it on test would be a second, subtler leak.

OUT  artifacts/baseline.pkl
     artifacts/feature_metadata.json
     evaluation/baseline_metrics.json
     evaluation/preds/baseline_{val,test}.parquet
"""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.metrics import best_f1_threshold, point_metrics, ranking_metrics
from ml.features import (
    BASELINE_CATEGORICAL_COLS,
    BASELINE_NUMERIC_COLS,
    EXCLUDED_HIGH_CARD_COLS,
    LOG_TRANSFORM_COLS,
    TARGET_COL,
    baseline_feature_columns,
)

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")
EVAL_DIR = Path("evaluation")
PREDS_DIR = EVAL_DIR / "preds"

RANDOM_STATE = 42
MISSING_LEVEL = "__MISSING__"

FEATURE_POLICY_STATEMENT = (
    "We use a constrained starter feature set, one-hot encode the manageable "
    "low-cardinality categorical features, and deliberately exclude "
    "high-cardinality identifier fields from the first linear baseline to "
    "control dimensionality and avoid unnecessary complexity. card1 alone "
    "would add ~12,000 dummy columns, and the numeric magnitude of an "
    "anonymised code carries no ordinal meaning a linear model can use. Those "
    "columns are left for the gradient-boosted model, which splits on them "
    "natively."
)


def fill_missing_categorical(df: "pd.DataFrame") -> "pd.DataFrame":
    """Collapse every spelling of "missing" onto one sentinel level.

    NOT SimpleImputer(strategy="constant"). SimpleImputer matches np.nan, but
    these columns arrive from parquet holding Python None, which it leaves
    untouched — None then survives into OneHotEncoder as a literal category.
    That still trains, but it is train/serve skew: a live /score request
    carrying np.nan would not match the learned `None` level and would fall
    through handle_unknown="ignore" to an all-zero row. Two spellings of
    missing, two encodings. pandas .notna() treats both as missing, so one
    sentinel covers both. Module-level so the pipeline stays picklable.
    """
    return df.astype(object).where(df.notna(), MISSING_LEVEL)


def build_preprocessor() -> ColumnTransformer:
    """Three branches. Every learned statistic here is fit on train only.

    Design decisions, each of which is specific to a LINEAR model:

    1. log1p on TransactionAmt (its own branch, not the generic numeric one).
       Right-skewed 0.25 -> 31,937 with median 68.95. A raw linear term is the
       wrong functional form; log1p makes the coefficient mean "per
       proportional increase in amount".

    2. add_indicator=True on both numeric branches. Missingness in this dataset
       is structural, not random: id_* is missing exactly where the identity
       left-join found no row, and D6-D14 are missing in blocks. Imputing the
       median and discarding that fact tells the model an imputed value was
       observed. The indicator keeps "we did not know" as its own signal.

    3. StandardScaler. L2 penalises coefficient magnitude, so with
       TransactionAmt spanning 5 orders of magnitude and C3 spanning 0-23 the
       penalty would be applied in arbitrary units and would shrink whichever
       feature happens to be measured large. lbfgs also converges poorly on
       unscaled input. This is not cosmetic.

    4. Categorical NaN -> its own "__MISSING__" level rather than the mode.
       Same argument as (2), and it is why the categorical branch does not need
       add_indicator: the level IS the indicator. See fill_missing_categorical
       for why this is not SimpleImputer.

    5. OneHotEncoder(handle_unknown="ignore", drop=None).
       - One-hot, never ordinal: label-encoding ProductCD as W=0,C=1,R=2,... 
         makes a linear model literally interpolate between product types.
       - drop=None keeps all levels. With an intercept that is collinear, and
         L2 absorbs it. drop="first" would be actively wrong here: an unseen
         category encodes as all-zeros, which under drop="first" is
         indistinguishable from the dropped reference level, silently
         relabelling unknowns as the reference category. Keeping every level
         gives "unknown" its own distinct all-zero signature.
       - handle_unknown="ignore" is currently a no-op on this data (after the
         feature-policy exclusions there are zero unseen categories in val or
         test) but it stays, because /score will eventually see live traffic
         where it is not a no-op.
    """
    plain_numeric = [c for c in BASELINE_NUMERIC_COLS if c not in LOG_TRANSFORM_COLS]

    log_branch = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])

    numeric_branch = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])

    categorical_branch = Pipeline([
        ("impute", FunctionTransformer(fill_missing_categorical,
                                       feature_names_out="one-to-one")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)),
    ])

    return ColumnTransformer(
        [
            ("amt", log_branch, LOG_TRANSFORM_COLS),
            ("num", numeric_branch, plain_numeric),
            ("cat", categorical_branch, BASELINE_CATEGORICAL_COLS),
        ],
        remainder="drop",  # anything not classified is dropped, loudly, by design
        verbose_feature_names_out=True,
    )


def build_pipeline() -> Pipeline:
    """Preprocessor + estimator as ONE object. This is the leakage control.

    class_weight="balanced" because at 3.5% prevalence an unweighted model at
    threshold 0.5 predicts almost nothing positive and reports recall ~0. It
    does not change the ranking (so PR-AUC is barely affected); it makes the
    point metrics non-degenerate and the probabilities roughly centred. Note
    that it also DESTROYS calibration — the outputs are not honest
    probabilities. Calibration is a separate step (ml/calibrate.py); nothing
    downstream should treat this model's score as P(fraud).
    """
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            # no n_jobs: lbfgs ignores it for binary LR and only spawns loky
            # workers that fail noisily at teardown.
            random_state=RANDOM_STATE,
        )),
    ])


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")


def extract_learned_stats(pipe: Pipeline) -> dict:
    """Pull the fitted statistics out so they are inspectable without unpickling."""
    prep = pipe.named_steps["prep"]
    stats: dict = {}

    for branch, cols in (("amt", LOG_TRANSFORM_COLS),
                         ("num", [c for c in BASELINE_NUMERIC_COLS
                                  if c not in LOG_TRANSFORM_COLS])):
        imp = prep.named_transformers_[branch].named_steps["impute"]
        stats.setdefault("imputer_medians", {}).update(
            {c: float(v) for c, v in zip(cols, imp.statistics_)}
        )

    ohe = prep.named_transformers_["cat"].named_steps["ohe"]
    stats["onehot_categories"] = {
        col: [str(v) for v in cats]
        for col, cats in zip(BASELINE_CATEGORICAL_COLS, ohe.categories_)
    }
    return stats


def save_predictions(name: str, df: pd.DataFrame, y_prob: np.ndarray) -> Path:
    out = pd.DataFrame({
        "TransactionID": df["TransactionID"].to_numpy(),
        "TransactionDT": df["TransactionDT"].to_numpy(),
        "isFraud": df[TARGET_COL].to_numpy(),
        "p_fraud": y_prob.astype("float32"),
    })
    path = PREDS_DIR / f"baseline_{name}.parquet"
    out.to_parquet(path, index=False)
    return path


def main() -> None:
    for d in (ARTIFACTS_DIR, EVAL_DIR, PREDS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    feature_cols = baseline_feature_columns()
    train, val, test = (load_split(n) for n in ("train", "val", "test"))

    print(f"Features in : {len(feature_cols)} "
          f"({len(BASELINE_NUMERIC_COLS)} numeric, "
          f"{len(BASELINE_CATEGORICAL_COLS)} categorical, "
          f"{len(EXCLUDED_HIGH_CARD_COLS)} excluded by feature policy)")
    for name, d in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:>5}: {len(d):>7,} rows | fraud {d[TARGET_COL].mean():.4f}")

    pipe = build_pipeline()
    # The one and only .fit in this file, and it sees train only.
    pipe.fit(train[feature_cols], train[TARGET_COL])

    n_iter = int(np.max(pipe.named_steps["clf"].n_iter_))
    max_iter = pipe.named_steps["clf"].max_iter
    converged = n_iter < max_iter
    print(f"lbfgs iterations: {n_iter} / {max_iter} "
          f"({'converged' if converged else 'DID NOT CONVERGE'})")

    feature_names = [str(n) for n in
                     pipe.named_steps["prep"].get_feature_names_out()]
    print(f"Features out: {len(feature_names)}")

    probs = {n: pipe.predict_proba(d[feature_cols])[:, 1]
             for n, d in (("val", val), ("test", test))}

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

    metrics["model"] = "LogisticRegression(l2, C=1.0, class_weight=balanced)"
    metrics["selected_threshold"] = threshold
    metrics["threshold_selected_on"] = "val"
    metrics["converged"] = converged
    metrics["lbfgs_iterations"] = n_iter
    metrics["notes"] = [
        "PR-AUC is the headline metric: threshold-free and appropriate at 3.5% prevalence.",
        "class_weight=balanced makes point metrics non-degenerate but destroys "
        "calibration. Brier here is NOT a fair calibration reading; see ml/calibrate.py.",
        "Threshold selected on val and applied unchanged to test.",
    ]
    (EVAL_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    metadata = {
        "model": "baseline_logistic_regression",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_policy": FEATURE_POLICY_STATEMENT,
        "features": feature_names,
        "n_features_out": len(feature_names),
        "input_columns": feature_cols,
        "numeric_columns": BASELINE_NUMERIC_COLS,
        "categorical_columns": BASELINE_CATEGORICAL_COLS,
        "log_transformed_columns": LOG_TRANSFORM_COLS,
        "excluded_high_cardinality_columns": EXCLUDED_HIGH_CARD_COLS,
        "missing_level": MISSING_LEVEL,
        "fitted_on": "train split only",
        "learned_statistics": extract_learned_stats(pipe),
        "versions": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "random_state": RANDOM_STATE,
    }
    (ARTIFACTS_DIR / "feature_metadata.json").write_text(json.dumps(metadata, indent=2))
    joblib.dump(pipe, ARTIFACTS_DIR / "baseline.pkl")

    print("\n--- baseline ---")
    for name in ("val", "test"):
        m = metrics["split"][name]
        p = m["at_val_selected_threshold"]
        cm = p["confusion_matrix"]
        print(f"{name:>5}  PR-AUC {m['pr_auc']:.4f}  ROC-AUC {m['roc_auc']:.4f}  "
              f"| @t={p['threshold']:.2f} P {p['precision']:.4f} R {p['recall']:.4f} "
              f"F1 {p['f1']:.4f} | TP {cm['tp']} FP {cm['fp']} FN {cm['fn']} TN {cm['tn']}")
    print("\nWrote artifacts/baseline.pkl, artifacts/feature_metadata.json,")
    print("      evaluation/baseline_metrics.json, evaluation/preds/baseline_{val,test}.parquet")


if __name__ == "__main__":
    main()
