"""
Load artifacts, score one transaction or a batch. Imported by risk_service.

PARTIAL FEATURE VECTORS ARE SUPPORTED ON PURPOSE
  The model takes 392 features. No API caller will have all of them, and
  demanding them would make the endpoint undemonstrable. LightGBM learns a
  default direction for NaN at every split, so an absent feature is handled the
  same way it was handled in training -- as missing, which is a real and
  frequent state in this dataset (76% of transactions have no identity row).

  So a caller sends what it has and the rest is NaN. This is not a hack; it is
  the same code path that scored 76% of the training set. The response reports
  how many features were actually supplied, because a prediction from four
  features is not the same evidence as a prediction from four hundred and the
  caller should be able to see which one it got.

CATEGORICAL DTYPES COME FROM THE ARTIFACT, NEVER FROM THE REQUEST
  The category->code mapping was learned on train and pickled with the model.
  Rebuilding it from request data would reintroduce exactly the silent
  scrambling that ml/train_lightgbm.py's fit_categorical_dtypes exists to
  prevent -- and at serving time, with one row, pandas would assign code 0 to
  whatever value happened to arrive.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "lightgbm.pkl"
CALIBRATOR_PATH = ARTIFACTS_DIR / "calibrator.pkl"


class ModelUnavailable(RuntimeError):
    """Artifacts are missing. Raised rather than silently returning a number."""


@lru_cache(maxsize=1)
def load_artifacts() -> dict:
    """Load once per process. Cached because unpickling LightGBM is not cheap."""
    if not MODEL_PATH.exists():
        raise ModelUnavailable(
            f"{MODEL_PATH} not found — run `make train` before serving")
    bundle = joblib.load(MODEL_PATH)

    calibrator = None
    if CALIBRATOR_PATH.exists():
        calibrator = joblib.load(CALIBRATOR_PATH)

    return {
        "model": bundle["model"],
        "categorical_dtypes": bundle["categorical_dtypes"],
        "feature_columns": bundle["feature_columns"],
        "calibrator": (calibrator or {}).get("calibrator"),
        "calibrator_name": (calibrator or {}).get("name", "uncalibrated"),
        "threshold": (calibrator or {}).get("threshold"),
    }


def _frame(features: dict, artifacts: dict) -> pd.DataFrame:
    """One row, every model column, unsupplied values as NaN."""
    cols = artifacts["feature_columns"]
    row = {c: features.get(c, np.nan) for c in cols}
    df = pd.DataFrame([row], columns=cols)

    for col, dtype in artifacts["categorical_dtypes"].items():
        df[col] = df[col].astype(dtype)

    # Everything else must be numeric; a string in a numeric column is a
    # malformed request, not a category.
    for col in cols:
        if col not in artifacts["categorical_dtypes"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def score_transaction(features: dict) -> dict:
    """Calibrated fraud probability for one transaction.

    `p_fraud` is the calibrated value when a calibrator is available, and is
    what every downstream consumer should use. The raw score is returned
    alongside it so the two are never conflated.
    """
    artifacts = load_artifacts()
    known = [c for c in artifacts["feature_columns"] if c in features]

    df = _frame(features, artifacts)
    raw = float(artifacts["model"].predict_proba(df)[:, 1][0])

    calibrator = artifacts["calibrator"]
    calibrated = float(calibrator.predict(np.array([raw]))[0]) if calibrator else raw

    return {
        "p_fraud": calibrated,
        "p_fraud_raw": raw,
        "calibrator": artifacts["calibrator_name"],
        "features_supplied": len(known),
        "features_expected": len(artifacts["feature_columns"]),
        "unrecognised_fields": sorted(set(features) - set(artifacts["feature_columns"])),
    }
