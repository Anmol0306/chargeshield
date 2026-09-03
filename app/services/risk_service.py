"""Wraps ml/predict.py. The only place the API touches the model.

One module between the API and the model so that swapping the model, adding a
cache, or degrading when artifacts are missing is a change in one file rather
than in every endpoint.
"""

from __future__ import annotations

from ml.predict import ModelUnavailable, load_artifacts, score_transaction

__all__ = ["ModelUnavailable", "score", "is_ready", "model_info"]


def is_ready() -> bool:
    try:
        load_artifacts()
        return True
    except Exception:
        return False


def model_info() -> dict:
    """Non-sensitive description of what is loaded, for /health."""
    try:
        a = load_artifacts()
        return {
            "loaded": True,
            "n_features": len(a["feature_columns"]),
            "calibrator": a["calibrator_name"],
        }
    except Exception as exc:
        return {"loaded": False, "reason": type(exc).__name__}


def score(features: dict) -> dict:
    return score_transaction(features)
