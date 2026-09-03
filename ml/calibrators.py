"""
Calibrator classes, in their own importable module.

WHY THIS IS NOT INSIDE ml/calibrate.py
  These objects get pickled into artifacts/calibrator.pkl. When they were
  defined in ml/calibrate.py and that file was run as `python -m ml.calibrate`,
  pickle recorded their class path as `__main__.PlattCalibrator` -- so the
  artifact could only be unpickled by a process that also happened to be
  running calibrate.py as __main__. Every other entry point, including the API
  and the demo, raised AttributeError on load. See FAILURES.md 06.

  A module that is imported and never executed as a script has a stable
  qualified name, so anything pickled from here loads anywhere.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


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
