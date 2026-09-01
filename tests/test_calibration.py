"""Calibration correctness and the val_fit/val_pick discipline."""
import pathlib

import numpy as np
import pandas as pd
import pytest

from ml.calibrate import (
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    chronological_val_split,
    expected_calibration_error,
)

PREDS = pathlib.Path("evaluation/preds")


@pytest.fixture(scope="module")
def val_preds():
    p = PREDS / "lightgbm_val.parquet"
    if not p.exists():
        pytest.skip("run `make train` first")
    return pd.read_parquet(p)


def test_val_split_is_chronological_and_disjoint(val_preds):
    fit, pick = chronological_val_split(val_preds)
    assert fit["TransactionDT"].max() < pick["TransactionDT"].min()
    assert not set(fit["TransactionID"]) & set(pick["TransactionID"])
    assert len(fit) + len(pick) == len(val_preds)


def test_calibrator_is_fit_on_val_fit_only(val_preds):
    """A calibrator fit on val_fit must be unchanged by later seeing val_pick."""
    fit, pick = chronological_val_split(val_preds)
    cal = PlattCalibrator().fit(fit["p_fraud"].to_numpy(), fit["isFraud"].to_numpy())
    before = cal.params()
    cal.predict(pick["p_fraud"].to_numpy())
    assert cal.params() == before, "predicting mutated the fitted calibrator"


def test_platt_is_strictly_monotonic(val_preds):
    """Platt cannot reorder. If it could, PR-AUC/ROC-AUC would silently move and
    the calibration step would be quietly changing the model's decisions."""
    fit, _ = chronological_val_split(val_preds)
    cal = PlattCalibrator().fit(fit["p_fraud"].to_numpy(), fit["isFraud"].to_numpy())
    p = np.linspace(0.001, 0.999, 500)
    out = cal.predict(p)
    assert np.all(np.diff(out) > 0), "Platt output is not strictly increasing"


def test_isotonic_is_non_decreasing(val_preds):
    fit, _ = chronological_val_split(val_preds)
    cal = IsotonicCalibrator().fit(fit["p_fraud"].to_numpy(), fit["isFraud"].to_numpy())
    p = np.linspace(0.001, 0.999, 500)
    assert np.all(np.diff(cal.predict(p)) >= 0)


def test_calibrated_output_is_a_valid_probability(val_preds):
    fit, pick = chronological_val_split(val_preds)
    for cls in (PlattCalibrator, IsotonicCalibrator):
        cal = cls().fit(fit["p_fraud"].to_numpy(), fit["isFraud"].to_numpy())
        out = cal.predict(pick["p_fraud"].to_numpy())
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert not np.isnan(out).any()


def test_identity_calibrator_is_the_identity():
    p = np.array([0.0, 0.13, 0.5, 0.99, 1.0])
    np.testing.assert_array_equal(IdentityCalibrator().fit(p, p > 0.5).predict(p), p)


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    """Teeth: if ECE does not read ~0 on data constructed to be calibrated, the
    metric is wrong and every calibration claim built on it is wrong."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, 200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01


def test_ece_detects_a_miscalibrated_predictor():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, 200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    # Halve every probability: ranking is untouched, calibration is destroyed.
    assert expected_calibration_error(y, p * 0.5) > 0.10
