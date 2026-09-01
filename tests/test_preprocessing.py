"""Preprocessing leakage controls.

These tests exist to make one claim checkable: every learned statistic in the
baseline is fit on train and only on train, and transforming a row does not
depend on which other rows are transformed alongside it.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from ml.features import (
    BASELINE_CATEGORICAL_COLS,
    BASELINE_NUMERIC_COLS,
    EXCLUDED_HIGH_CARD_COLS,
    ID_COLS,
    STARTER_IDENTITY_COLS,
    STARTER_TRANSACTION_COLS,
    TARGET_COL,
    baseline_feature_columns,
)
from ml.train_baseline import MISSING_LEVEL, build_preprocessor

PROCESSED = pathlib.Path("data/processed")
FIT_ROWS = 20_000  # enough for stable medians, fast enough to run every commit


@pytest.fixture(scope="module")
def frames():
    try:
        train = pd.read_parquet(PROCESSED / "train.parquet")
        val = pd.read_parquet(PROCESSED / "val.parquet")
    except FileNotFoundError:
        pytest.skip("run `make data` first")
    return train, val


@pytest.fixture(scope="module")
def fitted(frames):
    train, _ = frames
    prep = build_preprocessor()
    prep.fit(train[baseline_feature_columns()].head(FIT_ROWS))
    return prep


# --- the partition itself -------------------------------------------------

def test_feature_partition_covers_starter_set_exactly():
    declared = BASELINE_NUMERIC_COLS + BASELINE_CATEGORICAL_COLS + EXCLUDED_HIGH_CARD_COLS
    starter = STARTER_TRANSACTION_COLS + STARTER_IDENTITY_COLS
    assert sorted(declared) == sorted(starter)
    assert len(declared) == len(set(declared)), "a column is classified twice"


def test_target_and_ids_are_not_features():
    cols = baseline_feature_columns()
    assert TARGET_COL not in cols
    for c in ID_COLS:
        assert c not in cols


def test_high_cardinality_columns_are_excluded():
    """The feature policy is a claim about dimensionality. Check it holds."""
    cols = baseline_feature_columns()
    for c in EXCLUDED_HIGH_CARD_COLS:
        assert c not in cols, f"{c} is high-cardinality and must not reach the linear model"


# --- fit-on-train-only ----------------------------------------------------

def test_imputer_medians_come_from_train_not_pooled(frames, fitted):
    """The fitted medians must equal the train medians, and pooling train+val
    must actually change them — otherwise this test has no teeth."""
    train, val = frames
    imp = fitted.named_transformers_["num"].named_steps["impute"]
    plain = [c for c in BASELINE_NUMERIC_COLS if c != "TransactionAmt"]

    fitted_medians = dict(zip(plain, imp.statistics_))
    train_head = train[plain].head(FIT_ROWS)

    differs = 0
    for col in plain:
        expected = train_head[col].median()
        if np.isnan(expected):
            continue
        assert fitted_medians[col] == pytest.approx(expected, rel=1e-6, abs=1e-6), \
            f"{col}: fitted median is not the train median"
        pooled = pd.concat([train_head[col], val[col]]).median()
        if not np.isclose(pooled, expected, rtol=1e-6, atol=1e-6):
            differs += 1

    assert differs > 0, (
        "pooling train+val produced identical medians for every column, so this "
        "test cannot detect pooling. Tighten it."
    )


def test_transform_is_row_independent(frames, fitted):
    """Strongest available proof that no cross-row statistic is computed at
    transform time: transforming a slice must give bitwise-identical rows to
    transforming the whole frame."""
    _, val = frames
    cols = baseline_feature_columns()

    full = fitted.transform(val[cols])
    slice_ = fitted.transform(val[cols].head(100))

    np.testing.assert_array_equal(full[:100], slice_)


def test_transform_does_not_refit(frames, fitted):
    """Transforming val must not mutate the fitted statistics."""
    _, val = frames
    imp = fitted.named_transformers_["num"].named_steps["impute"]
    before = imp.statistics_.copy()
    fitted.transform(val[baseline_feature_columns()])
    np.testing.assert_array_equal(before, imp.statistics_)


def test_no_nan_survives_transform(frames, fitted):
    _, val = frames
    out = fitted.transform(val[baseline_feature_columns()])
    assert not np.isnan(out).any(), "NaN reached the model matrix"


# --- unseen categories ----------------------------------------------------

def test_unseen_category_encodes_as_all_zeros(frames, fitted):
    """handle_unknown='ignore' must degrade to 'no information', not raise, and
    must not be confusable with a real level (which is why drop=None)."""
    _, val = frames
    cols = baseline_feature_columns()
    row = val[cols].head(1).copy()

    ohe = fitted.named_transformers_["cat"].named_steps["ohe"]
    widths = [len(c) for c in ohe.categories_]
    start = int(sum(widths[:BASELINE_CATEGORICAL_COLS.index("ProductCD")]))
    width = len(ohe.categories_[BASELINE_CATEGORICAL_COLS.index("ProductCD")])

    row["ProductCD"] = "__NEVER_SEEN__"
    encoded = fitted.named_transformers_["cat"].transform(row[BASELINE_CATEGORICAL_COLS])

    block = encoded[0, start:start + width]
    assert block.sum() == 0, "unseen category did not encode as an all-zero block"


def test_missing_categorical_becomes_its_own_level(fitted):
    ohe = fitted.named_transformers_["cat"].named_steps["ohe"]
    cats = {c: [str(v) for v in levels]
            for c, levels in zip(BASELINE_CATEGORICAL_COLS, ohe.categories_)}
    # M-columns are 26-67% missing; __MISSING__ must be a real level, not imputed away.
    assert MISSING_LEVEL in cats["M1"]
    assert MISSING_LEVEL in cats["DeviceType"]


# --- the persisted metadata -----------------------------------------------

def test_feature_metadata_is_consistent():
    p = pathlib.Path("artifacts/feature_metadata.json")
    if not p.exists():
        pytest.skip("run `make baseline` first")
    meta = json.loads(p.read_text())

    assert TARGET_COL not in meta["features"]
    for c in ID_COLS:
        assert c not in meta["features"]
    for c in EXCLUDED_HIGH_CARD_COLS:
        assert c not in meta["input_columns"]
    assert meta["n_features_out"] == len(meta["features"])
    assert meta["fitted_on"] == "train split only"


def test_none_and_nan_encode_identically(frames, fitted):
    """Train/serve skew guard. Parquet gives Python None; a live API request
    gives np.nan. Both must land on the same __MISSING__ level, or the served
    model silently disagrees with the trained one."""
    _, val = frames
    cat = fitted.named_transformers_["cat"]

    row = val[BASELINE_CATEGORICAL_COLS].head(1).copy()
    with_none = row.copy()
    with_none["M1"] = None
    with_nan = row.copy()
    with_nan["M1"] = np.nan

    np.testing.assert_array_equal(cat.transform(with_none), cat.transform(with_nan))
