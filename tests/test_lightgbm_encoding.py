"""Categorical encoding leakage controls for the gradient-boosted model.

The failure these guard against is silent: it raises nothing, warns nothing,
and only shows up as quietly worse predictions.
"""
import pathlib

import numpy as np
import pandas as pd
import pytest

from ml.features import (
    EXCLUDED_HIGH_CARD_COLS,
    ID_COLS,
    LGBM_CATEGORICAL_COLS,
    TARGET_COL,
    V_COLS,
    lightgbm_feature_columns,
)
from ml.train_lightgbm import apply_categorical_dtypes, fit_categorical_dtypes

PROCESSED = pathlib.Path("data/processed")


@pytest.fixture(scope="module")
def frames():
    try:
        train = pd.read_parquet(PROCESSED / "train.parquet")
        val = pd.read_parquet(PROCESSED / "val.parquet")
    except FileNotFoundError:
        pytest.skip("run `make data` first")
    if "V1" not in train.columns:
        pytest.skip("re-run `make data` to pull the V-columns")
    return train, val


@pytest.fixture(scope="module")
def dtypes(frames):
    train, _ = frames
    return fit_categorical_dtypes(train)


def test_feature_set_shape():
    cols = lightgbm_feature_columns()
    assert len(cols) == len(set(cols))
    assert TARGET_COL not in cols
    for c in ID_COLS:
        assert c not in cols
    assert all(v in cols for v in V_COLS)


def test_tree_receives_what_the_baseline_excluded():
    """The baseline/tree contrast is the experiment. If someone 'fixes' the
    baseline exclusions by applying them here too, the comparison is gone."""
    cols = lightgbm_feature_columns()
    for c in EXCLUDED_HIGH_CARD_COLS:
        assert c in cols


def test_category_codes_are_identical_across_splits(frames, dtypes):
    """The core guard. One mapping, learned on train, applied everywhere."""
    train, val = frames
    tr = apply_categorical_dtypes(train[lightgbm_feature_columns()], dtypes)
    va = apply_categorical_dtypes(val[lightgbm_feature_columns()], dtypes)

    for col in LGBM_CATEGORICAL_COLS:
        assert list(tr[col].cat.categories) == list(va[col].cat.categories), (
            f"{col}: train and val disagree on the category->code mapping. "
            "LightGBM stores splits as integer codes, so this silently "
            "scrambles categories at inference."
        )


def test_naive_per_split_astype_would_actually_differ(frames):
    """Teeth for the test above: prove the bug it prevents is real on THIS data,
    not a theoretical concern."""
    train, val = frames
    disagreements = [
        col for col in LGBM_CATEGORICAL_COLS
        if list(train[col].astype("category").cat.categories)
        != list(val[col].astype("category").cat.categories)
    ]
    assert disagreements, (
        "per-split .astype('category') produced identical mappings for every "
        "column, so this test cannot detect the bug. Tighten it."
    )


def test_unseen_category_becomes_nan(frames, dtypes):
    """Unseen levels follow the model's learned missing branch rather than
    colliding with a real category. Analogue of handle_unknown='ignore'."""
    _, val = frames
    row = val[lightgbm_feature_columns()].head(1).copy()
    row["ProductCD"] = "__NEVER_SEEN__"
    encoded = apply_categorical_dtypes(row, dtypes)
    assert pd.isna(encoded["ProductCD"].iloc[0])


def test_apply_does_not_mutate_input(frames, dtypes):
    _, val = frames
    subset = val[lightgbm_feature_columns()].head(500)
    before = subset["ProductCD"].dtype
    apply_categorical_dtypes(subset, dtypes)
    assert subset["ProductCD"].dtype == before


def test_categorical_levels_come_from_train_only(frames, dtypes):
    train, val = frames
    for col in LGBM_CATEGORICAL_COLS:
        learned = set(dtypes[col].categories)
        assert learned == set(train[col].dropna().unique()), f"{col}: not train's levels"
        val_only = set(val[col].dropna().unique()) - set(train[col].dropna().unique())
        assert not (learned & val_only), f"{col}: a val-only level leaked into the mapping"
