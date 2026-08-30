"""Temporal integrity. If this fails, every metric downstream is invalid."""
import pandas as pd
import pytest

TRAIN = "data/processed/train.parquet"
VAL = "data/processed/val.parquet"
TEST = "data/processed/test.parquet"


@pytest.fixture(scope="module")
def splits():
    try:
        return (pd.read_parquet(TRAIN), pd.read_parquet(VAL), pd.read_parquet(TEST))
    except FileNotFoundError:
        pytest.skip("run `make data` first")


def test_no_temporal_leakage(splits):
    train, val, test = splits
    assert train["TransactionDT"].max() < val["TransactionDT"].min()
    assert val["TransactionDT"].max() < test["TransactionDT"].min()


def test_no_id_overlap(splits):
    train, val, test = splits
    ids = [set(d["TransactionID"]) for d in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])


def test_target_not_in_features(splits):
    train, _, _ = splits
    import json, pathlib
    p = pathlib.Path("artifacts/feature_metadata.json")
    if not p.exists():
        pytest.skip("train a model first")
    assert "isFraud" not in json.loads(p.read_text())["features"]
