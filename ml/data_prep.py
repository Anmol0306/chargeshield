"""
Load IEEE-CIS, join identity, downcast, chronological split.

IN   data/raw/train_transaction.csv, train_identity.csv
OUT  data/processed/{train,val,test}.parquet

CRITICAL
  - test_transaction.csv has no isFraud. All splits come from train_transaction.
  - Join on TransactionID (left join; most transactions have no identity row).
  - Downcast float64 -> float32 on read or you will OOM on 8GB.
  - Split by TransactionDT quantile from config/split.yaml. Never random.
  - Any transform that LEARNS (encoders, scalers, imputers) is fit on train only.
"""
"""
Load IEEE-CIS, join identity, downcast, chronological split.

IN   data/raw/train_transaction.csv, train_identity.csv
OUT  data/processed/{train,val,test}.parquet

CRITICAL
  - test_transaction.csv has no isFraud. All splits come from train_transaction.
  - Join on TransactionID, LEFT join from transaction (most rows have no
    identity match — that's real, ~76% of transactions in this dataset).
  - usecols on read: skip the 339 V-columns we're not using yet, rather than
    loading them and dropping — the parsing cost is what's expensive, not the
    dataframe once loaded.
  - Downcast float64 -> float32, int64 -> smallest int that fits.
  - Split by TransactionDT quantile from config/split.yaml. Never random.
  - Boundaries use < on the lower split and >= on the upper — never <= on
    both sides — so no transaction can land in two splits. Asserted, not
    just trusted.
"""

import json
from pathlib import Path

import pandas as pd
import yaml

from ml.features import (
    ID_COLS,
    STARTER_IDENTITY_COLS,
    STARTER_TRANSACTION_COLS,
    TARGET_COL,
)

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
SPLIT_CONFIG = Path("config/split.yaml")


def load_split_config() -> dict:
    with open(SPLIT_CONFIG) as f:
        return yaml.safe_load(f)


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32, int64 -> smallest fitting int. Object cols untouched."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    int_cols = df.select_dtypes(include=["int64"]).columns
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def load_raw() -> pd.DataFrame:
    tx_cols = ID_COLS + [TARGET_COL] + STARTER_TRANSACTION_COLS
    tx = pd.read_csv(RAW_DIR / "train_transaction.csv", usecols=tx_cols)

    id_cols = ["TransactionID"] + STARTER_IDENTITY_COLS
    identity = pd.read_csv(RAW_DIR / "train_identity.csv", usecols=id_cols)

    df = tx.merge(identity, on="TransactionID", how="left")
    return downcast(df)


def chronological_split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    dt = df["TransactionDT"]
    train_end = dt.quantile(cfg["train_end_q"])
    val_end = dt.quantile(cfg["val_end_q"])

    train = df[dt < train_end]
    val = df[(dt >= train_end) & (dt < val_end)]
    test = df[dt >= val_end]

    assert train["TransactionDT"].max() < val["TransactionDT"].min(), "train/val overlap"
    assert val["TransactionDT"].max() < test["TransactionDT"].min(), "val/test overlap"
    ids = [set(s["TransactionID"]) for s in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2]), \
        "TransactionID overlap across splits"

    return {"train": train, "val": val, "test": test}


def summarize(name: str, d: pd.DataFrame) -> None:
    print(
        f"  {name:>5}: {len(d):>7,} rows | "
        f"fraud rate {d[TARGET_COL].mean():.4f} | "
        f"DT [{d['TransactionDT'].min():.0f}, {d['TransactionDT'].max():.0f}]"
    )


def main() -> None:
    cfg = load_split_config()
    print(f"Split config: train_end_q={cfg['train_end_q']}, val_end_q={cfg['val_end_q']}")

    df = load_raw()
    print(f"Loaded + joined: {df.shape[0]:,} rows, {df.shape[1]} cols, "
          f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"Identity match rate: {df[STARTER_IDENTITY_COLS[0]].notna().mean():.2%}")

    splits = chronological_split(df, cfg)
    print("Splits:")
    for name, d in splits.items():
        summarize(name, d)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, d in splits.items():
        d.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
    print(f"Written to {OUT_DIR}/")

    meta = {"starter_transaction_cols": STARTER_TRANSACTION_COLS,
             "starter_identity_cols": STARTER_IDENTITY_COLS}
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/prep_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()