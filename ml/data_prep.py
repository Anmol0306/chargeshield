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
