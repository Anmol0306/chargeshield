"""
Feature construction. Fit on train, apply to val/test.

Starter set (skip the 339 V-columns on day one):
  TransactionAmt, ProductCD, card1-card6, addr1, addr2, dist1,
  C1-C14, D1-D15, M1-M9

LEAKAGE CHECKLIST — run before every training run:
  [ ] isFraud not in feature list
  [ ] no feature computed across train+test
  [ ] encoders fitted on train only
  [ ] no duplicate TransactionID across splits
"""
"""
Feature construction. Fit on train, apply to val/test — never the reverse.

STARTER_TRANSACTION_COLS is deliberately narrow: it skips the 339 V-columns
on day one. That's not just a modelling choice — it's also why data_prep.py
can use `usecols` on read and never pay to parse columns we're not using yet.
Widen this list on Day 2+ only if the baseline is solid and there's time.

LEAKAGE CHECKLIST — run before every training run:
  [ ] isFraud not in feature list
  [ ] no feature computed across train+test (fit encoders on train only)
  [ ] encoders/scalers fitted on train only, applied (not refit) to val/test
  [ ] no duplicate TransactionID across splits
"""

ID_COLS = ["TransactionID", "TransactionDT"]
TARGET_COL = "isFraud"

STARTER_TRANSACTION_COLS = [
    "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
    "D10", "D11", "D12", "D13", "D14", "D15",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

STARTER_IDENTITY_COLS = ["id_01", "id_02", "DeviceType", "DeviceInfo"]


def categorical_columns(cols: list[str]) -> list[str]:
    """Columns that are categorical, not numeric, among the starter set."""
    cat_prefixes = ("ProductCD", "card4", "card6", "M", "DeviceType", "DeviceInfo")
    return [c for c in cols if c.startswith(cat_prefixes)]