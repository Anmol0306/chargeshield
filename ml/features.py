"""
Feature construction. Fit on train, apply to val/test — never the reverse.

STARTER_TRANSACTION_COLS is deliberately narrow: it skips the 339 V-columns
on day one. That's not just a modelling choice — it's also why data_prep.py
can use `usecols` on read and never pay to parse columns we're not using yet.
Widen this list on Day 2+ only if the baseline is solid and there's time.

FEATURE POLICY FOR THE LINEAR BASELINE
  We use a constrained starter feature set, one-hot encode the manageable
  low-cardinality categoricals, and deliberately exclude high-cardinality
  identifier columns from the first linear baseline to control dimensionality
  and avoid unnecessary complexity. card1 alone would add ~12,000 dummy
  columns, and the numeric magnitude of an anonymised code carries no ordinal
  meaning a linear model can use. Those columns are left for the
  gradient-boosted model, which can split on them natively.

  This makes the baseline a deliberate FLOOR, not a best effort. It is
  supposed to be beaten.

MISSINGNESS IS NOT STATIONARY ACROSS THE SPLIT
  Measured on the current chronological split, the missing RATE itself drifts:
    M1     53.8% missing in train -> 26.8% in val
    M7-M9  66.9%                  -> 40.0%
    id_01  73.4%                  -> 82.3%  (identity join coverage moves the
                                             other way)
  So the missing-indicator features carry train/val distribution shift. That is
  a property of time-splitting real data, not a bug — but it is a reason not to
  over-read a val/test gap, and it must be stated rather than discovered by a
  reviewer.

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


# --------------------------------------------------------------------------
# Baseline feature partition.
#
# Explicit lists, not prefix matching. A prefix rule ("everything starting
# with M is categorical") happens to be correct today and will silently
# mis-bucket the first counter-example anyone adds. A list can be serialised
# into feature_metadata.json and asserted on in a test; a prefix function
# cannot.
# --------------------------------------------------------------------------

# Genuinely continuous or count-valued. Cardinalities and missing rates below
# are measured on data/processed/train.parquet (413,378 rows).
BASELINE_NUMERIC_COLS = [
    "TransactionAmt",                                    # 0.0% missing, heavy right skew
    "dist1",                                             # 61.9% missing
    "C1", "C2", "C3", "C4", "C5", "C6", "C7",            # counts, 0.0% missing
    "C8", "C9", "C10", "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",      # day-like timedeltas,
    "D9", "D10", "D11", "D12", "D13", "D14", "D15",      # 0.1% - 93.5% missing
    "id_01",                                             # 73.4% missing (join)
    "id_02",                                             # 74.0% missing (join)
]

# Low cardinality -> safe to one-hot. ~42 dummy columns in total including an
# explicit __MISSING__ level for each.
BASELINE_CATEGORICAL_COLS = [
    "ProductCD",                                         # 5 levels
    "card4",                                             # 4 levels (card network)
    "card6",                                             # 4 levels (debit/credit)
    "M1", "M2", "M3",                                    # 2 levels (T/F)
    "M4",                                                # 3 levels (M0/M1/M2)
    "M5", "M6", "M7", "M8", "M9",                        # 2 levels (T/F)
    "DeviceType",                                        # 2 levels
]

# Anonymised identifier codes. Numeric magnitude is meaningless (encoding them
# as floats imposes a false ordering a linear model will interpolate through),
# and the cardinality is too high to one-hot. Excluded from the linear
# baseline; the gradient-boosted model splits on them natively.
# Level counts measured on train.
EXCLUDED_HIGH_CARD_COLS = [
    "card1",       # 12,242 levels
    "DeviceInfo",  #  1,546 levels — and the ONLY column with unseen levels in
                   #  val (135, 0.33% of rows) or test (155, 0.56% of rows)
    "card2",       #    500 levels
    "addr1",       #    318 levels
    "card5",       #    110 levels
    "card3",       #    105 levels
    "addr2",       #     67 levels
]

# Log1p before scaling. TransactionAmt spans 0.25 -> 31,937 with a median of
# 68.95; a raw linear term asserts "each additional dollar shifts the log-odds
# by a constant", which is the wrong functional form for a log-normal amount.
# Named as a constant rather than hardcoded in the pipeline so that
# feature_metadata.json can record which columns were transformed.
LOG_TRANSFORM_COLS = ["TransactionAmt"]


def _assert_partition_is_total() -> None:
    """Every starter column must be classified exactly once.

    Asserted at import, not in a test, on purpose. The failure this catches is
    "someone widened the starter set and forgot to classify the new column".
    If that is only caught by pytest, the new column silently vanishes from the
    model and the training run still reports success. Failing at import makes
    it impossible to train on a feature set you have not classified.

    Same argument data_prep.py already makes for its split boundaries:
    asserted, not just trusted.
    """
    declared = BASELINE_NUMERIC_COLS + BASELINE_CATEGORICAL_COLS + EXCLUDED_HIGH_CARD_COLS
    starter = STARTER_TRANSACTION_COLS + STARTER_IDENTITY_COLS

    dupes = {c for c in declared if declared.count(c) > 1}
    assert not dupes, f"column classified more than once: {sorted(dupes)}"

    unclassified = set(starter) - set(declared)
    assert not unclassified, (
        f"starter columns with no baseline classification: {sorted(unclassified)}. "
        "Add each to BASELINE_NUMERIC_COLS, BASELINE_CATEGORICAL_COLS or "
        "EXCLUDED_HIGH_CARD_COLS."
    )

    unknown = set(declared) - set(starter)
    assert not unknown, f"classified columns not in the starter set: {sorted(unknown)}"

    assert TARGET_COL not in declared, "target leaked into the feature partition"
    assert not set(ID_COLS) & set(declared), "identifier columns leaked into features"


_assert_partition_is_total()


def baseline_feature_columns() -> list[str]:
    """Input columns the baseline pipeline consumes. Excludes ID cols and target."""
    return BASELINE_NUMERIC_COLS + BASELINE_CATEGORICAL_COLS
