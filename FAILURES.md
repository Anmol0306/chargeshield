# Failure log

Every entry: Problem / Cause / Fix / Lesson.
Written as it happens, not reconstructed on Sep 5.
This file is a deliverable — it is the evidence for "failure recovery".

---

## Failure 01 — Two spellings of "missing" in the baseline encoder
*Sep 1, `ml/train_baseline.py`*

**Problem:** `tests/test_preprocessing.py::test_missing_categorical_becomes_its_own_level`
failed: the fitted OneHotEncoder had learned levels `['T', 'None']` for `M1`,
with no `__MISSING__` level, despite a `SimpleImputer(strategy="constant",
fill_value="__MISSING__")` sitting directly upstream of it.

**Cause:** `SimpleImputer` matches `missing_values=np.nan`. The object columns
come out of parquet holding Python `None`, which is not `np.nan`, so the
imputer was a no-op and `None` survived into the encoder as a literal
category. Training was unaffected — `None` is still a distinct level, and the
metrics were valid — which is exactly why this would not have been noticed
without the test.

**Fix:** Replaced the imputer with a module-level `fill_missing_categorical`
using `df.where(df.notna(), "__MISSING__")`. pandas `.notna()` treats `None`
and `NaN` identically, so both collapse onto one sentinel. Added
`test_none_and_nan_encode_identically` as a regression guard.

**Lesson:** The bug was not a metrics bug, it was a **train/serve skew** bug,
and it would have surfaced in the API rather than in training. A live `/score`
request carries `np.nan`, which would not have matched the learned `None`
level and would have fallen through `handle_unknown="ignore"` to an all-zero
row — the model quietly ignoring a field it was trained on, with no error. Two
representations of the same concept encoded two different ways is a class of
bug that only a test comparing the two representations can catch. Assert on
the *fitted artifact*, not just on the score.
