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

---

## Failure 02 — Isotonic calibrator was not monotone at float32 precision
*Sep 1, `ml/calibrate.py`*

**Problem:** `test_isotonic_is_non_decreasing` failed on 69 of 500 probe
points, with the largest backward step at `-1.19e-07`.

**Cause:** Not a calibration bug — a dtype bug. Predictions are persisted to
parquet as float32 to keep the files small, and `IsotonicRegression` was fit
and evaluated at that precision, so adjacent steps of the fitted step function
differed by one ULP in the wrong direction.

**Fix:** Cast to float64 inside `IsotonicCalibrator.fit`/`.predict`. Storage
stays float32; the arithmetic does not.

**Lesson:** The magnitude was numerically irrelevant — no decision at any
threshold changes by 1e-07. Kept the fix anyway, because "a calibrator whose
output can decrease as the model score increases" is not a sentence worth
having to explain in a panel, and the cost of the fix was one cast. Storage
precision and arithmetic precision are separate decisions, and letting the
first silently set the second is how you get results that are *almost* right.