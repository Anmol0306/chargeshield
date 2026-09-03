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

---

## Failure 03 — The first cost curve said the product was unnecessary
*Sep 1, `ml/cost_curve.py`*

**Problem:** The first working sweep returned a cost-optimal policy of "contest
99% of disputes", beating the trivial contest-everything rule by ₹37,433 out of
₹394M — about 0.01%. Taken at face value: the model adds nothing and a merchant
should just contest every chargeback.

**Cause:** Not the loss function — the population. I was pricing every held-out
transaction as though it were a dispute, so the fraud base rate was 3.5%. Real
dispute queues are nothing like that: a chargeback arriving is already strong
evidence something went wrong. At 3.5% fraud, a median ₹6,028 dispute and a
₹500 representment cost, contesting is positive-EV almost regardless of score,
so no threshold can have content. The degeneracy was an artefact of the
population, not a property of the product.

**Fix:** Added a prior-shift sensitivity — re-calibrate the scores on the odds
scale and re-weight the population to assumed dispute-queue fraud rates of
20/35/50/65%. The model's edge over contest-everything rises ₹2 → ₹137 per
dispute and the contest rate falls 99% → 60%. Production bands are now derived
at a stated assumed queue rate (50%), not at the artefactual 3.5%. Also
excluded disputes above the amount cap from the sweep, since the policy engine
never decides those automatically.

**Lesson:** The arithmetic was correct and the answer was still meaningless,
which is the most dangerous combination — nothing fails, no test goes red, and
the number is simply about the wrong thing. Both the as-observed result and the
shifted ones are reported: deleting the embarrassing one and keeping the
flattering one would have been the easy move and the dishonest one. A
base-rate-dependent conclusion needs its base rate stated as an assumption and
swept, not inherited by accident from whatever data was lying around.

---

## Failure 04 — The policy comparison measured a policy we do not ship
*Sep 2, `ml/evaluate.py`, found in a pre-Day-4 audit*

**Problem:** The README reported "ChargeShield beats the best no-model policy by
₹84 per dispute" and called that "the value attributable to the ML". Re-running
the comparison with the actual policy engine gave **−₹18** — the shipped gate is
slightly *worse* than the static amount rule across the queue.

**Cause:** `ml/evaluate.py:152` defined the row labelled `chargeshield` as
`p_shift < bands["threshold"]` — a single global threshold on the score. The
real gate uses amount-dependent bands, an evidence gate, a fabrication check,
an amount cap and an economic floor. Worse, it *cannot* run in that module at
all: `evaluate.py` operates on transactions, and transactions carry no
`reason_code` and no evidence. So the row was not a degraded version of the
policy engine, it was a different object with the product's name on it.

**Fix:** Renamed the row to `global_cost_threshold_rule` with an explicit note
that it is not the shipped engine. Added `price_policies()` to
`app/services/batch_runner.py`, which prices the four policies on the dispute
queue using the actual `decide()` against real `isFraud` labels, plus a segment
decomposition showing where the gate wins (+₹69 on actionable disputes) and
where it pays for safety (−₹83 where evidence is missing).
`tests/test_artifacts.py` now fails if a `chargeshield` key reappears in
`evaluate.py`'s output, and asserts the batch row is produced by the real
`decide()` rather than by a label.

**Lesson:** The most dangerous number is the flattering one that nobody
re-derives. This survived three days and a commit message that quoted it,
because every test checked that the arithmetic was *internally* consistent and
none checked that the thing being priced was the thing being shipped. Naming is
part of correctness: a variable called `chargeshield` that is not ChargeShield
will be believed by everyone including its author. The corrected result is also
a better story — "the gate costs ₹18/dispute and here is exactly what that buys"
is defensible; "+₹84" was not true.

---

## Failure 05 — `evaluation/metrics.json` was not valid JSON
*Sep 2, same audit*

**Problem:** `evaluation/metrics.json` contained bare `NaN`, which no strict
JSON parser accepts. The Day-5 frontend would have failed to load it.

**Cause:** `expected_cost()` was called with `t=np.nan` when a policy mask was
supplied instead of a threshold, and `json.dumps` serialises NaN happily.
Python's own `json.loads` also *reads* it back happily, so every test passed.

**Fix:** `threshold` is now `None` when a mask is given, and
`tests/test_artifacts.py` parses every committed JSON artifact with
`parse_constant` set to raise — the check Python does not do by default.

**Lesson:** Round-tripping through the same library proves nothing about
interoperability. The test that mattered was the one that made Python behave
like a stricter consumer.

---

## Failure 06 — The calibrator artifact could only be loaded by the script that wrote it
*Sep 3, `ml/calibrate.py`, found while wiring the API*

**Problem:** `joblib.load("artifacts/calibrator.pkl")` raised
`AttributeError: Can't get attribute 'PlattCalibrator' on <module '__main__'>`
from every process except `python -m ml.calibrate` itself. The API could not
load the calibrator at all, so `/score` and the demo would both have failed.

**Cause:** The calibrator classes were defined in `ml/calibrate.py`, which is
executed as `__main__`. Pickle records a class by its qualified path, so the
artifact referenced `__main__.PlattCalibrator` — a name that only resolves
inside the process that happened to be running that file as a script.

**Fix:** Moved the classes to `ml/calibrators.py`, a module that is imported
and never executed, so the pickled path is `ml.calibrators.PlattCalibrator` and
resolves anywhere. Regenerated the artifact.

**Lesson:** Every test and every pipeline stage ran the calibrator inside the
process that created it, so nothing exercised the load path from a different
entry point. The bug was invisible until a second consumer existed, and it
would have appeared for the first time on demo day. Pickling anything defined
in a `__main__` script is a latent failure that waits for the second caller —
and "we always run it the same way" is exactly the assumption that stops being
true when you ship an API.

---

## Failure 07 — `AuditLog`'s default path could not be redirected
*Sep 3, `app/services/audit_service.py`, found while writing API tests*

**Problem:** API tests appended to the real `evidence/audit_log.jsonl`. The
obvious fix — monkeypatching `audit_service.DEFAULT_PATH` — had no effect.

**Cause:** `def __init__(self, path: Path = DEFAULT_PATH)` binds the module
constant at function-definition time, so patching the module attribute later
changes nothing.

**Fix:** `path: Path | None = None`, resolved against `DEFAULT_PATH` inside the
body.

**Lesson:** A default argument is evaluated once, at import. Using a module
constant as one silently makes it un-overridable — which is a testing problem
today and a deployment problem the first time the log needs to live somewhere
else.