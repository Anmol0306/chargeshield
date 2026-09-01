# ChargeShield

Cost-sensitive fraud risk scoring + bounded chargeback defense.

> ChargeShield scores transaction fraud risk on a temporally held-out public
> benchmark, then uses that calibrated score plus evidence availability to
> decide which chargebacks a merchant should contest — with a deterministic
> policy layer the LLM cannot override.

Razorpay AI Buildathon 2026 · Track 02, AI Risk Manager · Defense-only.

---

## Problem
A merchant facing a chargeback has three bad options: accept and lose, contest
blindly and burn ops cost on unwinnable cases, or review everything by hand.
Nobody tells them which disputes are worth fighting.

## Architecture
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## ML methodology
### Dataset
IEEE-CIS Fraud Detection (public benchmark). **Not** Razorpay data — the label
is externally supplied, not authored by this project.
<!-- TODO: cite the host's labelling-methodology post precisely, or drop the
     chargeback-derivation claim entirely. Do not attribute it to Kaggle's
     column description. -->
`TransactionAmt` is USD. Rupee figures are a labelled cost overlay at a stated
conversion rate (config/costs.yaml), not a property of the dataset.

### Time split
Chronological on `TransactionDT`: 70 / 15 / 15. Enforced by
`tests/test_split.py`, which fails on any temporal overlap.
All three splits come from `train_transaction.csv` — the competition's
`test_transaction.csv` has no labels.

### Data leakage controls
Structural, not disciplinary. Every learned transform is fit on train and only
on train, and the fitted object is the sole path from raw row to model input.

- Baseline: one sklearn `Pipeline`. `.fit` is called once; val/test only ever
  reach `.predict_proba`.
- LightGBM: one `CategoricalDtype` per categorical column, learned on train and
  applied unchanged to val/test. Per-split `.astype("category")` would assign
  integer codes per frame and silently scramble categories at inference.
- `tests/test_preprocessing.py` and `tests/test_lightgbm_encoding.py` assert
  both, including that the naive version would genuinely differ on this data —
  a leakage test that cannot fail is not a test.
- The operating threshold is selected on val and applied unchanged to test.
  Picking it on test is a second, subtler leak than the split itself.

### Model
| Model | Features | Test PR-AUC | Test ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| Logistic regression (baseline) | 46 in → 97 | 0.223 | 0.820 | 0.213 | 0.361 |
| LightGBM (primary) | 392 | **0.543** | 0.901 | 0.553 | 0.497 |

The baseline is a deliberate floor, not a weak attempt. It uses a constrained
starter feature set, one-hot encodes the manageable low-cardinality
categoricals, and deliberately excludes high-cardinality identifier fields to
control dimensionality — `card1` alone would add ~12,242 dummy columns, and an
anonymised code carries no ordinal meaning a linear model can use.

LightGBM receives exactly what the baseline was denied: those 7 identifier
columns plus the 339 V-columns. So the gap is not "one algorithm beat another";
it measures what the constraint cost. Four of the tree's top eight features by
gain — `DeviceInfo`, `card2`, `card1`, `addr1` — are columns the baseline
excluded, which is the direct evidence for that reading.

Val metrics are reported but are a **selection** estimate: early stopping and
the threshold sweep both read val. Test is the number quoted.

### Calibration
### Threshold selection

## Cost model
Illustrative merchant assumptions, swept across three scenarios. Not Razorpay's
figures.

## AI evidence responder
LLM output is an untrusted proposal. See "Policy engine".

## Policy engine
`app/policy/action_policy.py` — pure functions, no I/O, no LLM calls.

## Failure handling
See [FAILURES.md](FAILURES.md).

## Evaluation
| Metric | Population | Label source |
|---|---|---|
| Precision, recall, PR-AUC, Brier | IEEE-CIS held-out final 15% | Real (`isFraud`) |
| Expected loss vs threshold | same | Real × stated cost assumptions |
| Wasted representment effort | anchored disputes | Real (`isFraud` on anchor txn) |
| Evidence-gate correctness, schema validity, blocked proposals | ~50 controlled cases | Constructed by design |

## Limitations
**Read this section before the numbers above.**

- The dispute/evidence layer is synthetic, conforming to Razorpay's documented
  dispute entity shape. Each dispute is anchored to a real held-out transaction
  so it carries a real `isFraud` label.
- **No win-rate claim is made anywhere in this project.** Real chargeback
  outcomes require merchant-side dispute resolution labels, which are not
  public. Synthetic data is used only to demonstrate deterministic policy
  behaviour, never to claim predictive performance.
- Evidence-gate metrics measure policy correctness on constructed cases. They
  are true by construction and are reported as such.
- IEEE-CIS is US e-commerce, not Indian payments. The methodology transfers;
  the specific numbers do not.

## Running locally
```
make setup
make data        # requires data/raw/train_transaction.csv
make all
make api
```

## API
`POST /score` · `POST /disputes/analyze` · `POST /disputes/draft` ·
`POST /disputes/validate` · `POST /batch/run` · `GET /audit/{id}` · `GET /health`

## Future work
