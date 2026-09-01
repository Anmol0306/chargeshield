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
The cost model computes expected loss as `p x loss`. That arithmetic is only
honest if `p` is a real probability — if the transactions scored 0.30 are
fraudulent about 30% of the time. A model can rank well and still be badly
calibrated, so this is verified rather than assumed.

Candidates: uncalibrated, Platt, isotonic. Fit on `val_fit` (first 70% of val,
chronological), winner chosen by Brier on `val_pick` (last 30%, unseen by
both). Fitting and choosing on the same slice would always select isotonic.

| On test | Brier | log loss | ECE | ROC-AUC |
|---|---|---|---|---|
| Uncalibrated | 0.02205 | 0.09259 | 0.0077 | 0.9014 |
| Platt (selected) | 0.02214 | 0.09233 | **0.0050** | 0.9014 |

**Those aggregate numbers flatter the model, and the reliability plot is why.**
92% of test rows score below 0.05, so an overall ECE is mostly a measurement of
the region where no decision is ever made. Restricted to `p >= 0.10` — the band
where the policy engine actually chooses between CONTEST, ACCEPT and
HUMAN_REVIEW — calibration is roughly **ten times worse**:

| On test, p ≥ 0.10 | n | ECE | expected frauds | actual | bias |
|---|---|---|---|---|---|
| Uncalibrated | 4,011 | 0.0512 | 1,644 | 1,762 | −6.7% |
| Platt (selected) | 5,748 | 0.0478 | 2,244 | 1,969 | **+14.0%** |

Platt was kept anyway, and the reason is portfolio-level: summed across all
transactions the uncalibrated model expects 2,489 frauds against an actual
3,083 — **under-predicting total fraud by 19.3%**, which distorts summed
expected loss far more than a 14% over-count confined to 5,748 rows. Platt
lands at +4.8% portfolio-wide. The residual bias also errs toward ACCEPT
(don't contest), the conservative direction for a defence-only product.

**Stated plainly: the number to quote is ECE 0.048 in the decision region, not
0.005 overall.** Calibration was worth verifying more than it was worth
applying, and the remaining miscalibration is a known limitation of the cost
figures rather than something the cost figures hide.

ROC-AUC is identical before and after because Platt is strictly monotonic and
cannot reorder. That is asserted in code, not assumed.

Reliability curve: `evaluation/charts/calibration.png`.

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
