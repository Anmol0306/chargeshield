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
### Model
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
