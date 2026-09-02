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
Chosen by minimising expected INR cost per dispute on **val**, applied unchanged
to test. Never selected on test. Disputes above the amount cap are excluded from
the sweep, since the policy engine sends them to HUMAN_REVIEW regardless.

Shipped bands (`artifacts/policy_bands.json`, balanced scenario, assumed 50%
queue fraud rate): threshold **0.910**, review band **[0.595, 1.000]**, amount
cap ₹25,000.

The review band is derived, not guessed: it is the region where expected cost
is within one human review (₹150) of the optimum. It came out wide — 0.405 —
because the cost curve is genuinely flat there. That is an honest result and an
unresolved one: as it stands the band swallows the ACCEPT action entirely,
which contradicts the illustrative table in `docs/ARCHITECTURE.md`. Reconciling
the two is Day 3 work, and the band was **not** narrowed by adjusting the human
review cost to taste.

## Cost model
Illustrative merchant assumptions, swept across three scenarios. Not Razorpay's
figures. Full detail in `config/costs.yaml` and `evaluation/cost_curve.json`.

ChargeShield does not block transactions, so the two errors are not "blocked a
good customer" and "missed a fraud". They are:

| Error | Cost | Shape |
|---|---|---|
| Contest a dispute that was genuinely fraudulent | representment ops cost | fixed |
| Accept a dispute that was legitimate and winnable | the forfeited amount | scales with the transaction |

This inverts the polarity of a fraud screen: **CONTEST at low `p(fraud)`**,
ACCEPT at high. The cost-optimal threshold is analytically amount-dependent —
`p*(A) = 1 − c/(w·A)` — which is why the policy engine has an amount cap.

### The base rate is an assumption, and it decides the answer
Pricing every held-out transaction as a dispute gives a 3.5% fraud rate, at
which contesting is positive-EV almost regardless of score: the optimum is
"contest 99% of everything" and the model beats that trivial rule by ₹2 per
dispute. **That is an artefact of the population, not a finding about the
product** — a real chargeback queue is far more adverse, because a chargeback
arriving is already evidence.

So the sweep is repeated under prior shift (scores re-calibrated on the odds
scale, population re-weighted):

| Assumed dispute-queue fraud rate | t* | share contested | ₹ saved/dispute vs contest-all |
|---|---|---|---|
| 3.5% (as observed — artefactual) | 0.890 | 99.0% | 2 |
| 20% | 0.895 | 91.8% | 27 |
| 35% | 0.910 | 84.2% | 57 |
| 50% | 0.910 | 74.5% | 93 |
| 65% | 0.885 | 59.8% | 137 |

Both the embarrassing result and the informative ones are reported. Shipped
policy bands are derived at a **stated** assumed 50% queue rate, not at 3.5%.

### What is assumed, not measured
- `assumed_win_rate_if_legitimate` (0.50 / 0.70 / 0.85). Pricing a forfeited
  winnable case is impossible without it. **No win-rate claim is made anywhere
  in this project** — this is a stated assumption, swept, and never reported as
  a result.
- `assumed_dispute_fraud_rate` (0.50). IEEE-CIS has no dispute queue.
- Rupee figures inherit the decision-region calibration error (ECE ≈ 0.048).
- Headline figures are **per dispute**. Totals assume every held-out
  transaction is disputed, which is false, and are labelled where they appear.

Charts: `evaluation/charts/cost_curve.png`, `evaluation/charts/cost_prevalence.png`.

## AI evidence responder
LLM output is an untrusted proposal. See "Policy engine".

## Policy engine
`app/policy/action_policy.py` — pure functions, no I/O, no LLM calls, no clock,
no randomness. `decide()` cannot reach the network because it has no client and
cannot read a threshold from disk because thresholds arrive as an argument.
`tests/test_policy.py::test_policy_is_pure` enforces this by monkeypatching
`socket` and `open` out from under it, rather than by asserting a comment.

Override order, and why:

| # | Rule | Action | Why this position |
|---|---|---|---|
| 1 | `amount_cap_exceeded` | HUMAN_REVIEW | Exposure limit first — it holds even if the score is wrong |
| 2 | `proposal_cited_evidence_not_on_file` | HUMAN_REVIEW, proposal rejected | Before sufficiency: otherwise a fabricated citation slips through whenever real evidence happens to be complete |
| 3 | `dispute_too_small_to_repay_representment` | ACCEPT | `w·A ≤ c` — no score or evidence can change the answer, so escalating wastes ₹150 |
| 4 | `required_evidence_missing` | HUMAN_REVIEW | Cannot substantiate a contest |
| 5 | `inside_cost_review_band` | HUMAN_REVIEW | Automating is worth less than asking |
| 6 | `fraud_probability_above_band` | ACCEPT | Contesting likely-genuine fraud burns cost to lose |
| 7 | `proposal_honoured` | CONTEST | Cost-minimising, evidence on file |

Bands are amount-dependent and derived from the cost model, not chosen — see
`docs/ARCHITECTURE.md`. Every decision records which rule fired; an
unexplainable decision is not auditable, and an unauditable gate is decoration.

### Batch results over the anchored dispute queue
5,013 disputes, queue fraud rate 48.65% (`evaluation/batch_results.json`).

**The one dispute-side metric with real ground truth:**

| Wasted representment effort | |
|---|---|
| Contested | 1,803 |
| …of which genuinely fraudulent (real `isFraud`) | 672 → **37.3%** |
| Contest-everything would waste | 48.7% |
| **Relative reduction** | **23.4%** |

**Do not quote the HUMAN_REVIEW share as a finding.** It is 56.1%, dominated by
`required_evidence_missing`, and evidence availability in the synthetic queue is
a parameter (`p_required_evidence_present = 0.70`), not an observation. The
informative view is the slice where evidence is complete and the model actually
drives the decision (n=2,477): CONTEST 72.8%, ACCEPT 13.6%, HUMAN_REVIEW 13.6%.

## Failure handling
See [FAILURES.md](FAILURES.md).

## Evaluation
Full output: `evaluation/metrics.json`. Charts: `pr_curve.png`,
`error_analysis.png`.

### Held-out test (final 15%, chronological)
| Model | PR-AUC | ROC-AUC | threshold | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Logistic regression | 0.223 | 0.820 | 0.79 | 0.213 | 0.361 | 0.267 |
| LightGBM (Platt-calibrated) | **0.543** | 0.901 | 0.25 | 0.552 | 0.498 | 0.523 |

Each model at its own val-selected threshold. The baseline's scores are
rescaled by `class_weight="balanced"` and are not on the same axis as the
calibrated LightGBM's, so only PR-AUC and ROC-AUC are directly comparable.

### Policy comparison
Two comparisons exist and they say different things. Read both.

**On the dispute queue, with the REAL policy engine** (`evaluation/batch_results.json`) —
5,013 anchored disputes, realised cost against real `isFraud` labels:

| Policy | ₹/dispute | vs defend-all | vs static rule |
|---|---|---|---|
| Defend none | 12,464 | −3,631 | −3,640 |
| **ChargeShield** | 8,843 | −9 | **−18** |
| Defend all | 8,833 | 0 | −8 |
| Static amount rule | **8,825** | +8 | 0 |

**Across the whole queue the policy engine is ₹18/dispute worse than the best
no-model policy.** That is the honest number and it is reported first.

The aggregate hides two opposite effects, and the decomposition is the point:

| Segment | n | ChargeShield vs static rule |
|---|---|---|
| Evidence complete **and** under the amount cap | 2,209 | **+₹69/dispute** |
| Evidence complete | 2,477 | +₹49 |
| Evidence incomplete | 2,536 | −₹83 |
| All disputes | 5,013 | −₹18 |

Where the gate has what it needs, it wins. Where evidence is missing it loses,
because it pays ₹150 for a human rather than filing a representment it cannot
substantiate. **That is a safety property bought deliberately, not a modelling
failure** — and human review costs ₹84/dispute across the queue, larger than
the entire spread between all four policies.

Two caveats that matter more than the numbers:
- The share of disputes with incomplete evidence is a **parameter**
  (`p_required_evidence_present = 0.70` in `ml/link_disputes.py`), not an
  observation. The −₹18 net moves with a dial, not with the model.
- Human review is modelled as an analyst choosing the cost-minimising action
  given the same `p` the model saw — not an oracle. That is *conservative* for
  ChargeShield, since ChargeShield is the policy that routes work to humans.

**On transactions, with a global-threshold rule** (`evaluation/metrics.json`) —
this is *not* the shipped engine. Transactions carry no `reason_code` and no
evidence, so the gate cannot run on them. A single cost-optimal threshold on
the score beats the static amount rule by ₹84/dispute there. That number
describes a threshold rule, not this product, and an earlier version of this
README wrongly attributed it to ChargeShield. See FAILURES.md 04.

The static amount rule is the honest adversary throughout: it is the
cost-optimal policy available with **no model at all** (contest iff expected
recovery beats the representment cost), and it already captures nearly
everything defend-all does.

### Top failure mode
**Recall on `ProductCD == "W"` is 0.213, against 0.703 on every other product
type.** W is 79% of held-out transactions and carries **86% of all unrecovered
fraud value**.

The obvious reading — "the model fails when identity data is missing" — is
**wrong, and confounded**: every W transaction lacks identity data
(Jaccard 0.994 between the two slices). The 2×2 separates them:

| | identity present | identity missing |
|---|---|---|
| **W** | n=0 | recall 0.213 (n=69,468) |
| **non-W** | recall 0.703 (n=18,668) | recall 0.704 (n=445) |

Non-W transactions that also lack identity data still recall at ~0.70. Identity
missingness is not the driver; `W` is a population the model reads poorly.
Caveat: the disambiguating cell has only 27 frauds, so this is suggestive
rather than conclusive. Reproduced by `confound_check` in `ml/evaluate.py`.

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
