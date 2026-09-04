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
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection),
a public benchmark of real e-commerce transactions contributed by Vesta
Corporation. **Not Razorpay data.** The `isFraud` label is externally supplied;
this project did not author it.

`TransactionAmt` is USD. Rupee figures are a labelled cost overlay at a stated
conversion rate (`config/costs.yaml`), not a property of the dataset.

#### What `isFraud` means — and why this project does not depend on the answer
A description of the labelling methodology circulates widely and is attributed
to the competition host: that a reported chargeback marks a transaction fraud,
that later transactions linked by card, user account, email or billing address
are marked fraud too, that transactions with nothing reported within 120 days
are marked legitimate, and that genuinely fraudulent activity which was never
reported is therefore labelled legitimate.

**That attribution is repeated here as unverified.** It appears in secondary
sources rather than one I could open: the host discussion thread
([101203](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203))
requires a Kaggle account, and this project does not cite sources it has not
read. If you can open it, verify it — do not take this paragraph as
confirmation.

**The project is built so that it does not matter.** Nothing here requires the
label to be chargeback-derived. What it requires is weaker and verifiable:

- a real, externally-supplied binary label on real transactions, which is what
  makes the fraud metrics honest rather than self-scored, and
- that same label carried onto anchored disputes, which is what makes *wasted
  representment effort* a measurement rather than a simulation.

If the labelling methodology turns out to be something else entirely, every
number in this README still stands, because none of them is derived from it.
What would change is only how closely the modelled task resembles a real
chargeback queue — which is already declared as a limitation below, alongside
the fact that IEEE-CIS is US e-commerce rather than Indian payments.

If the widely-repeated description *is* accurate, it implies label noise in a
known direction: unreported fraud is labelled legitimate, so measured precision
is a pessimistic estimate. This project does not claim that correction.

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

**The deterministic path was written first** (`app/services/template_response.py`),
so the LLM is an enhancement to a system that already works rather than a
dependency of it. `propose()` never raises: no key, no network, a timeout,
malformed JSON, or schema violations twice in a row all end at the template.
The demo runs with no network and no credential at all.

The template *cannot* fabricate evidence — `cited_evidence` is derived from the
evidence object, so no code path produces a citation for a document that does
not exist. That makes it the safe floor; an LLM proposal can only be worse on
that axis, which is why the gate checks fabrication on every proposal
regardless of source.

**`llm_service` validates shape, not content.** It deliberately does *not*
strip fabricated citations before handing the proposal on — if it repaired
them, the policy engine's fabrication rule would never fire and there would be
no audit record showing the model tried it.

### What is sent to the provider
Reason code, dispute amount, the evidence **field names** on file, and the
requirement list. Deliberately not sent: document ids (opaque, useless to the
model) and **`p_fraud`** — the model's job is evidence assessment and drafting;
the probability is the deterministic system's input. Keeping them apart means
the LLM cannot launder a score into a decision.

### Credentials
`LLM_API_KEY` from the process environment (`OPENAI_API_KEY` accepted as a
fallback). **No module in this project reads `.env`.** The key is never logged,
printed, placed in an exception message, written to the audit log, or returned;
every string this module logs passes through a scrubber as defence in depth,
and provider exceptions are logged by type rather than message. Enforced by
`tests/test_failure_modes.py`, which raises a provider error containing the key
and asserts it never reaches `caplog`.

`LLM_MODEL` and `LLM_BASE_URL` make the provider swappable — point `LLM_BASE_URL`
at any OpenAI-compatible endpoint (OpenAI, Groq, Google AI Studio) with no code
change. JSON mode is the one call parameter that varies between them, so a 400
naming `response_format` triggers one retry without it rather than a failure;
a 429 or 401 deliberately does not.

```bash
make llm-check    # one call per scenario, reports credential presence only
```

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
See [FAILURES.md](FAILURES.md) for what broke during the build and why.

```bash
make demo                     # everything, ~180 lines
make demo ARGS=--pause        # stop between scenarios — use this to record
make demo ARGS="--only 2"     # just the fabrication case
make demo ARGS=--list         # list the scenarios
```

Runs with **no network and no API credential**. The full run does not fit on
one terminal screen, so `--pause` waits for Enter between scenarios — each one
fits in frame and can be narrated separately.

Eight scenarios, each asserting its expected action and rule. **The script
exits non-zero if any outcome changes**, so the thing being recorded and the
thing being tested are the same artifact — a policy change that would
invalidate the video breaks the build instead of being discovered on playback
(`tests/test_demo.py`).

| # | Scenario | Result |
|---|---|---|
| 01 | Required evidence missing | HUMAN_REVIEW · `required_evidence_missing` |
| 02 | Proposal cites evidence never collected | HUMAN_REVIEW, proposal rejected · `proposal_cited_evidence_not_on_file` |
| 03 | Provider returns malformed JSON | retries once → template → CONTEST |
| 04 | Provider unreachable | template → CONTEST |
| 05 | Above the exposure cap | HUMAN_REVIEW · `amount_cap_exceeded` |
| 06 | Likely genuine fraud | ACCEPT · `fraud_probability_above_band` |
| 07 | Too small to repay a representment | ACCEPT · `dispute_too_small_to_repay_representment` |
| 08 | Same p(fraud), two amounts | ₹6,070 → HUMAN_REVIEW; ₹2,000 → ACCEPT |

Scenario 02 is the one to watch: the evidence set is **complete**, so the
contest would otherwise be allowed. The proposal invents an
`access_activity_log` that was never collected, and the gate names it.

Scenario 08 shows the cost model rather than describing it — identical score,
identical evidence, different amount, different action, because the review band
is derived per dispute from `1 − (c±h)/(w·A)`.

`tests/test_demo.py` also asserts that **every policy rule that can fire has a
demo scenario**, so the demo cannot silently fall behind the engine. That test
is what caught scenarios 07 and 08 being missing.

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
```bash
git clone <repo> && cd chargeshield
# place train_transaction.csv and train_identity.csv in data/raw/
python3 -m venv .venv && make setup
make all      # ~83s end to end
make api      # http://127.0.0.1:8000
```

`make all` runs: `data → baseline → train → calibrate → cost → evaluate →
link → batch → test`. `make setup` uses `.venv/bin/python` when present and
falls back to `python3`, so no target requires an activated venv.

**Verified from a clean clone** (Sep 4): fresh `git clone`, fresh venv,
`requirements.txt` only, Kaggle CSVs linked in. `make all` completed in 83s
with 159 tests passing, and every headline number reproduced **bit-exactly**
against the committed artifacts — baseline PR-AUC 0.223152, LightGBM 0.543100,
best iteration 610, Platt selected, decision-region ECE 0.047840, 5,013
disputes at queue fraud rate 0.486535, wasted representment 0.372712,
ChargeShield ₹8,842.56/dispute vs static rule ₹8,824.66. The API served
`/health`, `/score`, `/metrics` and the fabrication demo from that clone with
no extra setup.

The only things not in the repo are the Kaggle CSVs (1.35GB, gitignored) and
`.env` (secrets). Everything else is regenerated.

## API
```
make api      # uvicorn app.main:app --reload  → http://127.0.0.1:8000/docs
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model artifacts loaded; whether an LLM credential is present (presence only, never any part of the value) |
| `POST /score` | Calibrated `p_fraud` from a **partial** feature vector, plus the amount-dependent bands |
| `POST /disputes/analyze` | Proposal (LLM or template) → policy decision → audit record |
| `POST /disputes/draft` | Representment draft, returned **inside** the decision. Never submits anything |
| `POST /disputes/validate` | Gate a caller-supplied proposal. No LLM. **This is the demo endpoint** |
| `POST /batch/run` | Re-run the gate over the dispute queue (`?limit=`, `?write_audit=`) |
| `GET /audit/{id}` | The record for a decision |

**No endpoint returns a bare proposal.** A caller cannot obtain the LLM's
suggestion without also obtaining the action the policy engine reached — an
endpoint that returned ungated output would defeat the entire design.

`/score` takes partial feature vectors because no caller has all 392. LightGBM
learns a NaN direction at every split, so an absent feature is handled exactly
as it was in training, where 76% of rows had no identity data. The response
reports `features_supplied` so a caller can see whether it got a prediction
from four features or four hundred.

### The demo, in one call
```bash
curl -s localhost:8000/disputes/validate -H 'content-type: application/json' -d '{
  "dispute_id":"disp_demo","reason_code":"NON_RECEIPT","amount_inr":6070,
  "p_fraud":0.05,"evidence":{"shipping_proof":["doc_a"],"billing_proof":["doc_b"]},
  "proposal":{"decision":"CONTEST","evidence_status":"SUFFICIENT",
    "cited_evidence":["shipping_proof","access_activity_log"],"missing_evidence":[],
    "reasoning_summary":"3-D Secure authentication confirms the cardholder.",
    "draft_representment":"The transaction was authenticated via 3-D Secure."}}'
```
The evidence set is complete, so the contest would otherwise be allowed. The
proposal cites an authentication record that was never collected:

```
action  HUMAN_REVIEW
rule    proposal_cited_evidence_not_on_file
fabricated_evidence  ["access_activity_log"]
```

## Future work
