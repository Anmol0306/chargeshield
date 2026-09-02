# Architecture

```
IEEE-CIS transaction ──► fraud model ──► calibrated P(fraud)
                                              │
synthetic dispute ────► reason code ──────────┤
(Razorpay schema,       evidence set          │
 anchored to a real                           ▼
 held-out TransactionID)              ┌───────────────┐
                                      │ LLM proposes  │  untrusted
                                      └───────┬───────┘
                                              ▼
                                      ┌───────────────┐
                                      │ POLICY ENGINE │  deterministic
                                      │ • evidence    │
                                      │ • cost thresh │
                                      │ • amount cap  │
                                      └───────┬───────┘
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                          CONTEST          ACCEPT       HUMAN_REVIEW
                              └───────────────┴───────────────┘
                                              ▼
                                          audit log
```

## Decision table
Bands are **amount-dependent**, derived in `app/policy/thresholds.py` from the
cost model. Examples below use a ₹6,070 dispute (the held-out median), where
the indifference point is `p* = 0.882` and the review band is `[0.847, 0.918]`.

| P(fraud) | Amount | Reason | Evidence | Action | Rule |
|---|---|---|---|---|---|
| any | ₹40,000 | any | any | HUMAN_REVIEW | `amount_cap_exceeded` |
| any | any | any | proposal cites evidence not on file | HUMAN_REVIEW, proposal rejected | `proposal_cited_evidence_not_on_file` |
| any | ₹400 | any | any | ACCEPT | `dispute_too_small_to_repay_representment` |
| 0.08 | ₹6,070 | Non-receipt | missing `shipping_proof` | HUMAN_REVIEW | `required_evidence_missing` |
| 0.88 | ₹6,070 | any | complete | HUMAN_REVIEW | `inside_cost_review_band` |
| 0.97 | ₹6,070 | Fraud claim | complete | ACCEPT | `fraud_probability_above_band` |
| 0.08 | ₹6,070 | Non-receipt | complete | CONTEST | `proposal_honoured` |

### Why p ≈ 0.47 is *not* a review case
An earlier version of this table sent `p = 0.47` to HUMAN_REVIEW as an
"uncertainty band". That intuition is wrong and the cost model says so: at
p = 0.47 on a ₹6,070 dispute, the expected cost of contesting is far below the
expected cost of accepting, so contesting is clearly correct and there is
nothing for a human to adjudicate.

What warrants a human is proximity to the **cost indifference point**, not
proximity to p = 0.5. Probability uncertainty and decision uncertainty are
different quantities. For this dispute the indifference point sits at
p* = 0.882, so the review band is `[0.847, 0.918]` — and it moves with the
amount, because a larger dispute is worth contesting at higher fraud risk:

| Dispute amount | p* | Review band |
|---|---|---|
| ₹2,000 | 0.643 | [0.536, 0.750] |
| ₹6,070 | 0.882 | [0.847, 0.918] |
| ₹20,000 | 0.964 | [0.954, 0.975] |

The table was changed to match the arithmetic, not the other way round.

## Failure isolation
External integrations are adapters. Core scoring and policy run with no
network access. Razorpay adapter down → queue + human review, scoring unaffected.
LLM down → deterministic template, policy gate unaffected.
