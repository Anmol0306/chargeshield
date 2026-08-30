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
| P(fraud) | Reason | Evidence | Action | Why |
|---|---|---|---|---|
| 0.91 | Fraud claim | complete | ACCEPT | Likely genuinely fraudulent — contesting burns cost to lose |
| 0.08 | Non-receipt | complete | CONTEST | Legitimate txn, defensible |
| 0.08 | Non-receipt | missing delivery proof | HUMAN_REVIEW | Can't substantiate |
| 0.47 | any | any | HUMAN_REVIEW | Uncertainty band |
| any | any | amount > cap | HUMAN_REVIEW | Exposure limit |

## Failure isolation
External integrations are adapters. Core scoring and policy run with no
network access. Razorpay adapter down → queue + human review, scoring unaffected.
LLM down → deterministic template, policy gate unaffected.
