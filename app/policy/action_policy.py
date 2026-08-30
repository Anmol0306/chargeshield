"""
THE GATE. Pure functions only — no I/O, no LLM calls. That is what makes
"the LLM cannot move money" a structural claim rather than a promise.

decide(proposal, evidence, fraud_prob, amount) -> PolicyDecision

Overrides, in order:
  1. amount > auto_action_amount_cap_inr        -> HUMAN_REVIEW
  2. required evidence missing                  -> HUMAN_REVIEW
  3. review_band_low <= p <= review_band_high   -> HUMAN_REVIEW
  4. p > review_band_high (likely real fraud)   -> ACCEPT  (contesting burns cost to lose)
  5. otherwise                                  -> honour the proposal
"""
