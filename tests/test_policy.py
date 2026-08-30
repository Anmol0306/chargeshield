"""
The gate. These are the tests you show on camera.

  test_blocks_contest_when_evidence_missing
      LLM says CONTEST, delivery_proof absent -> HUMAN_REVIEW

  test_blocks_hallucinated_evidence
      LLM cites 3DS authentication, evidence set has no 3DS field -> REJECT

  test_accepts_when_fraud_probability_high
      p=0.94 -> ACCEPT, not CONTEST. Contesting genuine fraud burns cost to lose.

  test_amount_cap_forces_review
      amount > cap -> HUMAN_REVIEW regardless of everything else

  test_policy_is_pure
      decide() makes no network call and no file read
"""
