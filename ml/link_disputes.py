"""
THE SPINE. Anchor each synthetic dispute to a real held-out TransactionID.

Each dispute carries the transaction's real isFraud label AND the model's
calibrated fraud probability. This is what connects the two halves and gives
the dispute layer a real label underneath it.

Enables the one dispute-side metric measured against real ground truth:
  wasted representment effort = disputes contested where isFraud == 1

Dispute fields mirror Razorpay's dispute entity:
  id, payment_id, amount, currency, reason_code, respond_by, status,
  phase, evidence{}

OUT  data/processed/disputes.json
CLAIM LIMIT: this measures wasted effort. It does NOT measure win rate.
"""
