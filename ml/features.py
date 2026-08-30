"""
Feature construction. Fit on train, apply to val/test.

Starter set (skip the 339 V-columns on day one):
  TransactionAmt, ProductCD, card1-card6, addr1, addr2, dist1,
  C1-C14, D1-D15, M1-M9

LEAKAGE CHECKLIST — run before every training run:
  [ ] isFraud not in feature list
  [ ] no feature computed across train+test
  [ ] encoders fitted on train only
  [ ] no duplicate TransactionID across splits
"""
