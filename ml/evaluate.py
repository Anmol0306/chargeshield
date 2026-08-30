"""
Final held-out evaluation + error analysis.

Metrics: precision, recall, F1, PR-AUC, ROC-AUC, Brier, confusion matrix.
Policy comparison: defend-none / defend-all / static-rule / ChargeShield.

Error analysis is the section that makes you sound like someone who works on
models rather than someone who trained one once. Slice FP/FN by amount band,
ProductCD, and identity-present vs missing. Name the top failure mode.

OUT  evaluation/metrics.json, evaluation/charts/*.png
"""
