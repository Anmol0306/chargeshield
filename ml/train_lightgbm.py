"""
LightGBM primary model.

OUT  artifacts/lgbm.pkl, artifacts/feature_metadata.json,
     evaluation/lgbm_metrics.json

Report PR-AUC, not accuracy — base rate is ~3.5%.
If PR-AUC > 0.90 on the temporal split, STOP and hunt for leakage.
"""
