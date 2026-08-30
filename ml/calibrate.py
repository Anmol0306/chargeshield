"""
Calibrate raw scores on the VALIDATION slice (never test).
Compare uncalibrated vs Platt (sigmoid) vs isotonic; pick by Brier score.

Required because the policy layer consumes probabilities, not rankings.
A cost-optimal threshold on uncalibrated scores is meaningless.

OUT  artifacts/calibrator.pkl, evaluation/charts/calibration.png
"""
