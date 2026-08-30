"""
Sweep threshold 0..1, compute expected INR loss per scenario in config/costs.yaml.
Pick the argmin. Write the chosen bands back for the policy engine.

expected_loss(t) = FP(t) * fp_cost + FN(t) * fn_cost

OUT  evaluation/cost_curve.json, evaluation/charts/cost_curve.png
"""
