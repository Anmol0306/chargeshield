"""Guards for two defects found in the Day-3 audit that would return silently.

1. evaluation/metrics.json contained NaN, which is valid Python json.dumps
   output but INVALID JSON. Every strict parser -- browser JSON.parse, most
   API clients -- rejects it. It passed every test we had because Python's own
   json.loads accepts NaN by default.

2. ml/evaluate.py labelled a global-threshold rule "chargeshield", so the
   README reported the value of a policy that is not the shipped one.
"""
import glob
import json
import pathlib

import pytest

ARTIFACT_GLOBS = ["evaluation/*.json", "artifacts/*.json", "evidence/*.json"]


def _strict(path: str) -> dict:
    """Reject NaN/Infinity the way a browser or a JSON API would."""
    def boom(const):
        raise ValueError(f"invalid JSON constant {const!r} in {path}")
    return json.loads(pathlib.Path(path).read_text(), parse_constant=boom)


@pytest.mark.parametrize("path", sorted(
    p for g in ARTIFACT_GLOBS for p in glob.glob(g)))
def test_artifact_is_strictly_valid_json(path):
    _strict(path)


def test_evaluate_does_not_claim_to_price_the_real_gate():
    """ml/evaluate.py works on transactions, which carry no reason_code and no
    evidence, so it CANNOT run the real gate. It must not imply otherwise."""
    p = pathlib.Path("evaluation/metrics.json")
    if not p.exists():
        pytest.skip("run `make evaluate` first")
    policies = _strict(str(p))["policy_comparison"]["policies"]
    assert "chargeshield" not in policies, (
        "ml/evaluate.py is labelling a global-threshold rule 'chargeshield'. "
        "The shipped policy engine is priced in evaluation/batch_results.json."
    )
    assert "global_cost_threshold_rule" in policies


def test_batch_results_price_the_real_gate():
    p = pathlib.Path("evaluation/batch_results.json")
    if not p.exists():
        pytest.skip("run `make batch` first")
    r = _strict(str(p))
    assert "chargeshield" in r["policy_comparison"]
    for k in ("defend_none", "defend_all", "static_amount_rule"):
        assert k in r["policy_comparison"]
    seg = r["policy_comparison_by_segment"]
    for k in ("evidence_complete", "evidence_incomplete",
              "actionable_complete_and_under_cap", "human_review_overhead"):
        assert k in seg


def test_chargeshield_row_is_produced_by_the_actual_decide_function():
    """Not a label check -- run the real gate and confirm price_policies uses it."""
    from app.policy.action_policy import decide
    from app.policy.thresholds import PolicyConfig
    from app.services.batch_runner import price_policies

    cfg = PolicyConfig(
        representment_cost_inr=500.0, assumed_win_rate_if_legitimate=0.70,
        human_review_cost_inr=150.0, auto_action_amount_cap_inr=25_000.0,
        scenario="balanced", assumed_dispute_fraud_rate=0.50,
        global_threshold=0.91, source="test")
    reqs = {"NON_RECEIPT": {"required": ["shipping_proof"], "optional": []}}

    # A dispute the gate must send to HUMAN_REVIEW (evidence missing) but which
    # a naive threshold rule would contest.
    d = {"id": "disp_1", "reason_code": "NON_RECEIPT", "evidence": {},
         "_chargeshield": {"p_fraud_calibrated": 0.01, "amount_inr": 6_070.0,
                           "anchor_is_fraud": False, "evidence_complete": False}}
    dec = decide(config=cfg, p_fraud=0.01, amount_inr=6_070.0,
                 reason_code="NON_RECEIPT", evidence={}, requirements=reqs)
    assert dec.action == "HUMAN_REVIEW"

    priced = price_policies([(d, dec)], cfg)
    # HUMAN_REVIEW costs h + the analyst's action; a pure contest rule does not.
    assert priced["chargeshield"]["per_dispute_inr"] != \
        priced["defend_all"]["per_dispute_inr"]


def test_canonical_indifference_threshold_has_one_implementation():
    """ml/cost_curve.analytic_threshold must delegate to app/policy/thresholds,
    or the cost analysis and the policy engine can silently disagree about the
    decision boundary."""
    from app.policy.thresholds import indifference_threshold
    from ml.cost_curve import analytic_threshold

    for amount in (500.0, 2_000.0, 6_070.0, 25_000.0, 100_000.0):
        assert analytic_threshold(amount, 500.0, 0.70) == \
            indifference_threshold(amount, 500.0, 0.70)
