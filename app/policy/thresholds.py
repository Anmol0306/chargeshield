"""Risk bands from the cost model. Cost-derived, not guessed.

This module is the ONLY place in the policy package that touches disk. It reads
the artifacts produced by ml/cost_curve.py and freezes them into a PolicyConfig
that app/policy/action_policy.py takes as an argument. That split is what makes
`decide()` a pure function, and what makes tests/test_policy.py's purity test
mean something rather than being a slogan.

THE BAND IS AMOUNT-DEPENDENT, AND THAT IS NOT A REFINEMENT -- IT IS THE FIX
  ml/cost_curve.py's first attempt reported a global review band of
  [0.595, 1.000], which is 0.405 wide and swallows the ACCEPT action entirely.
  The error was in what was being compared: it measured where one GLOBAL
  POLICY's total cost sits near the optimum. That is a statement about
  policies, not about an individual dispute.

  Per dispute, with amount A, representment cost c, assumed success rate w:

      E[cost | CONTEST] = c + A * (1 - w * (1 - p))
      E[cost | ACCEPT]  = A
      difference        = c - w * A * (1 - p)

  Escalate to a human when deciding automatically is barely better than the
  alternative -- when |difference| is smaller than what a human costs, h:

      1 - (c + h) / (w * A)  <  p  <  1 - (c - h) / (w * A)

  centred on the indifference point p*(A) = 1 - c / (w * A). At the median
  INR 6,070 dispute that is roughly [0.85, 0.92]. Narrow, and derived rather
  than tuned.

WHY THIS CONTRADICTS THE ORIGINAL ARCHITECTURE SKETCH
  docs/ARCHITECTURE.md originally sent p ~ 0.47 to HUMAN_REVIEW as an
  "uncertainty band". That intuition is wrong: at p = 0.47 on a INR 6,000
  dispute, contesting is clearly correct and there is nothing for a human to
  adjudicate. What warrants a human is proximity to the COST indifference
  point, not proximity to p = 0.5. Probability uncertainty and decision
  uncertainty are different quantities. The architecture was updated to match
  the arithmetic, not the other way round.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

BANDS_PATH = Path("artifacts/policy_bands.json")
COSTS_PATH = Path("config/costs.yaml")


def indifference_threshold(amount_inr: float, representment_cost_inr: float,
                           win_rate: float) -> float:
    """p*(A) = 1 - c / (w * A), clipped to [0, 1]. THE canonical implementation.

    Module-level and dependency-free on purpose: ml/cost_curve.py imports this
    rather than carrying its own copy. Two implementations of the decision
    boundary is how the cost analysis and the policy engine end up silently
    disagreeing about what the model recommends.
    """
    denom = win_rate * amount_inr
    if denom <= 0:
        return 0.0
    return min(1.0, max(0.0, 1.0 - representment_cost_inr / denom))


def review_band(amount_inr: float, representment_cost_inr: float, win_rate: float,
                human_review_cost_inr: float) -> tuple[float, float]:
    """The p-range around p*(A) where automating is worth less than asking."""
    denom = win_rate * amount_inr
    if denom <= 0:
        return 0.0, 0.0
    low = 1.0 - (representment_cost_inr + human_review_cost_inr) / denom
    high = 1.0 - (representment_cost_inr - human_review_cost_inr) / denom
    return min(1.0, max(0.0, low)), min(1.0, max(0.0, high))


@dataclass(frozen=True)
class PolicyConfig:
    """Immutable. Everything decide() needs, and nothing it could mutate."""

    representment_cost_inr: float
    assumed_win_rate_if_legitimate: float
    human_review_cost_inr: float
    auto_action_amount_cap_inr: float
    scenario: str
    assumed_dispute_fraud_rate: float
    global_threshold: float
    source: str

    def indifference_threshold(self, amount_inr: float) -> float:
        """p*(A) = 1 - c / (w * A), clipped to [0, 1].

        Below this, contesting is cheaper in expectation. Above it, accepting
        is. A dispute smaller than c / w can never repay a representment, so
        p* clips to 0 and the dispute is always accepted.
        """
        return indifference_threshold(
            amount_inr, self.representment_cost_inr,
            self.assumed_win_rate_if_legitimate)

    def review_band(self, amount_inr: float) -> tuple[float, float]:
        """The p-range around p*(A) where automating is worth less than asking.

        Width is set by human_review_cost_inr: a cheap analyst widens the band,
        an expensive one narrows it. That is the honest reading of "we are not
        confident enough to act here" -- confidence measured in rupees, not in
        distance from 0.5.
        """
        return review_band(
            amount_inr, self.representment_cost_inr,
            self.assumed_win_rate_if_legitimate, self.human_review_cost_inr)


def load_policy_config(bands_path: Path = BANDS_PATH,
                       costs_path: Path = COSTS_PATH) -> PolicyConfig:
    """The one I/O call in this package. Everything downstream is pure."""
    bands = json.loads(bands_path.read_text())
    costs = yaml.safe_load(costs_path.read_text())
    scenario = bands["scenario"]
    sc = costs["dispute_economics"]["scenarios"][scenario]

    return PolicyConfig(
        representment_cost_inr=float(sc["representment_cost_inr"]),
        assumed_win_rate_if_legitimate=float(sc["assumed_win_rate_if_legitimate"]),
        human_review_cost_inr=float(costs["policy"]["human_review_cost_inr"]),
        auto_action_amount_cap_inr=float(bands["auto_action_amount_cap_inr"]),
        scenario=scenario,
        assumed_dispute_fraud_rate=float(bands["assumed_dispute_fraud_rate"]),
        global_threshold=float(bands["threshold"]),
        source=bands["source"],
    )
