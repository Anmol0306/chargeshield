"""Dispute-defence cost model.

The polarity here is inverted relative to a fraud screen and the loss function
has four cases, so it is hand-checked against arithmetic done on paper rather
than against itself.
"""
import numpy as np
import pandas as pd
import pytest

from ml.cost_curve import (
    analytic_threshold,
    expected_cost,
    prevalence_weights,
    prior_shift,
)


def frame(p, y, amt):
    return pd.DataFrame({"p_fraud_calibrated": np.array(p, float),
                         "isFraud": np.array(y, int),
                         "amount_inr": np.array(amt, float)})


# --- the loss function, checked by hand -----------------------------------

def test_loss_function_matches_hand_arithmetic():
    """Four disputes, one of each case. c=500, w=0.70, A=10,000 throughout.
    CONTEST iff p < t; use t=0.5.

      p=0.1 y=0  -> CONTEST legit : 500 + 0.30*10000 = 3,500
      p=0.1 y=1  -> CONTEST fraud : 500 +      10000 = 10,500
      p=0.9 y=0  -> ACCEPT        :             10000 = 10,000
      p=0.9 y=1  -> ACCEPT        :             10000 = 10,000
                                          total = 34,000  (8,500 per dispute)
    """
    df = frame([0.1, 0.1, 0.9, 0.9], [0, 1, 0, 1], [10_000] * 4)
    r = expected_cost(df, t=0.5, c=500, w=0.70)
    assert r["total_inr"] == pytest.approx(34_000.0)
    assert r["per_dispute_inr"] == pytest.approx(8_500.0)
    assert r["n_contested"] == 2
    assert r["wasted_representments"] == 1              # the p=0.1, y=1 row
    assert r["wasted_representment_cost_inr"] == pytest.approx(500.0)
    assert r["forfeited_winnable"] == 1                 # the p=0.9, y=0 row
    assert r["forfeited_winnable_amount_inr"] == pytest.approx(10_000.0)


def test_polarity_is_contest_when_p_is_low():
    """The inversion that makes this a defence product and not a fraud screen.
    A confidently-legitimate dispute must be contested, not accepted."""
    df = frame([0.01], [0], [10_000])
    contested = expected_cost(df, t=0.5, c=500, w=0.70)
    accepted = expected_cost(df, t=0.0, c=500, w=0.70)
    assert contested["n_contested"] == 1
    assert accepted["n_contested"] == 0
    assert contested["total_inr"] < accepted["total_inr"], (
        "contesting a legitimate dispute must cost less than eating it"
    )


def test_contesting_real_fraud_is_pure_waste():
    """Cost of contesting a fraud exceeds accepting it by exactly c."""
    df = frame([0.99], [1], [10_000])
    accepted = expected_cost(df, t=0.0, c=500, w=0.70)
    contested = expected_cost(df, t=1.01, c=500, w=0.70)
    assert contested["total_inr"] - accepted["total_inr"] == pytest.approx(500.0)


def test_analytic_threshold_matches_numerical_indifference():
    """p*(A) = 1 - c/(w*A) must be the point where the two actions cost the same."""
    c, w, a = 500.0, 0.70, 10_000.0
    p_star = analytic_threshold(a, c, w)
    assert p_star == pytest.approx(1 - c / (w * a))

    # At p just below p*, contesting wins; just above, accepting wins.
    for delta, cheaper_to_contest in ((-0.02, True), (+0.02, False)):
        p = p_star + delta
        df = frame([p], [0], [a])
        contest = c + (1 - w) * a
        accept = a
        # expected cost under uncertainty about y, weighting by p
        exp_contest = (1 - p) * contest + p * (c + a)
        assert (exp_contest < accept) == cheaper_to_contest


def test_small_disputes_are_never_worth_contesting():
    """c=500 against a INR 500 dispute: even a certain win does not repay it."""
    assert analytic_threshold(500.0, 500.0, 0.70) == 0.0


# --- prior shift ----------------------------------------------------------

def test_prior_shift_is_identity_at_the_same_base_rate():
    p = np.array([0.01, 0.2, 0.5, 0.9, 0.99])
    np.testing.assert_allclose(prior_shift(p, 0.035, 0.035), p, rtol=1e-9)


def test_prior_shift_raises_probabilities_for_an_adverse_queue():
    p = np.array([0.05, 0.3, 0.8])
    shifted = prior_shift(p, 0.035, 0.50)
    assert np.all(shifted > p), "a more fraudulent queue must raise every score"
    assert np.all(np.diff(shifted) > 0), "prior shift must preserve ranking"


def test_prior_shift_stays_a_probability():
    p = np.linspace(1e-6, 1 - 1e-6, 1000)
    for pi in (0.01, 0.2, 0.5, 0.9):
        s = prior_shift(p, 0.035, pi)
        assert s.min() >= 0.0 and s.max() <= 1.0
        assert not np.isnan(s).any()


def test_prevalence_weights_produce_the_target_rate():
    y = np.array([1] * 35 + [0] * 965)          # 3.5% prevalence
    for pi in (0.20, 0.50, 0.65):
        q = prevalence_weights(y.astype(bool), 0.035, pi)
        assert (q * y).sum() / q.sum() == pytest.approx(pi, rel=1e-9)


def test_weighted_cost_reduces_to_unweighted_when_weights_are_one():
    df = frame([0.1, 0.6, 0.9], [0, 1, 0], [5_000, 8_000, 12_000])
    a = expected_cost(df, 0.5, 500, 0.7)
    b = expected_cost(df, 0.5, 500, 0.7, weights=np.ones(3))
    assert a["total_inr"] == pytest.approx(b["total_inr"])
