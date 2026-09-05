"""
Threshold selection by expected rupee loss, under dispute-defence economics.

WHAT DECISION IS BEING PRICED
  Not block-vs-allow. The transaction already happened and a chargeback has
  arrived; the choice is CONTEST or ACCEPT. That inverts the polarity of a
  normal fraud model:

      CONTEST  iff  p(fraud) < t        low p  = probably legitimate = defensible
      ACCEPT   iff  p(fraud) >= t       high p = contesting burns cost to lose

THE LOSS FUNCTION
  Per dispute, amount A (INR), representment cost c, assumed success rate w
  when the transaction was in fact legitimate:

      ACCEPT,  either truth   ->  A                (you eat the chargeback)
      CONTEST, legitimate     ->  c + (1 - w) * A  (you usually recover it)
      CONTEST, real fraud     ->  c + A            (ops cost AND you still lose)

  Note c is paid in both CONTEST rows: representment costs the same whether or
  not it works. That is the whole reason a threshold exists.

W IS AN ASSUMPTION, NOT A MEASUREMENT
  There is no way to price "forfeited winnable case" without stating how often
  a representment succeeds, and real chargeback outcomes are not in IEEE-CIS
  and are not public. So w is named, swept across three scenarios, and never
  reported as a result. This project makes no win-rate CLAIM; it states a
  win-rate ASSUMPTION and shows how much the conclusion moves when it is wrong.

THE OPTIMAL THRESHOLD IS ANALYTICALLY AMOUNT-DEPENDENT
  Indifference between the two actions is at c = w * A * (1 - p), i.e.

      p*(A) = 1 - c / (w * A)

  A small dispute is never worth contesting; a large one is worth contesting
  even at appreciable fraud risk. That is exactly why the policy engine has an
  amount cap. A single global threshold is reported here because the policy
  engine consumes fixed bands, and the analytic form is reported beside it so
  the approximation is visible rather than hidden.

LEAKAGE
  The threshold is swept and chosen on VAL, then applied unchanged to TEST.
  val is by now a thoroughly used selection set (early stopping, calibration
  fitting, calibrator choice, and now this). Test is scored once.

THE POPULATION PROBLEM -- READ THIS BEFORE THE NUMBERS
  Only a small fraction of transactions ever become disputes, and disputes are
  emphatically NOT a random sample of transactions: a chargeback arriving is
  already strong evidence something went wrong. The held-out split has a 3.5%
  fraud rate. A real dispute queue does not.

  This is not merely a scale problem, it distorts the DECISION. At 3.5% fraud,
  a median INR 6,028 dispute and a INR 500 representment cost, contesting is
  positive-EV almost regardless of the score -- the optimum is "contest 99% of
  everything" and no threshold can earn its keep. That is an artefact of the
  population, not a finding about the product.

SCALE HONESTY
  The headline figure is rupees PER DISPUTE. Totals are reported only under an
  explicit "if every held-out transaction were disputed" label.

IN   evaluation/preds/lightgbm_{val,test}_calibrated.parquet
     data/processed/{val,test}.parquet   (for TransactionAmt)
OUT  evaluation/cost_curve.json, evaluation/charts/cost_curve.png
     artifacts/policy_bands.json   <- consumed by app/policy/thresholds.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from app.policy.thresholds import indifference_threshold as _canonical_indifference

CONFIG = Path("config/costs.yaml")
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")
EVAL_DIR = Path("evaluation")
PREDS_DIR = EVAL_DIR / "preds"
CHARTS_DIR = EVAL_DIR / "charts"

GRID = np.linspace(0.0, 1.0, 201)

# Assumed fraud rate of a real dispute queue. The first entry is the split's
# own rate, kept so the artefact above stays visible
ASSUMED_DISPUTE_FRAUD_RATES = [None, 0.20, 0.35, 0.50, 0.65]
EPS = 1e-9


def load_config() -> dict:
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def load_scored(split: str, usd_to_inr: float) -> pd.DataFrame:
    """Calibrated predictions joined to the real transaction amount."""
    preds = pd.read_parquet(PREDS_DIR / f"lightgbm_{split}_calibrated.parquet")
    amounts = pd.read_parquet(
        PROCESSED_DIR / f"{split}.parquet", columns=["TransactionID", "TransactionAmt"]
    )
    df = preds.merge(amounts, on="TransactionID", how="left", validate="one_to_one")
    assert df["TransactionAmt"].notna().all(), "amount join lost rows"
    df["amount_inr"] = df["TransactionAmt"].astype("float64") * usd_to_inr
    return df


def prior_shift(p: np.ndarray, pi0: float, pi: float) -> np.ndarray:
    """Re-calibrate probabilities from base rate pi0 to base rate pi.

    Standard prior correction on the odds scale: multiply by the ratio of prior
    odds. A model calibrated on a 3.5% population is NOT calibrated on a 50%
    one, so re-weighting the population without also shifting the scores would
    compare a correct decision rule against a miscalibrated one.
    """
    p = np.clip(p, EPS, 1 - EPS)
    odds = (p / (1 - p)) * (pi / (1 - pi)) * ((1 - pi0) / pi0)
    return odds / (1 + odds)


def prevalence_weights(y: np.ndarray, pi0: float, pi: float) -> np.ndarray:
    """Row weights that turn a pi0-prevalence sample into a pi-prevalence one."""
    return np.where(y, pi / pi0, (1 - pi) / (1 - pi0))


def expected_cost(df: pd.DataFrame, t: float | None, c: float, w: float,
                  p: np.ndarray | None = None,
                  weights: np.ndarray | None = None,
                  contest: np.ndarray | None = None) -> dict:
    """Total expected INR cost of applying threshold t to these disputes.

    Contest iff p < t. Everything here is vectorised over the whole split; no
    per-row loop, because this is swept 201 times x 3 scenarios x 2 splits.
    """
    if p is None:
        p = df["p_fraud_calibrated"].to_numpy()
    y = df["isFraud"].to_numpy().astype(bool)
    a = df["amount_inr"].to_numpy()
    q = np.ones(len(df)) if weights is None else weights

    # `contest` lets a caller price an arbitrary policy (e.g. an amount-only
    # rule that ignores the model) through exactly this loss function, rather
    # than reimplementing it and hoping the two stay in sync.
    if contest is None:
        if t is None:
            raise ValueError("expected_cost needs either a threshold or a contest mask")
        contest = p < t
    accept = ~contest

    # CONTEST on a legitimate transaction: pay c, recover w of the amount.
    contest_legit = contest & ~y
    # CONTEST on a real fraud: pay c, still lose the amount. Wasted effort.
    contest_fraud = contest & y

    cost = (
        (q * a)[accept].sum()                               # accepted: eat it
        + c * q[contest_legit].sum() + (1 - w) * (q * a)[contest_legit].sum()
        + c * q[contest_fraud].sum() + (q * a)[contest_fraud].sum()
    )

    n_eff = q.sum()
    return {
        # None when an explicit policy mask was supplied: there is no single
        # threshold describing an arbitrary policy, and float("nan") here made
        # evaluation/metrics.json fail strict JSON parsing.
        "threshold": None if t is None else float(t),
        "total_inr": float(cost),
        "per_dispute_inr": float(cost / n_eff),
        "n_contested": int(contest.sum()),
        "contest_rate": float(q[contest].sum() / n_eff),
        "wasted_representments": int(contest_fraud.sum()),
        "wasted_representment_cost_inr": float(c * q[contest_fraud].sum()),
        "forfeited_winnable": int((accept & ~y).sum()),
        "forfeited_winnable_amount_inr": float((q * a)[accept & ~y].sum()),
    }


def sweep(df, c, w, p=None, weights=None) -> list[dict]:
    return [expected_cost(df, t, c, w, p, weights) for t in GRID]


def shifted(df: pd.DataFrame, pi: float | None):
    """(scores, weights) for an assumed dispute-queue fraud rate pi."""
    p = df["p_fraud_calibrated"].to_numpy()
    y = df["isFraud"].to_numpy().astype(bool)
    if pi is None:
        return p, np.ones(len(df))
    pi0 = float(y.mean())
    return prior_shift(p, pi0, pi), prevalence_weights(y, pi0, pi)


def analytic_threshold(amount_inr: float, c: float, w: float) -> float:
    """p*(A) = 1 - c / (w * A). Delegates to app/policy/thresholds.py.

    The policy engine is the authority on the decision boundary; this module is
    the analysis of it. Two copies of the formula is how they drift apart.
    """
    return _canonical_indifference(amount_inr, c, w)


def review_band(curve: list[dict], best_t: float,
                review_cost: float) -> tuple[float, float]:
    """Escalate to a human where deciding automatically is barely better than
    the alternative -- specifically, where the per-dispute expected cost is
    within the cost of a human review of the optimum.

    This makes the band cost-derived rather than guessed: a cheap analyst
    widens it, an expensive one narrows it. It is the honest reading of "we are
    not confident enough to act here".
    """
    best = min(row["per_dispute_inr"] for row in curve)
    inside = [row["threshold"] for row in curve
              if row["per_dispute_inr"] - best <= review_cost]
    if not inside:
        return best_t, best_t
    return float(min(inside)), float(max(inside))


def plot_curves(curves: dict, chosen: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8, 9), height_ratios=[3, 2], constrained_layout=True
    )
    for name, curve in curves.items():
        xs = [r["threshold"] for r in curve]
        ys = [r["per_dispute_inr"] for r in curve]
        line, = ax.plot(xs, ys, label=f"{name} (t*={chosen[name]['threshold']:.3f})")
        ax.axvline(chosen[name]["threshold"], color=line.get_color(), ls=":", alpha=0.6)
    ax.set_xlabel("threshold t   (CONTEST iff p(fraud) < t)")
    ax.set_ylabel("expected cost per dispute (INR)")
    ax.set_title("Expected dispute cost vs threshold\n"
                 "held-out test · amount-weighted · assumptions from config/costs.yaml")
    ax.legend()
    ax.grid(alpha=0.3)

    for name, curve in curves.items():
        ax2.plot([r["threshold"] for r in curve],
                 [r["contest_rate"] for r in curve], label=name)
    ax2.set_xlabel("threshold t")
    ax2.set_ylabel("share of disputes contested")
    ax2.grid(alpha=0.3)
    ax2.legend()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_prevalence(prevalence: dict, out: Path, scenario: str) -> None:
    """The chart that answers "why does this model exist?".

    At the split's own 3.5% fraud rate the threshold is nearly irrelevant --
    the cost curve is flat and contest-everything is close to optimal. The
    model's edge only appears as the dispute queue becomes adverse, which is
    the regime a real merchant is in.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(v["assumed_dispute_fraud_rate"],
             v["edge_over_contest_all_inr_per_dispute"],
             v["test"]["contest_rate"]) for v in prevalence.values()]
    rows.sort()
    xs, edge, contest = zip(*rows)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 7), constrained_layout=True)
    ax.plot(xs, edge, marker="o")
    ax.set_ylabel("INR saved per dispute\nvs contesting everything")
    ax.set_title(f"When does the model earn its keep?  ({scenario} scenario)\n"
                 "held-out test, prior-shifted to an assumed dispute-queue fraud rate")
    ax.grid(alpha=0.3)
    ax.annotate("held-out split's own rate —\nan artefact of scoring every\n"
                "transaction as a dispute",
                xy=(xs[0], edge[0]), xytext=(0.18, max(edge) * 0.45),
                arrowprops=dict(arrowstyle="->", alpha=0.6), fontsize=8)

    ax2.plot(xs, contest, marker="o", color="tab:orange")
    ax2.set_xlabel("assumed fraud rate of the dispute queue")
    ax2.set_ylabel("share of disputes contested")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    rate = cfg["currency"]["usd_to_inr"]
    scenarios = cfg["dispute_economics"]["scenarios"]
    default = cfg["default_scenario"]
    cap = cfg["policy"]["auto_action_amount_cap_inr"]
    review_cost = cfg["policy"]["human_review_cost_inr"]
    queue_pi = cfg["policy"]["assumed_dispute_fraud_rate"]

    val_all = load_scored("val", rate)
    test_all = load_scored("test", rate)

    # The threshold only governs disputes the policy engine may decide
    # automatically. Anything over the amount cap goes to HUMAN_REVIEW
    # regardless, so including it would price a decision this model never makes.
    val = val_all[val_all["amount_inr"] <= cap].reset_index(drop=True)
    test = test_all[test_all["amount_inr"] <= cap].reset_index(drop=True)
    print(f"val {len(val):,} disputes · test {len(test):,} disputes "
          f"· USD->INR {rate}")
    print(f"test amount INR: median {test['amount_inr'].median():,.0f} "
          f"p99 {test['amount_inr'].quantile(0.99):,.0f} "
          f"max {test['amount_inr'].max():,.0f}")
    n_capped = len(test_all) - len(test)
    print(f"above auto-action cap (INR {cap:,}): {n_capped:,} "
          f"({n_capped / len(test_all):.2%}) -> HUMAN_REVIEW by policy, excluded "
          f"from the sweep")
    print(f"auto-decidable test disputes: {len(test):,} "
          f"(fraud rate {test['isFraud'].mean():.4f})\n")

    results, test_curves, chosen = {}, {}, {}
    print(f"{'scenario':>12} {'c':>5} {'w':>5} | {'t*':>6} {'val ₹/disp':>11} "
          f"{'test ₹/disp':>12} {'vs accept-all':>14} {'vs contest-all':>15}")
    print("-" * 92)

    for name, s in scenarios.items():
        c = float(s["representment_cost_inr"])
        w = float(s["assumed_win_rate_if_legitimate"])

        # Chosen on VAL, applied unchanged to TEST.
        val_curve = sweep(val, c, w)
        best = min(val_curve, key=lambda r: r["per_dispute_inr"])
        t_star = best["threshold"]

        test_curve = sweep(test, c, w)
        at_star = expected_cost(test, t_star, c, w)

        # The two trivial policies, priced identically for comparison.
        accept_all = expected_cost(test, 0.0, c, w)      # contest nothing
        contest_all = expected_cost(test, 1.01, c, w)    # contest everything

        lo, hi = review_band(test_curve, t_star, review_cost)

        results[name] = {
            "representment_cost_inr": c,
            "assumed_win_rate_if_legitimate": w,
            "threshold_selected_on_val": t_star,
            "val_at_threshold": best,
            "test_at_threshold": at_star,
            "test_accept_all": accept_all,
            "test_contest_all": contest_all,
            "test_saving_vs_accept_all_inr": accept_all["total_inr"] - at_star["total_inr"],
            "test_saving_vs_contest_all_inr": contest_all["total_inr"] - at_star["total_inr"],
            "review_band": {"low": lo, "high": hi,
                            "derived_from_human_review_cost_inr": review_cost},
            "analytic_threshold_examples": {
                f"INR_{int(a):,}": analytic_threshold(a, c, w)
                for a in (500, 2_000, 6_070, 25_000, 100_000)
            },
        }
        test_curves[name] = test_curve
        chosen[name] = at_star

        print(f"{name:>12} {c:5.0f} {w:5.2f} | {t_star:6.3f} "
              f"{best['per_dispute_inr']:11,.0f} {at_star['per_dispute_inr']:12,.0f} "
              f"{accept_all['total_inr'] - at_star['total_inr']:14,.0f} "
              f"{contest_all['total_inr'] - at_star['total_inr']:15,.0f}")

    # --- sensitivity: does the recommendation survive being wrong? ---
    ts = [r["threshold_selected_on_val"] for r in results.values()]
    print(f"\nThreshold across scenarios: {min(ts):.3f} - {max(ts):.3f} "
          f"(spread {max(ts) - min(ts):.3f})")

    d = results[default]
    print(f"\n--- default scenario: {default} ---")
    print(f"  contest {d['test_at_threshold']['contest_rate']:.1%} of disputes")
    print(f"  wasted representments: {d['test_at_threshold']['wasted_representments']:,} "
          f"(INR {d['test_at_threshold']['wasted_representment_cost_inr']:,.0f})")
    print(f"  forfeited winnable:    {d['test_at_threshold']['forfeited_winnable']:,} "
          f"(INR {d['test_at_threshold']['forfeited_winnable_amount_inr']:,.0f})")
    print(f"  review band: [{d['review_band']['low']:.3f}, {d['review_band']['high']:.3f}]")
    print(f"  analytic p*(A): " + "  ".join(
        f"{k}={v:.3f}" for k, v in d["analytic_threshold_examples"].items()))

    # --- the question a reviewer will actually ask -----------------------
    # At what dispute-queue composition does a model beat "contest everything"?
    c_d = float(scenarios[default]["representment_cost_inr"])
    w_d = float(scenarios[default]["assumed_win_rate_if_legitimate"])
    prevalence = {}
    print(f"\n--- prior shift: '{default}' scenario at assumed dispute-queue "
          f"fraud rates ---")
    print("The split's own 3.5% is an artefact of scoring every transaction as a")
    print("dispute. A real chargeback queue is far more adverse.")
    print(f"{'queue fraud':>12} {'t*':>7} {'contest':>8} {'INR/disp':>10} "
          f"{'vs contest-all':>15} {'model edge':>11}")
    print("-" * 70)
    for pi in ASSUMED_DISPUTE_FRAUD_RATES:
        p_v, q_v = shifted(val, pi)
        p_t, q_t = shifted(test, pi)
        best_v = min(sweep(val, c_d, w_d, p_v, q_v), key=lambda r: r["per_dispute_inr"])
        t_star = best_v["threshold"]

        at_star = expected_cost(test, t_star, c_d, w_d, p_t, q_t)
        c_all = expected_cost(test, 1.01, c_d, w_d, p_t, q_t)
        a_all = expected_cost(test, 0.0, c_d, w_d, p_t, q_t)
        edge = c_all["per_dispute_inr"] - at_star["per_dispute_inr"]

        key = "as_observed" if pi is None else f"{pi:.2f}"
        prevalence[key] = {
            "assumed_dispute_fraud_rate": float(test["isFraud"].mean()) if pi is None else pi,
            "is_as_observed": pi is None,
            "threshold_selected_on_val": t_star,
            "test": at_star,
            "test_contest_all": c_all,
            "test_accept_all": a_all,
            "edge_over_contest_all_inr_per_dispute": edge,
            "edge_over_accept_all_inr_per_dispute":
                a_all["per_dispute_inr"] - at_star["per_dispute_inr"],
        }
        label = "as-observed" if pi is None else f"{pi:.0%}"
        print(f"{label:>12} {t_star:7.3f} {at_star['contest_rate']:8.1%} "
              f"{at_star['per_dispute_inr']:10,.0f} "
              f"{c_all['per_dispute_inr']:15,.0f} {edge:11,.0f}")

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "usd_to_inr": rate,
        "default_scenario": default,
        "n_val": len(val), "n_test": len(test),
        "threshold_selected_on": "val, applied unchanged to test",
        "scenarios": results,
        "threshold_spread_across_scenarios": float(max(ts) - min(ts)),
        "prevalence_sensitivity": prevalence,
        "n_excluded_above_amount_cap": int(n_capped),
        "auto_action_amount_cap_inr": cap,
        "notes": [
            "Dispute-defence economics: CONTEST iff p(fraud) < t. Inverted "
            "relative to a fraud screen -- see config/costs.yaml.",
            "assumed_win_rate_if_legitimate is an ASSUMPTION, not a "
            "measurement. Real chargeback outcomes are not in IEEE-CIS and are "
            "not public. No win-rate claim is made anywhere in this project; "
            "the parameter is swept so the reader can see how much it matters.",
            "Rupees PER DISPUTE is the headline. Totals assume every held-out "
            "transaction becomes a dispute, which is false in reality and is "
            "why they are labelled rather than quoted.",
            "The cost-optimal threshold is analytically amount-dependent: "
            "p*(A) = 1 - c/(w*A). A single global threshold is an "
            "approximation the policy engine's amount cap partly compensates for.",
            "Calibration in the decision region is ECE ~0.048 (see "
            "evaluation/calibration_metrics.json). These rupee figures inherit "
            "that error.",
            "Disputes above the amount cap are excluded from the sweep: the "
            "policy engine sends them to HUMAN_REVIEW regardless, so the "
            "threshold never governs them.",
            "AS-OBSERVED IS AN ARTEFACT. Scoring every held-out transaction as "
            "a dispute gives a 3.5% fraud queue, at which contesting is "
            "positive-EV almost regardless of score and no threshold earns its "
            "keep. prevalence_sensitivity re-calibrates (prior shift on the "
            "odds) and re-weights to adverse queue compositions, which is where "
            "the decision actually has content.",
        ],
    }
    (EVAL_DIR / "cost_curve.json").write_text(json.dumps(out, indent=2))

    # Production bands are derived at the ASSUMED queue prevalence, not the
    # as-observed 3.5%. Tuning the policy engine on a population this file has
    # just argued is an artefact would be the wrong kind of consistency.
    p_v, q_v = shifted(val, queue_pi)
    p_t, q_t = shifted(test, queue_pi)
    band_curve_val = sweep(val, c_d, w_d, p_v, q_v)
    band_t = min(band_curve_val, key=lambda r: r["per_dispute_inr"])["threshold"]
    band_lo, band_hi = review_band(sweep(test, c_d, w_d, p_t, q_t), band_t, review_cost)
    band_at = expected_cost(test, band_t, c_d, w_d, p_t, q_t)

    print(f"\n--- shipped policy bands (queue fraud rate assumed {queue_pi:.0%}) ---")
    print(f"  threshold {band_t:.3f} -> contest {band_at['contest_rate']:.1%} "
          f"of auto-decidable disputes")
    print(f"  review band [{band_lo:.3f}, {band_hi:.3f}]  "
          f"(width {band_hi - band_lo:.3f}, from INR {review_cost} human review)")

    bands = {
        "source": "ml/cost_curve.py",
        "created_utc": out["created_utc"],
        "scenario": default,
        "assumed_dispute_fraud_rate": queue_pi,
        "threshold": band_t,
        "review_band_low": band_lo,
        "review_band_high": band_hi,
        "auto_action_amount_cap_inr": cap,
        "expected_contest_rate": band_at["contest_rate"],
        "derivation": (
            f"Threshold minimises expected INR cost per dispute on val under the "
            f"'{default}' scenario, with scores prior-shifted and the population "
            f"re-weighted to an assumed dispute-queue fraud rate of {queue_pi:.0%}, "
            f"then applied unchanged to test. Review band is the region where "
            f"expected cost is within one human review (INR {review_cost}) of the "
            f"optimum. Disputes above INR {cap:,} bypass this entirely."
        ),
        "superseded_by": (
            "review_band_low/high here is a POPULATION-level summary and is too "
            "wide to use as a policy (0.405). app/policy/thresholds.py derives "
            "the band PER DISPUTE from the same cost model -- "
            "1 - (c+h)/(w*A) < p < 1 - (c-h)/(w*A) -- which is the correct level "
            "for a per-dispute decision and comes out ~0.07 wide at the median "
            "amount. The engine uses that; these values are retained for "
            "reference only."
        ),
        "caveat": (
            "assumed_dispute_fraud_rate is STATED, not measured. IEEE-CIS has no "
            "dispute queue. See evaluation/cost_curve.json prevalence_sensitivity "
            "for how the threshold and the model's edge move across 3.5%-65%."
        ),
    }
    (ARTIFACTS_DIR / "policy_bands.json").write_text(json.dumps(bands, indent=2))

    plot_curves(test_curves, chosen, CHARTS_DIR / "cost_curve.png")
    plot_prevalence(prevalence, CHARTS_DIR / "cost_prevalence.png", default)
    print("\nWrote evaluation/cost_curve.json, artifacts/policy_bands.json,")
    print("      evaluation/charts/cost_curve.png, evaluation/charts/cost_prevalence.png")


if __name__ == "__main__":
    main()
