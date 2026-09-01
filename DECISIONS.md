# Decisions

What I cut, and why. Five days, one person.
A defensible cut is engineering; an undefended one is scope creep.

| Date | Decision | Reason | Cost accepted |
|---|---|---|---|
| Aug 30 | SQLite over PostgreSQL | Single-process batch analytical workload. Postgres would be resume decoration. | No concurrent writes |
| Aug 30 | Plain HTML over React | ~2h vs ~1 day for the same screen. Metrics don't depend on the UI. | Less polished demo |
| Aug 30 | No agent framework | Workflow is retrieve → classify → propose → validate → gate. A graph library adds indirection, not capability. | — |
| Sep 1 | Exclude 7 high-cardinality ID columns (`card1`, `DeviceInfo`, `card2`, `addr1`, `card5`, `card3`, `addr2`) from the linear baseline | Anonymised codes: numeric magnitude is meaningless to a linear model, and one-hot on `card1` alone is +12,242 columns fit on noise. 46 inputs -> 97 features instead of ~13,000. | Baseline understates achievable performance. Deliberate — it is a floor, and LightGBM splits on these natively |
| Sep 1 | `class_weight="balanced"` on the baseline LR | At 3.5% prevalence an unweighted model at t=0.5 predicts ~nothing positive and reports recall ~0. Makes point metrics non-degenerate. | Destroys calibration. Baseline scores are rankings, not probabilities — calibration is a separate step |
| Sep 1 | Threshold selected on val, applied unchanged to test | Selecting the operating point on test is a second, subtler leak than the split itself. | Test F1 is lower than a test-tuned number would be. That is the point |
| Sep 1 | LightGBM gets the 339 V-columns and all 7 identifiers; the baseline keeps neither | Makes the two models a controlled contrast rather than an algorithm bake-off. The gap measures what the baseline's constraint cost. | Two feature sets to maintain; `ml/metrics.py` shared so scoring is identical |
| Sep 1 | Native LightGBM categoricals via one train-fitted `CategoricalDtype`, not per-split `.astype("category")` | pandas assigns codes per frame in order of appearance; LightGBM stores splits as integer codes. Per-split casting silently scrambles categories at inference. Verified on this data — the mappings genuinely differ. | A learned artifact that must ship with the model |
| Sep 1 | `card1/2/3/5`, `addr1/2` numeric for the tree, not categorical | 12,242 levels overfits LightGBM's categorical splitter; a tree can carve an arbitrary partition of a numeric axis given enough splits. | `DeviceInfo` (string, 1,546 levels) has no numeric fallback and stays categorical |
| Sep 1 | No `scale_pos_weight` on LightGBM | Threshold comes from the val PR curve anyway, so re-weighting buys nothing and damages the probability scale `ml/calibrate.py` needs. The baseline needed `class_weight` only to make t=0.5 non-degenerate. | Point metrics at t=0.5 are degenerate; irrelevant, we don't use t=0.5 |
| Sep 1 | Calibrators fit on `val_fit` (first 70% of val), winner chosen on `val_pick` (last 30%) | Fitting Platt and isotonic on val and then picking by Brier on the same val always picks isotonic — it has more freedom and partly fits val's noise. That is a formality, not a comparison. | val now does three jobs (early stopping, calibrator fit, method selection) and is not a held-out estimate |
| Sep 1 | Selection by Brier, not ECE | Brier is a proper scoring rule, so it rewards being correct *and* honest about uncertainty. ECE alone is gamed by a model that predicts the base rate for everything. | ECE still reported — it is the interpretable number |
| Sep 1 | Kept Platt despite a near-tie | ECE 0.0077 → 0.0050 on test is a real halving of the calibration gap; Brier is a wash (0.02205 → 0.02214). Two parameters, cannot overfit. | Honest framing: calibration was worth *verifying* more than it was worth *applying* |
