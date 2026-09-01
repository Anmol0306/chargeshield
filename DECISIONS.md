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
