# BorderScore design notes

## Problem
A lender entering a new country has a few hundred labelled loans at best, a risk function that differs somewhat from existing markets, and a regulator-like expectation that scores are calibrated (ECE matters as much as AUC, because scores feed pricing and limits). The system must (1) decide how to bootstrap a model, (2) change models safely, (3) notice when a market drifts.

## Key decisions and trade-offs

| Decision | Chosen | Alternative | Why / cost |
|---|---|---|---|
| Main model | One pooled model with one-hot country | Per-country models; hierarchical model | Pooling shares strength with the small market; per-country models would be starved. Cost: one bad country can affect others, and the country effect is only additive for trees/NN unless splits pick it up. |
| New-market strategy | Compared 5 strategies empirically | Commit to one a priori | Which transfer works depends on how different the market is; the experiment is the product. Cost: fixed hyper-parameters, 3 seeds only, results are on synthetic data. |
| Promotion gate | Strict AUC gain **and** strict ECE gain, champion re-scored on current fixed holdout | Compare stored metrics; AUC only | Re-scoring removes holdout drift between runs; ECE is guarded because discrimination-only gates ship miscalibrated models. Cost: conservative, so equal models never swap and a tiny AUC gain with slightly worse ECE is rejected. No significance test yet (see roadmap). |
| Selection vs gating data | Challenger picked on validation, gated on holdout | Pick on holdout | Avoids selection leakage into the gate. Cost: holdout is reused across runs, so it slowly overfits to the process. |
| ECE bins | Equal-frequency | Equal-width | Defaults are 4-14%, equal-width bins collapse into one. Cost: bin edges vary by model. |
| Drift reference | Holdout (unseen by the model) | Training data | Training scores are overfit, which would make live data look drifted. |
| Orchestration | Plain DAG runner (tested) + Airflow DAG over the same task table | Airflow only | Runs and tests with no services. Cost: the Airflow DAG file is not covered by automated tests. |
| Task communication | Files in a run directory | XCom / in-memory | Works across Airflow workers and is inspectable. Cost: no schema validation or lineage tooling. |
| Tracking | MLflow with SQLite and local artifacts | Server + object store | Zero-setup and reproducible. Cost: single-writer, not for concurrent teams. |
| Model storage | joblib pickles in a file registry | MLflow Model Registry / ONNX | Simple, keeps wrapper classes intact. Cost: pickles are unsafe for untrusted sources and tied to library versions. |

## Known limitations
- Data is synthetic; the source model transfers unusually well because countries share the same feature set and similar risk functions. Real markets differ in bureau coverage and data availability.
- PSI cannot see pure label shift; the report surfaces it only through default rate vs mean score and ECE. In production, labels arrive months late, so that signal is delayed.
- No policy layer: scores are not yet turned into approve/decline decisions or expected-loss estimates.
- No fairness/segment analysis, no model explainability.

## One-year roadmap
1. **Q1: Decisioning.** Add a policy simulator (cutoff by country, approval rate vs expected loss, with LGD and APR assumptions) and make the gate report expected-loss impact next to AUC/ECE.
2. **Q1: Statistical gating.** Bootstrap confidence intervals for the AUC/ECE difference so the gate requires significance instead of a point-estimate win; add a per-country no-regression guardrail.
3. **Q2: Calibration layer.** Per-country isotonic/Platt recalibration on a small local sample, evaluated in the cold-start curve (likely the cheapest win at low n).
4. **Q2: Transfer research.** Hierarchical/Bayesian country effects, domain-adaptation (importance weighting using covariate shift estimates), and a "transferability" score that predicts from a small sample whether to pool, warm-start or fine-tune.
5. **Q3: Production monitoring.** Scheduled monitoring on real batches, delayed-label performance tracking, alert routing, automatic retrain triggers tied to PSI and calibration drift; shadow deployment of challengers before gating.
6. **Q3: Platform hardening.** Move to MLflow Model Registry with stage transitions, containerised Airflow in CI (DAG import tests, an end-to-end smoke run), data contracts for features.
7. **Q4: Governance.** Model cards per country, fairness/segment reports, reason codes, audit trail from `history.jsonl` into a proper store, and a playbook for onboarding a country (minimum labelled sample size, exit criteria from the cold-start curve).
