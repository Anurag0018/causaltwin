# Notebooks — Phase 5 benchmark (placeholder)

## Planned: `benchmark_vs_lgbm.ipynb`

Reproducible benchmark of CausalTwin against a **correlational LightGBM
baseline**, run on the synthetic ground-truth SCM (the only tier where
interventional claims are valid — see the main README's honest-claims rule):

1. Generate the synthetic SCM at several seeds / lengths (`app.data_loader`).
2. Run PCMCI discovery (`app.discovery`) → report **SHD** and **Edge F1**
   vs the known ground truth.
3. Estimate treatment→KPI effects (`app.estimator`, LinearDML) → **ATE error**
   vs the generator's true coefficients.
4. Intervene: apply `do(X = x')` shifts and compare CausalTwin's rollout
   (`app.simulator`) against a LightGBM regressor trained on the same
   observational data → **intervention RMSE**, causal vs correlational,
   including under a distribution shift (changed load regime).
5. Inject upstream faults → **root-cause Top-1 / Top-3 recall**
   (`app.root_cause`).

All numbers produced here are the ones that go in the main README's
benchmark table. No number ships without this notebook reproducing it.
