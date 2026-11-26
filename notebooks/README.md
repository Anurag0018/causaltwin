# Notebooks — Phase 5 benchmark

## `benchmark.ipynb` — CausalTwin vs the correlational LightGBM baseline

Readable front-end for the Phase 5 benchmark. The executable source of
truth is `backend/scripts/benchmark_baseline.py`; the notebook runs the
same `run_benchmark()` and carries the measured results
(`backend/scripts/benchmark_results.json`).

The comparison, on the synthetic ground-truth SCM (the only tier where
interventional claims are valid — see the main README's honest-claims
rule), with both models trained on the same normal-operation data:

1. **Intervention prediction** — sustained `do(fan_speed_nc)` overrides
   (ground truth via `app.faults` step hooks) → intervention RMSE,
   causal vs correlational.
2. **In-distribution prediction** — held-out natural run, no
   interventions → plain prediction RMSE for both (the honesty check).
3. **Root cause** — the Phase 4 fault set → causal RCA Top-1 / Top-3 vs
   LightGBM global gain-importance ranking.

Headline (2026-11-26): intervention RMSE **8.21** (causal) vs **18.46**
(LightGBM) points; RCA Top-1 **1.000** vs **0.250**. Full tables are in
the notebook and in the main README's Results section. No number ships
without the script reproducing it.
