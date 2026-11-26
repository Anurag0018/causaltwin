"""Phase 5 — CausalTwin vs a correlational LightGBM baseline.

The project's thesis is "prediction != intervention". This benchmark is
designed to test exactly that, fairly, on the synthetic ground-truth SCM
(the only tier where interventional claims are valid):

  Both models are trained on the SAME normal-operation data
  (generate(n=2000, seed=0)) and compared on three tasks:

  a) INTERVENTION PREDICTION (headline). Predict the failure_risk CHANGE
     from sustained do(fan_speed_nc = natural + delta) overrides, delta
     in {+40, +80, +120} rpm, under two scenarios: normal operation and
     the Phase 4 cooling fault (coolant_temp +1.2 z). Ground truth comes
     from app.faults.simulate_scm step hooks (the true dynamics, same
     seed, settled window means) — the Phase 4 protocol.
       - Causal entry: the Phase 4 structural simulator (own-lag AR
         equations + exact steady-state shift propagation).
       - Baseline: a LightGBM regressor predicting failure_risk from the
         current state snapshot (all other nodes at t, incl. the lever
         and power_draw) — the standard "just ML" KPI model. Its only
         available what-if is the one practitioners actually use: bump
         the lever feature in the base run's snapshots and re-predict,
         every other feature held at its natural value (a single-output
         correlational model has no mechanism to move the mediators).
     Metric: RMSE of predicted vs actual failure_risk change, per
     intervention and overall.

  b) IN-DISTRIBUTION PREDICTION (honesty check). Same models, held-out
     natural run (seed=1), no interventions: one-step / snapshot
     failure_risk RMSE. If the correlational model wins here, that is
     reported plainly — it is the expected half of the thesis.

  c) ROOT-CAUSE COMPARISON. The Phase 4 fault set (4 fault nodes x
     seeds 0/1/2, same windows/protocol):
       - Causal RCA: residual z-score ranking, Phase 4 protocol
         (recomputed here; cross-checked against phase4_results.json).
       - Baseline: the LightGBM model's GLOBAL gain importance ranking.
         Importance is one ranking for all incidents — it cannot localise
         a specific incident. That asymmetry is real and is stated, not
         hidden; the baseline is scored exactly as a practitioner would
         use it (rank candidate causes by importance).

Protocol, fault set, windows and seeds are fixed in FULL_CONFIG below
(pre-registered — they are the Phase 4 values, not tuned for this run).

    python scripts/benchmark_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import data_loader, faults
from app.root_cause import rank_root_causes
from app.simulator import StructuralEquations

TARGET = "failure_risk"
LEVER = "fan_speed_nc"
FAN_SCALE = 80.0  # rpm per z (data_loader._SCALE)

FULL_CONFIG = {
    # --- training (shared by both models) ---------------------------------
    "n_train": 2000,
    "train_seed": 0,
    # --- (b) in-distribution ----------------------------------------------
    "n_test": 2000,
    "test_seed": 1,
    # --- (a) interventions --------------------------------------------------
    "intervention_seed": 0,
    "fan_deltas_rpm": [40.0, 80.0, 120.0],
    "scenarios": ["normal", "cooling_fault"],
    "cooling_fault": {"node": "coolant_temp", "delta_z": 1.2},
    "intervention_window": [1000, 1600],
    "settled_window": [1250, 1600],
    # --- (c) root cause (the Phase 4 pre-registered set) --------------------
    "rca_n": 2000,
    "rca_faults": [
        {"node": "coolant_temp", "delta_z": 1.2},
        {"node": "hpc_outlet_temp", "delta_z": 1.2},
        {"node": "vibration", "delta_z": 1.2},
        {"node": "oil_pressure", "delta_z": -1.2},
    ],
    "rca_seeds": [0, 1, 2],
    "rca_window": [1000, 1060],
    # --- baseline hyperparameters (fixed, not tuned on the benchmark) -------
    "lgbm_params": {"n_estimators": 500, "learning_rate": 0.05,
                    "num_leaves": 31, "random_state": 0, "verbose": -1},
}

FEATURES = [c for c in data_loader.NODES if c != TARGET]


# --------------------------------------------------------------------- models
def train_baseline(train_df, params: dict):
    """The correlational baseline: LightGBM snapshot regressor.

    Predicts failure_risk(t) from the current state (all other nodes at
    the same t, including the lever and the pure correlate power_draw).
    """
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**params)
    model.fit(train_df[FEATURES], train_df[TARGET])
    return model


def baseline_counterfactual_change(model, base_df, delta_rpm: float,
                                   settled: slice) -> float:
    """What the snapshot model answers to 'what if the lever were higher?'

    Bump the lever feature by delta in the base run's settled-window
    snapshots, re-predict, and difference against the model's own
    predictions on the unmodified snapshots. Every other feature stays
    at its natural value — the only counterfactual a single-output
    correlational regressor can express.
    """
    snap = base_df.iloc[settled]
    bumped = snap.copy()
    bumped[LEVER] = bumped[LEVER] + float(delta_rpm)
    pred_nat = float(model.predict(snap[FEATURES]).mean())
    pred_do = float(model.predict(bumped[FEATURES]).mean())
    return pred_do - pred_nat


# -------------------------------------------------------------------- sections
def in_distribution_rmse(se: StructuralEquations, model, cfg: dict) -> dict:
    """(b) Held-out natural run, no interventions, for both models."""
    test_df, _, _ = data_loader.generate(n_steps=cfg["n_test"],
                                         seed=cfg["test_seed"])
    # Baseline: snapshot predictions on the held-out run.
    pred = model.predict(test_df[FEATURES])
    base_rmse = float(np.sqrt(np.mean((pred - test_df[TARGET].to_numpy()) ** 2)))
    # Causal: one-step structural predictions (residual = actual - pred).
    resids = np.array([se.residuals_at(test_df, t)[TARGET]
                       for t in range(2, len(test_df))])
    causal_rmse = float(np.sqrt(np.mean(resids ** 2)))
    return {"test_seed": cfg["test_seed"], "n_test": cfg["n_test"],
            "causal_rmse": causal_rmse, "baseline_rmse": base_rmse,
            "note": "baseline predicts failure_risk(t) from the state at t; "
                    "causal predicts it from the structural parents at their "
                    "true lags (one-step). Each is scored the way that model "
                    "is actually used."}


def intervention_benchmark(se: StructuralEquations, model, cfg: dict) -> dict:
    """(a) Predicted vs actual failure_risk change under do(lever)."""
    start, end = cfg["intervention_window"]
    settled = slice(*cfg["settled_window"])
    seed = cfg["intervention_seed"]
    fault_spec = cfg["cooling_fault"]

    bases: dict[str, tuple] = {}
    if "normal" in cfg["scenarios"]:
        df_nat, _ = faults.simulate_scm(n_steps=cfg["n_train"], seed=seed)
        bases["normal"] = (df_nat, None)
    if "cooling_fault" in cfg["scenarios"]:
        fault = {**fault_spec, "start": start, "end": end}
        df_flt, _ = faults.simulate_scm(n_steps=cfg["n_train"], seed=seed,
                                        fault=fault)
        bases["cooling_fault"] = (df_flt, fault)

    rows = []
    for scenario, (df_base, fault) in bases.items():
        base_fr = float(df_base[TARGET].iloc[settled].mean())
        for delta_rpm in cfg["fan_deltas_rpm"]:
            step = {"node": LEVER, "delta_z": delta_rpm / FAN_SCALE,
                    "start": start, "end": end}
            df_do, _ = faults.simulate_scm(n_steps=cfg["n_train"], seed=seed,
                                           fault=fault, step=step)
            actual = float(df_do[TARGET].iloc[settled].mean()) - base_fr
            causal_pred = float(se.intervene(LEVER, delta_rpm, TARGET)
                                ["target_shift"])
            base_pred = baseline_counterfactual_change(model, df_base,
                                                       delta_rpm, settled)
            rows.append({
                "scenario": scenario, "fan_delta_rpm": float(delta_rpm),
                "actual_change": actual,
                "causal_predicted_change": causal_pred,
                "baseline_predicted_change": base_pred,
                "causal_abs_error": abs(causal_pred - actual),
                "baseline_abs_error": abs(base_pred - actual),
            })
            print(f"    [{scenario:>13}] fan +{delta_rpm:5.0f} rpm: "
                  f"actual {actual:+7.2f} | causal {causal_pred:+7.2f} | "
                  f"baseline {base_pred:+7.2f}")

    def _rmse(key, subset):
        return float(np.sqrt(np.mean([r[key] ** 2 for r in subset])))

    out = {"rows": rows,
           "causal_rmse": _rmse("causal_abs_error", rows),
           "baseline_rmse": _rmse("baseline_abs_error", rows),
           "per_scenario": {}}
    for scenario in bases:
        sub = [r for r in rows if r["scenario"] == scenario]
        out["per_scenario"][scenario] = {
            "causal_rmse": _rmse("causal_abs_error", sub),
            "baseline_rmse": _rmse("baseline_abs_error", sub),
        }
    return out


def root_cause_benchmark(model, cfg: dict) -> dict:
    """(c) Causal RCA vs LightGBM global gain importance, Phase 4 set."""
    # --- causal RCA, Phase 4 protocol, recomputed -------------------------
    win_start, win_end = cfg["rca_window"]
    causal_cases = []
    for spec in cfg["rca_faults"]:
        for seed in cfg["rca_seeds"]:
            fault = {**spec, "start": win_start, "end": win_end}
            df_fault, _ = faults.simulate_scm(n_steps=cfg["rca_n"],
                                              seed=seed, fault=fault)
            ranking = rank_root_causes(
                df_fault, data_loader.GROUND_TRUTH_EDGES,
                t=win_end - 1, window=win_end - win_start,
                fit_df=df_fault.iloc[:win_start])
            rank = next(r["rank"] for r in ranking
                        if r["node"] == spec["node"])
            causal_cases.append({"fault_node": spec["node"], "seed": seed,
                                 "true_node_rank": int(rank),
                                 "top1_hit": bool(rank == 1),
                                 "top3_hit": bool(rank <= 3)})
    causal = {"cases": causal_cases,
              "top1": float(np.mean([c["top1_hit"] for c in causal_cases])),
              "top3": float(np.mean([c["top3_hit"] for c in causal_cases]))}

    # Cross-check against the published Phase 4 numbers when available.
    p4_path = Path(__file__).resolve().parent / "phase4_results.json"
    if p4_path.is_file():
        p4 = json.loads(p4_path.read_text())
        causal["phase4_results_crosscheck"] = {
            "phase4_top1": p4["rca"]["phase4"]["top1_accuracy"],
            "phase4_top3": p4["rca"]["phase4"]["top3_accuracy"],
            "matches": bool(
                p4["rca"]["phase4"]["top1_accuracy"] == causal["top1"]
                and p4["rca"]["phase4"]["top3_accuracy"] == causal["top3"]),
        }

    # --- baseline: global gain importance ---------------------------------
    gains = model.booster_.feature_importance(importance_type="gain")
    importance_ranking = [f for f, _ in sorted(zip(FEATURES, gains),
                                               key=lambda kv: -kv[1])]
    imp_cases = []
    for spec in cfg["rca_faults"]:
        for seed in cfg["rca_seeds"]:
            rank = importance_ranking.index(spec["node"]) + 1
            imp_cases.append({"fault_node": spec["node"], "seed": seed,
                              "true_node_rank": int(rank),
                              "top1_hit": bool(rank == 1),
                              "top3_hit": bool(rank <= 3)})
    importance = {
        "ranking": importance_ranking,
        "gain": {f: float(g) for f, g in zip(FEATURES, gains)},
        "cases": imp_cases,
        "top1": float(np.mean([c["top1_hit"] for c in imp_cases])),
        "top3": float(np.mean([c["top3_hit"] for c in imp_cases])),
        "note": "ONE global ranking scored against every incident: "
                "importance has no per-incident view, so its rank for a "
                "fault node is the same in all 12 cases. That is the "
                "standard way practitioners use feature importance for "
                "diagnosis, and the asymmetry vs per-incident RCA is "
                "part of what is being measured — stated, not hidden.",
    }
    return {"causal": causal, "importance": importance,
            "fault_set": cfg["rca_faults"], "seeds": cfg["rca_seeds"],
            "window": cfg["rca_window"]}


# ------------------------------------------------------------------------- main
def run_benchmark(cfg: dict | None = None) -> dict:
    """Run the full Phase 5 comparison; returns the results dict."""
    cfg = {**FULL_CONFIG, **(cfg or {})}
    t0 = time.time()
    results = {
        "phase": "Phase 5 — CausalTwin vs correlational LightGBM baseline",
        "protocol": {
            "training": f"both models trained on generate(n={cfg['n_train']}, "
                        f"seed={cfg['train_seed']}) — normal operation only",
            "baseline": "LightGBM snapshot regressor: failure_risk(t) from "
                        "all other nodes at t (incl. the lever and "
                        "power_draw); counterfactual = bump the lever "
                        "feature, other features held natural",
            "causal": "Phase 4 structural simulator: own-lag AR equations "
                      "fit on the same training data (ground-truth edges), "
                      "exact steady-state shift propagation",
            "intervention_truth": "app.faults.simulate_scm step hooks, "
                                  "same seed, settled-window means",
            "config": {k: v for k, v in cfg.items()},
        },
    }

    train_df, ground_truth, _ = data_loader.generate(
        n_steps=cfg["n_train"], seed=cfg["train_seed"])
    se = StructuralEquations().fit(train_df, ground_truth)
    model = train_baseline(train_df, cfg["lgbm_params"])

    print("== (b) In-distribution prediction (held-out natural run) ==")
    results["in_distribution"] = in_distribution_rmse(se, model, cfg)
    b = results["in_distribution"]
    print(f"  causal one-step RMSE {b['causal_rmse']:.3f} | "
          f"baseline snapshot RMSE {b['baseline_rmse']:.3f}")

    print("\n== (a) Intervention prediction (do(fan_speed_nc), settled change) ==")
    results["intervention"] = intervention_benchmark(se, model, cfg)
    a = results["intervention"]
    print(f"  => intervention change RMSE: causal {a['causal_rmse']:.3f} | "
          f"baseline {a['baseline_rmse']:.3f} points")

    print("\n== (c) Root cause: causal RCA vs LightGBM gain importance ==")
    results["root_cause"] = root_cause_benchmark(model, cfg)
    c = results["root_cause"]
    print(f"  causal RCA:      Top-1 {c['causal']['top1']:.3f} | "
          f"Top-3 {c['causal']['top3']:.3f}")
    print(f"  importance rank: Top-1 {c['importance']['top1']:.3f} | "
          f"Top-3 {c['importance']['top3']:.3f}")
    print(f"  importance ranking: {' > '.join(c['importance']['ranking'])}")
    if "phase4_results_crosscheck" in c["causal"]:
        xc = c["causal"]["phase4_results_crosscheck"]
        print(f"  cross-check vs phase4_results.json: "
              f"{'MATCH' if xc['matches'] else 'MISMATCH'} "
              f"(Phase 4 published Top-1 {xc['phase4_top1']:.3f})")

    results["runtime_seconds"] = round(time.time() - t0, 2)
    return results


def main():
    results = run_benchmark()
    out = Path(__file__).resolve().parent / "benchmark_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out} (total runtime {results['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
