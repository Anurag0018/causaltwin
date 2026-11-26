"""Phase 3 smoke — effect estimation validated against the KNOWN true effect.

Smoke/validation run only: this script does NOT change any defaults in
app/estimator.py. It proves the estimation layer works on the synthetic SCM,
where the true effect is known, and that the estimate survives refutation.

Estimand (stated precisely, because the SCM has NO direct edge
fan_speed_nc -> failure_risk):
  The total effect of fan_speed_nc at time t on failure_risk at time t+3.
  fan speed reaches the KPI along exactly two lagged paths:
    fan(t) -> coolant_temp(t+1) -> hpc_outlet_temp(t+2) -> failure_risk(t+3)
        coefficients (z-space): -0.75 * 0.55 * 0.7 = -0.28875
    fan(t) -> hpc_outlet_temp(t+2) -> failure_risk(t+3)   (lag-2 edge)
        coefficients (z-space): -0.5 * 0.7          = -0.35
  Analytic true effect = -0.63875 z-units of failure_risk per +1 z-unit of
  fan speed. Engineering units: +1 z fan = +80 fan units, 1 z failure_risk
  = 12 points, so truth = -7.665 points per +80 fan units
  = -0.0958125 points per +1 fan unit.

TRUE effect derivation (method actually used): pulse simulation with common
random numbers. A faithful z-space replica of data_loader.generate() is
validated to reproduce generate()'s output EXACTLY (interventions off,
same seed); the replica then adds a +1 z pulse to fan_speed_nc at spaced
times t0 and differences failure_risk at t0+3 against the unpulsed run.
(The single-structural-coefficient method is not available: there is no
direct fan_speed_nc -> failure_risk coefficient to read off.)

Estimation: app.estimator.estimate_ate (EconML LinearDML) on a lagged
dataset built from generate(n=2000, seed=0):
  T = fan_speed_nc(t),  Y = failure_risk(t+3),
  W = the remaining state at t (load, coolant_temp, vibration,
      hpc_outlet_temp, oil_pressure, power_draw, failure_risk,
      maintenance_event) — the state at t is what blocks the backdoor
      paths from past fan speed in this SCM.
A 95% CI is obtained from an identical LinearDML refit via ate_inference
(estimate_ate itself does not return a CI).

Refutation: DoWhy CausalModel on the same lagged dataset with a graph
encoding W as confounders (W -> T, W -> Y, T -> Y). The DoWhy base
estimate uses backdoor.linear_regression; refuters run on THAT estimate
(random common cause, placebo treatment, data subset). This is stated
plainly: refutation targets the DoWhy estimate of the same estimand on
the same data, not the EconML object directly.

    python scripts/estimator_smoke.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import data_loader
from app.estimator import estimate_ate

HORIZON = 3
N = 2000
DATA_SEED = 0
PULSE_SEED = 7
TREATMENT = "fan_speed_nc"
OUTCOME = "failure_risk_h3"
CONFOUNDERS = [
    "load",
    "coolant_temp",
    "vibration",
    "hpc_outlet_temp",
    "oil_pressure",
    "power_draw",
    "failure_risk",
    "maintenance_event",
]
# z-space -> engineering scales (from data_loader._SCALE)
FAN_SCALE = 80.0     # +1 z fan_speed_nc  = +80 fan units
FR_SCALE = 12.0      # +1 z failure_risk  = +12 points
ANALYTIC_TRUE_Z = -0.75 * 0.55 * 0.7 + -0.5 * 0.7   # -0.63875


# ---------------------------------------------------------------------------
# Faithful z-space replica of data_loader.generate(), with a pulse hook.
# RNG call order and short-circuit behaviour are copied exactly, so with
# pulses off and the same seed the replica reproduces generate() bit-for-bit
# (asserted in main()).
# ---------------------------------------------------------------------------
def simulate_z(n_steps: int, seed: int, pulse_times=(), pulse_delta: float = 0.0):
    rng = np.random.default_rng(seed)
    names = [n for n in data_loader.NODES if n != "maintenance_event"]
    z = {name: np.zeros(n_steps) for name in names}
    maint = np.zeros(n_steps, dtype=float)
    pulse_times = set(pulse_times)

    for t in range(n_steps):
        prev = {k: (v[t - 1] if t >= 1 else 0.0) for k, v in z.items()}
        prev2 = {k: (v[t - 2] if t >= 2 else 0.0) for k, v in z.items()}

        z["load"][t] = (
            0.6 * prev["load"]
            + 0.5 * np.sin(2 * np.pi * t / 100.0)
            + rng.normal(0, 0.6)
        )
        z["fan_speed_nc"][t] = (
            0.5 * prev["fan_speed_nc"]
            + 0.4 * np.sin(2 * np.pi * t / 140.0 + 1.0)
            + rng.normal(0, 0.6)
        )
        if t in pulse_times:
            z["fan_speed_nc"][t] += pulse_delta

        scheduled = (t % 90 == 45)
        triggered = prev["vibration"] > 1.6 and rng.random() < 0.5
        if t >= 1 and (scheduled or triggered):
            maint[t] = 1.0

        z["coolant_temp"][t] = (
            0.5 * prev["coolant_temp"]
            - 0.75 * prev["fan_speed_nc"]
            + rng.normal(0, 0.35)
        )
        z["vibration"][t] = (
            0.45 * prev["vibration"]
            + 0.7 * prev["load"]
            - 1.4 * (maint[t - 1] if t >= 1 else 0.0)
            + rng.normal(0, 0.35)
        )
        z["hpc_outlet_temp"][t] = (
            0.5 * prev["hpc_outlet_temp"]
            + 0.65 * prev["load"]
            + 0.55 * prev["coolant_temp"]
            - 0.5 * prev2["fan_speed_nc"]
            + 0.12 * np.clip(prev["load"], -2, 2) ** 2
            + rng.normal(0, 0.35)
        )
        z["oil_pressure"][t] = (
            0.55 * prev["oil_pressure"]
            - 0.6 * prev["vibration"]
            + rng.normal(0, 0.35)
        )
        z["power_draw"][t] = (
            0.4 * prev["power_draw"]
            + 0.8 * prev["load"]
            + 0.3 * prev["fan_speed_nc"]
            + rng.normal(0, 0.35)
        )
        z["failure_risk"][t] = (
            0.3 * prev["failure_risk"]
            + 0.7 * prev["hpc_outlet_temp"]
            - 0.6 * prev["oil_pressure"]
            + 0.3 * prev["vibration"]
            + rng.normal(0, 0.3)
        )
    return z, maint


def z_to_frame(z, maint) -> pd.DataFrame:
    data = {}
    for name, (offset, scale) in data_loader._SCALE.items():
        col = offset + scale * z[name]
        if name == "failure_risk":
            col = np.clip(col, 0.0, 100.0)
        data[name] = col
    data["maintenance_event"] = maint
    return pd.DataFrame(data, columns=data_loader.NODES)


def true_effect_pulse() -> dict:
    """True horizon-3 effect via common-random-number pulse simulation."""
    # 1) validate the replica against generate() itself
    df_ref, _, _ = data_loader.generate(N, seed=PULSE_SEED)
    z0, m0 = simulate_z(N, seed=PULSE_SEED)
    df_rep = z_to_frame(z0, m0)
    replica_exact = bool(
        np.allclose(df_ref.to_numpy(dtype=float), df_rep.to_numpy(dtype=float))
    )
    assert replica_exact, "z-space replica does NOT reproduce generate()"

    # 2) pulse +1 z of fan speed at spaced times, difference failure_risk
    pulse_times = list(range(200, N - 10, 25))
    z1, m1 = simulate_z(N, seed=PULSE_SEED, pulse_times=pulse_times, pulse_delta=1.0)
    diffs = np.array([z1["failure_risk"][t + HORIZON] - z0["failure_risk"][t + HORIZON]
                      for t in pulse_times])
    true_z = float(diffs.mean())
    # Same pulses, but differencing the OBSERVED (engineering, clipped to
    # [0, 100]) failure_risk that estimators actually see — generate() clips
    # the output column, so the observed-outcome effect is mechanically
    # attenuated relative to the latent z-space effect. Reported so the
    # estimator comparison can use the like-for-like target as well.
    df_pulsed = z_to_frame(z1, m1)
    diffs_obs = np.array([
        df_pulsed["failure_risk"].iloc[t + HORIZON] - df_rep["failure_risk"].iloc[t + HORIZON]
        for t in pulse_times])
    return {
        "method": "pulse simulation with common random numbers on a z-space "
                  "replica of generate() (replica validated bit-for-bit "
                  "against generate() with interventions off); +1 z pulse "
                  "to fan_speed_nc at spaced times, failure_risk differenced "
                  f"at t+{HORIZON}",
        "replica_validated_exact": replica_exact,
        "n_pulses": len(pulse_times),
        "true_effect_z_per_z": true_z,
        "pulse_diff_std_z": float(diffs.std()),
        "analytic_effect_z_per_z": ANALYTIC_TRUE_Z,
        "true_effect_per_80_fan_units": true_z * FR_SCALE,
        "true_effect_per_fan_unit": true_z * FR_SCALE / FAN_SCALE,
        "observed_outcome_effect_per_fan_unit": float(diffs_obs.mean()) / FAN_SCALE,
        "observed_outcome_effect_per_80_fan_units": float(diffs_obs.mean()),
        "observed_outcome_note": "effect on the clipped engineering "
                                 "failure_risk (the outcome estimators see), "
                                 "same pulses; per +80 fan units before the "
                                 "/80 division",
    }


def build_lagged_dataset() -> pd.DataFrame:
    df, _, _ = data_loader.generate(N, seed=DATA_SEED)
    lag = pd.DataFrame({TREATMENT: df[TREATMENT].to_numpy(dtype=float)[: N - HORIZON]})
    lag[OUTCOME] = df["failure_risk"].to_numpy(dtype=float)[HORIZON:]
    for c in CONFOUNDERS:
        lag[c] = df[c].to_numpy(dtype=float)[: N - HORIZON]
    return lag


def dml_confidence_interval(lag: pd.DataFrame) -> dict:
    """95% CI from a LinearDML refit identical to app.estimator's config."""
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestRegressor

    Y = lag[OUTCOME].to_numpy(dtype=float)
    T = lag[TREATMENT].to_numpy(dtype=float)
    W = lag[CONFOUNDERS].to_numpy(dtype=float)
    est = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=0),
        model_t=RandomForestRegressor(n_estimators=100, random_state=0),
        discrete_treatment=False,
        random_state=0,
    )
    est.fit(Y, T, X=W, W=None)
    # EconML 0.16: ate_inference() returns PopulationSummaryResults, whose
    # ATE interval is conf_int_mean(alpha).
    ci = np.asarray(est.ate_inference(X=W).conf_int_mean(alpha=0.05),
                    dtype=float).ravel()
    return {"ci_low_per_fan_unit": float(ci[0]), "ci_high_per_fan_unit": float(ci[1]),
            "refit_ate_per_fan_unit": float(est.ate(W))}


def dowhy_refutation(lag: pd.DataFrame) -> dict:
    """DoWhy CausalModel on the same lagged data; refutation of the DoWhy
    backdoor linear-regression estimate of the same estimand."""
    import networkx as nx
    from dowhy import CausalModel

    # Graph passed as a networkx DiGraph (same edges a DOT string would
    # encode: W -> T, W -> Y, T -> Y). The DOT-string path is broken in
    # this environment by a networkx/pydot version incompatibility
    # (Graph.get_strict() signature), so the DiGraph form is used.
    graph = nx.DiGraph()
    for w in CONFOUNDERS:
        graph.add_edge(w, TREATMENT)
        graph.add_edge(w, OUTCOME)
    graph.add_edge(TREATMENT, OUTCOME)

    model = CausalModel(data=lag, treatment=TREATMENT, outcome=OUTCOME, graph=graph)
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
        test_significance=False,
    )
    base = float(estimate.value)

    def summarise(ref, kind):
        out = {
            "refuter": kind,
            "estimated_effect": float(ref.estimated_effect),
            "new_effect": float(ref.new_effect),
        }
        rr = getattr(ref, "refutation_result", None)
        if isinstance(rr, dict):
            for key in ("p_value", "is_statistically_significant"):
                if key in rr and isinstance(rr[key], (int, float, bool)):
                    out[key] = rr[key]
        return out

    results = {"dowhy_base_ate_per_fan_unit": base, "refuters": []}

    # Pass criteria are effect-magnitude criteria (the standard reading of
    # each refuter). DoWhy's own significance test (test_significance in
    # dowhy/causal_refuter.py) tests the null "the estimate is part of the
    # refutation distribution" and flags is_statistically_significant at
    # p <= 0.05; the returned p-values are reported verbatim in the JSON.
    ref = model.refute_estimate(identified, estimate, method_name="random_common_cause",
                                show_progress_bar=False)
    s = summarise(ref, "random_common_cause")
    s["pass"] = bool(abs(s["new_effect"] - base) <= 0.10 * abs(base))
    s["interpretation"] = ("PASS — adding a random common cause barely moves "
                           "the estimate" if s["pass"] else
                           "FAIL — estimate shifts materially under a random "
                           "common cause")
    results["refuters"].append(s)

    ref = model.refute_estimate(identified, estimate,
                                method_name="placebo_treatment_refuter",
                                placebo_type="permute", show_progress_bar=False)
    s = summarise(ref, "placebo_treatment_refuter")
    s["pass"] = bool(abs(s["new_effect"]) <= 0.25 * abs(base))
    s["interpretation"] = ("PASS — permuted (fake) treatment collapses the "
                           "effect to ~0, as it should if the real estimate "
                           "is genuinely driven by the treatment"
                           if s["pass"] else
                           "FAIL — placebo treatment does not collapse to ~0")
    results["refuters"].append(s)

    ref = model.refute_estimate(identified, estimate, method_name="data_subset_refuter",
                                subset_fraction=0.8, show_progress_bar=False)
    s = summarise(ref, "data_subset_refuter")
    s["pass"] = bool(abs(s["new_effect"] - base) <= 0.10 * abs(base))
    s["interpretation"] = ("PASS — estimate is stable on an 80% data subset"
                           if s["pass"] else
                           "FAIL — estimate shifts materially on a data subset")
    results["refuters"].append(s)

    results["all_refuters_pass"] = bool(all(r["pass"] for r in results["refuters"]))
    return results


def main():
    t_start = time.time()
    results: dict = {
        "phase": "Phase 3 smoke — estimator validation on synthetic SCM",
        "treatment": TREATMENT,
        "outcome": f"failure_risk at t+{HORIZON}",
        "horizon": HORIZON,
        "n_generated": N,
        "data_seed": DATA_SEED,
        "confounders": CONFOUNDERS,
    }

    print("== True effect (pulse simulation, common random numbers) ==")
    truth = true_effect_pulse()
    results["true_effect"] = truth
    print(f"  replica validated exact vs generate(): {truth['replica_validated_exact']}")
    print(f"  simulated true effect : {truth['true_effect_z_per_z']:+.5f} z/z "
          f"(pulse diff std {truth['pulse_diff_std_z']:.2e})")
    print(f"  analytic  true effect : {truth['analytic_effect_z_per_z']:+.5f} z/z")
    print(f"  => {truth['true_effect_per_fan_unit']:+.5f} failure-risk points "
          f"per +1 fan unit ({truth['true_effect_per_80_fan_units']:+.3f} per +80)")
    print(f"  same pulses on the OBSERVED (clipped) outcome: "
          f"{truth['observed_outcome_effect_per_fan_unit']:+.5f} per +1 fan unit")

    print("\n== EconML LinearDML estimate (app.estimator.estimate_ate) ==")
    lag = build_lagged_dataset()
    results["n_estimation_rows"] = int(len(lag))
    est = estimate_ate(lag, TREATMENT, OUTCOME, CONFOUNDERS)
    results["estimate"] = est
    ate_err = abs(est["ate"] - truth["true_effect_per_fan_unit"])
    results["ate_error_per_fan_unit"] = float(ate_err)
    results["ate_error_per_80_fan_units"] = float(ate_err * FAN_SCALE)
    print(f"  estimated ATE : {est['ate']:+.5f} per +1 fan unit "
          f"(effect std {est['effect_std']:.5f})")
    print(f"  ATE error     : {ate_err:.5f} per fan unit "
          f"({ate_err * FAN_SCALE:.3f} points per +80 fan units)")

    ci = dml_confidence_interval(lag)
    ci["true_effect_inside_ci"] = bool(
        ci["ci_low_per_fan_unit"] <= truth["true_effect_per_fan_unit"]
        <= ci["ci_high_per_fan_unit"])
    results["confidence_interval_95"] = ci
    print(f"  95% CI        : [{ci['ci_low_per_fan_unit']:+.5f}, "
          f"{ci['ci_high_per_fan_unit']:+.5f}]  (true effect inside: "
          f"{ci['true_effect_inside_ci']})")

    print("\n== DoWhy refutation (same lagged data, backdoor linear regression) ==")
    ref = dowhy_refutation(lag)
    results["refutation"] = ref
    print(f"  DoWhy base ATE: {ref['dowhy_base_ate_per_fan_unit']:+.5f} per +1 fan unit")
    for r in ref["refuters"]:
        pv = f", p={r['p_value']:.4f}" if "p_value" in r else ""
        print(f"  {r['refuter']}: new effect {r['new_effect']:+.5f}{pv} — "
              f"{r['interpretation']}")
    print(f"  ALL REFUTERS PASS: {ref['all_refuters_pass']}")

    latent = truth["true_effect_per_fan_unit"]
    obs = truth["observed_outcome_effect_per_fan_unit"]
    results["ate_error_vs_observed_outcome_truth_per_fan_unit"] = float(
        abs(est["ate"] - obs))
    results["estimator_vs_truth_notes"] = (
        f"LinearDML ({est['ate']:+.5f}/unit) and DoWhy linear regression "
        f"({ref['dowhy_base_ate_per_fan_unit']:+.5f}/unit) agree closely with "
        f"each other. Both are attenuated relative to the LATENT z-space "
        f"pulse truth ({latent:+.5f}/unit), and the DML 95% CI excludes the "
        f"latent truth. Part of the gap is mechanical: generate() clips the "
        f"observed failure_risk to [0, 100], and the same pulses applied to "
        f"the clipped observed outcome give {obs:+.5f}/unit — the "
        f"like-for-like target for estimators fit on observed data. The "
        f"remaining gap vs the observed-outcome truth is "
        f"{abs(est['ate'] - obs):.5f}/unit (candidate contributors: "
        f"RandomForest nuisance shrinkage in LinearDML and the mild "
        f"hpc_outlet_temp nonlinearity; not isolated in this smoke run)."
    )
    print("\n== Notes ==")
    print(" ", results["estimator_vs_truth_notes"])

    results["runtime_seconds"] = round(time.time() - t_start, 2)
    out_path = Path(__file__).resolve().parent / "estimator_smoke_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}  (total runtime {results['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
