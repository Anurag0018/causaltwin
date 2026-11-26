"""Intervention recommendation engine (Phase 4, blueprint §3.4).

Given a fitted StructuralEquations and a required failure_risk reduction
(in points), search a grid of sustained changes to the actionable lever
and return the SMALLEST change whose predicted steady-state effect meets
the target — plus the alternatives that were tried and rejected, so the
explanation card can name one honestly.

The headline alternative is `power_draw`: it is strongly *correlated*
with failure_risk (both are driven by load and fan speed) but has no
causal path to it, so the causal simulator predicts exactly no effect
from changing it. That contrast — correlation says act, causation says
don't bother — is the project's core thesis, measured rather than asserted.
"""
from __future__ import annotations

from .simulator import StructuralEquations

# Default grid of sustained fan_speed_nc changes, in rpm (fan mean is
# 2250 rpm, 1 z = 80 rpm, schema bounds 2100-2400).
DEFAULT_GRID = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
                100.0, 110.0, 120.0, 130.0, 140.0, 150.0]

# Phase 3 (scripts/estimator_smoke.py): the fan_speed_nc -> failure_risk
# effect estimate passed all 3 DoWhy refuters (random common cause,
# placebo, data subset). The recommendation inherits that refutation
# status. (Phase 3's numeric CI is NOT reused here: it bounds the
# horizon-3 effect, while the recommendation is a steady-state effect —
# the card's CI is a bootstrap of the simulator's own prediction, below.)
PHASE3_REFUTATION_STATUS = "passed"


def _bootstrap_reduction_ci(df, edges, lever, delta, target,
                            n_bootstrap: int = 40, seed: int = 0,
                            block: int = 100) -> tuple[float, float]:
    """Block-bootstrap 95% CI for the predicted reduction at `delta`.

    Resamples contiguous blocks of the fitting data (preserving local
    time structure), refits the structural equations, and takes the
    2.5/97.5 percentiles of the predicted steady-state reduction.
    Returns the CI in POINTS of the target variable.

    Block length matters: at block=50 the ~40 block boundaries per
    resample corrupt enough lagged features to bias the bootstrap
    predictions systematically low (measured: CI [18.4, 21.2] around a
    21.6 point estimate); block=100 restores a CI that brackets the
    full-data estimate ([19.4, 22.1]). 100 steps is also several times
    the system's longest memory (AR + seasonality), so blocks are
    near-independent.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(df)
    n_blocks = int(np.ceil(n / block))
    reductions = []
    for _ in range(n_bootstrap):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        boot = df.iloc[idx].reset_index(drop=True)
        se_b = StructuralEquations().fit(boot, edges)
        shift = se_b.intervene(lever, float(delta), target)["target_shift"]
        reductions.append(-float(shift))
    lo, hi = np.percentile(reductions, [2.5, 97.5])
    return float(lo), float(hi)


def recommend_intervention(
    se: StructuralEquations,
    target_reduction_points: float,
    lever: str = "fan_speed_nc",
    target: str = "failure_risk",
    grid: list[float] | None = None,
    graph_version: int = 1,
    df=None,
    edges=None,
    n_bootstrap: int = 40,
) -> dict:
    """Smallest sustained `lever` increase predicted to cut `target` by
    at least `target_reduction_points` (steady-state, via the fitted
    structural equations). Returns a RecommendationResponse-compatible
    dict plus the full candidate table under "candidates"."""
    grid = list(grid) if grid is not None else list(DEFAULT_GRID)
    baseline = float(se.means[target])
    lever_mean = float(se.means[lever])

    candidates = []
    chosen = None
    for delta in grid:
        shift = se.intervene(lever, float(delta), target)["target_shift"]
        reduction = -float(shift)
        meets = reduction >= target_reduction_points
        candidates.append({
            "param": lever, "delta": float(delta),
            "predicted_reduction_points": reduction, "meets_target": bool(meets),
        })
        if meets and chosen is None:
            chosen = candidates[-1]

    # Alternative that was tried and rejected: power_draw has no causal
    # path to failure_risk, so its predicted effect is exactly 0.
    alt_shift = se.intervene("power_draw", 50.0, target)["target_shift"]
    negligible = {
        "param": "power_draw", "delta": 50.0,
        "predicted_reduction_points": -float(alt_shift),
    }

    if chosen is None:
        raise ValueError(
            f"no grid change to {lever} reaches the target reduction of "
            f"{target_reduction_points} points (largest tried: {grid[-1]})"
        )

    delta = chosen["delta"]
    reduction = chosen["predicted_reduction_points"]
    ci_95 = None
    if df is not None and edges is not None and n_bootstrap > 0:
        lo_pts, hi_pts = _bootstrap_reduction_ci(df, edges, lever, delta, target,
                                                 n_bootstrap=n_bootstrap)
        ci_95 = (round(lo_pts / baseline * 100.0, 1),
                 round(hi_pts / baseline * 100.0, 1))
    return {
        "param": lever,
        "direction": "Increasing",
        "pct": round(delta / lever_mean * 100.0, 1),
        "kpi": target,
        "effect_pct": round(reduction / baseline * 100.0, 1),
        "ci_95": ci_95,
        "path": ("fan_speed_nc drives coolant_temp, which drives "
                 "hpc_outlet_temp, which drives failure_risk"),
        "negligible_param": negligible["param"],
        "graph_version": graph_version,
        "refutation_status": PHASE3_REFUTATION_STATUS,
        # --- detail beyond the card schema -------------------------------
        "delta": delta,
        "lever_baseline": lever_mean,
        "target_baseline": baseline,
        "target_reduction_points": float(target_reduction_points),
        "predicted_reduction_points": reduction,
        "negligible_candidate": negligible,
        "candidates": candidates,
    }
