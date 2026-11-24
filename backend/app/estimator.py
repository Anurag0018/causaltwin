"""Causal effect estimation (Phase 3): EconML LinearDML for treatment->KPI pairs.

Scoped deliberately (blueprint §3.3, v3): DML is run on actionable
treatment -> KPI pairs with named confounders — not indiscriminately on
every edge of the graph.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_ate(df: pd.DataFrame, treatment: str, outcome: str, confounders: list[str]) -> dict:
    """Estimate the ATE of `treatment` on `outcome` adjusting for `confounders`.

    Uses EconML LinearDML. Nuisance models: RandomForest (classifier for a
    binary treatment, regressor otherwise). Returns the ATE plus a summary
    of the per-unit effect distribution.
    """
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    Y = df[outcome].to_numpy(dtype=float)
    T = df[treatment].to_numpy(dtype=float)
    W = df[confounders].to_numpy(dtype=float) if confounders else None

    binary = len(np.unique(T)) == 2
    model_t = (
        RandomForestClassifier(n_estimators=100, random_state=0)
        if binary
        else RandomForestRegressor(n_estimators=100, random_state=0)
    )
    est = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=0),
        model_t=model_t,
        discrete_treatment=binary,
        random_state=0,
    )
    if W is not None:
        est.fit(Y, T, X=W, W=None)
        effects = np.asarray(est.effect(W), dtype=float)
        ate = float(est.ate(W))
    else:
        est.fit(Y, T)
        effects = np.asarray(est.effect(), dtype=float)
        ate = float(est.ate())
    return {
        "treatment": treatment,
        "outcome": outcome,
        "confounders": list(confounders),
        "ate": ate,
        "effect_mean": float(effects.mean()),
        "effect_std": float(effects.std()),
    }
