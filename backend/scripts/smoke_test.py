"""Step Zero smoke test — run this before writing any project code.

    python scripts/smoke_test.py

Imports the full causal stack, prints versions, then EXECUTES one tiny job
per library (PCMCI run, LinearDML fit, DoWhy identify_effect). Import-only
checks are not enough: numpy/scipy C-extension conflicts in this stack
surface when code runs, not when it imports.
"""
from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def main() -> int:
    for pkg in ["numpy", "scipy", "scikit-learn", "pandas", "tigramite", "dowhy", "econml", "numba"]:
        print(f"{pkg}=={version(pkg)}")

    # --- Tigramite: tiny PCMCI run on the synthetic SCM --------------------
    from app import data_loader
    from app.discovery import run_discovery

    df, ground_truth, _ = data_loader.generate(n_steps=400, seed=0)
    edges = run_discovery(df, tau_max=2, pc_alpha=0.01)
    print(f"PCMCI ok: {len(edges)} lagged edges discovered (ground truth: {len(ground_truth)})")

    # --- EconML: tiny LinearDML fit ----------------------------------------
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    Y = rng.normal(size=200)
    T = rng.binomial(1, 0.5, 200)
    X = rng.normal(size=(200, 2))
    est = LinearDML(
        model_y=RandomForestRegressor(n_estimators=10, random_state=0),
        model_t=RandomForestRegressor(n_estimators=10, random_state=0),
        random_state=0,
    )
    est.fit(Y, T, X=X)
    print(f"LinearDML ok: mean effect = {float(est.effect(X[:5]).mean()):.4f}")

    # --- DoWhy: identify an effect ------------------------------------------
    from dowhy import CausalModel

    df2 = pd.DataFrame({"T": T, "Y": Y, "X0": X[:, 0], "X1": X[:, 1]})
    model = CausalModel(data=df2, treatment="T", outcome="Y", common_causes=["X0", "X1"])
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    print(f"DoWhy ok: identify_effect -> {type(identified).__name__}")

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
