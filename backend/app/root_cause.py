"""Root-cause attribution (Phase 4): residual z-score ranking.

Fit the structural equations, then score an anomalous timestep by how far
each node's actual value sits from its structural prediction, in units of
that node's residual standard deviation. Upstream culprits are the nodes
with the largest |z|.

Phase 4 hardening (scripts/phase4_end_to_end.py documents the failure
that motivated it): two protocol options that match how a twin is
actually deployed —

  fit_df:  fit the reference equations on a CLEAN reference period
           (e.g. history before the fault window) instead of the series
           being scored. Fitting on data that contains a sustained fault
           biases the faulted node's AR coefficient toward 1 — the model
           learns to *expect* the elevated level, and the true culprit's
           innovation z-score collapses (measured: +3.4 expected, +1.0
           observed under the scaffold protocol).
  window:  average residuals / z-scores over a window of timesteps
           ending at t instead of scoring a single instant. A one-step
           coincidence (e.g. a binary maintenance event firing exactly at
           t, z = +3.9) averages out; a sustained fault does not.
"""
from __future__ import annotations

import pandas as pd

from .simulator import StructuralEquations


def rank_root_causes(df: pd.DataFrame, edges, t: int | None = None,
                     window: int = 1, fit_df: pd.DataFrame | None = None) -> list[dict]:
    """Rank all nodes by |residual z-score| at time t (default: last row).

    With window > 1, residuals and z-scores are averaged over the
    `window` timesteps ending at t. With fit_df given, the structural
    equations (and the per-node scales) come from that reference data
    while the residuals are scored on `df`.
    """
    if t is None:
        t = len(df) - 1
    ref = fit_df if fit_df is not None else df
    se = StructuralEquations().fit(ref, edges)

    stds = {}
    for node in df.columns:
        if node == "maintenance_event":
            stds[node] = float(ref[node].std()) or 1.0
        elif se.equations[node]["features"]:
            stds[node] = se.residual_std(node)
        else:
            stds[node] = float(ref[node].std()) or 1.0

    ts = list(range(max(0, t - window + 1), t + 1))
    rows = []
    for node in df.columns:
        resids = [se.residuals_at(df, tt)[node] for tt in ts]
        resid = float(sum(resids) / len(resids))
        z = resid / stds[node] if stds[node] else 0.0
        rows.append({"node": node, "residual": resid, "z_score": float(z)})
    rows.sort(key=lambda r: abs(r["z_score"]), reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows
