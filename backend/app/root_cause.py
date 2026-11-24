"""Root-cause attribution (Phase 4): residual z-score ranking.

Fit the structural equations, then score an anomalous timestep by how far
each node's actual value sits from its structural prediction, in units of
that node's residual standard deviation. Upstream culprits are the nodes
with the largest |z|.
"""
from __future__ import annotations

import pandas as pd

from .simulator import StructuralEquations


def rank_root_causes(df: pd.DataFrame, edges, t: int | None = None) -> list[dict]:
    """Rank all nodes by |residual z-score| at time t (default: last row)."""
    if t is None:
        t = len(df) - 1
    se = StructuralEquations().fit(df, edges)
    residuals = se.residuals_at(df, t)

    rows = []
    for node, resid in residuals.items():
        if node == "maintenance_event":
            std = float(df[node].std()) or 1.0
        else:
            std = se.residual_std(node) if se.equations[node]["features"] else (float(df[node].std()) or 1.0)
        rows.append({
            "node": node,
            "residual": float(resid),
            "z_score": float(resid / std) if std else 0.0,
        })
    rows.sort(key=lambda r: abs(r["z_score"]), reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows
