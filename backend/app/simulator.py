"""Structural equations + intervention propagation (Phases 3-4).

Each node gets a linear structural equation fitted by OLS on its parents at
their specified lags:  X_j(t) = sum_k coef_k * X_src(k)(t - lag_k) + eps.
Because the fitted system is linear, the mean effect of clamping a node by
`delta` propagates exactly: shifts are pushed through the equations in
topological order of the collapsed (lag-ignored) DAG. This is the scaffold's
do(X = x') rollout; multi-step trajectory simulation is a Phase 4 extension.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class StructuralEquations:
    def __init__(self):
        self.equations: dict[str, dict] = {}   # target -> {features: [(src, lag)], coef, intercept, resid_std}
        self.edges: list[tuple] = []
        self.means: dict[str, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, edges) -> "StructuralEquations":
        self.edges = [(s, t, int(lag)) for (s, t, lag) in
                      (e if not isinstance(e, dict) else (e["source"], e["target"], e["lag"]) for e in edges)]
        self.means = {c: float(df[c].mean()) for c in df.columns}

        parents: dict[str, list[tuple[str, int]]] = {}
        for s, t, lag in self.edges:
            parents.setdefault(t, []).append((s, lag))

        max_lag = max((lag for _, _, lag in self.edges), default=0)
        for target in df.columns:
            feats = parents.get(target, [])
            if not feats:
                self.equations[target] = {"features": [], "coef": np.array([]),
                                          "intercept": self.means[target], "resid_std": float(df[target].std())}
                continue
            X = np.column_stack([df[s].shift(lag).to_numpy(dtype=float) for s, lag in feats])
            y = df[target].to_numpy(dtype=float)
            mask = np.ones(len(df), dtype=bool)
            mask[:max_lag] = False
            model = LinearRegression().fit(X[mask], y[mask])
            resid = y[mask] - model.predict(X[mask])
            self.equations[target] = {
                "features": feats,
                "coef": np.asarray(model.coef_, dtype=float),
                "intercept": float(model.intercept_),
                "resid_std": float(resid.std()) if len(resid) else 0.0,
            }
        return self

    # ------------------------------------------------------------- intervene
    def collapsed_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for s, t, _ in self.edges:
            g.add_edge(s, t)
        return g

    def shift_propagation(self, node: str, delta: float) -> dict[str, float]:
        """Mean shift at every node after clamping `node` by `delta`."""
        g = self.collapsed_graph()
        order = list(nx.topological_sort(g)) if g.number_of_nodes() else []
        shift: dict[str, float] = {n: 0.0 for n in self.equations}
        shift[node] = float(delta)
        for target in order:
            if target == node:
                continue
            eq = self.equations.get(target)
            if not eq or not eq["features"]:
                continue
            total = 0.0
            for (src, _lag), coef in zip(eq["features"], eq["coef"]):
                total += float(coef) * shift.get(src, 0.0)
            shift[target] = total
        return shift

    def intervene(self, node: str, delta: float, target: str) -> dict:
        shifts = self.shift_propagation(node, delta)
        return {
            "node": node,
            "delta": float(delta),
            "target": target,
            "target_shift": float(shifts.get(target, 0.0)),
            "all_shifts": {k: float(v) for k, v in shifts.items()},
        }

    # -------------------------------------------------------------- residuals
    def residuals_at(self, df: pd.DataFrame, t: int) -> dict[str, float]:
        """Residual of every node at time t (actual - structural prediction).

        Parentless nodes use deviation from their mean, so every node gets a
        residual and root-cause ranking can cover the whole graph.
        """
        out = {}
        for target, eq in self.equations.items():
            if not eq["features"]:
                out[target] = float(df[target].iloc[t] - self.means[target])
                continue
            pred = eq["intercept"]
            ok = True
            for (src, lag), coef in zip(eq["features"], eq["coef"]):
                if t - lag < 0:
                    ok = False
                    break
                pred += float(coef) * float(df[src].iloc[t - lag])
            out[target] = float(df[target].iloc[t] - pred) if ok else 0.0
        return out

    def residual_std(self, target: str) -> float:
        eq = self.equations.get(target, {})
        if eq.get("features"):
            return float(eq["resid_std"]) or 1.0
        return 1.0
