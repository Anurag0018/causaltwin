"""Structural equations + intervention propagation (Phases 3-4).

Each node gets a linear structural equation fitted by OLS on its parents at
their specified lags PLUS its own lag-1 (autoregressive) term:
    X_j(t) = a_j * X_j(t-1) + sum_k coef_k * X_src(k)(t - lag_k) + eps.
The own-lag term was added in Phase 4: the generator's true equations all
carry one (0.3-0.6), so the scaffold's parent-only fit suffered omitted-
variable bias, and — more importantly — a *sustained* clamp's steady-state
effect is amplified by the factor 1 / (1 - a_j) at every node it passes
through. Ignoring that made shift_propagation systematically underpredict
sustained interventions (validated in scripts/phase4_end_to_end.py).

Because the fitted system is linear, the steady-state mean effect of
clamping a node by `delta` is exact: shifts are pushed through the
equations in topological order of the collapsed (lag-ignored) DAG, with
each node's incoming shift divided by (1 - a_j). This is the do(X = x')
steady-state rollout; per-timestep trajectory simulation lives in
app/faults.py (the SCM replica used for fault injection / ground truth).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class StructuralEquations:
    def __init__(self):
        self.equations: dict[str, dict] = {}   # target -> {features: [(src, lag)], coef, self_coef, intercept, resid_std}
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
                                          "self_coef": None,
                                          "intercept": self.means[target], "resid_std": float(df[target].std())}
                continue
            # Joint OLS: own lag-1 first, then the parent features. Fitting
            # them jointly (not sequentially) is what removes the omitted-
            # variable bias of the scaffold's parent-only fit.
            own_lag = df[target].shift(1).to_numpy(dtype=float)
            X = np.column_stack(
                [own_lag] + [df[s].shift(lag).to_numpy(dtype=float) for s, lag in feats]
            )
            y = df[target].to_numpy(dtype=float)
            mask = np.ones(len(df), dtype=bool)
            mask[: max(max_lag, 1)] = False
            model = LinearRegression().fit(X[mask], y[mask])
            resid = y[mask] - model.predict(X[mask])
            self.equations[target] = {
                "features": feats,
                "coef": np.asarray(model.coef_[1:], dtype=float),
                "self_coef": float(model.coef_[0]),
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
        """STEADY-STATE mean shift at every node after clamping `node` by `delta`.

        A sustained clamp settles to the fixed point of the linear system:
        shift_j = (sum_k coef_k * shift_src_k) / (1 - self_coef_j), solved
        exactly in topological order of the collapsed DAG (self-loops do not
        affect the order across distinct nodes). The clamped node's own
        shift is `delta` by definition.
        """
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
            a = eq.get("self_coef")
            if a is not None:
                # Guard against a (near-)unit root fit, which would explode.
                denom = max(1.0 - float(a), 0.05)
                total /= denom
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
            a = eq.get("self_coef")
            if a is not None:
                if t - 1 < 0:
                    ok = False
                else:
                    pred += float(a) * float(df[target].iloc[t - 1])
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
