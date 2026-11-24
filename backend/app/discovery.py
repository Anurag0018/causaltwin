"""Causal discovery (Phase 2): Tigramite PCMCI + ParCorr, and graph metrics.

PCMCI+ (the blueprint's target algorithm) extends PCMCI with contemporaneous
orientation; this scaffold uses PCMCI over lagged links (tau_min = 1), which
is the part scored against the lagged ground truth. Moving to
`run_pcmciplus` is a Phase 2 refinement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_discovery(df: pd.DataFrame, tau_max: int = 2, pc_alpha: float = 0.01):
    """Run PCMCI and return discovered lagged edges.

    Returns a list of dicts {source, target, lag, p_value} for every
    significant lagged link (self-links excluded).
    """
    import tigramite.data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    var_names = list(df.columns)
    dataframe = pp.DataFrame(df.to_numpy(dtype=float), var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"))
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)

    graph = results["graph"]
    p_matrix = results["p_matrix"]
    edges = []
    n = len(var_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for lag in range(1, tau_max + 1):
                if graph[i, j, lag] == "-->":
                    edges.append({
                        "source": var_names[i],
                        "target": var_names[j],
                        "lag": int(lag),
                        "p_value": float(p_matrix[i, j, lag]),
                    })
    return edges


def _as_tuples(edges):
    out = set()
    for e in edges:
        if isinstance(e, dict):
            out.add((e["source"], e["target"], int(e["lag"])))
        else:
            s, t, lag = e
            out.add((s, t, int(lag)))
    return {(s, t, lag) for (s, t, lag) in out if s != t}


def edge_metrics(discovered, ground_truth) -> dict:
    """Precision / recall / Edge F1 and an SHD-style count vs ground truth.

    SHD-style count = extra (false positive) + missing (false negative)
    lagged edges. Self-links are ignored on both sides.
    """
    disc = _as_tuples(discovered)
    truth = _as_tuples(ground_truth)
    tp = len(disc & truth)
    fp = len(disc - truth)
    fn = len(truth - disc)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "edge_f1": round(f1, 4),
        "shd_style": fp + fn,
    }
