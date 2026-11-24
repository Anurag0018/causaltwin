"""Confidence / explanation layer (blueprint §3.5).

Every recommendation ships with a plain-language explanation card in the
official SNH-AI-01 example format — effect size, uncertainty, causal path,
and the lever that does *not* help, never a bare probability.
"""
from __future__ import annotations


def format_explanation(rec: dict) -> str:
    param = rec.get("param", "parameter X")
    pct = rec.get("pct", 0)
    kpi = rec.get("kpi", "failure probability")
    effect = rec.get("effect_pct", 0)
    direction = rec.get("direction", "Reducing")
    ci = rec.get("ci_95")
    ci_txt = f" [95% CI {ci[0]:.1f} to {ci[1]:.1f}]" if ci else ""
    path = rec.get("path", "")
    path_txt = f", because {path}" if path else ""
    other = rec.get("negligible_param")
    other_txt = f" Changing {other} produces negligible improvement." if other else ""
    version = rec.get("graph_version", 1)
    refutation = rec.get("refutation_status", "passed")
    return (
        f"{direction} {param} by {pct:g}% is predicted to reduce {kpi} by {effect:g}%"
        f"{ci_txt}{path_txt}.{other_txt} "
        f"Graph v{version}, refutation: {refutation}."
    )
