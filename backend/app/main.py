"""CausalTwin FastAPI app — thin endpoints over the engine modules.

Endpoints (blueprint §5, Phase 6 surface):
  GET  /health          service + causal-stack versions
  POST /discover        synthetic data -> PCMCI discovery + metrics vs ground truth
  POST /intervene       fit structural equations on the ground-truth graph, apply do(X)
  POST /recommend       smallest sustained lever change predicted to cut failure_risk
                        by the requested amount, with its explanation card (Phase 4)
  GET  /graph/versions  registered graph versions (+ optional diff)
  POST /explain         explanation card for a recommendation
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import data_loader
from .discovery import edge_metrics, run_discovery
from .explain import format_explanation
from .recommend import recommend_intervention
from .schemas import EdgeSpec, InterventionRequest, NodeSpec, RecommendationResponse, TwinGraph
from .simulator import StructuralEquations
from .updater import GraphRegistry

app = FastAPI(title="CausalTwin", version="0.1.0")

# The console normally reaches the API through the Next.js /api rewrite
# (same-origin, no CORS needed). CORS is enabled for the direct mode, where
# NEXT_PUBLIC_API_URL points the browser straight at this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = GraphRegistry()


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _ground_truth_graph() -> TwinGraph:
    nodes = [
        NodeSpec(id="load", role="exogenous"),
        NodeSpec(id="fan_speed_nc", role="actionable_lever", unit="rpm", bounds=(2100, 2400)),
        NodeSpec(id="coolant_temp", role="intermediate_state", unit="degR"),
        NodeSpec(id="vibration", role="intermediate_state", unit="mm/s"),
        NodeSpec(id="hpc_outlet_temp", role="intermediate_state", unit="degR"),
        NodeSpec(id="oil_pressure", role="intermediate_state", unit="psi"),
        NodeSpec(id="power_draw", role="intermediate_state", unit="kW"),
        NodeSpec(id="failure_risk", role="target_kpi", unit="index_0_100"),
        NodeSpec(id="maintenance_event", role="event", unit="binary"),
    ]
    edges = [EdgeSpec(source=s, target=t, lag=lag) for s, t, lag in data_loader.GROUND_TRUTH_EDGES]
    return TwinGraph(nodes=nodes, edges=edges, max_lag=2)


registry.register(_ground_truth_graph())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "causaltwin",
        "versions": {
            "numpy": np.__version__,
            "tigramite": _pkg("tigramite"),
            "dowhy": _pkg("dowhy"),
            "econml": _pkg("econml"),
        },
    }


class DiscoverRequest(BaseModel):
    n_steps: int = 600
    tau_max: int = 2
    seed: int = 0
    pc_alpha: float = 0.01


@app.post("/discover")
def discover(req: DiscoverRequest):
    df, ground_truth, events = data_loader.generate(n_steps=req.n_steps, seed=req.seed)
    discovered = run_discovery(df, tau_max=req.tau_max, pc_alpha=req.pc_alpha)
    metrics = edge_metrics(discovered, ground_truth)
    return {
        "n_steps": req.n_steps,
        "tau_max": req.tau_max,
        "discovered_edges": discovered,
        "ground_truth_edges": [
            {"source": s, "target": t, "lag": lag} for s, t, lag in ground_truth
        ],
        "metrics": metrics,
        "n_events": len(events),
    }


@app.post("/intervene")
def intervene(req: InterventionRequest):
    df, ground_truth, _ = data_loader.generate(n_steps=req.n_steps, seed=req.seed)
    se = StructuralEquations().fit(df, ground_truth)
    return se.intervene(req.node, req.delta, req.target)


class RecommendRequest(BaseModel):
    n_steps: int = 2000
    seed: int = 0
    target_reduction_points: float = 5.0
    lever: str = "fan_speed_nc"


@app.post("/recommend")
def recommend(req: RecommendRequest):
    df, ground_truth, _ = data_loader.generate(n_steps=req.n_steps, seed=req.seed)
    se = StructuralEquations().fit(df, ground_truth)
    rec = recommend_intervention(
        se,
        target_reduction_points=req.target_reduction_points,
        lever=req.lever,
        graph_version=registry.list_versions()[-1],
        df=df,
        edges=ground_truth,
    )
    card_fields = {k: rec[k] for k in (
        "param", "pct", "kpi", "effect_pct", "ci_95", "path",
        "negligible_param", "graph_version", "refutation_status", "direction")}
    rec["explanation"] = format_explanation(card_fields)
    return rec


@app.get("/graph/versions")
def graph_versions():
    return {"versions": registry.list_versions()}


@app.get("/graph/diff")
def graph_diff(from_version: int, to_version: int):
    return registry.diff(from_version, to_version)


@app.post("/explain")
def explain(rec: RecommendationResponse):
    payload = rec.model_dump()
    payload["explanation"] = format_explanation(payload)
    return payload
