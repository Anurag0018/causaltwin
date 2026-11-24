"""CausalTwin scaffold test suite (run: python -m pytest from backend/)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import data_loader
from app.discovery import edge_metrics, run_discovery
from app.explain import format_explanation
from app.main import app
from app.root_cause import rank_root_causes
from app.schemas import EdgeSpec, TwinGraph
from app.simulator import StructuralEquations
from app.updater import GraphRegistry


# ---------------------------------------------------------------- generator
def test_generator_deterministic_and_shape():
    df1, edges1, events1 = data_loader.generate(n_steps=300, seed=3)
    df2, edges2, events2 = data_loader.generate(n_steps=300, seed=3)
    assert df1.shape == (300, len(data_loader.NODES))
    assert list(df1.columns) == data_loader.NODES
    assert df1.equals(df2)
    assert edges1 == edges2
    assert events1 == events2
    assert set(df1["maintenance_event"].unique()) <= {0.0, 1.0}
    assert df1["maintenance_event"].sum() > 0, "event stream should contain maintenance events"
    assert any(e["type"] == "maintenance" and e["is_treatment"] for e in events1)


def test_ground_truth_sanity():
    nodes = set(data_loader.NODES)
    for s, t, lag in data_loader.GROUND_TRUTH_EDGES:
        assert s in nodes and t in nodes and s != t
        assert 1 <= lag <= 2
    targets = {t for _, t, _ in data_loader.GROUND_TRUTH_EDGES}
    assert "failure_risk" in targets


# ---------------------------------------------------------------- discovery
def test_discovery_metrics_on_synthetic():
    df, ground_truth, _ = data_loader.generate(n_steps=600, seed=0)
    discovered = run_discovery(df, tau_max=2, pc_alpha=0.01)
    metrics = edge_metrics(discovered, ground_truth)
    assert metrics["edge_f1"] >= 0.4, f"Edge F1 too low: {metrics}"


def test_edge_metrics_perfect_and_empty():
    gt = data_loader.GROUND_TRUTH_EDGES
    perfect = edge_metrics(gt, gt)
    assert perfect["edge_f1"] == 1.0 and perfect["shd_style"] == 0
    empty = edge_metrics([], gt)
    assert empty["recall"] == 0.0 and empty["false_negatives"] == len(gt)


# ---------------------------------------------------------------- simulator
def test_intervention_direction():
    """More fan speed -> more cooling -> lower failure risk (generator sign)."""
    df, ground_truth, _ = data_loader.generate(n_steps=1200, seed=0)
    se = StructuralEquations().fit(df, ground_truth)
    result = se.intervene("fan_speed_nc", delta=80.0, target="failure_risk")
    assert result["target_shift"] < 0, result


# --------------------------------------------------------------- root cause
def test_root_cause_ranking_covers_all_nodes():
    df, ground_truth, _ = data_loader.generate(n_steps=400, seed=1)
    ranking = rank_root_causes(df, ground_truth)
    ranked_nodes = {r["node"] for r in ranking}
    assert ranked_nodes == set(df.columns)
    assert [r["rank"] for r in ranking] == list(range(1, len(ranking) + 1))
    zs = [abs(r["z_score"]) for r in ranking]
    assert zs == sorted(zs, reverse=True)


# ------------------------------------------------------------------ updater
def test_graph_registry_version_and_diff():
    reg = GraphRegistry()
    g1 = TwinGraph(edges=[EdgeSpec(source="a", target="b", lag=1)])
    g2 = TwinGraph(edges=[EdgeSpec(source="a", target="b", lag=1),
                          EdgeSpec(source="b", target="c", lag=2)])
    v1 = reg.register(g1)
    v2 = reg.register(g2)
    assert (v1, v2) == (1, 2)
    assert reg.list_versions() == [1, 2]
    diff = reg.diff(v1, v2)
    assert diff["added"] == [{"source": "b", "target": "c", "lag": 2}]
    assert diff["removed"] == []
    assert diff["unchanged_count"] == 1


# ------------------------------------------------------------------ explain
def test_explain_format():
    card = format_explanation({
        "param": "fan_speed_nc", "pct": 7, "kpi": "failure_risk",
        "effect_pct": 12, "ci_95": (8.0, 16.0),
        "path": "fan_speed_nc drives coolant_temp, which drives hpc_outlet_temp",
        "negligible_param": "oil_pressure",
        "graph_version": 1, "refutation_status": "passed",
    })
    assert "Reducing fan_speed_nc by 7%" in card
    assert "reduce failure_risk by 12%" in card
    assert "negligible improvement" in card
    assert "Graph v1, refutation: passed." in card


# ---------------------------------------------------------------------- api
def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["service"] == "causaltwin"
    for pkg in ["numpy", "tigramite", "dowhy", "econml"]:
        assert body["versions"][pkg] != "not installed"
