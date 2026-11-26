"""Phase 4 tests: fault injection, recommendation engine, explanation card,
and root-cause ranking on an injected fault."""
import numpy as np
from fastapi.testclient import TestClient

from app import data_loader, faults
from app.explain import format_explanation
from app.main import app
from app.recommend import recommend_intervention
from app.root_cause import rank_root_causes
from app.simulator import StructuralEquations


# ------------------------------------------------------------- fault replica
def test_simulate_scm_reproduces_generate_without_hooks():
    df_ref, _, events_ref = data_loader.generate(n_steps=300, seed=3)
    df_sim, events_sim = faults.simulate_scm(n_steps=300, seed=3)
    assert np.allclose(df_ref.to_numpy(dtype=float), df_sim.to_numpy(dtype=float))
    assert events_ref == events_sim


def test_fault_injection_shifts_target_node_and_propagates():
    fault = {"node": "coolant_temp", "delta_z": 1.2, "start": 100, "end": 160}
    df_clean, _ = faults.simulate_scm(n_steps=300, seed=0)
    df_fault, events = faults.simulate_scm(n_steps=300, seed=0, fault=fault)
    # coolant_temp scale is 25 degR per z: inside the window the faulted
    # series sits ~1.2 z (30 degR) above clean, and downstream
    # hpc_outlet_temp rises in the following steps.
    ct_diff = (df_fault["coolant_temp"] - df_clean["coolant_temp"]).iloc[110:160]
    assert ct_diff.mean() > 20.0
    hpc_diff = (df_fault["hpc_outlet_temp"] - df_clean["hpc_outlet_temp"]).iloc[110:160]
    assert hpc_diff.mean() > 0.0
    assert any(e["type"] == "fault_injection" and e["node"] == "coolant_temp"
               for e in events)


# --------------------------------------------------------------------- RCA
def test_rca_ranks_injected_fault_in_top3():
    fault = {"node": "coolant_temp", "delta_z": 1.2, "start": 500, "end": 560}
    df_fault, _ = faults.simulate_scm(n_steps=800, seed=0, fault=fault)
    _, ground_truth, _ = data_loader.generate(n_steps=10, seed=0)
    ranking = rank_root_causes(df_fault, ground_truth, t=530)
    top3 = [r["node"] for r in ranking[:3]]
    assert "coolant_temp" in top3, ranking


def test_rca_clean_reference_fit_and_window_rank_fault_first():
    """Phase 4 protocol: equations fit on the pre-fault reference period,
    scoring averaged over the fault window — the true node should win."""
    fault = {"node": "coolant_temp", "delta_z": 1.2, "start": 500, "end": 560}
    df_fault, _ = faults.simulate_scm(n_steps=800, seed=0, fault=fault)
    _, ground_truth, _ = data_loader.generate(n_steps=10, seed=0)
    ranking = rank_root_causes(df_fault, ground_truth, t=559, window=60,
                               fit_df=df_fault.iloc[:500])
    assert ranking[0]["node"] == "coolant_temp", ranking


# -------------------------------------------------------------- recommend
def _fitted_se():
    df, ground_truth, _ = data_loader.generate(n_steps=1200, seed=0)
    return StructuralEquations().fit(df, ground_truth), df, ground_truth


def test_recommend_returns_smallest_sufficient_change():
    se, _, _ = _fitted_se()
    target = 5.0
    rec = recommend_intervention(se, target_reduction_points=target)
    assert rec["predicted_reduction_points"] >= target
    # every smaller grid step must fail the target (smallest sufficient)
    smaller = [c for c in rec["candidates"] if c["delta"] < rec["delta"]]
    assert smaller and all(not c["meets_target"] for c in smaller)
    # and the chosen delta's prediction must match the simulator exactly
    direct = se.intervene("fan_speed_nc", rec["delta"], "failure_risk")
    assert abs(-direct["target_shift"] - rec["predicted_reduction_points"]) < 1e-9


def test_recommend_endpoint_and_card():
    client = TestClient(app)
    resp = client.post("/recommend", json={"target_reduction_points": 5.0,
                                           "n_steps": 600, "seed": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["param"] == "fan_speed_nc"
    assert body["predicted_reduction_points"] >= 5.0
    card = body["explanation"]
    assert "fan_speed_nc" in card
    assert f"reduce failure_risk by {body['effect_pct']:g}%" in card
    assert "Changing power_draw produces negligible improvement." in card
    assert "refutation: passed." in card


def test_explanation_card_contains_predicted_effect():
    se, df, ground_truth = _fitted_se()
    rec = recommend_intervention(se, target_reduction_points=5.0,
                                 df=df, edges=ground_truth, n_bootstrap=10)
    card = format_explanation(rec)
    assert f"Increasing fan_speed_nc by {rec['pct']:g}%" in card
    assert f"reduce failure_risk by {rec['effect_pct']:g}%" in card
    assert "[95% CI" in card
