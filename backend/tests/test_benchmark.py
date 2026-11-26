"""Phase 5 tests: the benchmark's core comparison runs on a tiny config
and returns the full metric structure (values are not asserted beyond
finiteness — the benchmark's numbers live in benchmark_results.json)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_baseline import run_benchmark  # noqa: E402

TINY_CONFIG = {
    "n_train": 400, "train_seed": 0,
    "n_test": 400, "test_seed": 1,
    "intervention_seed": 0,
    "fan_deltas_rpm": [80.0],
    "scenarios": ["normal"],
    "intervention_window": [200, 320],
    "settled_window": [260, 320],
    "rca_n": 400,
    "rca_faults": [{"node": "coolant_temp", "delta_z": 1.2}],
    "rca_seeds": [0],
    "rca_window": [200, 240],
    "lgbm_params": {"n_estimators": 50, "learning_rate": 0.1,
                    "num_leaves": 15, "random_state": 0, "verbose": -1},
}


def test_benchmark_tiny_config_returns_all_metric_keys():
    results = run_benchmark(TINY_CONFIG)

    ind = results["in_distribution"]
    assert np.isfinite(ind["causal_rmse"]) and ind["causal_rmse"] > 0
    assert np.isfinite(ind["baseline_rmse"]) and ind["baseline_rmse"] > 0

    interv = results["intervention"]
    assert np.isfinite(interv["causal_rmse"])
    assert np.isfinite(interv["baseline_rmse"])
    assert len(interv["rows"]) == 1
    row = interv["rows"][0]
    for key in ("scenario", "fan_delta_rpm", "actual_change",
                "causal_predicted_change", "baseline_predicted_change",
                "causal_abs_error", "baseline_abs_error"):
        assert key in row
    # the fan actually cools the machine: the true change must be negative
    assert row["actual_change"] < 0

    rc = results["root_cause"]
    for side in ("causal", "importance"):
        assert 0.0 <= rc[side]["top1"] <= 1.0
        assert 0.0 <= rc[side]["top3"] <= 1.0
        assert len(rc[side]["cases"]) == 1
    assert len(rc["importance"]["ranking"]) == 8  # all non-target nodes
