"""Phase 2 de-risk — PCMCI discovery smoke run on REAL NASA CMAPSS FD001.

Purpose: prove discovery runs end-to-end on real observational data and
produces a structurally sensible graph. This is a SMOKE RUN, not a
benchmark: CMAPSS has no ground-truth DAG, so per the project's
honest-claims rule NO accuracy metrics are computed or claimed here —
only structure, runtime and plausibility are reported. (All accuracy
claims in CausalTwin come from the synthetic SCM.)

Method:
  * FD001 is loaded with app.data_loader.load_cmapss(drop_constant=True),
    which drops the constant FD001 sensors via the loader's own
    select_sensors() machinery.
  * One analysis series: the longest-lived engine unit, extracted with
    app.data_loader.prepare_series(). Columns = the 3 op settings + the
    14 non-constant sensors (unit/cycle/RUL excluded).
  * Stationarity context: app.data_loader.stationarity_report() (ADF) on
    those columns for the chosen unit.
  * Attempt 1 — RAW series into app.discovery.run_discovery
    (tau_max=2, pc_alpha=0.001, the Phase 2 sweep's winning alpha).
  * Attempt 2 — FIRST-DIFFERENCED series (x_t - x_{t-1}, first row
    dropped), same discovery settings. CMAPSS run-to-failure sensors
    carry a strong degradation trend, so most raw series are expected to
    be non-stationary per ADF; differencing is the simple, documented
    remedy. BOTH attempts are reported — nothing is silently transformed.

Results are written to cmapss_discovery_smoke.json next to this script.

    python scripts/cmapss_discovery_smoke.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import data_loader
from app.discovery import run_discovery

TAU_MAX = 2
PC_ALPHA = 0.001  # winning alpha from the Phase 2 synthetic tuning sweep

RESULTS_PATH = Path(__file__).resolve().parent / "cmapss_discovery_smoke.json"


def run_quiet(df, tau_max, pc_alpha):
    """run_discovery with Tigramite's verbose progress output suppressed."""
    buf = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        edges = run_discovery(df, tau_max=tau_max, pc_alpha=pc_alpha)
    return edges, time.perf_counter() - start


def main() -> int:
    print("=== CMAPSS FD001 discovery smoke run (structure only, no accuracy claims) ===")

    df = data_loader.load_cmapss(drop_constant=True)
    print(f"loaded FD001 (drop_constant=True): shape={df.shape}, "
          f"units={df['unit'].nunique()}")

    # Longest-lived unit = most time steps for discovery.
    life = df.groupby("unit")["cycle"].max()
    unit_id = int(life.idxmax())
    n_cycles = int(life.max())
    print(f"chosen unit: {unit_id} (longest life, {n_cycles} cycles)")

    sensor_cols = [c for c in df.columns if c.startswith("s")]
    op_cols = [c for c in df.columns if c.startswith("op_setting_")]
    analysis_cols = op_cols + sensor_cols
    print(f"analysis columns ({len(analysis_cols)}): {analysis_cols}")

    series = data_loader.prepare_series(df, unit_id, columns=analysis_cols)
    series = series.drop(columns=["cycle"])
    print(f"series shape: {series.shape}")

    # FD001 has a single operating condition, so within ONE unit some op
    # settings are exactly constant. A zero-variance column breaks both
    # ADF ("x is constant") and ParCorr, so constant-within-unit columns
    # are excluded here in the script (documented, not silent) and the
    # discovery runs on the varying columns only.
    variances = series.var(numeric_only=True)
    constant_cols = [c for c in analysis_cols if variances[c] == 0.0]
    if constant_cols:
        print(f"constant within unit {unit_id} (excluded): {constant_cols}")
        series = series.drop(columns=constant_cols)
        analysis_cols = [c for c in analysis_cols if c not in constant_cols]
        print(f"series shape after exclusion: {series.shape}; columns: {analysis_cols}")

    adf = data_loader.stationarity_report(series, columns=analysis_cols)
    n_stat = sum(1 for v in adf.values() if v["stationary_bool"])
    print(f"ADF on raw unit-{unit_id} series: {n_stat}/{len(adf)} columns stationary "
          f"(non-stationary: {[c for c, v in adf.items() if not v['stationary_bool']]})")

    attempts = []

    # --- Attempt 1: raw series -------------------------------------------------
    edges_raw, rt_raw = run_quiet(series, TAU_MAX, PC_ALPHA)
    print(f"\n[raw] runtime={rt_raw:.1f}s edges={len(edges_raw)}")
    for e in edges_raw:
        print(f"  {e['source']} -> {e['target']} @ lag {e['lag']} (p={e['p_value']:.4g})")
    attempts.append({
        "name": "raw",
        "transform": "none — series exactly as loaded",
        "input_shape": list(series.shape),
        "runtime_s": round(rt_raw, 2),
        "n_edges": len(edges_raw),
        "edges": edges_raw,
    })

    # --- Attempt 2: first-differenced series (documented transform) ------------
    differenced = series.diff().dropna().reset_index(drop=True)
    adf_diff = data_loader.stationarity_report(differenced, columns=analysis_cols)
    n_stat_diff = sum(1 for v in adf_diff.values() if v["stationary_bool"])
    print(f"\nADF after first differencing: {n_stat_diff}/{len(adf_diff)} columns stationary")
    edges_diff, rt_diff = run_quiet(differenced, TAU_MAX, PC_ALPHA)
    print(f"[first_difference] runtime={rt_diff:.1f}s edges={len(edges_diff)}")
    for e in edges_diff:
        print(f"  {e['source']} -> {e['target']} @ lag {e['lag']} (p={e['p_value']:.4g})")
    attempts.append({
        "name": "first_difference",
        "transform": "first difference x_t - x_{t-1} per column, first row dropped "
                     "(remedy for the degradation trend flagged by ADF on the raw series)",
        "input_shape": list(differenced.shape),
        "runtime_s": round(rt_diff, 2),
        "n_edges": len(edges_diff),
        "edges": edges_diff,
    })

    out = {
        "dataset": "NASA CMAPSS FD001 (train_FD001.txt), observational — no ground truth, "
                   "no accuracy metrics claimed (honest-claims rule)",
        "unit": unit_id,
        "unit_cycles": n_cycles,
        "excluded_constant_within_unit": constant_cols,
        "columns": analysis_cols,
        "tau_max": TAU_MAX,
        "pc_alpha": PC_ALPHA,
        "adf_raw": adf,
        "adf_first_difference": adf_diff,
        "attempts": attempts,
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
