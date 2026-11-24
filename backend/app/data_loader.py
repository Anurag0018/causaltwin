"""Synthetic SCM generator + NASA CMAPSS hook (Phase 1).

The synthetic system is a machine-like telemetry twin with NAMED nodes and a
KNOWN lagged ground-truth DAG, so causal discovery can be scored exactly
(Phase 2: SHD / Edge F1) and interventions have a checkable sign.

Dynamics are generated in standardised (z) space as a lagged structural
system, then mapped to engineering-looking units with positive affine
transforms (affine maps preserve the causal graph and effect signs).
`maintenance_event` is a binary event series aligned to the same timeline:
maintenance is scheduled periodically and also triggered when vibration is
high, and it causally reduces next-step vibration. Alarm events are recorded
when HPC outlet temperature crosses a threshold (they are observations of the
system, not causes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NODES = [
    "load",
    "fan_speed_nc",
    "coolant_temp",
    "vibration",
    "hpc_outlet_temp",
    "oil_pressure",
    "power_draw",
    "failure_risk",
    "maintenance_event",
]

# (source, target, lag) — self-lag (autoregressive) terms are NOT listed;
# discovery metrics ignore self-links on both sides.
GROUND_TRUTH_EDGES = [
    ("load", "power_draw", 1),
    ("fan_speed_nc", "power_draw", 1),
    ("fan_speed_nc", "coolant_temp", 1),
    ("coolant_temp", "hpc_outlet_temp", 1),
    ("load", "hpc_outlet_temp", 1),
    ("fan_speed_nc", "hpc_outlet_temp", 2),
    ("load", "vibration", 1),
    ("maintenance_event", "vibration", 1),
    ("vibration", "oil_pressure", 1),
    ("hpc_outlet_temp", "failure_risk", 1),
    ("oil_pressure", "failure_risk", 1),
    ("vibration", "failure_risk", 1),
]

# z-space -> engineering units (positive affine maps only)
_SCALE = {
    "load": (70.0, 15.0),
    "fan_speed_nc": (2250.0, 80.0),
    "coolant_temp": (620.0, 25.0),
    "vibration": (4.0, 1.5),
    "hpc_outlet_temp": (1100.0, 70.0),
    "oil_pressure": (60.0, 8.0),
    "power_draw": (400.0, 60.0),
    "failure_risk": (30.0, 12.0),
}


def generate(n_steps: int = 1200, seed: int = 0):
    """Generate synthetic telemetry with a known ground-truth DAG.

    Returns (df, ground_truth_edges, events):
      df    — DataFrame with one column per node in NODES (incl. the binary
              `maintenance_event` column), in engineering units.
      edges — list of (source, target, lag) tuples (no self-links).
      events— list of dicts {ts, type, node, is_treatment}.
    Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    z = {name: np.zeros(n_steps) for name in NODES if name != "maintenance_event"}
    maint = np.zeros(n_steps, dtype=float)
    events = []

    for t in range(n_steps):
        prev = {k: (v[t - 1] if t >= 1 else 0.0) for k, v in z.items()}
        prev2 = {k: (v[t - 2] if t >= 2 else 0.0) for k, v in z.items()}

        # --- exogenous drivers -------------------------------------------------
        z["load"][t] = (
            0.6 * prev["load"]
            + 0.5 * np.sin(2 * np.pi * t / 100.0)          # sinusoidal seasonality
            + rng.normal(0, 0.6)
        )
        z["fan_speed_nc"][t] = (
            0.5 * prev["fan_speed_nc"]
            + 0.4 * np.sin(2 * np.pi * t / 140.0 + 1.0)
            + rng.normal(0, 0.6)
        )

        # --- event stream: scheduled + vibration-triggered maintenance ---------
        scheduled = (t % 90 == 45)
        triggered = prev["vibration"] > 1.6 and rng.random() < 0.5
        if t >= 1 and (scheduled or triggered):
            maint[t] = 1.0
            events.append({"ts": int(t), "type": "maintenance",
                           "node": "vibration", "is_treatment": True})

        # --- intermediate state -------------------------------------------------
        z["coolant_temp"][t] = (
            0.5 * prev["coolant_temp"]
            - 0.75 * prev["fan_speed_nc"]
            + rng.normal(0, 0.35)
        )
        z["vibration"][t] = (
            0.45 * prev["vibration"]
            + 0.7 * prev["load"]
            - 1.4 * (maint[t - 1] if t >= 1 else 0.0)
            + rng.normal(0, 0.35)
        )
        z["hpc_outlet_temp"][t] = (
            0.5 * prev["hpc_outlet_temp"]
            + 0.65 * prev["load"]
            + 0.55 * prev["coolant_temp"]
            - 0.5 * prev2["fan_speed_nc"]
            + 0.12 * np.clip(prev["load"], -2, 2) ** 2      # mild nonlinearity
            + rng.normal(0, 0.35)
        )
        if z["hpc_outlet_temp"][t] > 2.3:
            events.append({"ts": int(t), "type": "alarm",
                           "node": "hpc_outlet_temp", "is_treatment": False})
        z["oil_pressure"][t] = (
            0.55 * prev["oil_pressure"]
            - 0.6 * prev["vibration"]
            + rng.normal(0, 0.35)
        )
        z["power_draw"][t] = (
            0.4 * prev["power_draw"]
            + 0.8 * prev["load"]
            + 0.3 * prev["fan_speed_nc"]
            + rng.normal(0, 0.35)
        )

        # --- target KPI -----------------------------------------------------------
        z["failure_risk"][t] = (
            0.3 * prev["failure_risk"]
            + 0.7 * prev["hpc_outlet_temp"]
            - 0.6 * prev["oil_pressure"]
            + 0.3 * prev["vibration"]
            + rng.normal(0, 0.3)
        )

    data = {}
    for name, (offset, scale) in _SCALE.items():
        col = offset + scale * z[name]
        if name == "failure_risk":
            col = np.clip(col, 0.0, 100.0)
        data[name] = col
    data["maintenance_event"] = maint
    df = pd.DataFrame(data, columns=NODES)
    return df, list(GROUND_TRUTH_EDGES), events


def adf_report(df: pd.DataFrame) -> dict:
    """Augmented Dickey-Fuller stationarity check per column.

    Returns {column: {statistic, p_value, stationary}} — used by the
    preprocessing stage before discovery (blueprint §3.2).
    """
    from statsmodels.tsa.stattools import adfuller

    report = {}
    for col in df.columns:
        stat, p_value, *_ = adfuller(df[col].dropna().to_numpy(), autolag="AIC")
        report[col] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "stationary": bool(p_value < 0.05),
        }
    return report


def load_cmapss(path: str) -> pd.DataFrame:
    """NASA CMAPSS loader stub (Phase 1 extension).

    CMAPSS is observational run-to-failure data: in this project it is used
    for discovery / root-cause demonstration only — interventional claims
    come from the synthetic SCM above. Full parsing lands in Phase 1.
    """
    raise NotImplementedError(
        "CMAPSS loading is a Phase 1 extension; the weekend scaffold "
        "validates on the synthetic SCM (see README)."
    )
