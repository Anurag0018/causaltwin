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

from pathlib import Path

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


# ---------------------------------------------------------------------------
# NASA CMAPSS (Phase 1 — implemented)
#
# CMAPSS is observational run-to-failure data: in this project it is used
# for discovery / root-cause demonstration only — interventional claims
# come from the synthetic SCM above.
# ---------------------------------------------------------------------------

CMAPSS_COLUMNS = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors that are constant (zero variance) across FD001 — the standard
# FD001 constant-sensor list. They carry no signal, so modelling code
# usually drops them; the loader keeps them by default and only drops
# them when asked (drop_constant=True / select_sensors()).
CMAPSS_CONSTANT_SENSORS_FD001 = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CMAPSS_PATH = _REPO_ROOT / "data" / "train_FD001.txt"


def load_cmapss(path=None, *, rul_cap: int | None = 125,
                drop_constant: bool = False) -> pd.DataFrame:
    """Load a NASA CMAPSS training file (e.g. train_FD001.txt).

    The raw files are whitespace-separated with no header and 26 columns:
    unit, cycle, 3 operational settings, then sensors s1..s21.

    Args:
        path: file to load. Defaults to <repo root>/data/train_FD001.txt.
        rul_cap: piecewise-RUL cap (standard practice: 125 cycles — RUL is
            clipped at this value because engines are effectively "healthy"
            far from failure). Pass None for the raw, uncapped RUL.
        drop_constant: if True, drop the known constant FD001 sensors
            (s1, s5, s6, s10, s16, s18, s19 — see
            CMAPSS_CONSTANT_SENSORS_FD001). Default False: the full raw
            frame is returned; use select_sensors() to drop them later.

    Returns:
        DataFrame sorted by (unit, cycle) with numeric dtypes, the 26 raw
        columns plus an `RUL` column (max cycle of the unit − cycle,
        capped at rul_cap if given). RUL is 0 at each unit's last cycle.

    Raises:
        FileNotFoundError: with download/placement instructions if the
            file does not exist.
    """
    file_path = Path(path) if path is not None else DEFAULT_CMAPSS_PATH
    if not file_path.is_file():
        raise FileNotFoundError(
            f"CMAPSS data file not found: {file_path}\n"
            "Download the NASA C-MAPSS 'Turbofan Engine Degradation "
            "Simulation Data Set' (data.nasa.gov — C-MAPSS Jet Engine "
            "Simulated Data; also mirrored at data.phmsociety.org/nasa), "
            "unzip it, and place train_FD001.txt in the project's data/ "
            f"directory (expected at {DEFAULT_CMAPSS_PATH})."
        )
    df = pd.read_csv(file_path, sep=r"\s+", header=None,
                     names=CMAPSS_COLUMNS, engine="python")
    # Trailing whitespace in some distributions yields an all-NaN extra
    # column set; with fixed names anything beyond the 26 known columns is
    # dropped defensively via the column list itself.
    df = df[CMAPSS_COLUMNS]
    for col in CMAPSS_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)

    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    if rul_cap is not None:
        df["RUL"] = df["RUL"].clip(upper=rul_cap)

    if drop_constant:
        df = select_sensors(df)
    return df


def select_sensors(df: pd.DataFrame, *,
                   constant_sensors=CMAPSS_CONSTANT_SENSORS_FD001) -> pd.DataFrame:
    """Return a copy of a CMAPSS frame without the constant FD001 sensors.

    Only sensors actually present are dropped, so this is safe on any
    subset frame. All non-sensor columns (unit, cycle, op settings, RUL)
    are kept.
    """
    drop = [c for c in constant_sensors if c in df.columns]
    return df.drop(columns=drop)


def stationarity_report(df: pd.DataFrame, columns=None) -> dict:
    """ADF stationarity report for selected columns (preprocessing, §3.1/§3.2).

    Returns {column: {adf_statistic, p_value, stationary_bool}} where
    stationary_bool is p_value < 0.05. `columns` defaults to all numeric
    columns except unit/cycle/RUL (identifiers and the derived target are
    not stationarity-tested).
    """
    from statsmodels.tsa.stattools import adfuller

    if columns is None:
        skip = {"unit", "cycle", "RUL"}
        columns = [c for c in df.columns
                   if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    report = {}
    for col in columns:
        stat, p_value, *_ = adfuller(df[col].dropna().to_numpy(), autolag="AIC")
        report[col] = {
            "adf_statistic": float(stat),
            "p_value": float(p_value),
            "stationary_bool": bool(p_value < 0.05),
        }
    return report


def prepare_series(df: pd.DataFrame, unit_id, columns=None) -> pd.DataFrame:
    """Extract one engine unit's time series in cycle order (preprocessing).

    Returns a copy containing `cycle` plus the requested columns (default:
    all columns), sorted by cycle with a fresh RangeIndex — the per-unit
    series shape the discovery stage consumes.

    Raises:
        ValueError: if unit_id is not present in df.
    """
    unit_df = df[df["unit"] == unit_id]
    if unit_df.empty:
        raise ValueError(
            f"unit {unit_id!r} not present in frame "
            f"(units available: {sorted(df['unit'].unique())[:10]}...)"
        )
    unit_df = unit_df.sort_values("cycle").reset_index(drop=True)
    if columns is not None:
        keep = ["cycle"] + [c for c in columns if c != "cycle"]
        unit_df = unit_df[keep]
    return unit_df
