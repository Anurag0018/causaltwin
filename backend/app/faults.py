"""Fault injection + ground-truth intervention runs (Phase 4).

`data_loader.generate()` has no fault/intervention hooks, so this module
carries a faithful z-space replica of the generator (the same approach the
Phase 3 estimator smoke validated bit-for-bit) with two hooks:

  fault = {"node", "delta_z", "start", "end"}
      A sustained mechanism/level shift: `delta_z` (in z-space units) is
      ADDED to the node's computed value at every timestep in
      [start, end). All cross-node dependencies in the SCM are lagged >= 1,
      so applying the shift at the end of the timestep is dynamically
      exact: the node's own series shows the shift from `start` on, and
      descendants feel it from the next step through the TRUE equations.
      This models e.g. a cooling degradation (coolant_temp running hot)
      or a sensor/mechanism bias — and because it propagates through the
      true dynamics, the downstream anomaly pattern is the real one.

  step = {"node", "delta_z", "start", "end"}
      A sustained do() override used as WHAT-IF GROUND TRUTH: during
      [start, end) the node's z-value is SET to its natural (hook-free,
      same-seed) trajectory + `delta_z` — i.e. the lever is clamped
      `delta_z` above what it would naturally have been, while every
      other node evolves through the true equations. Implemented as two
      passes over the same seed (natural pass first). Noise draws stay
      aligned between the passes as long as the hooks do not change the
      vibration path (the maintenance trigger is the only data-dependent
      RNG consumer); the Phase 4 scenario — coolant fault + fan step —
      satisfies this, and the script asserts the resulting fan elevation.

With no hooks, simulate_scm(n, seed) reproduces generate(n, seed)
exactly (asserted in the test suite): same RNG call order and
short-circuit behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_loader

_Z_NODES = [n for n in data_loader.NODES if n != "maintenance_event"]


def _run_z(n_steps: int, seed: int, fault: dict | None = None,
           step: dict | None = None, z_natural: dict | None = None):
    """One pass of the SCM dynamics in z-space, with optional hooks."""
    rng = np.random.default_rng(seed)
    z = {name: np.zeros(n_steps) for name in _Z_NODES}
    maint = np.zeros(n_steps, dtype=float)
    events = []

    for t in range(n_steps):
        prev = {k: (v[t - 1] if t >= 1 else 0.0) for k, v in z.items()}
        prev2 = {k: (v[t - 2] if t >= 2 else 0.0) for k, v in z.items()}

        z["load"][t] = (
            0.6 * prev["load"]
            + 0.5 * np.sin(2 * np.pi * t / 100.0)
            + rng.normal(0, 0.6)
        )
        z["fan_speed_nc"][t] = (
            0.5 * prev["fan_speed_nc"]
            + 0.4 * np.sin(2 * np.pi * t / 140.0 + 1.0)
            + rng.normal(0, 0.6)
        )

        scheduled = (t % 90 == 45)
        triggered = prev["vibration"] > 1.6 and rng.random() < 0.5
        if t >= 1 and (scheduled or triggered):
            maint[t] = 1.0
            events.append({"ts": int(t), "type": "maintenance",
                           "node": "vibration", "is_treatment": True})

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
            + 0.12 * np.clip(prev["load"], -2, 2) ** 2
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
        z["failure_risk"][t] = (
            0.3 * prev["failure_risk"]
            + 0.7 * prev["hpc_outlet_temp"]
            - 0.6 * prev["oil_pressure"]
            + 0.3 * prev["vibration"]
            + rng.normal(0, 0.3)
        )

        # --- hooks (end of timestep; exact because all cross-node
        # dependencies are lagged >= 1) --------------------------------
        if fault is not None and fault["start"] <= t < fault["end"]:
            z[fault["node"]][t] += float(fault["delta_z"])
        if step is not None and step["start"] <= t < step["end"]:
            z[step["node"]][t] = z_natural[step["node"]][t] + float(step["delta_z"])

    return z, maint, events


def _to_frame(z, maint) -> pd.DataFrame:
    data = {}
    for name, (offset, scale) in data_loader._SCALE.items():
        col = offset + scale * z[name]
        if name == "failure_risk":
            col = np.clip(col, 0.0, 100.0)
        data[name] = col
    data["maintenance_event"] = maint
    return pd.DataFrame(data, columns=data_loader.NODES)


def simulate_scm(n_steps: int = 2000, seed: int = 0,
                 fault: dict | None = None, step: dict | None = None):
    """Simulate the synthetic SCM with optional fault / step hooks.

    Returns (df, events) in engineering units, same shape as
    data_loader.generate(). With fault=step=None the output equals
    generate(n_steps, seed) exactly. Hook windows are [start, end) in
    timesteps; deltas are in z-space units (fan_speed_nc: 1 z = 80 rpm;
    failure_risk: 1 z = 12 points; see data_loader._SCALE).
    """
    for hook in (fault, step):
        if hook is not None:
            if hook["node"] not in _Z_NODES:
                raise ValueError(f"hook node must be one of {_Z_NODES}, got {hook['node']!r}")
            if not (0 <= hook["start"] < hook["end"] <= n_steps):
                raise ValueError(f"hook window must satisfy 0 <= start < end <= {n_steps}")

    z_natural = None
    if step is not None:
        z_natural, _, _ = _run_z(n_steps, seed)
    z, maint, events = _run_z(n_steps, seed, fault=fault, step=step,
                              z_natural=z_natural)
    if fault is not None:
        events.append({"ts": int(fault["start"]), "type": "fault_injection",
                       "node": fault["node"], "is_treatment": False,
                       "delta_z": float(fault["delta_z"]),
                       "end": int(fault["end"])})
    if step is not None:
        events.append({"ts": int(step["start"]), "type": "intervention",
                       "node": step["node"], "is_treatment": True,
                       "delta_z": float(step["delta_z"]),
                       "end": int(step["end"])})
    return _to_frame(z, maint), events
