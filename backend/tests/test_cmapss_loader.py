"""Tests for the CMAPSS loader + preprocessing helpers (Phase 1).

Uses a small hand-made CMAPSS-format fixture: whitespace-separated, no
header, 26 columns (unit, cycle, 3 op settings, s1..s21). Rows are
deliberately written out of (unit, cycle) order to test sorting.
"""
import numpy as np
import pandas as pd
import pytest

from app.data_loader import (
    CMAPSS_COLUMNS,
    CMAPSS_CONSTANT_SENSORS_FD001,
    load_cmapss,
    prepare_series,
    select_sensors,
    stationarity_report,
)


def _row(unit, cycle, base=100.0):
    vals = [unit, cycle, 0.1, 0.2, 100.0] + [base + i + cycle * 0.5 for i in range(21)]
    return " ".join(str(v) for v in vals)


@pytest.fixture()
def cmapss_file(tmp_path):
    """3 units: unit 1 has 5 cycles, unit 2 has 3, unit 3 has 200 (for cap)."""
    lines = []
    # unit 2 first, cycles shuffled — loader must sort
    for c in [2, 1, 3]:
        lines.append(_row(2, c, base=200.0))
    for c in [5, 3, 1, 4, 2]:
        lines.append(_row(1, c, base=100.0))
    for c in range(1, 201):
        lines.append(_row(3, c, base=300.0))
    p = tmp_path / "train_FD001.txt"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_column_names_and_count(cmapss_file):
    df = load_cmapss(cmapss_file)
    assert list(df.columns) == CMAPSS_COLUMNS + ["RUL"]
    assert len(df.columns) == 27  # 26 raw + RUL
    assert len(df) == 5 + 3 + 200
    assert pd.api.types.is_numeric_dtype(df["s1"])
    assert pd.api.types.is_integer_dtype(df["unit"])


def test_sorted_by_unit_cycle(cmapss_file):
    df = load_cmapss(cmapss_file)
    pairs = list(zip(df["unit"], df["cycle"]))
    assert pairs == sorted(pairs)


def test_rul_correctness(cmapss_file):
    df = load_cmapss(cmapss_file, rul_cap=None)
    u1 = df[df["unit"] == 1].sort_values("cycle")
    assert list(u1["RUL"]) == [4, 3, 2, 1, 0]          # max_cycle=5
    assert u1["RUL"].iloc[0] == 5 - 1                  # first cycle: max-1
    assert u1["RUL"].iloc[-1] == 0                     # last cycle: 0
    u2 = df[df["unit"] == 2].sort_values("cycle")
    assert list(u2["RUL"]) == [2, 1, 0]


def test_rul_capped_at_125(cmapss_file):
    df = load_cmapss(cmapss_file)  # default cap 125
    u3 = df[df["unit"] == 3].sort_values("cycle")
    assert u3["RUL"].max() == 125                     # raw max would be 199
    assert u3["RUL"].iloc[-1] == 0
    df_raw = load_cmapss(cmapss_file, rul_cap=None)
    assert df_raw[df_raw["unit"] == 3]["RUL"].max() == 199


def test_drop_constant_and_select_sensors(cmapss_file):
    df_full = load_cmapss(cmapss_file)
    for s in CMAPSS_CONSTANT_SENSORS_FD001:
        assert s in df_full.columns                    # raw frame keeps them
    df_sel = select_sensors(df_full)
    for s in CMAPSS_CONSTANT_SENSORS_FD001:
        assert s not in df_sel.columns
    assert "RUL" in df_sel.columns and "s2" in df_sel.columns
    df_dropped = load_cmapss(cmapss_file, drop_constant=True)
    assert list(df_dropped.columns) == list(df_sel.columns)


def test_missing_file_error_message(tmp_path):
    missing = tmp_path / "train_FD001.txt"
    with pytest.raises(FileNotFoundError) as exc:
        load_cmapss(missing)
    msg = str(exc.value)
    assert "train_FD001.txt" in msg
    assert "data" in msg.lower() and "download" in msg.lower()


def test_stationarity_report(cmapss_file):
    df = load_cmapss(cmapss_file)
    report = stationarity_report(df, columns=["s2", "s3"])
    assert set(report.keys()) == {"s2", "s3"}
    for entry in report.values():
        assert set(entry.keys()) == {"adf_statistic", "p_value", "stationary_bool"}
        assert isinstance(entry["stationary_bool"], bool)
        assert 0.0 <= entry["p_value"] <= 1.0


def test_prepare_series(cmapss_file):
    df = load_cmapss(cmapss_file)
    series = prepare_series(df, 1, columns=["s2", "RUL"])
    assert list(series["cycle"]) == [1, 2, 3, 4, 5]
    assert list(series.columns) == ["cycle", "s2", "RUL"]
    with pytest.raises(ValueError):
        prepare_series(df, 999)
