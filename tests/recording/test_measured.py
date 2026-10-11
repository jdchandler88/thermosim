"""Measured-data CSV loading (delimiters, decimal commas, clock times, units) and comparison.

Assumed format for the Inkbird-style fixture (fixtures/inkbird_style_fahrenheit.csv): a logger
export with a header row "Date,Time,Temperature(°F)", one reading per row, Date as
YYYY-MM-DD, Time as HH:MM:SS, temperature in degrees Fahrenheit, readings every 20 s. The date
column is ignored; time is elapsed from the first row. Tilt-style exports (ISO timestamps in
a "Timestamp" column, "Temp" in °F) are handled by the same code path.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from thermosim.core.types import Sample
from thermosim.recording import MeasuredSeries, compare, load_measured_csv

FIXTURES = Path(__file__).parent / "fixtures"
Builder = Callable[..., Sample]


def test_comma_csv_in_seconds_celsius() -> None:
    series = load_measured_csv(FIXTURES / "comma_seconds.csv")
    assert series.label == "comma_seconds"
    np.testing.assert_allclose(series.t, [0, 30, 60, 90, 120, 150, 180])
    np.testing.assert_allclose(
        series.T, np.array([20.0, 24.5, 29.1, 33.0, 36.2, 39.0, 41.3]) + 273.15
    )
    assert series.t.dtype == np.float64


def test_semicolon_csv_decimal_commas_minutes_drops_nan_and_sorts() -> None:
    series = load_measured_csv(FIXTURES / "semicolon_minutes_decimal_comma.csv", time_unit="min")
    # Rows are 0, 2, 1, (empty), 4 minutes; the empty row is dropped and the rest sorted.
    np.testing.assert_allclose(series.t, [0.0, 60.0, 120.0, 240.0])
    # Default temp_unit is Celsius: 20.5 C etc.
    np.testing.assert_allclose(series.T, [293.65, 295.15, 296.15, 300.65], atol=1e-9)


def test_clock_time_fahrenheit_with_missing_reading() -> None:
    series = load_measured_csv(FIXTURES / "clock_fahrenheit.csv", temp_unit="F")
    np.testing.assert_allclose(series.t, [0.0, 30.0, 60.0, 90.0, 150.0])
    assert series.T[0] == pytest.approx(293.15)  # 68 F
    assert series.T[-1] == pytest.approx((80.2 - 32.0) * 5.0 / 9.0 + 273.15)


def test_inkbird_style_export() -> None:
    series = load_measured_csv(FIXTURES / "inkbird_style_fahrenheit.csv", temp_unit="F")
    np.testing.assert_allclose(series.t, [0.0, 20.0, 40.0, 60.0, 80.0])
    assert series.T[0] == pytest.approx((64.8 - 32.0) * 5.0 / 9.0 + 273.15)
    assert np.all(np.diff(series.T) > 0)


def test_explicit_column_names_and_kelvin(tmp_path: Path) -> None:
    path = tmp_path / "explicit.csv"
    path.write_text("junk,t,T\nx,0,300\nx,10,305\n", encoding="utf-8")
    series = load_measured_csv(path, time_col="t", temp_col="T", temp_unit="K")
    np.testing.assert_allclose(series.t, [0.0, 10.0])
    np.testing.assert_allclose(series.T, [300.0, 305.0])


def test_unknown_column_name_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "cols.csv"
    path.write_text("t,T\n0,20\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"cols\.csv.*no column named 'nope'"):
        load_measured_csv(path, time_col="nope")


def test_bad_value_error_names_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("t,T\n0,20\n30,abc\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.csv:3"):
        load_measured_csv(path)


def test_headerless_numeric_file_uses_first_two_columns(tmp_path: Path) -> None:
    path = tmp_path / "plain.csv"
    path.write_text("0,20.0\n60,21.5\n120,23.0\n", encoding="utf-8")
    series = load_measured_csv(path, time_unit="min")
    np.testing.assert_allclose(series.t, [0.0, 3600.0, 7200.0])
    np.testing.assert_allclose(series.T, [293.15, 294.65, 296.15])
    with pytest.raises(ValueError, match="no header row"):
        load_measured_csv(path, time_col="t")


def test_clock_times_wrap_past_midnight(tmp_path: Path) -> None:
    path = tmp_path / "midnight.csv"
    path.write_text("time,temp\n23:59:30,10\n00:00:30,11\n00:01:30,12\n", encoding="utf-8")
    series = load_measured_csv(path)
    np.testing.assert_allclose(series.t, [0.0, 60.0, 120.0])


def test_no_usable_rows_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("t,T\n0,\n10,nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        load_measured_csv(path)


def test_compare_perfect_model_has_zero_error(make_sample: Builder) -> None:
    samples = [
        make_sample(float(t), 293.15 + 0.1 * t, {"center": 293.15 + 0.1 * t}) for t in range(201)
    ]
    measured = MeasuredSeries(
        t=np.arange(0.0, 181.0, 30.0),
        T=293.15 + 0.1 * np.arange(0.0, 181.0, 30.0),
        label="m",
    )
    result = compare(samples, "center", measured, target_temp_K=293.15 + 5.0)
    assert result.rmse_K == pytest.approx(0.0, abs=1e-12)
    assert result.max_abs_K == pytest.approx(0.0, abs=1e-12)
    assert result.bias_K == pytest.approx(0.0, abs=1e-12)
    assert result.time_to_target_sim_s == pytest.approx(50.0)
    assert result.time_to_target_meas_s == pytest.approx(50.0)
    assert result.time_to_target_error_rel == pytest.approx(0.0, abs=1e-12)


def test_compare_offset_gives_bias_and_time_to_target_error(make_sample: Builder) -> None:
    samples = [
        make_sample(float(t), 293.15 + 0.1 * t, {"center": 293.15 + 0.1 * t}) for t in range(201)
    ]
    grid = np.arange(0.0, 181.0, 30.0)
    # Measured runs 1 K hotter everywhere, so it reaches the target 10 s earlier.
    measured = MeasuredSeries(t=grid, T=293.15 + 0.1 * grid + 1.0, label="m")
    result = compare(samples, "center", measured, target_temp_K=293.15 + 6.0)
    assert result.bias_K == pytest.approx(-1.0)
    assert result.rmse_K == pytest.approx(1.0)
    assert result.max_abs_K == pytest.approx(1.0)
    assert result.time_to_target_sim_s == pytest.approx(60.0)
    assert result.time_to_target_meas_s == pytest.approx(50.0)
    assert result.time_to_target_error_rel == pytest.approx(0.2)


def test_compare_uses_only_overlapping_range(make_sample: Builder) -> None:
    samples = [make_sample(float(t), 300.0, {"center": 300.0}) for t in range(0, 101)]
    measured = MeasuredSeries(
        t=np.array([0.0, 50.0, 100.0, 500.0]),
        T=np.array([300.0, 300.0, 300.0, 9999.0]),  # the last point is outside the overlap
        label="m",
    )
    result = compare(samples, "center", measured, target_temp_K=None)
    assert result.rmse_K == pytest.approx(0.0)
    assert result.max_abs_K == pytest.approx(0.0)
    assert result.time_to_target_sim_s is None
    assert result.time_to_target_meas_s is None
    assert result.time_to_target_error_rel is None


def test_compare_rejects_disjoint_ranges(make_sample: Builder) -> None:
    samples = [make_sample(float(t), 300.0) for t in range(10)]
    measured = MeasuredSeries(t=np.array([100.0, 200.0]), T=np.array([300.0, 301.0]), label="m")
    with pytest.raises(ValueError, match="do not overlap"):
        compare(samples, "mean", measured, target_temp_K=None)
