"""Measured temperature data: CSV loading and comparison against a simulated probe."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from thermosim.core.types import F64, Sample
from thermosim.recording.summary import first_crossing_time, probe_series, sample_times

_KELVIN_OFFSET = 273.15
_DAY_S = 86400.0
_HALF_DAY_S = 43200.0
# Cell values treated as "no reading": such rows are dropped.
_MISSING = frozenset({"", "nan", "na", "n/a", "-", "--", "null", "none"})
_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?$")


@dataclass(frozen=True)
class MeasuredSeries:
    t: F64  # s
    T: F64  # K
    label: str


@dataclass(frozen=True)
class Comparison:
    rmse_K: float
    max_abs_K: float
    bias_K: float
    time_to_target_sim_s: float | None
    time_to_target_meas_s: float | None
    time_to_target_error_rel: float | None


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def _is_time_name(name: str) -> bool:
    n = _norm(name)
    return n in {"t", "s"} or n.startswith(("time", "second", "sec", "minute", "min", "elapsed"))


def _is_temp_name(name: str) -> bool:
    return _norm(name).startswith("temp")


def _is_t_name(name: str) -> bool:
    return _norm(name) == "t"


def _detect_delimiter(line: str) -> str:
    """Tab, then semicolon, then comma. Semicolon files often use decimal commas."""
    if "\t" in line:
        return "\t"
    if ";" in line:
        return ";"
    return ","


def _numeric(text: str, delimiter: str) -> float | None:
    """The value of a plain number, or None. A decimal comma is accepted unless ',' delimits."""
    s = text.strip()
    if delimiter != "," and "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _clock_seconds(text: str) -> float | None:
    """Clock time 'HH:MM[:SS[.frac]]' as seconds of day, or an ISO date-time as epoch seconds."""
    s = text.strip()
    m = _CLOCK_RE.match(s)
    if m is not None:
        hours, minutes, seconds = m.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds or 0.0)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _to_kelvin(value: float, unit: Literal["C", "F", "K"]) -> float:
    if unit == "C":
        return value + _KELVIN_OFFSET
    if unit == "F":
        return (value - 32.0) * 5.0 / 9.0 + _KELVIN_OFFSET
    return value


def _numberish(text: str, delimiter: str) -> bool:
    """A missing marker, a number, or a clock/date-time: anything a data cell may hold."""
    return (
        text.strip().lower() in _MISSING
        or _numeric(text, delimiter) is not None
        or _clock_seconds(text) is not None
    )


def _find_column(path: Path, headers: list[str], name: str, exclude: int | None = None) -> int:
    """Index of the column called `name`: exact match first, then case-insensitive."""
    wanted = name.strip()
    candidates = [j for j in range(len(headers)) if j != exclude]
    for j in candidates:
        if headers[j].strip() == wanted:
            return j
    for j in candidates:
        if headers[j].strip().lower() == wanted.lower():
            return j
    raise ValueError(f"{path}: no column named {name!r}; columns are {headers}")


def load_measured_csv(
    path: Path,
    time_col: str | None = None,
    temp_col: str | None = None,
    temp_unit: Literal["C", "F", "K"] = "C",
    time_unit: Literal["s", "min"] = "s",
) -> MeasuredSeries:
    """Read a measured temperature series from CSV and convert it to SI (s, K).

    Delimiter: tab, semicolon or comma (first one present on the first line). Decimal commas
    are accepted with non-comma delimiters. Time is a number in `time_unit`, or a clock time
    (HH:MM[:SS], or an ISO date-time) which is converted to seconds from the first row.
    Without `time_col`/`temp_col`, columns are found by name (time: t, time, seconds,
    minutes...; temperature: temp*, or T); failing that, the first two numeric columns.
    Rows with missing values are dropped; the result is sorted by time.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        raw = f.read()
    first_line = next((ln for ln in raw.splitlines() if ln.strip()), None)
    if first_line is None:
        raise ValueError(f"{path}: file is empty")
    delimiter = _detect_delimiter(first_line)

    reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    records: list[tuple[int, list[str]]] = []
    for row in reader:
        if any(cell.strip() for cell in row):
            records.append((reader.line_num, row))

    first_lineno, first_row = records[0]
    has_header = not all(_numberish(c, delimiter) for c in first_row)
    headers: list[str] | None = None
    data = records
    if has_header:
        headers = [c.strip() for c in first_row]
        data = records[1:]
    elif time_col is not None or temp_col is not None:
        raise ValueError(
            f"{path}:{first_lineno}: column names were given but the file has no header row"
        )

    ncols = max((len(row) for _, row in records), default=0)

    def cell(row: list[str], j: int | None) -> str:
        assert j is not None
        return row[j] if j < len(row) else ""

    def numberish(text: str) -> bool:
        return _numberish(text, delimiter)

    t_idx: int | None = None
    T_idx: int | None = None
    if headers is not None:
        if time_col is not None:
            t_idx = _find_column(path, headers, time_col)
        else:
            t_idx = next((j for j, h in enumerate(headers) if _is_time_name(h)), None)
        if temp_col is not None:
            T_idx = _find_column(path, headers, temp_col, exclude=t_idx)
        else:
            candidates = [j for j, h in enumerate(headers) if j != t_idx and _is_temp_name(h)]
            candidates += [j for j, h in enumerate(headers) if j != t_idx and _is_t_name(h)]
            T_idx = candidates[0] if candidates else None

    if t_idx is None or T_idx is None:
        numeric_cols = [
            j
            for j in range(ncols)
            if all(numberish(cell(row, j)) for _, row in data)
            and any(cell(row, j).strip() for _, row in data)
        ]
        if t_idx is None and T_idx is None:
            if len(numeric_cols) < 2:
                raise ValueError(f"{path}: need a time and a temperature column; found none")
            t_idx, T_idx = numeric_cols[0], numeric_cols[1]
        elif t_idx is None:
            others = [j for j in numeric_cols if j != T_idx]
            if not others:
                raise ValueError(f"{path}: no numeric time column found")
            t_idx = others[0]
        else:
            others = [j for j in numeric_cols if j != t_idx]
            if not others:
                raise ValueError(f"{path}: no numeric temperature column found")
            T_idx = others[0]

    # Time style is decided by the first usable time cell: plain number or clock time.
    use_clock: bool | None = None
    time_factor = 60.0 if time_unit == "min" else 1.0
    times: list[float] = []
    temps: list[float] = []
    for lineno, row in data:
        t_text = cell(row, t_idx)
        T_text = cell(row, T_idx)
        if t_text.strip().lower() in _MISSING or T_text.strip().lower() in _MISSING:
            continue
        if use_clock is None:
            use_clock = _numeric(t_text, delimiter) is None
        if use_clock:
            t_val = _clock_seconds(t_text)
            if t_val is None:
                raise ValueError(f"{path}:{lineno}: cannot read clock time from {t_text!r}")
        else:
            num = _numeric(t_text, delimiter)
            if num is None:
                raise ValueError(f"{path}:{lineno}: cannot read time from {t_text!r}")
            t_val = num * time_factor
        T_val = _numeric(T_text, delimiter)
        if T_val is None:
            raise ValueError(f"{path}:{lineno}: cannot read temperature from {T_text!r}")
        if np.isnan(t_val) or np.isnan(T_val):
            continue
        times.append(t_val)
        temps.append(_to_kelvin(T_val, temp_unit))

    if not times:
        raise ValueError(f"{path}: no rows with both a time and a temperature")

    t_arr = np.asarray(times, dtype=np.float64)
    if use_clock:
        # Clock times wrap at midnight; undo that, then count from the first row.
        offset = 0.0
        adjusted = t_arr.copy()
        for i in range(1, len(t_arr)):
            if t_arr[i] < t_arr[i - 1] and t_arr[i - 1] - t_arr[i] > _HALF_DAY_S:
                offset += _DAY_S
            adjusted[i] = t_arr[i] + offset
        t_arr = adjusted - adjusted[0]
    order = np.argsort(t_arr, kind="stable")
    return MeasuredSeries(
        t=t_arr[order],
        T=np.asarray(temps, dtype=np.float64)[order],
        label=path.stem,
    )


def compare(
    samples: list[Sample], probe: str, measured: MeasuredSeries, target_temp_K: float | None
) -> Comparison:
    """Compare a simulated probe against a measured series.

    The simulation is interpolated onto the measured times inside the overlapping time range.
    Bias is mean(sim - measured). With a target, time-to-target is found in both series by the
    same interpolation, and the relative error is (sim - meas) / meas.
    """
    if not samples:
        raise ValueError("no simulated samples to compare")
    t_sim = sample_times(samples)
    v_sim = probe_series(samples, probe)
    lo = max(float(t_sim[0]), float(measured.t[0]))
    hi = min(float(t_sim[-1]), float(measured.t[-1]))
    in_range = (measured.t >= lo) & (measured.t <= hi)
    if not np.any(in_range):
        raise ValueError(
            f"simulated range [{t_sim[0]:g}, {t_sim[-1]:g}] s and measured range "
            f"[{measured.t[0]:g}, {measured.t[-1]:g}] s do not overlap"
        )
    t_m = measured.t[in_range]
    T_m = measured.T[in_range]
    err = np.interp(t_m, t_sim, v_sim) - T_m

    tts_sim: float | None = None
    tts_meas: float | None = None
    rel: float | None = None
    if target_temp_K is not None:
        tts_sim = first_crossing_time(t_sim, v_sim, target_temp_K)
        tts_meas = first_crossing_time(measured.t, measured.T, target_temp_K)
        if tts_sim is not None and tts_meas is not None and tts_meas != 0.0:
            rel = (tts_sim - tts_meas) / tts_meas

    return Comparison(
        rmse_K=float(np.sqrt(np.mean(err**2))),
        max_abs_K=float(np.max(np.abs(err))),
        bias_K=float(np.mean(err)),
        time_to_target_sim_s=tts_sim,
        time_to_target_meas_s=tts_meas,
        time_to_target_error_rel=rel,
    )
