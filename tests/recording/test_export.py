"""CSV export (metric and US units, BOM, LF endings) and save_run / load_run round trips."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from thermosim.core.types import EnergyFlows, RunResult, Sample, Snapshot
from thermosim.recording import SnapshotStore, export_csv, load_run, save_run
from thermosim.recording.recorder import ENERGY_FIELDS

Builder = Callable[..., Sample]
RunFactory = Callable[[list[Sample], list[Snapshot]], RunResult]
SnapFactory = Callable[..., Snapshot]
SameSnapshot = Callable[[Snapshot, Snapshot], None]

_BASE_HEADERS_METRIC = [
    "t [s]",
    "T_mean [°C]",
    "T_min [°C]",
    "T_max [°C]",
    "T_shell_mean [°C]",
    "command",
    "power_delivered [kW]",
    "power_gross [kW]",
    "fuel_kg [kg]",
    "electric_kwh [kWh]",
    "probe_center [°C]",
]


def _series(make_sample: Builder) -> list[Sample]:
    return [
        make_sample(
            float(i) * 10.0,
            293.15 + i,
            {"center": 293.15 + i + 0.5},
            power_gross=2000.0,
            cumulative=EnergyFlows(source_liquid=1055.05585262 * i, loss_shell_outer=1000.0 * i),
        )
        for i in range(4)
    ]


def _read(path: Path, encoding: str = "utf-8-sig") -> list[list[str]]:
    with open(path, encoding=encoding, newline="") as f:
        return list(csv.reader(f))


def test_metric_csv_header_units_and_values(make_sample: Builder, tmp_path: Path) -> None:
    path = tmp_path / "metric.csv"
    export_csv(_series(make_sample), path, "metric")
    rows = _read(path)
    header = rows[0]
    assert header[: len(_BASE_HEADERS_METRIC)] == _BASE_HEADERS_METRIC
    assert header[len(_BASE_HEADERS_METRIC) :] == [f"energy_{f} [kJ]" for f in ENERGY_FIELDS]
    assert len(rows) == 1 + 4
    first = dict(zip(header, rows[1], strict=True))
    assert float(first["t [s]"]) == 0.0
    assert float(first["T_mean [°C]"]) == pytest.approx(20.0)
    assert float(first["power_gross [kW]"]) == pytest.approx(2.0)
    last = dict(zip(header, rows[-1], strict=True))
    assert float(last["t [s]"]) == 30.0
    assert float(last["energy_loss_shell_outer [kJ]"]) == pytest.approx(3.0)
    assert float(last["energy_source_liquid [kJ]"]) == pytest.approx(1055.05585262 * 3 / 1000)


def test_us_csv_header_and_values(make_sample: Builder, tmp_path: Path) -> None:
    path = tmp_path / "us.csv"
    export_csv(_series(make_sample), path, "us")
    rows = _read(path)
    header = rows[0]
    assert header[0] == "t [s]"
    assert "T_mean [°F]" in header
    assert "power_gross [BTU/hr]" in header
    assert "energy_source_liquid [BTU]" in header
    first = dict(zip(header, rows[1], strict=True))
    assert float(first["T_mean [°F]"]) == pytest.approx(68.0)  # 293.15 K
    assert float(first["power_gross [BTU/hr]"]) == pytest.approx(2000.0 * 3.412141633)
    last = dict(zip(header, rows[-1], strict=True))
    assert float(last["energy_source_liquid [BTU]"]) == pytest.approx(3.0)


def test_csv_bytes_have_bom_and_lf_only(make_sample: Builder, tmp_path: Path) -> None:
    path = tmp_path / "bytes.csv"
    export_csv(_series(make_sample), path)
    data = path.read_bytes()
    assert data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data
    assert data.endswith(b"\n")
    assert "°".encode() in data


def test_csv_with_no_samples_has_header_only(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    export_csv([], path)
    rows = _read(path)
    assert len(rows) == 1
    assert rows[0][0] == "t [s]"


def test_save_and_load_run_round_trip(
    make_sample: Builder,
    snapshot_factory: SnapFactory,
    run_factory: RunFactory,
    same_snapshot: SameSnapshot,
    tmp_path: Path,
) -> None:
    samples = [
        make_sample(
            float(i) * 1.5,
            293.0 + 0.3 * i,
            {"center": 293.0 + 0.4 * i, "edge": 292.1 + i / 7.0},
            power_gross=1234.5 + i / 3.0,
            cumulative=EnergyFlows(
                source_liquid=0.1 * i,
                loss_surface_evap=1.0 / 3.0 * i,
                stored_liquid=2.0 * i,
            ),
            fuel_kg=0.001 * i,
            electric_kwh=0.5 * i / 7.0,
        )
        for i in range(12)
    ]
    snaps = [
        snapshot_factory(float(i) * 5.0, seed=i, lid=i % 2 == 0, velocity=i == 2) for i in range(5)
    ]
    result = run_factory(samples, snaps)

    directory = tmp_path / "run"
    save_run(result, directory)
    for name in ("summary.json", "timeseries.csv", "samples.npz", "snapshots.npz", "scenario.toml"):
        assert (directory / name).is_file(), name

    summary_json = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["stop_reason"] == "target_reached"
    timeseries_header = _read(directory / "timeseries.csv")[0]
    assert timeseries_header[0] == "t [s]"
    assert "T_mean [°C]" in timeseries_header

    loaded = load_run(directory)
    assert loaded.summary == result.summary
    assert loaded.samples == samples
    assert loaded.scenario_toml == result.scenario_toml
    assert isinstance(loaded.snapshots, SnapshotStore)
    assert len(loaded.snapshots) == len(snaps)
    assert loaded.snapshots.times() == result.snapshots.times()
    for i, original in enumerate(snaps):
        same_snapshot(loaded.snapshots[i], original)


def test_save_run_accepts_any_snapshot_store_like(
    make_sample: Builder, snapshot_factory: SnapFactory, run_factory: RunFactory, tmp_path: Path
) -> None:
    class ListStore:
        def __init__(self, items: list[Snapshot]) -> None:
            self._items = items

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, i: int) -> Snapshot:
            return self._items[i]

        def times(self) -> list[float]:
            return [s.t for s in self._items]

        def nearest(self, t: float) -> Snapshot:
            return min(self._items, key=lambda s: abs(s.t - t))

    snaps = [snapshot_factory(float(i), seed=i) for i in range(3)]
    samples = [make_sample(0.0, 300.0), make_sample(1.0, 301.0)]
    base = run_factory(samples, [])
    result = RunResult(
        summary=base.summary, samples=samples, snapshots=ListStore(snaps), scenario_toml="x = 1\n"
    )
    save_run(result, tmp_path / "run")
    loaded = load_run(tmp_path / "run")
    assert len(loaded.snapshots) == 3
    assert loaded.scenario_toml == "x = 1\n"
