"""CSV export and run save/load (summary.json, timeseries.csv, samples.npz, snapshots.npz)."""

from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from thermosim.config.units import UnitSystem, to_display, unit_label
from thermosim.core.types import (
    F64,
    EnergyFlows,
    RunResult,
    Sample,
    StopReason,
    Summary,
)
from thermosim.recording.recorder import ENERGY_FIELDS, Recorder
from thermosim.recording.snapshots import SnapshotStore

_J_PER_BTU = 1055.05585262  # BTU (IT)
_TEMP_COLUMNS = ("T_mean", "T_min", "T_max", "T_shell_mean")
_SAMPLE_SCALARS = (
    "t",
    "T_mean",
    "T_min",
    "T_max",
    "T_shell_mean",
    "command",
    "power_delivered",
    "power_gross",
    "fuel_kg",
    "electric_kwh",
)


def _csv_column(name: str, values: F64, system: UnitSystem) -> tuple[str, F64]:
    """Header and converted values for one column of Recorder.to_arrays()."""
    if name == "t":
        label = unit_label("time", system)
        return f"t [{label}]", values
    if name == "command":
        return "command", values
    if name in _TEMP_COLUMNS or name.startswith("probe_"):
        label = unit_label("temperature", system)
        converted = np.asarray(to_display(values, "temperature", system), dtype=np.float64)
        return f"{name} [{label}]", converted
    if name in ("power_delivered", "power_gross"):
        label = unit_label("power", system)
        converted = np.asarray(to_display(values, "power", system), dtype=np.float64)
        return f"{name} [{label}]", converted
    if name == "fuel_kg":
        return "fuel_kg [kg]", values
    if name == "electric_kwh":
        return "electric_kwh [kWh]", values
    if name.startswith("energy_"):
        if system == "metric":
            return f"{name} [kJ]", values / 1000.0
        return f"{name} [BTU]", values / _J_PER_BTU
    raise KeyError(f"no CSV mapping for column {name!r}")


def export_csv(samples: list[Sample], path: Path, system: UnitSystem = "metric") -> None:
    """Write the time series to CSV, in the given unit system.

    UTF-8 with BOM (so Excel shows the degree sign), LF line endings.
    """
    recorder = Recorder()
    for sample in samples:
        recorder.add(sample)
    arrays = recorder.to_arrays()
    headers: list[str] = []
    columns: list[F64] = []
    for name, values in arrays.items():
        header, converted = _csv_column(name, values, system)
        headers.append(header)
        columns.append(converted)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(headers)
        for row in zip(*columns, strict=True):
            writer.writerow([str(float(v)) for v in row])


# ---------------------------------------------------------------- run save / load
def _summary_to_dict(summary: Summary) -> dict[str, Any]:
    return {
        "stop_reason": summary.stop_reason.value,
        "sim_time_s": summary.sim_time_s,
        "wall_time_s": summary.wall_time_s,
        "time_to_target_s": summary.time_to_target_s,
        "target_temp_K": summary.target_temp_K,
        "target_probe": summary.target_probe,
        "final_probes": {k: float(v) for k, v in summary.final_probes.items()},
        "final_T_mean": summary.final_T_mean,
        "energy": {f.name: float(getattr(summary.energy, f.name)) for f in fields(EnergyFlows)},
        "energy_balance_rel_error": summary.energy_balance_rel_error,
        "fuel_kg": summary.fuel_kg,
        "electric_kwh": summary.electric_kwh,
        "heating_efficiency": summary.heating_efficiency,
    }


def _summary_from_dict(d: dict[str, Any]) -> Summary:
    return Summary(
        stop_reason=StopReason(d["stop_reason"]),
        sim_time_s=float(d["sim_time_s"]),
        wall_time_s=float(d["wall_time_s"]),
        time_to_target_s=None if d["time_to_target_s"] is None else float(d["time_to_target_s"]),
        target_temp_K=float(d["target_temp_K"]),
        target_probe=str(d["target_probe"]),
        final_probes={str(k): float(v) for k, v in d["final_probes"].items()},
        final_T_mean=float(d["final_T_mean"]),
        energy=EnergyFlows(**{k: float(v) for k, v in d["energy"].items()}),
        energy_balance_rel_error=float(d["energy_balance_rel_error"]),
        fuel_kg=float(d["fuel_kg"]),
        electric_kwh=float(d["electric_kwh"]),
        heating_efficiency=float(d["heating_efficiency"]),
    )


def _save_samples(samples: list[Sample], path: Path) -> None:
    names = list(samples[0].probes.keys()) if samples else []
    arrays: dict[str, Any] = {
        name: np.array([getattr(s, name) for s in samples], dtype=np.float64)
        for name in _SAMPLE_SCALARS
    }
    arrays["probe_names"] = np.array(names, dtype=str)
    arrays["probes"] = np.array(
        [[s.probes[p] for p in names] for s in samples], dtype=np.float64
    ).reshape(len(samples), len(names))
    arrays["energy"] = np.array(
        [[float(getattr(s.cumulative, f)) for f in ENERGY_FIELDS] for s in samples],
        dtype=np.float64,
    ).reshape(len(samples), len(ENERGY_FIELDS))
    with open(path, "wb") as f:
        np.savez_compressed(f, **arrays)


def _load_samples(path: Path) -> list[Sample]:
    with np.load(path) as data:
        names = [str(n) for n in data["probe_names"].tolist()]
        scalars = {name: data[name].tolist() for name in _SAMPLE_SCALARS}
        probe_rows = data["probes"].tolist()
        energy_rows = data["energy"].tolist()
    samples: list[Sample] = []
    for i in range(len(scalars["t"])):
        samples.append(
            Sample(
                t=scalars["t"][i],
                probes=dict(zip(names, probe_rows[i], strict=True)),
                T_mean=scalars["T_mean"][i],
                T_min=scalars["T_min"][i],
                T_max=scalars["T_max"][i],
                T_shell_mean=scalars["T_shell_mean"][i],
                command=scalars["command"][i],
                power_delivered=scalars["power_delivered"][i],
                power_gross=scalars["power_gross"][i],
                cumulative=EnergyFlows(**dict(zip(ENERGY_FIELDS, energy_rows[i], strict=True))),
                fuel_kg=scalars["fuel_kg"][i],
                electric_kwh=scalars["electric_kwh"][i],
            )
        )
    return samples


def save_run(result: RunResult, directory: Path) -> None:
    """Write summary.json, timeseries.csv, snapshots.npz and scenario.toml into `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_summary_to_dict(result.summary), f, indent=2)
    export_csv(result.samples, directory / "timeseries.csv", "metric")
    _save_samples(result.samples, directory / "samples.npz")
    snapshots = result.snapshots
    if not isinstance(snapshots, SnapshotStore):
        store = SnapshotStore()
        for i in range(len(snapshots)):
            store.add(snapshots[i])
        snapshots = store
    snapshots.save(directory / "snapshots.npz")
    with open(directory / "scenario.toml", "w", encoding="utf-8", newline="") as f:
        f.write(result.scenario_toml)


def load_run(directory: Path) -> RunResult:
    """Read back a run written by save_run."""
    with open(directory / "summary.json", encoding="utf-8") as f:
        summary = _summary_from_dict(json.load(f))
    samples = _load_samples(directory / "samples.npz")
    snapshots = SnapshotStore.load(directory / "snapshots.npz")
    with open(directory / "scenario.toml", encoding="utf-8", newline="") as f:
        scenario_toml = f.read()
    return RunResult(
        summary=summary,
        samples=samples,
        snapshots=snapshots,
        scenario_toml=scenario_toml,
    )
