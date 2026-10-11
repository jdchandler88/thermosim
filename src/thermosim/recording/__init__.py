"""Recording, snapshots, summary, CSV export and measured-data comparison (CONTRACTS section 11).

WP00 stub: every function raises NotImplementedError until WP07 implements it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from thermosim.config.units import UnitSystem
from thermosim.core.types import (
    F64,
    RunResult,
    Sample,
    Snapshot,
    SnapshotStoreLike,
    StopReason,
    Summary,
)


class Recorder:
    def add(self, sample: Sample) -> None:
        """Append one time-series sample."""
        raise NotImplementedError

    @property
    def samples(self) -> list[Sample]:
        """All samples recorded so far, in order."""
        raise NotImplementedError

    def to_arrays(self) -> dict[str, F64]:
        """Columns: t, T_mean, ..., probe_<name>, energy_<field>."""
        raise NotImplementedError


class SnapshotStore(SnapshotStoreLike):
    def __init__(self, max_in_memory: int | None = None, spill_dir: Path | None = None) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int) -> Snapshot:
        raise NotImplementedError

    def times(self) -> list[float]:
        raise NotImplementedError

    def nearest(self, t: float) -> Snapshot:
        raise NotImplementedError

    def add(self, snap: Snapshot) -> None:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Save to an .npz file."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> SnapshotStore:
        """Load from an .npz file written by save()."""
        raise NotImplementedError


def summarize(
    samples: list[Sample],
    target_temp_K: float,
    target_probe: str,
    stop_reason: StopReason,
    wall_time_s: float,
) -> Summary:
    """Build the run summary from the time series."""
    raise NotImplementedError


def export_csv(samples: list[Sample], path: Path, system: UnitSystem = "metric") -> None:
    """Write the time series to CSV, in the given unit system."""
    raise NotImplementedError


def save_run(result: RunResult, directory: Path) -> None:
    """Write summary.json, timeseries.csv, snapshots.npz and scenario.toml into `directory`."""
    raise NotImplementedError


def load_run(directory: Path) -> RunResult:
    """Read back a run written by save_run."""
    raise NotImplementedError


@dataclass(frozen=True)
class MeasuredSeries:
    t: F64  # s
    T: F64  # K
    label: str


def load_measured_csv(
    path: Path,
    time_col: str | None = None,
    temp_col: str | None = None,
    temp_unit: Literal["C", "F", "K"] = "C",
    time_unit: Literal["s", "min"] = "s",
) -> MeasuredSeries:
    """Read a measured temperature series from CSV and convert it to SI (s, K)."""
    raise NotImplementedError


@dataclass(frozen=True)
class Comparison:
    rmse_K: float
    max_abs_K: float
    bias_K: float
    time_to_target_sim_s: float | None
    time_to_target_meas_s: float | None
    time_to_target_error_rel: float | None


def compare(
    samples: list[Sample], probe: str, measured: MeasuredSeries, target_temp_K: float | None
) -> Comparison:
    """Compare a simulated probe against a measured series."""
    raise NotImplementedError
