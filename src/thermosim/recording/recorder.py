"""Append-only time-series recorder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields

import numpy as np

from thermosim.core.types import F64, EnergyFlows, Sample

_BASE_COLUMNS = (
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
ENERGY_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(EnergyFlows))
_N_BASE = len(_BASE_COLUMNS)
_INITIAL_CAPACITY = 256


def column_names(probe_names: Sequence[str]) -> list[str]:
    """Column order used by Recorder.to_arrays and the CSV export."""
    return [
        *_BASE_COLUMNS,
        *(f"probe_{p}" for p in probe_names),
        *(f"energy_{f}" for f in ENERGY_FIELDS),
    ]


class Recorder:
    """Stores the numeric columns of each sample in a geometrically grown NumPy buffer.

    `add` is amortized O(1). `Sample` objects are rebuilt from the buffer when `samples` is read,
    so later in-place changes to the caller's probe dict or EnergyFlows do not reach the record.
    """

    def __init__(self) -> None:
        self._probe_names: tuple[str, ...] | None = None
        self._buf: F64 = np.empty((_INITIAL_CAPACITY, _N_BASE + len(ENERGY_FIELDS)))
        self._n = 0

    def add(self, sample: Sample) -> None:
        """Append one time-series sample. Probe names must match the first sample's."""
        names = tuple(sample.probes.keys())
        if self._probe_names is None:
            self._probe_names = names
            self._buf = np.empty((_INITIAL_CAPACITY, len(column_names(names))), dtype=np.float64)
        elif len(names) != len(self._probe_names) or set(names) != set(self._probe_names):
            raise ValueError(
                f"sample probes {sorted(names)} differ from recorded probes "
                f"{sorted(self._probe_names)}"
            )
        if self._n == self._buf.shape[0]:
            grown = np.empty((2 * self._buf.shape[0], self._buf.shape[1]), dtype=np.float64)
            grown[: self._n] = self._buf[: self._n]
            self._buf = grown
        probes = sample.probes
        cum = sample.cumulative
        row: list[float] = [
            sample.t,
            sample.T_mean,
            sample.T_min,
            sample.T_max,
            sample.T_shell_mean,
            sample.command,
            sample.power_delivered,
            sample.power_gross,
            sample.fuel_kg,
            sample.electric_kwh,
        ]
        assert self._probe_names is not None
        row.extend(probes[p] for p in self._probe_names)
        row.extend(getattr(cum, f) for f in ENERGY_FIELDS)
        self._buf[self._n] = row
        self._n += 1

    def _sample_at(self, i: int) -> Sample:
        vals = self._buf[i].tolist()
        names = self._probe_names or ()
        p0 = _N_BASE
        e0 = p0 + len(names)
        return Sample(
            t=vals[0],
            probes={name: vals[p0 + k] for k, name in enumerate(names)},
            T_mean=vals[1],
            T_min=vals[2],
            T_max=vals[3],
            T_shell_mean=vals[4],
            command=vals[5],
            power_delivered=vals[6],
            power_gross=vals[7],
            cumulative=EnergyFlows(**dict(zip(ENERGY_FIELDS, vals[e0:], strict=True))),
            fuel_kg=vals[8],
            electric_kwh=vals[9],
        )

    @property
    def samples(self) -> list[Sample]:
        """All samples recorded so far, in order."""
        return [self._sample_at(i) for i in range(self._n)]

    def to_arrays(self) -> dict[str, F64]:
        """Columns: t, T_mean, ..., probe_<name>, energy_<field>."""
        names = column_names(self._probe_names or ())
        data = self._buf[: self._n]
        return {name: data[:, i].copy() for i, name in enumerate(names)}
