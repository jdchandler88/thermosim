"""In-memory snapshot store with optional spill-to-disk and .npz persistence."""

from __future__ import annotations

import bisect
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from thermosim.core.types import Snapshot, SnapshotStoreLike

_CACHE_SIZE = 64  # snapshots kept decoded after being read back from a spill chunk

_ARRAY_FIELDS = (
    "slice_T",
    "x_centers",
    "z_centers",
    "wall_left_z",
    "wall_left_T",
    "wall_right_z",
    "wall_right_T",
    "base_x",
    "base_T",
)
_OPTIONAL_ARRAY_FIELDS = ("velocity_u", "velocity_w")


def _encode(prefix: str, snap: Snapshot, out: dict[str, Any]) -> None:
    """Write one snapshot into `out` under keys starting with `prefix`."""
    out[prefix + "t"] = np.float64(snap.t)
    for name in _ARRAY_FIELDS:
        out[prefix + name] = np.asarray(getattr(snap, name))
    out[prefix + "lid_present"] = np.bool_(snap.lid_T is not None)
    out[prefix + "lid_T"] = np.float64(np.nan if snap.lid_T is None else snap.lid_T)
    out[prefix + "T_mean"] = np.float64(snap.T_mean)
    out[prefix + "power_delivered"] = np.float64(snap.power_delivered)
    out[prefix + "probe_names"] = np.array(list(snap.probes.keys()), dtype=str)
    out[prefix + "probe_values"] = np.array(list(snap.probes.values()), dtype=np.float64)
    for name in _OPTIONAL_ARRAY_FIELDS:
        value = getattr(snap, name)
        out[prefix + name + "_present"] = np.bool_(value is not None)
        if value is not None:
            out[prefix + name] = np.asarray(value)


def _decode(prefix: str, data: Mapping[str, Any]) -> Snapshot:
    names = [str(n) for n in data[prefix + "probe_names"].tolist()]
    values = [float(v) for v in data[prefix + "probe_values"].tolist()]
    lid_T = float(data[prefix + "lid_T"]) if bool(data[prefix + "lid_present"]) else None
    optional: dict[str, Any] = {}
    for name in _OPTIONAL_ARRAY_FIELDS:
        present = bool(data[prefix + name + "_present"])
        optional[name] = data[prefix + name] if present else None
    arrays = {name: data[prefix + name] for name in _ARRAY_FIELDS}
    return Snapshot(
        t=float(data[prefix + "t"]),
        slice_T=arrays["slice_T"],
        x_centers=arrays["x_centers"],
        z_centers=arrays["z_centers"],
        wall_left_z=arrays["wall_left_z"],
        wall_left_T=arrays["wall_left_T"],
        wall_right_z=arrays["wall_right_z"],
        wall_right_T=arrays["wall_right_T"],
        base_x=arrays["base_x"],
        base_T=arrays["base_T"],
        lid_T=lid_T,
        probes=dict(zip(names, values, strict=True)),
        T_mean=float(data[prefix + "T_mean"]),
        power_delivered=float(data[prefix + "power_delivered"]),
        velocity_u=optional["velocity_u"],
        velocity_w=optional["velocity_w"],
    )


class SnapshotStore(SnapshotStoreLike):
    """Time-ordered snapshots.

    With `max_in_memory` and `spill_dir` set, the oldest snapshots are written to .npz chunk
    files once more than `max_in_memory` are held; they are read back on access through a
    small LRU cache. Without `spill_dir` everything stays in memory.
    """

    def __init__(self, max_in_memory: int | None = None, spill_dir: Path | None = None) -> None:
        if max_in_memory is not None and max_in_memory < 1:
            raise ValueError("max_in_memory must be at least 1")
        self._max = max_in_memory
        self._spill_dir = spill_dir
        self._times: list[float] = []
        # spilled chunks, oldest first: (file, index of its first snapshot)
        self._chunks: list[tuple[Path, int]] = []
        self._chunk_starts: list[int] = []
        self._n_spilled = 0
        self._mem: list[Snapshot] = []  # snapshots with index >= _n_spilled
        self._cache: OrderedDict[int, Snapshot] = OrderedDict()
        self._token = uuid.uuid4().hex[:12]
        self._n_chunks = 0

    def __len__(self) -> int:
        return len(self._times)

    def __getitem__(self, i: int) -> Snapshot:
        n = len(self._times)
        idx = i + n if i < 0 else i
        if not 0 <= idx < n:
            raise IndexError(f"snapshot index {i} out of range for {n} snapshots")
        if idx >= self._n_spilled:
            return self._mem[idx - self._n_spilled]
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        c = bisect.bisect_right(self._chunk_starts, idx) - 1
        path, start = self._chunks[c]
        with np.load(path) as data:
            snap = _decode(f"{idx - start}__", data)
        self._cache[idx] = snap
        if len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return snap

    def times(self) -> list[float]:
        return list(self._times)

    def nearest(self, t: float) -> Snapshot:
        if not self._times:
            raise ValueError("no snapshots stored")
        idx = bisect.bisect_left(self._times, t)
        candidates = [j for j in (idx - 1, idx) if 0 <= j < len(self._times)]
        best = min(candidates, key=lambda j: abs(self._times[j] - t))
        return self[best]

    def add(self, snap: Snapshot) -> None:
        self._times.append(snap.t)
        self._mem.append(snap)
        if self._max is not None and self._spill_dir is not None:
            chunk = max(1, self._max // 2)
            while len(self._mem) > self._max:
                self._spill(min(chunk, len(self._mem)))

    def _spill(self, count: int) -> None:
        assert self._spill_dir is not None
        self._spill_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {}
        for k, snap in enumerate(self._mem[:count]):
            _encode(f"{k}__", snap, arrays)
        path = self._spill_dir / f"snapshots_{self._token}_{self._n_chunks:06d}.npz"
        self._n_chunks += 1
        with open(path, "wb") as f:
            np.savez(f, **arrays)
        self._chunks.append((path, self._n_spilled))
        self._chunk_starts.append(self._n_spilled)
        self._n_spilled += count
        self._mem = self._mem[count:]

    def save(self, path: Path) -> None:
        """Save all snapshots (including spilled ones) to a single compressed .npz file."""
        arrays: dict[str, Any] = {"count": np.int64(len(self._times))}
        for i in range(len(self._times)):
            _encode(f"{i}__", self[i], arrays)
        with open(path, "wb") as f:
            np.savez_compressed(f, **arrays)

    @classmethod
    def load(cls, path: Path) -> SnapshotStore:
        """Load from an .npz file written by save()."""
        store = cls()
        with np.load(path) as data:
            count = int(data["count"])
            for i in range(count):
                store.add(_decode(f"{i}__", data))
        return store
