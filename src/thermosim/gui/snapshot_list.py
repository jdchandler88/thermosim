"""Simple list-backed snapshot store (satisfies `SnapshotStoreLike`)."""

from __future__ import annotations

from bisect import bisect_left

from thermosim.core.types import Snapshot


class ListSnapshotStore:
    """Snapshots in time order, with nearest-time lookup."""

    def __init__(self) -> None:
        self._snaps: list[Snapshot] = []
        self._times: list[float] = []

    def add(self, snap: Snapshot) -> None:
        self._snaps.append(snap)
        self._times.append(snap.t)

    def __len__(self) -> int:
        return len(self._snaps)

    def __getitem__(self, i: int) -> Snapshot:
        return self._snaps[i]

    def times(self) -> list[float]:
        return list(self._times)

    def nearest(self, t: float) -> Snapshot:
        if not self._snaps:
            raise IndexError("no snapshots")
        i = bisect_left(self._times, t)
        if i == 0:
            return self._snaps[0]
        if i == len(self._times):
            return self._snaps[-1]
        before, after = self._times[i - 1], self._times[i]
        return self._snaps[i] if (after - t) < (t - before) else self._snaps[i - 1]
