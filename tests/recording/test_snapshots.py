"""SnapshotStore: indexing, nearest lookup, spill to disk, and .npz save/load."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from thermosim.core.types import Snapshot
from thermosim.recording import SnapshotStore

SnapFactory = Callable[..., Snapshot]
SameSnapshot = Callable[[Snapshot, Snapshot], None]


def _snapshots(factory: SnapFactory, n: int) -> list[Snapshot]:
    return [
        factory(float(i) * 5.0, seed=i, lid=(i % 2 == 0), velocity=(i % 3 == 0)) for i in range(n)
    ]


def test_index_len_and_times(snapshot_factory: SnapFactory, same_snapshot: SameSnapshot) -> None:
    snaps = _snapshots(snapshot_factory, 4)
    store = SnapshotStore()
    for s in snaps:
        store.add(s)
    assert len(store) == 4
    assert store.times() == [0.0, 5.0, 10.0, 15.0]
    same_snapshot(store[2], snaps[2])
    same_snapshot(store[-1], snaps[-1])
    with pytest.raises(IndexError):
        _ = store[4]


def test_nearest_uses_closest_time(snapshot_factory: SnapFactory) -> None:
    store = SnapshotStore()
    for s in _snapshots(snapshot_factory, 3):
        store.add(s)  # times 0, 5, 10
    assert store.nearest(4.9).t == 5.0
    assert store.nearest(2.4).t == 0.0
    assert store.nearest(-100.0).t == 0.0
    assert store.nearest(1000.0).t == 10.0
    assert store.nearest(7.5).t == 5.0  # tie goes to the earlier snapshot


def test_nearest_on_empty_store_raises() -> None:
    with pytest.raises(ValueError):
        SnapshotStore().nearest(0.0)


def test_rejects_bad_max_in_memory() -> None:
    with pytest.raises(ValueError):
        SnapshotStore(max_in_memory=0)


def test_spills_oldest_to_chunks_and_reads_back(
    snapshot_factory: SnapFactory, same_snapshot: SameSnapshot, tmp_path: Path
) -> None:
    spill = tmp_path / "spill"
    snaps = _snapshots(snapshot_factory, 25)
    store = SnapshotStore(max_in_memory=6, spill_dir=spill)
    for s in snaps:
        store.add(s)
    assert list(spill.glob("*.npz")), "old snapshots should have been written to disk"
    assert len(store._mem) <= 6  # only the newest stay in memory
    assert len(store) == 25
    assert store.times() == [s.t for s in snaps]
    for i in (0, 1, 7, 13, 24, -25):
        same_snapshot(store[i], snaps[i])
    # Reading again goes through the LRU cache and still matches.
    same_snapshot(store[0], snaps[0])
    same_snapshot(store.nearest(31.0), snaps[6])


def test_no_spill_without_spill_dir(snapshot_factory: SnapFactory, tmp_path: Path) -> None:
    store = SnapshotStore(max_in_memory=2)
    for s in _snapshots(snapshot_factory, 10):
        store.add(s)
    assert len(store) == 10
    assert len(store._mem) == 10
    assert not list(tmp_path.iterdir())


def test_save_load_round_trip(
    snapshot_factory: SnapFactory, same_snapshot: SameSnapshot, tmp_path: Path
) -> None:
    snaps = _snapshots(snapshot_factory, 7)
    store = SnapshotStore(max_in_memory=3, spill_dir=tmp_path / "spill")
    for s in snaps:
        store.add(s)
    target = tmp_path / "snapshots.npz"
    store.save(target)
    loaded = SnapshotStore.load(target)
    assert len(loaded) == len(snaps)
    assert loaded.times() == store.times()
    for i, original in enumerate(snaps):
        same_snapshot(loaded[i], original)


def test_save_load_empty_store(tmp_path: Path) -> None:
    target = tmp_path / "empty.npz"
    SnapshotStore().save(target)
    loaded = SnapshotStore.load(target)
    assert len(loaded) == 0
