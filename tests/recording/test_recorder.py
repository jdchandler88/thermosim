"""Recorder: columns, round-trip of samples, isolation and append speed."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest

from thermosim.core.types import EnergyFlows, Sample
from thermosim.recording import Recorder
from thermosim.recording.recorder import ENERGY_FIELDS, column_names

Builder = Callable[..., Sample]


def test_columns_and_order(make_sample: Builder) -> None:
    rec = Recorder()
    for i in range(3):
        rec.add(
            make_sample(
                float(i),
                293.15 + i,
                {"center": 300.0 + i, "edge": 299.0 + i},
                cumulative=EnergyFlows(source_liquid=10.0 * i, loss_shell_outer=float(i)),
            )
        )
    arrays = rec.to_arrays()
    expected = column_names(["center", "edge"])
    assert list(arrays) == expected
    assert expected[:10] == [
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
    ]
    assert expected[10:12] == ["probe_center", "probe_edge"]
    assert expected[12:] == [f"energy_{f}" for f in ENERGY_FIELDS]
    np.testing.assert_array_equal(arrays["t"], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(arrays["probe_edge"], [299.0, 300.0, 301.0])
    np.testing.assert_array_equal(arrays["energy_source_liquid"], [0.0, 10.0, 20.0])
    assert all(a.dtype == np.float64 for a in arrays.values())


def test_empty_recorder(make_sample: Builder) -> None:
    rec = Recorder()
    assert rec.samples == []
    arrays = rec.to_arrays()
    assert list(arrays) == column_names(())
    assert all(a.shape == (0,) for a in arrays.values())


def test_samples_round_trip(make_sample: Builder) -> None:
    rec = Recorder()
    originals = [
        make_sample(
            float(i) * 0.5,
            293.0 + 0.1 * i,
            {"a": 1.0 + i, "b": 2.0 - i},
            power_gross=1234.5 + i,
            cumulative=EnergyFlows(source_liquid=float(i), stored_liquid=0.25 * i),
            fuel_kg=0.001 * i,
            electric_kwh=0.5 * i,
        )
        for i in range(20)
    ]
    for s in originals:
        rec.add(s)
    assert rec.samples == originals


def test_probe_set_must_match(make_sample: Builder) -> None:
    rec = Recorder()
    rec.add(make_sample(0.0, probes={"a": 1.0}))
    with pytest.raises(ValueError, match="differ"):
        rec.add(make_sample(1.0, probes={"b": 1.0}))


def test_record_is_isolated_from_caller_mutation(make_sample: Builder) -> None:
    rec = Recorder()
    flows = EnergyFlows(source_liquid=5.0)
    probes = {"a": 300.0}
    rec.add(make_sample(0.0, probes=probes, cumulative=flows))
    flows.source_liquid = 999.0
    probes["a"] = -1.0
    recorded = rec.samples[0]
    assert recorded.cumulative.source_liquid == 5.0
    assert recorded.probes["a"] == 300.0
    assert rec.to_arrays()["probe_a"][0] == 300.0


def test_append_1e5_samples_under_one_second(make_sample: Builder) -> None:
    samples = [
        make_sample(float(i), 293.0 + 1e-4 * i, {"center": 293.0, "edge": 292.0})
        for i in range(100_000)
    ]
    rec = Recorder()
    start = time.perf_counter()
    for s in samples:
        rec.add(s)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"appending 1e5 samples took {elapsed:.2f} s"
    assert rec.to_arrays()["t"].shape == (100_000,)
