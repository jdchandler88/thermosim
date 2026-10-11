"""Builders for synthetic samples, snapshots and run results used across the recording tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from thermosim.core.types import (
    EnergyFlows,
    RunResult,
    Sample,
    Snapshot,
    StopReason,
    Summary,
)

Builder = Callable[..., Sample]


@pytest.fixture
def make_sample() -> Builder:
    """Sample at time t with T_mean (K) and optional probes {name: K}."""

    def build(
        t: float,
        T_mean: float = 293.15,
        probes: dict[str, float] | None = None,
        *,
        power_gross: float = 2500.0,
        power_delivered: float = 1000.0,
        cumulative: EnergyFlows | None = None,
        fuel_kg: float = 0.0,
        electric_kwh: float = 0.0,
    ) -> Sample:
        return Sample(
            t=t,
            probes=dict(probes) if probes is not None else {"center": T_mean},
            T_mean=T_mean,
            T_min=T_mean - 1.0,
            T_max=T_mean + 1.0,
            T_shell_mean=T_mean + 2.0,
            command=1.0,
            power_delivered=power_delivered,
            power_gross=power_gross,
            cumulative=cumulative if cumulative is not None else EnergyFlows(),
            fuel_kg=fuel_kg,
            electric_kwh=electric_kwh,
        )

    return build


def make_snapshot(t: float, seed: int = 0, *, lid: bool = True, velocity: bool = False) -> Snapshot:
    """A small deterministic snapshot; slice_T has one NaN (a non-liquid cell)."""
    rng = np.random.default_rng(seed)
    slice_T = (293.0 + rng.random((4, 3))).astype(np.float32)
    slice_T[0, 0] = np.nan
    return Snapshot(
        t=t,
        slice_T=slice_T,
        x_centers=np.linspace(-0.01, 0.01, 4),
        z_centers=np.linspace(0.0, 0.05, 3),
        wall_left_z=np.linspace(0.0, 0.06, 5),
        wall_left_T=(293.0 + rng.random(5)).astype(np.float32),
        wall_right_z=np.linspace(0.0, 0.06, 5),
        wall_right_T=(293.0 + rng.random(5)).astype(np.float32),
        base_x=np.linspace(-0.01, 0.01, 4),
        base_T=(293.0 + rng.random(4)).astype(np.float32),
        lid_T=(301.5 + seed) if lid else None,
        probes={"center": 300.0 + seed, "edge": 299.5 + seed},
        T_mean=295.0 + seed,
        power_delivered=900.0 + seed,
        velocity_u=(rng.random((4, 3)) * 0.01).astype(np.float32) if velocity else None,
        velocity_w=(rng.random((4, 3)) * 0.01).astype(np.float32) if velocity else None,
    )


def make_run(samples: list[Sample], snapshots: list[Snapshot]) -> RunResult:
    from thermosim.recording import SnapshotStore

    store = SnapshotStore()
    for snap in snapshots:
        store.add(snap)
    cumulative = samples[-1].cumulative if samples else EnergyFlows()
    summary = Summary(
        stop_reason=StopReason.TARGET_REACHED,
        sim_time_s=samples[-1].t if samples else 0.0,
        wall_time_s=1.25,
        time_to_target_s=None,
        target_temp_K=350.0,
        target_probe="center",
        final_probes=dict(samples[-1].probes) if samples else {},
        final_T_mean=samples[-1].T_mean if samples else float("nan"),
        energy=cumulative,
        energy_balance_rel_error=1e-7,
        fuel_kg=0.0,
        electric_kwh=0.25,
        heating_efficiency=0.4,
    )
    scenario = 'schema_version = 1\nname = "synthetic"\n# comment kept verbatim\n'
    return RunResult(summary=summary, samples=samples, snapshots=store, scenario_toml=scenario)


@pytest.fixture
def snapshot_factory() -> Callable[..., Snapshot]:
    return make_snapshot


@pytest.fixture
def same_snapshot() -> Callable[[Snapshot, Snapshot], None]:
    """Assert two snapshots are identical, including dtypes and None-ness."""

    def check(a: Snapshot, b: Snapshot) -> None:
        assert a.t == b.t
        assert a.lid_T == b.lid_T
        assert a.T_mean == b.T_mean
        assert a.power_delivered == b.power_delivered
        assert a.probes == b.probes
        for name in (
            "slice_T",
            "x_centers",
            "z_centers",
            "wall_left_z",
            "wall_left_T",
            "wall_right_z",
            "wall_right_T",
            "base_x",
            "base_T",
            "velocity_u",
            "velocity_w",
        ):
            x = getattr(a, name)
            y = getattr(b, name)
            if x is None or y is None:
                assert x is None and y is None, name
                continue
            assert x.dtype == y.dtype, name
            np.testing.assert_array_equal(x, y, err_msg=name)

    return check


@pytest.fixture
def run_factory() -> Callable[[list[Sample], list[Snapshot]], RunResult]:
    return make_run
