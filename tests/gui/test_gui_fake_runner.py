"""FakeRunner behaviour against the Runner protocol."""

from __future__ import annotations

import math
import time

import pytest
from gui_helpers import short_scenario

from thermosim.core.errors import ModeError
from thermosim.core.protocols import (
    FinishedEvent,
    ProgressEvent,
    Runner,
    RunnerState,
    SampleEvent,
    SnapshotEvent,
)
from thermosim.core.types import RunMode, StopReason
from thermosim.gui.fake_runner import FakeRunner, slice_dimensions
from thermosim.gui.runner_factory import make_fake_runner
from thermosim.gui.snapshot_list import ListSnapshotStore


def _drain(runner: FakeRunner, timeout: float = 10.0) -> list:  # type: ignore[type-arg]
    events: list = []  # type: ignore[type-arg]
    deadline = time.time() + timeout
    while time.time() < deadline:
        events.extend(runner.poll_events())
        if runner.result is not None and runner.state != RunnerState.RUNNING:
            events.extend(runner.poll_events())
            return events
        time.sleep(0.005)
    raise AssertionError("runner did not finish")


def test_factory_returns_fake_runner_implementing_protocol() -> None:
    runner = make_fake_runner(short_scenario(), RunMode.FAST)
    assert isinstance(runner, FakeRunner)
    assert isinstance(runner, Runner)
    assert runner.state == RunnerState.IDLE


def test_fast_run_emits_events_and_result() -> None:
    sc = short_scenario(60.0)
    runner = FakeRunner(sc, RunMode.FAST, fast_rate=6000.0, tick_s=0.005)
    runner.start()
    events = _drain(runner)
    kinds = {type(e) for e in events}
    assert {SampleEvent, SnapshotEvent, ProgressEvent, FinishedEvent} <= kinds
    result = runner.result
    assert result is not None
    assert result.summary.stop_reason in (StopReason.MAX_TIME, StopReason.TARGET_REACHED)
    assert result.summary.sim_time_s == pytest.approx(60.0, abs=1.5)
    assert len(result.snapshots) >= 10
    assert result.samples[0].t == 0.0
    assert result.scenario_toml.startswith("#")
    assert runner.state == RunnerState.FINISHED
    # monotone heating
    means = [s.T_mean for s in result.samples]
    assert means == sorted(means)
    # exact energy balance
    assert result.summary.energy_balance_rel_error < 1e-9


def test_slice_dimensions_follow_scenario() -> None:
    sc = short_scenario()
    _radius, fill, dx, nx, nz = slice_dimensions(sc)
    assert nx == math.ceil(sc.vessel.inner_diameter_m / dx)
    assert nz == math.ceil(fill / dx)
    runner = FakeRunner(sc, RunMode.FAST, fast_rate=6000.0, tick_s=0.005)
    runner.start()
    _drain(runner)
    assert runner.result is not None
    snap = runner.result.snapshots[0]
    assert snap.slice_T.shape == (nx, nz)
    assert len(snap.x_centers) == nx
    assert len(snap.z_centers) == nz
    assert snap.lid_T is not None  # example has a lid


def test_stop_mid_run_emits_user_stop() -> None:
    sc = short_scenario(3600.0)
    runner = FakeRunner(sc, RunMode.FAST, fast_rate=50.0, tick_s=0.005)
    runner.start()
    time.sleep(0.15)
    runner.stop()
    _drain(runner)
    assert runner.result is not None
    assert runner.result.summary.stop_reason == StopReason.USER_STOP
    assert runner.state == RunnerState.STOPPED
    assert runner.result.summary.sim_time_s < 3600.0


def test_pause_and_resume_hold_sim_time() -> None:
    sc = short_scenario(3600.0)
    runner = FakeRunner(sc, RunMode.FAST, fast_rate=200.0, tick_s=0.005)
    runner.start()
    time.sleep(0.1)
    runner.pause()
    assert runner.state == RunnerState.PAUSED
    time.sleep(0.05)
    runner.poll_events()
    time.sleep(0.15)
    assert all(not isinstance(e, ProgressEvent) for e in runner.poll_events())
    runner.resume()
    assert runner.state == RunnerState.RUNNING
    time.sleep(0.1)
    assert any(isinstance(e, ProgressEvent) for e in runner.poll_events())
    runner.stop()
    _drain(runner)


def test_mode_errors_in_fast_mode() -> None:
    runner = FakeRunner(short_scenario(), RunMode.FAST)
    with pytest.raises(ModeError):
        runner.set_power_command(0.5)
    with pytest.raises(ModeError):
        runner.set_speed(2.0)


def test_realtime_power_command_changes_heating() -> None:
    sc = short_scenario(3600.0)
    runner = FakeRunner(sc, RunMode.REALTIME, speed=100.0, tick_s=0.005)
    runner.set_power_command(0.0)
    runner.start()
    time.sleep(0.3)
    runner.set_speed(0.1)
    runner.stop()
    _drain(runner)
    assert runner.result is not None
    # no power: no heating at all
    assert runner.result.summary.final_T_mean == pytest.approx(sc.liquid.initial_temp_K, abs=1e-6)


def test_list_snapshot_store_nearest() -> None:
    sc = short_scenario(30.0)
    runner = FakeRunner(sc, RunMode.FAST, fast_rate=6000.0, tick_s=0.005)
    runner.start()
    _drain(runner)
    assert runner.result is not None
    store = runner.result.snapshots
    assert isinstance(store, ListSnapshotStore)
    assert store.nearest(-5).t == store.times()[0]
    assert store.nearest(1e9).t == store.times()[-1]
    assert store.nearest(7.4).t == 5.0
    assert store.nearest(7.6).t == 10.0
