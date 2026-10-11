"""Threaded runner: events, state machine, pacing, thread safety, headless entry point."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from thermosim.config.schema import Scenario
from thermosim.core.errors import ModeError, SolverError
from thermosim.core.protocols import (
    ErrorEvent,
    FinishedEvent,
    ProgressEvent,
    Runner,
    RunnerEvent,
    RunnerState,
    SampleEvent,
    SnapshotEvent,
)
from thermosim.core.types import RunMode, StopReason
from thermosim.sim import SimulationRunner, run_headless
from thermosim.sim.pacing import Pacer

Example = Callable[..., Scenario]


def _drain_until(
    runner: SimulationRunner, pred: Callable[[list[RunnerEvent]], bool], timeout: float = 30.0
) -> list[RunnerEvent]:
    events: list[RunnerEvent] = []
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        events += runner.poll_events()
        if pred(events):
            return events
        time.sleep(0.002)
    raise AssertionError("timed out waiting for runner events")


def _finished(events: list[RunnerEvent]) -> bool:
    return any(isinstance(e, FinishedEvent | ErrorEvent) for e in events)


def test_runner_is_a_runner(example: Example) -> None:
    assert isinstance(SimulationRunner(example("small_pot_stove")), Runner)


def test_fast_run_to_target(example: Example) -> None:
    r = SimulationRunner(example("small_pot_stove"))
    assert r.state == RunnerState.IDLE and r.result is None
    r.start()
    events = _drain_until(r, _finished)
    r.join(5)
    kinds = {type(e) for e in events}
    assert {SampleEvent, SnapshotEvent, ProgressEvent, FinishedEvent} <= kinds
    assert r.state == RunnerState.FINISHED
    assert r.result is not None
    assert r.result.summary.stop_reason == StopReason.TARGET_REACHED
    assert r.result.summary.energy_balance_rel_error < 1e-6
    fin = next(e for e in events if isinstance(e, FinishedEvent))
    assert fin.result is r.result
    assert len(r.result.snapshots) >= 2
    assert "small" in r.result.scenario_toml.lower() or "name" in r.result.scenario_toml


def test_fast_mode_rejects_realtime_calls(example: Example) -> None:
    r = SimulationRunner(example("small_pot_stove"), RunMode.FAST)
    with pytest.raises(ModeError):
        r.set_power_command(0.5)
    with pytest.raises(ModeError):
        r.set_speed(2.0)


def test_invalid_state_calls_are_noops(example: Example) -> None:
    r = SimulationRunner(example("small_pot_stove"))
    r.pause()
    r.resume()
    r.stop()
    assert r.state == RunnerState.IDLE
    r.start()
    r.start()  # second start ignored
    _drain_until(r, _finished)
    r.join(5)
    r.pause()
    r.stop()
    assert r.state == RunnerState.FINISHED


def test_user_stop_in_fast_mode(example: Example) -> None:
    sc = example("kettle_propane_40L", cell_size_m=0.02)
    r = SimulationRunner(sc)
    r.start()
    _drain_until(r, lambda ev: any(isinstance(e, SampleEvent) and e.sample.t > 3 for e in ev))
    r.stop()
    events = _drain_until(r, _finished)
    assert r.state == RunnerState.STOPPED
    fin = next(e for e in events if isinstance(e, FinishedEvent))
    assert fin.result.summary.stop_reason == StopReason.USER_STOP


def test_pause_resume_fast(example: Example) -> None:
    sc = example("kettle_propane_40L", cell_size_m=0.02)
    r = SimulationRunner(sc)
    r.start()
    _drain_until(r, lambda ev: any(isinstance(e, SampleEvent) and e.sample.t > 2 for e in ev))
    r.pause()
    assert r.state == RunnerState.PAUSED
    time.sleep(0.2)
    r.poll_events()
    time.sleep(0.3)
    assert not [e for e in r.poll_events() if isinstance(e, SampleEvent)]
    r.resume()
    _drain_until(r, lambda ev: any(isinstance(e, SampleEvent) for e in ev))
    r.stop()
    _drain_until(r, _finished)


def test_pause_stops_simulated_time_realtime(example: Example) -> None:
    sc = example("small_pot_stove", mode=RunMode.REALTIME, realtime_speed=10.0)
    r = SimulationRunner(sc)
    r.start()
    _drain_until(r, lambda ev: any(isinstance(e, SampleEvent) and e.sample.t >= 2 for e in ev))
    r.pause()
    time.sleep(0.2)
    r.poll_events()
    time.sleep(0.5)
    assert not [e for e in r.poll_events() if isinstance(e, SampleEvent)]
    r.resume()
    _drain_until(r, lambda ev: any(isinstance(e, SampleEvent) for e in ev))
    r.stop()
    _drain_until(r, _finished)


def test_realtime_pacing_10x(example: Example) -> None:
    """Simulating 5 s at 10x takes 0.5 s (+-20 %)."""
    sc = example("small_pot_stove", mode=RunMode.REALTIME, realtime_speed=10.0)
    r = SimulationRunner(sc)
    assert r.mode == RunMode.REALTIME
    r.start()
    stamps: dict[float, float] = {}
    deadline = time.perf_counter() + 30
    while 5.0 not in stamps and time.perf_counter() < deadline:
        for e in r.poll_events():
            if isinstance(e, SampleEvent):
                stamps.setdefault(round(e.sample.t, 6), time.perf_counter())
        time.sleep(0.001)
    r.stop()
    _drain_until(r, _finished)
    wall = stamps[5.0] - stamps[0.0]
    assert 0.4 <= wall <= 0.6, wall


def test_realtime_set_speed_midrun_and_power(example: Example) -> None:
    sc = example("small_pot_stove", mode=RunMode.REALTIME, realtime_speed=5.0)
    r = SimulationRunner(sc)
    r.start()
    stamps: dict[float, float] = {}

    def pump() -> None:
        for e in r.poll_events():
            if isinstance(e, SampleEvent):
                stamps.setdefault(round(e.sample.t, 6), time.perf_counter())

    deadline = time.perf_counter() + 30
    while 4.0 not in stamps and time.perf_counter() < deadline:
        pump()
        time.sleep(0.001)
    r.set_speed(25.0)
    r.set_power_command(0.0)
    while 24.0 not in stamps and time.perf_counter() < deadline:
        pump()
        time.sleep(0.001)
    r.stop()
    events = _drain_until(r, _finished)
    rate_before = 4.0 / (stamps[4.0] - stamps[0.0])
    rate_after = 10.0 / (stamps[14.0] - stamps[4.0])
    assert rate_after > 2.5 * rate_before, (rate_before, rate_after)
    fin = next(e for e in events if isinstance(e, FinishedEvent))
    assert fin.result.samples[-1].command == 0.0


def test_no_deadlock_on_rapid_controls(example: Example) -> None:
    sc = example("small_pot_stove", mode=RunMode.REALTIME, realtime_speed=2.0)
    r = SimulationRunner(sc)
    t0 = time.perf_counter()
    r.start()
    r.pause()
    r.set_power_command(0.7)
    r.resume()
    r.set_speed(8.0)
    r.pause()
    r.resume()
    r.stop()
    events = _drain_until(r, _finished, timeout=10)
    r.join(10)
    assert time.perf_counter() - t0 < 10
    assert r.state == RunnerState.STOPPED
    assert any(isinstance(e, FinishedEvent) for e in events)


def test_stop_while_paused(example: Example) -> None:
    sc = example("small_pot_stove", mode=RunMode.REALTIME, realtime_speed=2.0)
    r = SimulationRunner(sc)
    r.start()
    _drain_until(r, lambda ev: any(isinstance(e, SampleEvent) for e in ev))
    r.pause()
    r.stop()
    _drain_until(r, _finished, timeout=10)
    assert r.state == RunnerState.STOPPED


def test_solver_error_gives_error_event(example: Example, monkeypatch: pytest.MonkeyPatch) -> None:
    from thermosim.solver.thermal import ThermalSolver

    def boom(*args: object, **kwargs: object) -> None:
        raise SolverError("synthetic failure")

    monkeypatch.setattr(ThermalSolver, "step", boom)
    r = SimulationRunner(example("small_pot_stove"))
    r.start()
    events = _drain_until(r, _finished)
    errs = [e for e in events if isinstance(e, ErrorEvent)]
    assert errs and "synthetic failure" in errs[0].message
    r.join(5)
    assert r.state == RunnerState.FAILED
    assert r.result is not None and r.result.summary.stop_reason == StopReason.ERROR


def test_run_headless_matches_runner_and_reports_progress(example: Example) -> None:
    sc = example("small_pot_stove")
    seen: list[ProgressEvent] = []
    result = run_headless(sc, seen.append)
    assert result.summary.stop_reason == StopReason.TARGET_REACHED
    assert seen and seen[-1].max_time_s == sc.simulation.max_sim_time_s
    r = SimulationRunner(sc)
    r.start()
    _drain_until(r, _finished)
    assert r.result is not None
    # The two runs split the simulation into chunks of different lengths (the chunk length follows
    # the wall-clock speed, so it changes with the machine's load), and a chunk end shortens the
    # step that crosses it; the explicit-Euler trajectories then differ by the integration error
    # (about 2e-6 relative). Do not demand more than that.
    assert r.result.summary.time_to_target_s == pytest.approx(
        result.summary.time_to_target_s, rel=1e-4
    )


def test_run_headless_raises_solver_error(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    from thermosim.solver.thermal import ThermalSolver

    def boom(*args: object, **kwargs: object) -> None:
        raise SolverError("synthetic failure")

    monkeypatch.setattr(ThermalSolver, "step", boom)
    with pytest.raises(SolverError):
        run_headless(example("small_pot_stove"))


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_pacer_integrates_speed_changes_and_pauses() -> None:
    clk = _Clock()
    p = Pacer(speed=2.0, clock=clk)
    clk.t += 1.0
    assert p.target() == pytest.approx(2.0)
    p.set_speed(10.0)
    clk.t += 1.0
    assert p.target() == pytest.approx(12.0)
    p.pause()
    clk.t += 5.0
    assert p.target() == pytest.approx(12.0)
    p.resume()
    clk.t += 0.5
    assert p.target() == pytest.approx(17.0)
    assert p.lag_wall_s(12.0) == pytest.approx(0.5)
    assert p.lag_wall_s(100.0) == 0.0
    p.set_speed(0.01)
    assert p.speed == pytest.approx(0.1)
