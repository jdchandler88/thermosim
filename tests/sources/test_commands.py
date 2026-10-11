"""Command sources: piecewise-constant schedule and thread-safe manual command (WP05)."""

from __future__ import annotations

import random
import threading
from types import SimpleNamespace

import pytest

from thermosim.config.io import example_scenarios
from thermosim.config.schema import PowerStep
from thermosim.sources.commands import ManualCommand, ScheduleCommand


def _schedule(*steps: tuple[float, float]) -> ScheduleCommand:
    scenario = SimpleNamespace(power_schedule=[PowerStep(t_s=t, command=c) for t, c in steps])
    return ScheduleCommand(scenario)  # type: ignore[arg-type]


def test_schedule_is_piecewise_constant() -> None:
    cmd = _schedule((0.0, 1.0), (10.0, 0.4), (25.0, 0.0))
    assert cmd.command(0.0, 300.0) == 1.0
    assert cmd.command(9.999, 300.0) == 1.0
    assert cmd.command(10.0, 300.0) == 0.4  # a step takes effect at its own time
    assert cmd.command(24.9, 300.0) == 0.4
    assert cmd.command(25.0, 300.0) == 0.0
    assert cmd.command(1.0e6, 300.0) == 0.0  # the last value holds forever


def test_schedule_ignores_probe_temperature() -> None:
    cmd = _schedule((0.0, 0.5))
    assert cmd.command(3.0, 280.0) == cmd.command(3.0, 400.0) == 0.5


def test_schedule_from_real_scenario_settings() -> None:
    scenario = example_scenarios()["kettle_propane_40L"]
    scenario = scenario.model_copy(
        update={
            "power_schedule": [
                PowerStep(t_s=0.0, command=1.0),
                PowerStep(t_s=600.0, command=0.25),
            ]
        }
    )
    cmd = ScheduleCommand(scenario)
    assert cmd.command(599.0, 0.0) == 1.0
    assert cmd.command(600.0, 0.0) == 0.25


def test_manual_command_default_and_set() -> None:
    cmd = ManualCommand()
    assert cmd.command(0.0, 300.0) == 0.0
    cmd.set(0.6)
    assert cmd.command(1.0, 300.0) == pytest.approx(0.6)


def test_manual_command_initial_value_is_clipped() -> None:
    assert ManualCommand(2.0).command(0.0, 0.0) == 1.0
    assert ManualCommand(-1.0).command(0.0, 0.0) == 0.0


def test_manual_command_set_clips_to_unit_interval() -> None:
    cmd = ManualCommand(0.5)
    cmd.set(3.0)
    assert cmd.command(0.0, 0.0) == 1.0
    cmd.set(-0.2)
    assert cmd.command(0.0, 0.0) == 0.0


def test_manual_command_two_threads_stress() -> None:
    cmd = ManualCommand()
    errors: list[BaseException] = []
    seen: list[float] = []
    stop = threading.Event()

    def writer(seed: int) -> None:
        rng = random.Random(seed)
        try:
            for _ in range(20000):
                cmd.set(rng.uniform(-0.5, 1.5))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def reader() -> None:
        try:
            while not stop.is_set():
                value = cmd.command(0.0, 300.0)
                seen.append(value)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    writers = [threading.Thread(target=writer, args=(s,)) for s in (1, 2)]
    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join()
    stop.set()
    reader_thread.join()

    assert not errors
    assert seen
    assert all(0.0 <= v <= 1.0 for v in seen)
