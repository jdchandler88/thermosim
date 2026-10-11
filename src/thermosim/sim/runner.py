"""Threaded runner implementing the Runner protocol, and the headless entry point
(CONTRACTS section 12). WP00 stub, implemented by WP09.
"""

from __future__ import annotations

from collections.abc import Callable

from thermosim.config.schema import Scenario
from thermosim.core.protocols import ProgressEvent, Runner, RunnerEvent, RunnerState
from thermosim.core.types import RunMode, RunResult


class SimulationRunner(Runner):
    mode: RunMode
    state: RunnerState

    def __init__(self, scenario: Scenario, mode: RunMode | None = None) -> None:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def pause(self) -> None:
        raise NotImplementedError

    def resume(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def set_power_command(self, value: float) -> None:
        raise NotImplementedError

    def set_speed(self, factor: float) -> None:
        raise NotImplementedError

    def poll_events(self, max_events: int = 1000) -> list[RunnerEvent]:
        raise NotImplementedError

    @property
    def result(self) -> RunResult | None:
        raise NotImplementedError


def run_headless(
    scenario: Scenario, progress: Callable[[ProgressEvent], None] | None = None
) -> RunResult:
    """Run a scenario to completion without a GUI and return the result."""
    raise NotImplementedError
