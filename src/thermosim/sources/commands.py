"""Command sources: the requested power fraction over time (WP05).

`ScheduleCommand` replays `Scenario.power_schedule`. `ManualCommand` is set by the user
(REALTIME mode) and is safe to set from one thread while the model reads it on another.
"""

from __future__ import annotations

import bisect
import threading
from typing import TYPE_CHECKING

from thermosim.core.protocols import CommandSource

if TYPE_CHECKING:
    from thermosim.config.schema import Scenario


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


class ScheduleCommand(CommandSource):
    """Piecewise-constant command: the value at t is the last step with t_s <= t."""

    def __init__(self, scenario: Scenario) -> None:
        steps = sorted(scenario.power_schedule, key=lambda step: step.t_s)
        if not steps:
            raise ValueError("power schedule is empty")
        self._times: list[float] = [float(step.t_s) for step in steps]
        self._values: list[float] = [_clip01(step.command) for step in steps]

    def command(self, t: float, state_probe_K: float) -> float:
        index = bisect.bisect_right(self._times, t) - 1
        return self._values[max(index, 0)]


class ManualCommand(CommandSource):
    """Thread-safe command set by the user; used in REALTIME mode."""

    def __init__(self, initial: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._value = _clip01(initial)

    def set(self, value: float) -> None:
        """Set the requested power fraction, 0..1 (values outside are clipped)."""
        clipped = _clip01(value)
        with self._lock:
            self._value = clipped

    def command(self, t: float, state_probe_K: float) -> float:
        with self._lock:
            return self._value
