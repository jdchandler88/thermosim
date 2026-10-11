"""Real-time pacing: maps wall-clock time to the target simulated time (WP09).

The target simulated time is ``t_sim0 + integral(speed d(wall))``, integrated piecewise so that
speed changes and pauses are exact. The clock is injectable for tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable

MIN_SPEED = 0.1


class Pacer:
    """Piecewise-linear wall-to-simulation time map. Not thread-safe; the owner locks."""

    def __init__(
        self,
        speed: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
        t_sim0: float = 0.0,
    ) -> None:
        self._clock = clock
        self._speed = max(float(speed), MIN_SPEED)
        self._t_sim = float(t_sim0)  # simulated time at self._t_wall
        self._t_wall = clock()
        self._paused = False

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def paused(self) -> bool:
        return self._paused

    def target(self, now: float | None = None) -> float:
        """Simulated time the run should have reached at wall time `now` (default: now)."""
        now = self._clock() if now is None else now
        if self._paused:
            return self._t_sim
        return self._t_sim + (now - self._t_wall) * self._speed

    def _roll(self) -> None:
        now = self._clock()
        self._t_sim = self.target(now)
        self._t_wall = now

    def set_speed(self, factor: float) -> None:
        """Change the speed from now on (clamped to at least 0.1)."""
        self._roll()
        self._speed = max(float(factor), MIN_SPEED)

    def pause(self) -> None:
        """Freeze the target."""
        self._roll()
        self._paused = True

    def resume(self) -> None:
        """Continue from the frozen target."""
        self._roll()
        self._paused = False

    def lag_wall_s(self, t_sim: float, now: float | None = None) -> float:
        """How many wall seconds the simulation (at `t_sim`) is behind the target (>= 0)."""
        return max(0.0, (self.target(now) - t_sim) / self._speed)
