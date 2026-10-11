"""Actuators: how a requested power fraction becomes the fraction actually delivered (WP05).

Every actuator maps a command in [0, 1] and a step [t, t+dt] to the exact average output
over that step, in [0, 1]. Stateful actuators (SSR cycle latch, valve lag) keep their state
between calls; `reset()` returns them to the initial state.
"""

from __future__ import annotations

import math


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


class VariableActuator:
    """Power follows the command directly: output = clip(command, 0, 1)."""

    def output(self, command: float, t: float, dt: float) -> float:
        return _clip01(command)

    def reset(self) -> None:
        """Stateless; nothing to reset."""


class OnOffActuator:
    """SSR time-proportioning over fixed cycles of length `cycle_period_s`.

    Within the cycle [nT, (n+1)T) the output is ON for the first `command * T` seconds. The
    command is latched at the start of each cycle, so a cycle keeps the duty it started with
    even if the command changes part-way through. A step that spans several cycles, or a
    partial one, gets the exact time-average of the on/off signal over that step.
    """

    def __init__(self, cycle_period_s: float) -> None:
        if cycle_period_s <= 0.0:
            raise ValueError("cycle_period_s must be positive")
        self._T = float(cycle_period_s)
        self._latches: dict[int, float] = {}

    def reset(self) -> None:
        self._latches.clear()

    def output(self, command: float, t: float, dt: float) -> float:
        T = self._T
        c = _clip01(command)
        n0 = math.floor(t / T)
        if (n0 + 1) * T <= t:  # guard against round-off just below a cycle boundary
            n0 += 1
        # Forget latches of cycles that have already ended.
        for done in [n for n in self._latches if n < n0]:
            del self._latches[done]

        if dt <= 0.0:
            duty = self._latches.setdefault(n0, c)
            phase = t - n0 * T
            return 1.0 if phase < duty * T else 0.0

        t_end = t + dt
        on_time = 0.0
        n = n0
        while n * T < t_end:
            start = n * T
            end = start + T
            duty = self._latches.setdefault(n, c)
            seg_lo = max(t, start)
            seg_hi = min(t_end, end)
            on_hi = min(seg_hi, start + duty * T)
            if on_hi > seg_lo:
                on_time += on_hi - seg_lo
            n += 1
        return on_time / dt


class GasValveActuator:
    """Modulated gas valve: turndown limit plus a first-order lag.

    The target is 0 below `min_fraction / 2`, `min_fraction` between `min_fraction / 2` and
    `min_fraction` (the burner's turndown limit), and the command itself above that. The
    output follows the target as dy/dt = (target - y) / tau with tau = `response_time_s`. For a
    constant target over dt the average is computed exactly from the analytic solution.
    """

    def __init__(self, min_fraction: float, response_time_s: float) -> None:
        if not 0.0 <= min_fraction <= 1.0:
            raise ValueError("min_fraction must be in [0, 1]")
        if response_time_s < 0.0:
            raise ValueError("response_time_s must be >= 0")
        self._min = float(min_fraction)
        self._tau = float(response_time_s)
        self._y = 0.0

    def reset(self) -> None:
        self._y = 0.0

    def target(self, command: float) -> float:
        c = _clip01(command)
        if c < 0.5 * self._min:
            return 0.0
        if c < self._min:
            return self._min
        return c

    def output(self, command: float, t: float, dt: float) -> float:
        target = self.target(command)
        y0 = self._y
        if dt <= 0.0:
            return y0
        tau = self._tau
        if tau == 0.0:
            self._y = target
            return target
        decay = math.exp(-dt / tau)
        # Average of target + (y0 - target) * exp(-s / tau) over s in [0, dt]. expm1 keeps
        # (1 - exp(-dt/tau)) accurate for steps much shorter than tau.
        one_minus_decay = -math.expm1(-dt / tau)
        average = target + (y0 - target) * tau * one_minus_decay / dt
        self._y = target + (y0 - target) * decay
        return average
