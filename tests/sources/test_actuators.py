"""Actuators: exact time-averages, SSR latching, gas valve lag and turndown (WP05)."""

from __future__ import annotations

import math
import random

import pytest

from thermosim.sources.actuators import GasValveActuator, OnOffActuator, VariableActuator


def test_variable_actuator_clips_command() -> None:
    act = VariableActuator()
    assert act.output(0.4, 0.0, 1.0) == pytest.approx(0.4)
    assert act.output(-0.3, 0.0, 1.0) == 0.0
    assert act.output(1.7, 0.0, 1.0) == 1.0
    act.reset()
    assert act.output(0.25, 5.0, 2.0) == pytest.approx(0.25)


def test_on_off_energy_over_100_cycles_equals_command_times_time() -> None:
    T = 2.0
    command = 0.37
    act = OnOffActuator(T)
    rng = random.Random(7)
    t = 0.0
    total_on = 0.0
    total_time = 0.0
    # Irregular steps: some shorter than a cycle, some spanning several cycles.
    while t < 100 * T - 1e-12:
        dt = min(rng.choice([0.05, 0.31, 0.9, 1.3, 2.0, 3.7, 7.1]), 100 * T - t)
        total_on += act.output(command, t, dt) * dt
        total_time += dt
        t += dt
    assert total_on == pytest.approx(command * total_time, rel=1e-12)


def test_on_off_single_step_spanning_many_cycles_is_exact_average() -> None:
    act = OnOffActuator(2.0)
    assert act.output(0.3, 0.0, 200.0) == pytest.approx(0.3, abs=1e-12)
    act.reset()
    # A step that starts mid-cycle and covers 50 whole cycles plus a partial one.
    assert act.output(0.5, 0.5, 101.0) == pytest.approx(0.5, abs=1e-12)


def test_on_off_partial_cycle_average() -> None:
    act = OnOffActuator(1.0)
    # Cycle [0, 1) is ON for [0, 0.5): the first half of the cycle is on.
    assert act.output(0.5, 0.0, 0.5) == pytest.approx(1.0)
    # Cycle [0.5, 1) is OFF for the rest of this second half.
    assert act.output(0.5, 0.5, 0.5) == pytest.approx(0.0)


def test_on_off_command_is_latched_at_cycle_start() -> None:
    act = OnOffActuator(1.0)
    # Cycle 0 latches 0.8: ON over [0, 0.8).
    assert act.output(0.8, 0.0, 0.5) == pytest.approx(1.0)
    # Command drops to 0.2 mid-cycle. Cycle 0 keeps its 0.8 duty: ON over [0.5, 0.8).
    assert act.output(0.2, 0.5, 0.5) == pytest.approx(0.6)


def test_on_off_future_cycle_latched_once_and_kept() -> None:
    act = OnOffActuator(1.0)
    # Cycle 0 latches 0.5: ON for its first half, which is the whole window [0, 0.5).
    assert act.output(0.5, 0.0, 0.5) == pytest.approx(1.0)
    # Spans cycle 1 ([1, 2)); cycle 1 latches 0.9 here and is ON over [1, 1.9).
    assert act.output(0.9, 0.5, 1.0) == pytest.approx(0.5)
    # A later call with a different command must reuse cycle 1's latched duty 0.9.
    assert act.output(0.1, 1.0, 1.0) == pytest.approx(0.9)


def test_on_off_zero_and_full_command() -> None:
    act = OnOffActuator(2.0)
    assert act.output(0.0, 0.0, 10.0) == 0.0
    assert act.output(1.0, 10.0, 10.0) == pytest.approx(1.0)


def test_on_off_instantaneous_query() -> None:
    act = OnOffActuator(1.0)
    assert act.output(0.5, 0.2, 0.0) == 1.0
    assert act.output(0.5, 0.7, 0.0) == 0.0


def test_on_off_rejects_non_positive_period() -> None:
    with pytest.raises(ValueError):
        OnOffActuator(0.0)


def test_gas_valve_step_response_reaches_63_2_percent_at_tau() -> None:
    tau = 2.0
    act = GasValveActuator(min_fraction=0.2, response_time_s=tau)
    # Average over the first tau seconds, then the state at tau from a tiny final step.
    act.output(1.0, 0.0, tau)
    state_at_tau = act.output(1.0, tau, 1e-9)
    assert state_at_tau == pytest.approx(1.0 - math.exp(-1.0), abs=1e-6)


def test_gas_valve_average_is_exact_analytic_integral() -> None:
    tau = 2.0
    act = GasValveActuator(min_fraction=0.2, response_time_s=tau)
    # From 0 with target 1, the average over [0, tau] is 1 - (1 - e^-1) * tau / tau = e^-1.
    assert act.output(1.0, 0.0, tau) == pytest.approx(math.exp(-1.0), abs=1e-12)


def test_gas_valve_average_matches_fine_integration() -> None:
    act = GasValveActuator(min_fraction=0.2, response_time_s=1.5)
    ref = GasValveActuator(min_fraction=0.2, response_time_s=1.5)
    # Two steps with different commands, compared with a fine Riemann sum of the state.
    dt = 0.01
    fine_total = 0.0
    t = 0.0
    for command, span in ((0.9, 3.0), (0.0, 2.0)):
        avg = act.output(command, t, span)
        steps = round(span / dt)
        acc = 0.0
        for _ in range(steps):
            acc += ref.output(command, t, dt) * dt
            t += dt
        fine_total = acc / span
        assert avg == pytest.approx(fine_total, abs=2e-4)


def test_gas_valve_turndown_regions() -> None:
    act = GasValveActuator(min_fraction=0.2, response_time_s=0.0)
    assert act.output(0.05, 0.0, 1.0) == 0.0  # below min/2 = 0.1: off
    act.reset()
    assert act.output(0.10, 0.0, 1.0) == pytest.approx(0.2)  # at min/2: clamped to min
    act.reset()
    assert act.output(0.15, 0.0, 1.0) == pytest.approx(0.2)  # between min/2 and min
    act.reset()
    assert act.output(0.5, 0.0, 1.0) == pytest.approx(0.5)  # above min: follows command
    act.reset()
    assert act.output(1.2, 0.0, 1.0) == pytest.approx(1.0)


def test_gas_valve_zero_lag_follows_target_exactly() -> None:
    act = GasValveActuator(min_fraction=0.0, response_time_s=0.0)
    assert act.output(0.33, 0.0, 5.0) == pytest.approx(0.33)
    assert act.output(0.0, 5.0, 5.0) == 0.0


def test_gas_valve_reset_returns_output_to_zero() -> None:
    act = GasValveActuator(min_fraction=0.2, response_time_s=1.0)
    act.output(1.0, 0.0, 10.0)
    act.reset()
    # Fresh output starts from 0: the average over a tiny step is ~0.
    assert act.output(1.0, 0.0, 1e-9) == pytest.approx(0.0, abs=1e-9)


def test_gas_valve_converges_to_target() -> None:
    act = GasValveActuator(min_fraction=0.2, response_time_s=2.0)
    act.output(0.6, 0.0, 100.0)
    assert act.output(0.6, 100.0, 1e-9) == pytest.approx(0.6, abs=1e-9)
