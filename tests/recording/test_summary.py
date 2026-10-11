"""summarize(): time-to-target interpolation, energy totals, efficiency and balance error."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from thermosim.core.types import EnergyFlows, Sample, StopReason
from thermosim.recording import summarize

Builder = Callable[..., Sample]


def test_time_to_target_exact_on_linear_ramp(make_sample: Builder) -> None:
    # 0.5 K/s ramp sampled every second; target 7.3 K above the start is reached at t = 14.6 s.
    samples = [make_sample(float(t), 293.15 + 0.5 * t) for t in range(0, 61)]
    summary = summarize(samples, 293.15 + 7.3, "mean", StopReason.TARGET_REACHED, 0.1)
    assert summary.time_to_target_s == pytest.approx(14.6, abs=1e-9)
    assert summary.target_probe == "mean"
    assert summary.stop_reason is StopReason.TARGET_REACHED


def test_time_to_target_on_named_probe(make_sample: Builder) -> None:
    # The mean ramps slowly; the probe ramps twice as fast and crosses first.
    samples = [
        make_sample(float(t), 293.15 + 0.1 * t, {"center": 293.15 + 0.2 * t, "edge": 293.15})
        for t in range(0, 101)
    ]
    summary = summarize(samples, 293.15 + 4.0, "center", StopReason.TARGET_REACHED, 0.0)
    assert summary.time_to_target_s == pytest.approx(20.0, abs=1e-9)
    assert summary.final_probes == {"center": pytest.approx(293.15 + 20.0), "edge": 293.15}
    assert summarize(samples, 293.15 + 4.0, "mean", StopReason.MAX_TIME, 0.0).time_to_target_s == (
        pytest.approx(40.0, abs=1e-9)
    )


def test_time_to_target_none_when_never_reached(make_sample: Builder) -> None:
    samples = [make_sample(float(t), 293.15 + 0.01 * t) for t in range(10)]
    summary = summarize(samples, 400.0, "mean", StopReason.MAX_TIME, 0.0)
    assert summary.time_to_target_s is None


def test_time_to_target_when_first_sample_is_already_hot(make_sample: Builder) -> None:
    samples = [make_sample(float(t), 350.0 + t) for t in range(5)]
    summary = summarize(samples, 340.0, "mean", StopReason.TARGET_REACHED, 0.0)
    assert summary.time_to_target_s == 0.0


def test_unknown_probe_raises(make_sample: Builder) -> None:
    samples = [make_sample(0.0, probes={"center": 300.0})]
    with pytest.raises(ValueError, match="unknown probe 'nope'"):
        summarize(samples, 310.0, "nope", StopReason.MAX_TIME, 0.0)


def test_energy_efficiency_and_balance(make_sample: Builder) -> None:
    # Constant 2000 W gross for 100 s: trapezoid gross energy is 200 000 J.
    flows = EnergyFlows(
        source_liquid=200_000.0,
        loss_shell_outer=20_000.0,
        stored_liquid=150_000.0,
        stored_shell=30_000.0,
    )
    samples = [
        make_sample(float(t), 300.0, power_gross=2000.0, cumulative=flows if t == 100 else None)
        for t in range(101)
    ]
    summary = summarize(samples, 350.0, "mean", StopReason.MAX_TIME, 2.0)
    assert summary.heating_efficiency == pytest.approx(0.75)
    assert summary.energy_balance_rel_error == pytest.approx(0.0, abs=1e-15)
    assert summary.energy.stored_liquid == 150_000.0
    assert summary.energy is not flows  # a copy, not the sample's object
    assert summary.sim_time_s == 100.0
    assert summary.wall_time_s == 2.0


def test_balance_error_is_relative_to_energy_in(make_sample: Builder) -> None:
    flows = EnergyFlows(source_liquid=200_000.0, stored_liquid=150_000.0, stored_shell=29_000.0)
    samples = [make_sample(0.0, 300.0), make_sample(1.0, 301.0, cumulative=flows)]
    summary = summarize(samples, 350.0, "mean", StopReason.MAX_TIME, 0.0)
    # balance = 200000 - 0 - 179000 = 21000 J, relative to 200000 J in
    assert summary.energy_balance_rel_error == pytest.approx(0.105)


def test_efficiency_with_ramping_gross_power_uses_trapezoid(make_sample: Builder) -> None:
    # power_gross ramps 0 -> 1000 W over 10 s: trapezoid = 5000 J.
    flows = EnergyFlows(stored_liquid=2500.0)
    samples = [
        make_sample(float(t), 300.0, power_gross=100.0 * t, cumulative=flows if t == 10 else None)
        for t in range(11)
    ]
    summary = summarize(samples, 350.0, "mean", StopReason.MAX_TIME, 0.0)
    assert summary.heating_efficiency == pytest.approx(0.5)


def test_fuel_and_electric_totals_come_from_last_sample(make_sample: Builder) -> None:
    samples = [
        make_sample(0.0, 300.0, fuel_kg=0.0, electric_kwh=0.0),
        make_sample(1.0, 301.0, fuel_kg=0.02, electric_kwh=0.5),
    ]
    summary = summarize(samples, 350.0, "mean", StopReason.USER_STOP, 0.0)
    assert summary.fuel_kg == 0.02
    assert summary.electric_kwh == 0.5
    assert summary.stop_reason is StopReason.USER_STOP


def test_summary_of_no_samples() -> None:
    summary = summarize([], 350.0, "mean", StopReason.ERROR, 0.5)
    assert summary.stop_reason is StopReason.ERROR
    assert summary.time_to_target_s is None
    assert summary.heating_efficiency == 0.0
    assert summary.sim_time_s == 0.0
