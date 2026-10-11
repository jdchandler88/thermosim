"""Model assembly: stepping, sampling, stop criteria, energy accounting, lumped agreement."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from thermosim.config.schema import Scenario
from thermosim.core.types import Sample, Snapshot, StopReason
from thermosim.physics.properties import wort_density, wort_specific_heat
from thermosim.sim.model import Model
from thermosim.sources.commands import ScheduleCommand

Example = Callable[..., Scenario]


def _model(sc: Scenario, **kw: object) -> Model:
    return Model(sc, ScheduleCommand(sc), **kw)  # type: ignore[arg-type]


def test_initial_state_vessel_at_ambient(example: Example) -> None:
    sc = example("kettle_propane_40L")
    m = _model(sc)
    amb = sc.atmosphere.ambient_temp_K
    assert np.all(m.state.T_shell == amb)
    assert m.state.T_lid == amb and m.state.T_headspace == amb
    liquid = m.state.T_liquid[m.domain.grid.is_liquid]
    assert np.all(liquid == sc.liquid.initial_temp_K)


def test_first_advance_returns_t0_sample_and_snapshot(example: Example) -> None:
    m = _model(example("small_pot_stove"))
    out = m.advance(0.0)
    assert [type(o) for o in out] == [Sample, Snapshot]
    assert all(o.t == 0.0 for o in out)


def test_samples_and_snapshots_are_regular_and_ordered(example: Example) -> None:
    sc = example("small_pot_stove", sample_interval_s=1.0, snapshot_interval_s=5.0)
    m = _model(sc)
    out = m.advance(20.0)
    samples = [o for o in out if isinstance(o, Sample)]
    snaps = [o for o in out if isinstance(o, Snapshot)]
    assert [s.t for s in samples] == pytest.approx(list(range(21)))
    assert [s.t for s in snaps] == pytest.approx([0.0, 5.0, 10.0, 15.0, 20.0])
    times = [o.t for o in out]
    assert times == sorted(times)
    assert m.state.t == pytest.approx(20.0)


def test_target_reached_gives_final_sample_and_balance(example: Example) -> None:
    m = _model(example("small_pot_stove"))
    out = m.advance(10_000.0)
    assert m.stop_reason == StopReason.TARGET_REACHED
    last = m.recorder.samples[-1]
    assert last.t == pytest.approx(m.state.t)
    assert isinstance(out[-1], Snapshot | Sample)
    assert last.T_mean >= m.scenario.target.temp_K or max(last.probes.values()) >= (
        m.scenario.target.temp_K
    )
    res = m.build_result(1.0)
    assert res.summary.energy_balance_rel_error < 1e-9
    assert res.summary.time_to_target_s is not None


def test_max_time_stop_when_power_off(example: Example) -> None:
    sc = example("small_pot_stove", max_sim_time_s=30.0)
    data = sc.model_dump(mode="python")
    data["power_schedule"] = [{"t_s": 0.0, "command": 0.0}]
    sc = Scenario.model_validate(data)
    m = _model(sc)
    m.advance(1000.0)
    assert m.stop_reason == StopReason.MAX_TIME
    assert m.state.t == pytest.approx(30.0)
    assert m.recorder.samples[-1].t == pytest.approx(30.0)
    assert m.build_result(0.1).summary.fuel_kg == 0.0


def test_plateau_stop(example: Example) -> None:
    # A huge allowed rise rate makes any trace "level"; the burner is on, so PLATEAU fires.
    sc = example("small_pot_stove", plateau_window_s=60.0, plateau_rate_K_per_min=10.0)
    m = _model(sc)
    m.advance(10_000.0)
    assert m.stop_reason == StopReason.PLATEAU
    assert m.state.t == pytest.approx(60.0, abs=1.5)


def test_no_plateau_when_power_off(example: Example) -> None:
    sc = example(
        "small_pot_stove",
        plateau_window_s=20.0,
        plateau_rate_K_per_min=10.0,
        max_sim_time_s=100.0,
    )
    data = sc.model_dump(mode="python")
    data["power_schedule"] = [{"t_s": 0.0, "command": 0.0}]
    m = _model(Scenario.model_validate(data))
    m.advance(1000.0)
    assert m.stop_reason == StopReason.MAX_TIME


def test_user_stop_finish(example: Example) -> None:
    m = _model(example("small_pot_stove"))
    m.advance(7.3)
    out = m.finish(StopReason.USER_STOP)
    assert m.stop_reason == StopReason.USER_STOP
    assert any(isinstance(o, Sample) and o.t == pytest.approx(7.3) for o in out)
    assert m.snapshots[len(m.snapshots) - 1].t == pytest.approx(7.3)


def test_energy_balance_is_roundoff_for_all_examples(example: Example) -> None:
    for name in ("kettle_electric_insulated", "kettle_propane_40L", "small_pot_stove"):
        m = _model(example(name))
        m.advance(300.0)
        e = m.cumulative
        assert abs(e.balance_error()) / e.total_in() < 1e-9, name
        assert np.all(np.isfinite(m.state.T_liquid))


def test_burner_electric_accounting(example: Example) -> None:
    mb = _model(example("small_pot_stove"))
    mb.advance(60.0)
    s = mb.recorder.samples[-1]
    assert s.fuel_kg > 0 and s.electric_kwh == 0
    me = _model(example("kettle_electric_insulated"))
    me.advance(60.0)
    s = me.recorder.samples[-1]
    assert s.electric_kwh > 0 and s.fuel_kg == 0
    assert s.electric_kwh == pytest.approx(5500.0 * 60.0 / 3.6e6, rel=0.02)


def test_solver_error_sets_error_stop(example: Example, monkeypatch: pytest.MonkeyPatch) -> None:
    from thermosim.core.errors import SolverError
    from thermosim.solver.thermal import ThermalSolver

    def boom(*args: object, **kwargs: object) -> None:
        raise SolverError("synthetic failure")

    m = _model(example("small_pot_stove"))
    monkeypatch.setattr(ThermalSolver, "step", boom)
    m.advance(10.0)
    assert m.stop_reason == StopReason.ERROR
    assert isinstance(m.error, SolverError)


def test_lumped_agreement_40L_propane_no_losses(example: Example) -> None:
    """Well mixed (k_eff_scale = 50), no wind, losses disabled: the time to target equals the
    hand estimate (energy to heat the liquid + the vessel) / delivered power within 3 %."""
    sc = example("kettle_propane_40L", max_sim_time_s=6000.0, cell_size_m=0.04)
    # The schema caps k_eff_scale at 10; bypass validation for this well-mixed test.
    sc = sc.model_copy(
        update={
            "model": sc.model.model_copy(update={"k_eff_scale": 50.0}),
            "atmosphere": sc.atmosphere.model_copy(update={"wind_speed_m_s": 0.0}),
        }
    )

    holder: list[Model] = []

    def no_losses(c: object) -> None:
        cache = holder[0]._cache
        for arr in (cache.h_out_full, cache.h_out_burner):
            arr.fill(0.0)
        c.h_dry.fill(0.0)  # type: ignore[attr-defined]
        c.h_surface_open.fill(0.0)  # type: ignore[attr-defined]
        c.q_evap_open.fill(0.0)  # type: ignore[attr-defined]

    m = _model(sc, coeff_hook=no_losses)
    holder.append(m)
    m.advance(sc.simulation.max_sim_time_s)
    assert m.stop_reason == StopReason.TARGET_REACHED
    res = m.build_result(1.0)
    t_sim = res.summary.time_to_target_s
    assert t_sim is not None

    # Hand estimate, using the liquid's properties at the mid temperature.
    T0, T1 = sc.liquid.initial_temp_K, sc.target.temp_K
    mid = 0.5 * (T0 + T1)
    vol = sc.liquid.volume_m3
    assert vol is not None
    mass = vol * float(wort_density(mid, 1.0))
    e_liquid = mass * float(wort_specific_heat(mid, 1.0)) * (T1 - T0)
    sh = m.domain.shell
    e_vessel = float(
        np.sum(sh.rho_c * sh.thickness * sh.area * (m.state.T_shell - sc.atmosphere.ambient_temp_K))
    )
    power = sc.heat_source.rated_power_W * sc.heat_source.efficiency  # type: ignore[union-attr]
    t_hand = (e_liquid + e_vessel) / power
    assert t_sim == pytest.approx(t_hand, rel=0.03)
    # And with every loss path disabled, nothing leaves.
    assert res.summary.energy.total_loss() < 1e-6 * res.summary.energy.total_in()
