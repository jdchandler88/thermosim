"""Flow mode (Phase 2, WP22): model coupling, energy, snapshots, runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from thermosim.config.schema import Scenario
from thermosim.core.types import Sample, Snapshot, StopReason
from thermosim.sim import SimulationRunner, run_headless
from thermosim.sim.model import Model
from thermosim.sources.commands import ScheduleCommand

Example = Callable[..., Scenario]


def flow_scenario(sc: Scenario, **model: Any) -> Scenario:
    update = {"liquid_transport": "flow", **model}
    return sc.model_copy(update={"model": sc.model.model_copy(update=update)})


def _model(sc: Scenario) -> Model:
    return Model(sc, ScheduleCommand(sc))


def test_effective_mode_has_no_flow_and_no_velocity(example: Example) -> None:
    m = _model(example("small_pot_stove"))
    assert m.flow is None
    out = m.advance(10.0)
    snaps = [o for o in out if isinstance(o, Snapshot)]
    assert snaps
    assert all(s.velocity_u is None and s.velocity_w is None for s in snaps)


def test_flow_mode_snapshots_carry_velocity_nan_outside_liquid(example: Example) -> None:
    sc = flow_scenario(example("small_pot_stove", snapshot_interval_s=10.0))
    m = _model(sc)
    out = m.advance(60.0)
    snaps = [o for o in out if isinstance(o, Snapshot)]
    assert len(snaps) >= 2
    for s in snaps:
        assert s.velocity_u is not None and s.velocity_w is not None
        assert s.velocity_u.shape == s.slice_T.shape == s.velocity_w.shape
        assert s.velocity_u.dtype == np.float32
        outside = np.isnan(s.slice_T)
        assert np.isnan(s.velocity_u[outside]).all()
        assert np.isnan(s.velocity_w[outside]).all()
        assert np.isfinite(s.velocity_u[~outside]).all()
        assert np.isfinite(s.velocity_w[~outside]).all()
    first, last = snaps[0], snaps[-1]
    assert np.nanmax(np.abs(first.velocity_w)) == 0.0  # fluid at rest at t = 0
    assert np.nanmax(np.abs(last.velocity_w)) > 1e-3  # the heated base drives a plume
    assert m.peak_speed > 1e-3


def test_flow_mode_uses_molecular_conductivity_plus_subgrid(example: Example) -> None:
    sc = flow_scenario(example("small_pot_stove"), smagorinsky_cs=0.17)
    m = _model(sc)
    m.advance(60.0)
    coeffs = m._coeffs
    liq = m.domain.grid.is_liquid
    k_liq = coeffs.k_liquid[liq]
    assert k_liq.min() > 0.55  # molecular, not a mixing closure of 10 W/(m K) and up
    # with the subgrid model on, the eddy conductivity adds to it where the flow shears
    assert m._cache.k_turb is not None
    assert (k_liq >= 0.55).all()
    assert k_liq.max() > k_liq.min()


def test_effective_mode_conductivity_unchanged_by_flow_settings(example: Example) -> None:
    base = _model(example("small_pot_stove"))
    other = _model(
        example("small_pot_stove").model_copy(
            update={
                "model": example("small_pot_stove").model.model_copy(update={"smagorinsky_cs": 0.3})
            }
        )
    )
    base.advance(30.0)
    other.advance(30.0)
    assert np.array_equal(base.state.T_liquid, other.state.T_liquid)
    assert base.state.t == other.state.t
    liq = base.domain.grid.is_liquid
    assert base._coeffs.k_liquid[liq].min() > 2.0  # the mixing closure


def test_thermal_dt_is_capped_by_flow_cfl(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    sc = flow_scenario(example("small_pot_stove"))
    m = _model(sc)
    assert m.flow is not None
    seen: list[tuple[float, float]] = []
    orig_step = m.flow.step
    orig_max = m.flow.max_dt

    def spy_step(state, T, props, dt):  # type: ignore[no-untyped-def]
        seen.append((dt, orig_max(state, props.nu)))
        return orig_step(state, T, props, dt)

    monkeypatch.setattr(m.flow, "step", spy_step)
    m.advance(120.0)
    assert len(seen) > 100
    assert all(dt <= cap * (1 + 1e-9) for dt, cap in seen)
    # and the cap actually bound at some point once the plume had developed
    assert any(dt >= 0.99 * cap for dt, cap in seen[len(seen) // 2 :])


def test_flow_mode_energy_balance_closes(example: Example) -> None:
    sc = flow_scenario(example("small_pot_stove"))
    m = _model(sc)
    m.advance(200.0)
    result = m.build_result(1.0)
    assert result.summary.energy_balance_rel_error < 1e-9
    assert m.cumulative.stored_liquid > 0.0


def test_flow_mode_reaches_target_on_coarse_grid(example: Example) -> None:
    eff = example("small_pot_stove", cell_size_m=0.03)
    flow = flow_scenario(eff)
    r_eff = run_headless(eff)
    r_flow = run_headless(flow)
    s = r_flow.summary
    assert s.stop_reason == StopReason.TARGET_REACHED
    assert s.energy_balance_rel_error < 1e-9
    assert r_eff.summary.time_to_target_s is not None and s.time_to_target_s is not None
    # same physics, different liquid transport: the times agree to within a modest margin
    assert s.time_to_target_s == pytest.approx(r_eff.summary.time_to_target_s, rel=0.15)


def test_runner_in_flow_mode_emits_velocity_snapshots(example: Example) -> None:
    sc = flow_scenario(example("small_pot_stove", cell_size_m=0.03, max_sim_time_s=90.0))
    runner = SimulationRunner(sc)
    runner.start()
    runner.join(120.0)
    events = runner.poll_events(100_000)
    snaps = [e.snapshot for e in events if hasattr(e, "snapshot")]
    samples = [e for e in events if hasattr(e, "sample")]
    assert snaps and samples
    assert all(isinstance(e.sample, Sample) for e in samples)
    assert snaps[-1].velocity_u is not None
    assert runner.result is not None
    assert runner.result.summary.energy_balance_rel_error < 1e-9


def test_flow_mode_roundtrips_through_scenario_toml(example: Example) -> None:
    from thermosim.config.io import scenario_from_toml, scenario_to_toml

    sc = flow_scenario(example("small_pot_stove"), smagorinsky_cs=0.12, poisson_tol=1e-5)
    back = scenario_from_toml(scenario_to_toml(sc))
    assert back.model.liquid_transport == "flow"
    assert back.model.smagorinsky_cs == 0.12
    assert back.model.poisson_tol == 1e-5
