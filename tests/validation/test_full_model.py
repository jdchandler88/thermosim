"""End-to-end validation of the 3-D model (WP10 Part B).

* the 3-D time to target against the lumped model with a well-mixed liquid;
* energy conservation of every example;
* a public rule of thumb for the 40 L propane example;
* grid convergence of the 40 L propane example (slow).

The numbers reported in ``docs/validation.md`` come from ``benchmarks/validation_report.py``.
"""

from __future__ import annotations

import math

import pytest

from thermosim.config.io import example_scenarios
from thermosim.core.types import RunResult, StopReason
from thermosim.physics import properties as pr
from thermosim.sim.runner import run_headless
from thermosim.validation.crosscheck import compare_with_lumped, richardson

EXAMPLES = example_scenarios()

# Cell sizes of the lumped comparison: coarse enough for the default test run (k_eff_scale = 100
# makes the explicit time step tiny), fine enough that the geometry is represented well.
LUMPED_CELL_M = {
    "small_pot_stove": 0.02,
    "kettle_electric_insulated": 0.02,
    "kettle_propane_40L": 0.04,
}


@pytest.mark.parametrize("name", sorted(LUMPED_CELL_M))
def test_well_mixed_3d_agrees_with_the_lumped_model(name: str) -> None:
    """k_eff_scale = 100 forces a well-mixed liquid; then the 3-D time to target must be within
    3 % of the lumped model's (they share the property and correlation functions)."""
    cmp = compare_with_lumped(EXAMPLES[name], k_eff_scale=100.0, cell_size_m=LUMPED_CELL_M[name])
    print(
        f"\n{name}: 3-D {cmp.time_3d_s:.0f} s, lumped {cmp.time_lumped_s:.0f} s, "
        f"error {100 * cmp.rel_error:+.2f} % (dx = {cmp.cell_size_m * 1e3:.0f} mm, "
        f"{cmp.wall_s:.1f} s)"
    )
    assert cmp.time_3d_s is not None and cmp.time_lumped_s is not None
    assert abs(cmp.rel_error) < 0.03
    assert cmp.energy_balance_rel_error < 1e-6


@pytest.mark.parametrize("name", sorted(LUMPED_CELL_M))
def test_lumped_agreement_without_forced_mixing_is_reported(name: str) -> None:
    """At k_eff_scale = 1 the liquid stratifies and the lumped model is only a rough guide; the
    error is informational (docs/validation.md). It must stay moderate."""
    cmp = compare_with_lumped(EXAMPLES[name], k_eff_scale=1.0, cell_size_m=0.02)
    print(f"\n{name}: k_eff_scale = 1: 3-D vs lumped error {100 * cmp.rel_error:+.1f} %")
    assert not math.isnan(cmp.rel_error)
    assert abs(cmp.rel_error) < 0.15


@pytest.fixture(scope="module")
def example_runs() -> dict[str, RunResult]:
    """Every example at the 20 mm smoke-test cell size, default model settings."""
    out = {}
    for name, sc in EXAMPLES.items():
        s = sc.model_copy(deep=True)
        s.simulation.cell_size_m = 0.02
        out[name] = run_headless(s)
    return out


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_energy_balance_of_every_example(example_runs: dict[str, RunResult], name: str) -> None:
    s = example_runs[name].summary
    assert s.stop_reason in (StopReason.TARGET_REACHED, StopReason.PLATEAU)
    assert s.energy_balance_rel_error < 1e-6


def test_propane_example_is_within_the_public_rule_of_thumb(
    example_runs: dict[str, RunResult],
) -> None:
    """40 L from 15 to 99 C on a 60k BTU/hr (17.6 kW) burner at 35 % efficiency.

    Ideal energy m cp dT = 40 kg * 4.19 kJ/(kg K) * 84 K = 14 MJ at 6.2 kW delivered is 38 min;
    losses (open surface, evaporation, wind-cooled wall) and heating the kettle add to that.
    Rules of thumb for outdoor propane kettles put it at 30-60 min."""
    sc = EXAMPLES["kettle_propane_40L"]
    s = example_runs["kettle_propane_40L"].summary
    assert s.time_to_target_s is not None
    minutes = s.time_to_target_s / 60.0
    ideal = (
        float(pr.wort_density(sc.liquid.initial_temp_K, 1.0))
        * sc.liquid.volume_m3  # type: ignore[operator]
        * float(pr.wort_specific_heat(0.5 * (sc.liquid.initial_temp_K + sc.target.temp_K), 1.0))
        * (sc.target.temp_K - sc.liquid.initial_temp_K)
        / (sc.heat_source.rated_power_W * sc.heat_source.efficiency)  # type: ignore[union-attr]
        / 60.0
    )
    print(f"\npropane 40 L: {minutes:.1f} min (ideal {ideal:.1f} min)")
    assert 30.0 <= minutes <= 60.0
    assert ideal < minutes < 1.8 * ideal


def test_richardson_extrapolation_on_a_known_sequence() -> None:
    # f(h) = 5 + 3 h^2 at h = 4, 2, 1 -> order 2, limit 5
    c = richardson([5 + 3 * 16.0, 5 + 3 * 4.0, 5 + 3 * 1.0])
    assert c.monotone
    assert c.order == pytest.approx(2.0)
    assert c.extrapolated == pytest.approx(5.0)
    assert c.fine_error == pytest.approx(3.0 / 5.0)
    assert not richardson([1.0, 0.5, 0.6]).monotone


@pytest.mark.slow
def test_grid_convergence_of_the_propane_example() -> None:
    """40 L propane at dx = 20, 10, 5 mm.

    The mean-temperature time to target is almost grid independent (the whole run is dominated
    by the energy balance): it must agree to 0.2 % between all three levels. The local
    temperatures converge monotonically with an observed order and a Richardson-extrapolated
    value, and the 5 mm result must be within 2 % (of the temperature rise) of it."""
    sc = EXAMPLES["kettle_propane_40L"]
    times, centre, wall = [], [], []
    t_probe = 1500.0
    for dx in (0.02, 0.01, 0.005):
        s = sc.model_copy(deep=True)
        s.simulation.cell_size_m = dx
        res = run_headless(s)
        assert res.summary.time_to_target_s is not None
        times.append(res.summary.time_to_target_s)
        smp = next(x for x in res.samples if abs(x.t - t_probe) < 0.5)
        centre.append(smp.probes["center"])
        wall.append(smp.probes["near_wall"])
    t0 = sc.liquid.initial_temp_K
    print(f"\ntime to target (s): {times}")
    assert max(times) - min(times) < 0.002 * times[-1]
    for name, vals in (("center", centre), ("near_wall", wall)):
        rise = [v - t0 for v in vals]
        conv = richardson(rise)
        print(
            f"{name}: rise {rise} -> monotone {conv.monotone}, order {conv.order:.2f}, "
            f"extrapolated {conv.extrapolated:.4f}, 5 mm error {100 * conv.fine_error:.3f} %"
        )
        assert conv.monotone, name
        assert 0.5 < conv.order < 4.0, name
        assert conv.fine_error < 0.02, name
