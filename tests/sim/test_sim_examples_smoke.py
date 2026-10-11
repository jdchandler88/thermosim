"""End-to-end smoke tests: every example scenario at dx = 0.02 m."""

from __future__ import annotations

import math
import time

import pytest

from thermosim.config.io import example_scenarios
from thermosim.core.types import StopReason
from thermosim.physics.properties import wort_density, wort_specific_heat
from thermosim.sim import run_headless

NAMES = sorted(example_scenarios())


@pytest.mark.parametrize("name", NAMES)
def test_example_runs_to_target_with_closed_energy_balance(name: str) -> None:
    sc = example_scenarios()[name]
    sc = sc.model_copy(
        update={"simulation": sc.simulation.model_copy(update={"cell_size_m": 0.02})}
    )
    t0 = time.perf_counter()
    result = run_headless(sc)
    wall = time.perf_counter() - t0
    s = result.summary
    assert s.stop_reason in (StopReason.TARGET_REACHED, StopReason.PLATEAU)
    assert s.energy_balance_rel_error < 1e-6
    assert wall < 30.0
    assert s.stop_reason == StopReason.TARGET_REACHED
    assert s.time_to_target_s is not None

    # Physical sanity. The liquid energy gain must equal rho cp V dT for the scenario volume,
    # and the losses (vessel, surface, lid) must be a moderate share of the input.
    sg = sc.liquid.specific_gravity
    T0, T1 = sc.liquid.initial_temp_K, s.final_T_mean
    mid = 0.5 * (T0 + T1)
    if sc.liquid.volume_m3 is not None:
        volume = sc.liquid.volume_m3
    else:
        assert sc.liquid.fill_height_m is not None
        volume = sc.liquid.fill_height_m * math.pi * (sc.vessel.inner_diameter_m / 2) ** 2
    e_hand = volume * float(wort_density(mid, sg)) * float(wort_specific_heat(mid, sg)) * (T1 - T0)
    assert s.energy.stored_liquid == pytest.approx(e_hand, rel=0.03)
    # The time to target cannot beat the lossless estimate m cp dT / P_delivered.
    p_deliv = result.samples[len(result.samples) // 2].power_delivered
    assert s.time_to_target_s >= 0.95 * e_hand / p_deliv
    loss_share = s.energy.total_loss() / s.energy.total_in()
    assert 0.0 < loss_share < 0.5
    assert 0.1 < s.heating_efficiency < 1.0
