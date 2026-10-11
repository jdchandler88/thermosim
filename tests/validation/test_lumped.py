"""The lumped reference model: integration accuracy, physical sanity and speed."""

from __future__ import annotations

import time

import numpy as np
import pytest

from thermosim.config import schema
from thermosim.config.io import example_scenarios
from thermosim.core.types import StopReason
from thermosim.physics import properties as pr
from thermosim.validation.analytic import lumped_newton
from thermosim.validation.lumped import LumpedModel, _Coeffs, quick_estimate

EXAMPLES = example_scenarios()


class _ConstantModel(LumpedModel):
    """Constant conductances and capacity with an algebraic shell: an exact Newton body."""

    G_FILM, G_OUT, G_DRY, G_OPEN, C_L = 80.0, 3.0, 1.0, 2.0, 2.0e5

    def __init__(self, scenario: schema.Scenario) -> None:
        super().__init__(scenario)
        self.shell_quasi_steady = True

    def _coefficients(self, T_liquid: float, T_shell: float, T_lid: float, firing: bool) -> _Coeffs:
        return _Coeffs(self.G_FILM, self.G_OUT, self.G_DRY, self.G_OPEN, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _C_liquid(self, T_liquid: float) -> float:
        return self.C_L


def _newton_scenario(name: str) -> schema.Scenario:
    sc = EXAMPLES[name].model_copy(deep=True)
    sc.lid = schema.Lid()
    sc.heat_source = sc.heat_source.model_copy(update={"rated_power_W": 300.0})
    sc.target = schema.Target(temp_K=373.15)  # never reached: steady state is ~50 K above ambient
    sc.simulation = sc.simulation.model_copy(
        update={"max_sim_time_s": 6000.0, "plateau_window_s": 1e9}
    )
    return sc


@pytest.mark.parametrize("name", ["kettle_electric_insulated", "kettle_propane_40L"])
def test_matches_analytic_newton_cooling(name: str) -> None:
    sc = _newton_scenario(name)
    model = _ConstantModel(sc)
    res = model.run()
    assert res.stop_reason is StopReason.MAX_TIME
    # Algebraic shell: liquid -(Gf)- shell -(Gl)- ambient, plus the open surface. The burner's
    # power is delivered to the liquid (the base outer loss is zero while it fires, so none of
    # it is lost on the way), as is an element's; the loss is the series conductance.
    Gf, Gl = model.G_FILM, model.G_OUT + model.G_DRY
    G = Gf * Gl / (Gf + Gl) + model.G_OPEN
    P_eff = model.power(1.0)
    ana = lumped_newton(
        res.t,
        sc.liquid.initial_temp_K,
        sc.atmosphere.ambient_temp_K,
        G,
        model.C_L,
        P_eff,
    )
    np.testing.assert_allclose(res.T_liquid, ana, atol=1e-6, rtol=0)


def test_examples_time_to_target_vs_hand_estimate() -> None:
    for name, sc in EXAMPLES.items():
        res = LumpedModel(sc).run()
        assert res.stop_reason is StopReason.TARGET_REACHED, name
        assert res.time_to_target_s is not None
        cfg = sc.heat_source
        eff = cfg.efficiency if isinstance(cfg, schema.Burner) else 1.0
        P = cfg.rated_power_W * eff
        T0, Tt = sc.liquid.initial_temp_K, sc.target.temp_K
        sg = sc.liquid.specific_gravity
        assert sc.liquid.volume_m3 is not None
        m = float(pr.wort_density(T0, sg)) * sc.liquid.volume_m3
        cp = float(pr.wort_specific_heat(0.5 * (T0 + Tt), sg))
        ideal = m * cp * (Tt - T0) / P
        # Losses and heating the vessel make it slower, but not absurdly so.
        assert ideal < res.time_to_target_s < 2.5 * ideal, name
        # The cold-start vessel may briefly cool a warm liquid (first minute); then it rises.
        assert np.all(np.diff(res.T_liquid[res.t >= 120.0]) >= -1e-9), name
        assert res.T_liquid[-1] == pytest.approx(Tt, abs=1e-6)
        assert res.t[-1] == pytest.approx(res.time_to_target_s)


def test_propane_example_in_public_rule_of_thumb_band() -> None:
    t = LumpedModel(EXAMPLES["kettle_propane_40L"]).run().time_to_target_s
    assert t is not None and 30 * 60 <= t <= 60 * 60


def test_quick_estimate_is_fast_and_close() -> None:
    for sc in EXAMPLES.values():
        t0 = time.perf_counter()
        quick = quick_estimate(sc)
        elapsed = time.perf_counter() - t0
        full = LumpedModel(sc).run().time_to_target_s
        assert quick is not None and full is not None
        assert elapsed < 1.0
        assert quick == pytest.approx(full, rel=0.01)


def test_more_power_is_faster_and_schedule_pauses() -> None:
    sc = EXAMPLES["small_pot_stove"]
    base = LumpedModel(sc).run().time_to_target_s
    hot = sc.model_copy(deep=True)
    hot.heat_source = hot.heat_source.model_copy(update={"rated_power_W": 5000.0})
    faster = LumpedModel(hot).run().time_to_target_s
    assert base is not None and faster is not None and faster < base

    off = sc.model_copy(deep=True)
    off.power_schedule = [
        schema.PowerStep(t_s=0.0, command=1.0),
        schema.PowerStep(t_s=300.0, command=0.0),
        schema.PowerStep(t_s=600.0, command=1.0),
    ]
    res = LumpedModel(off).run()
    paused = res.time_to_target_s
    assert paused is not None and paused > base + 250.0
    assert res.power_W[res.t < 300].max() > 0.0
    assert res.power_W[(res.t > 301) & (res.t < 599)].max() == 0.0


def test_plateau_and_max_time_stop_reasons() -> None:
    weak = EXAMPLES["small_pot_stove"].model_copy(deep=True)
    weak.heat_source = weak.heat_source.model_copy(update={"rated_power_W": 40.0})
    weak.simulation = weak.simulation.model_copy(update={"max_sim_time_s": 6 * 3600.0})
    res = LumpedModel(weak).run()
    assert res.stop_reason is StopReason.PLATEAU
    assert res.time_to_target_s is None
    assert res.T_liquid[-1] < weak.target.temp_K

    short = EXAMPLES["small_pot_stove"].model_copy(deep=True)
    short.simulation = short.simulation.model_copy(update={"max_sim_time_s": 100.0})
    res = LumpedModel(short).run()
    assert res.stop_reason is StopReason.MAX_TIME
    assert res.t[-1] == pytest.approx(100.0)
    assert quick_estimate(short) is None


def test_already_at_target() -> None:
    sc = EXAMPLES["small_pot_stove"].model_copy(deep=True)
    sc.liquid = sc.liquid.model_copy(update={"initial_temp_K": 370.0})
    res = LumpedModel(sc).run()
    assert res.time_to_target_s == 0.0
