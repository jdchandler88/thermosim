"""Coefficient assembly against direct calls of the correlations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from thermosim.config.schema import Scenario
from thermosim.core.protocols import SourcePower
from thermosim.core.types import OuterFace, ShellKind
from thermosim.physics import correlations as cr
from thermosim.physics import properties as pr
from thermosim.sim.coefficients import apply_source, build_coefficients, mean_liquid_T
from thermosim.sim.model import Model
from thermosim.sources.commands import ScheduleCommand

Example = Callable[..., Scenario]


def _warm_model(sc: Scenario, t: float = 60.0) -> Model:
    m = Model(sc, ScheduleCommand(sc))
    m.advance(t)
    return m


def test_all_coefficients_finite_and_sized(example: Example) -> None:
    for name in ("kettle_electric_insulated", "kettle_propane_40L", "small_pot_stove"):
        m = _warm_model(example(name), 30.0)
        c = m._coeffs
        for field in (
            "k_liquid",
            "rho_c_liquid",
            "h_film",
            "h_out_shell",
            "h_dry",
            "h_surface_open",
            "q_evap_open",
            "h_surface_covered",
            "q_cells",
            "q_shell",
        ):
            assert np.all(np.isfinite(getattr(c, field))), (name, field)
        assert c.h_film.shape == m.domain.shell.couple_shell.shape
        assert c.h_out_shell.shape == (m.domain.shell.n,)
        assert c.rho_c_liquid.shape == m.domain.grid.shape


def test_rho_c_matches_direct_property_evaluation(example: Example) -> None:
    sc = example("kettle_electric_insulated")
    m = _warm_model(sc, 30.0)
    build_coefficients(m.domain, m.state, sc, None, m._cache)  # refresh at the current state
    liq = m.domain.grid.is_liquid
    T = m.state.T_liquid[liq]
    sg = sc.liquid.specific_gravity
    direct = np.asarray(pr.wort_density(T, sg)) * np.asarray(pr.wort_specific_heat(T, sg))
    assert m._coeffs.rho_c_liquid[liq] == pytest.approx(direct, rel=1e-5)


def test_k_liquid_uniform_and_at_least_molecular(example: Example) -> None:
    sc = example("kettle_propane_40L")
    m = _warm_model(sc, 30.0)
    k = m._coeffs.k_liquid[m.domain.grid.is_liquid]
    assert np.ptp(k) == 0.0
    assert k[0] >= float(pr.wort_conductivity(m.recorder.samples[-1].T_mean, 1.0)) * 0.99


def test_outer_coefficient_uses_outer_diameter_and_wind(example: Example) -> None:
    sc = example("kettle_propane_40L")  # 2 m/s wind, uninsulated
    m = _warm_model(sc, 5.0)
    sh = m.domain.shell
    cache = m._cache
    side = sh.kind == int(ShellKind.SIDE)
    direct = cr.h_outer(
        m.state.T_shell,
        sc.atmosphere.ambient_temp_K,
        sh.outer_face,
        sh.char_length,
        sc.atmosphere.wind_speed_m_s,
        cache.p,
        sh.emissivity,
        diameter=sh.outer_diameter,
    )
    assert cache.h_out_full[side] == pytest.approx(np.asarray(direct)[side], rel=0.15)
    assert cache.h_out_full[side].min() > 5.0


def test_burner_base_outer_loss_switched_off_while_burning(example: Example) -> None:
    sc = example("kettle_propane_40L")
    m = _warm_model(sc, 5.0)
    base = m.domain.shell.kind == int(ShellKind.BASE)
    c, cache = m._coeffs, m._cache
    on = SourcePower(5000.0, 15000.0)
    apply_source(c, cache, on)
    assert np.all(c.h_out_shell[base] == 0.0)
    assert np.all(c.h_out_shell[~base] > 0.0)
    apply_source(c, cache, SourcePower(0.0, 0.0))
    assert np.all(c.h_out_shell[base] > 0.0)


def test_source_terms_sum_to_delivered_power(example: Example) -> None:
    for name, attr in (("kettle_propane_40L", "q_shell"), ("kettle_electric_insulated", "q_cells")):
        m = _warm_model(example(name), 2.0)
        apply_source(m._coeffs, m._cache, SourcePower(1234.0, 2000.0))
        assert getattr(m._coeffs, attr).sum() == pytest.approx(1234.0, rel=1e-9)
        other = "q_cells" if attr == "q_shell" else "q_shell"
        assert getattr(m._coeffs, other).sum() == 0.0


def test_covered_surface_uses_headspace_gap(example: Example) -> None:
    sc = example("kettle_electric_insulated")  # partial lid
    m = _warm_model(sc, 30.0)
    build_coefficients(m.domain, m.state, sc, None, m._cache)
    dom, st, c = m.domain, m.state, m._coeffs
    gap = dom.inner_height - dom.fill_height
    T_surf = st.T_liquid.reshape(-1)[dom.grid.surface_cells]
    direct = cr.h_headspace(T_surf, st.T_lid, sc.liquid.specific_gravity, m._cache.p, gap=gap)
    assert c.h_surface_covered == pytest.approx(np.asarray(direct), rel=1e-9)
    assert c.h_lid_inner > 0 and c.h_lid_outer > 0
    # open surface is evaporating
    assert np.all(c.q_evap_open >= 0.0)


def test_open_surface_and_dry_wall_without_lid(example: Example) -> None:
    sc = example("kettle_propane_40L")
    m = _warm_model(sc, 30.0)
    dom, c = m.domain, m._coeffs
    assert np.all(c.h_surface_open > 0.0)
    assert c.h_lid_inner == 0.0 and c.h_lid_outer == 0.0
    dry = dom.shell.dry_area > 0
    assert np.all(c.h_dry[dry] > 0.0)


def test_film_orientations(example: Example) -> None:
    sc = example("kettle_propane_40L")
    m = _warm_model(sc, 30.0)
    sh = m.domain.shell
    assert set(np.unique(m._cache.row_orient)) <= {int(OuterFace.UP), int(OuterFace.SIDE)}
    base_rows = sh.kind[sh.couple_shell] == int(ShellKind.BASE)
    assert np.all(m._cache.row_orient[base_rows] == int(OuterFace.UP))


def test_rebuild_updates_in_place(example: Example) -> None:
    sc = example("small_pot_stove")
    m = _warm_model(sc, 10.0)
    first = m._coeffs
    again = build_coefficients(m.domain, m.state, sc, None, m._cache)
    assert again is first


def test_k_effective_hot_side_is_capped_at_boiling_plus_30_k(example: Example) -> None:
    """A wall far above the boiling point must not inflate the mixing conductivity."""
    sc = example("kettle_propane_40L")
    m = _warm_model(sc, 30.0)
    cache = m._cache
    m.state.T_shell[cache.src_shell_idx] = 800.0  # absurdly hot burner footprint
    on = SourcePower(delivered_W=6000.0, gross_W=17000.0)
    c = build_coefficients(m.domain, m.state, sc, on, cache)
    T_mean = mean_liquid_T(m.domain, m.state.T_liquid)
    sg = sc.liquid.specific_gravity
    capped = float(cr.k_effective(T_mean, cache.T_sat + 30.0, m.domain.fill_height, sg, 1.0))
    assert c.k_liquid[m.domain.grid.is_liquid][0] == pytest.approx(capped, rel=2e-3)
    assert cache.T_sat == pytest.approx(373.12, abs=0.2)  # sea level, SG 1.000


def test_film_coefficient_boils_above_the_boiling_point(example: Example) -> None:
    """Wall nodes above T_sat get the (much larger) nucleate-boiling coefficient."""
    sc = example("kettle_propane_40L")
    m = _warm_model(sc, 30.0)
    cache = m._cache
    rows = np.flatnonzero(m.domain.shell.kind[m.domain.shell.couple_shell] == int(ShellKind.BASE))
    nodes = m.domain.shell.couple_shell[rows]
    base_T = m.state.T_shell.copy()
    m.state.T_shell[nodes] = cache.T_sat - 20.0
    cold = build_coefficients(m.domain, m.state, sc, None, cache).h_film[rows].copy()
    m.state.T_shell[nodes] = cache.T_sat + 60.0  # surface ~ T_sat + 10 K after the wall drop
    hot = build_coefficients(m.domain, m.state, sc, None, cache).h_film[rows].copy()
    m.state.T_shell[:] = base_T
    assert np.all(cold <= cr.H_FILM_MAX)
    assert np.all(hot > 2.0 * cold)
    assert np.all(np.isfinite(hot))


def test_heated_wall_does_not_superheat(example: Example) -> None:
    """The propane kettle's base shell stays within ~60 K of the boiling point (it was ~2000 K)."""
    sc = example("kettle_propane_40L", cell_size_m=0.02)
    m = Model(sc, ScheduleCommand(sc))
    base = m.domain.shell.kind == int(ShellKind.BASE)
    worst = 0.0
    for t in np.arange(30.0, 901.0, 30.0):
        m.advance(float(t))
        worst = max(worst, float(m.state.T_shell[base].max()))
    # mid-plane node: boiling surface (~T_sat + 10 K) plus the 25 K conduction drop in the base
    assert worst < m._cache.T_sat + 60.0
    assert m.error is None
