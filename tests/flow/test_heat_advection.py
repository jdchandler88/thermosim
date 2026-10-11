"""Advective heat flux: conservation, exactness for linear fields, TVD (WP21 validation case 4)."""

from __future__ import annotations

import numpy as np
import pytest
from flow_harness import box_grid

from thermosim.flow import FlowProps, FlowSolver
from thermosim.geometry.mesh import build_grid

PROPS = FlowProps(rho=1000.0, beta=2.0e-4, nu=1.0e-6, cp=4180.0, T_ref=330.0)


@pytest.fixture(scope="module")
def cylinder():
    return build_grid(0.1, 0.205, 0.01)  # partial top layer, staircase footprint


def _random_projected(fs, seed=0):
    rng = np.random.default_rng(seed)
    st = fs.initial_state()
    g = fs.grid
    st.u[...] = 0.05 * rng.standard_normal(st.u.shape) * g.mu
    st.v[...] = 0.05 * rng.standard_normal(st.v.shape) * g.mv
    st.w[...] = 0.05 * rng.standard_normal(st.w.shape) * g.mw
    fs.project(st)
    return st


@pytest.mark.parametrize("scheme", ["vanleer", "central", "upwind"])
def test_q_adv_sums_to_zero(cylinder, scheme):
    """sum(q_adv) = 0 to 1e-12 relative to sum(|q_adv|), also with a random temperature field."""
    fs = FlowSolver(cylinder, scalar_scheme=scheme)
    st = _random_projected(fs)
    rng = np.random.default_rng(1)
    T = np.where(cylinder.is_liquid, 330.0 + 10.0 * rng.random(cylinder.shape), 0.0)
    res = fs.step(st, T, PROPS, 0.05)
    assert np.abs(res.q_adv).sum() > 0.0
    assert abs(res.q_adv.sum()) <= 1e-12 * np.abs(res.q_adv).sum()
    assert not np.any(res.q_adv[~cylinder.is_liquid])
    assert res.q_adv.dtype == np.float64


def test_q_adv_sums_to_zero_float32(cylinder):
    fs = FlowSolver(cylinder, dtype=np.float32)
    st = _random_projected(fs)
    rng = np.random.default_rng(1)
    T = np.where(cylinder.is_liquid, 330.0 + 10.0 * rng.random(cylinder.shape), 0.0)
    res = fs.step(st, T, PROPS, 0.05)
    assert abs(res.q_adv.sum()) <= 1e-12 * np.abs(res.q_adv).sum()


def test_total_heat_is_conserved_over_many_steps(cylinder):
    """Buoyancy-driven flow plus flux-form advection (no conduction) keeps sum(rho cp V T)."""
    fs = FlowSolver(cylinder, smagorinsky_cs=0.1)
    st = fs.initial_state()
    rng = np.random.default_rng(2)
    T = np.where(cylinder.is_liquid, 330.0 + 5.0 * rng.random(cylinder.shape), 0.0)
    vol = cylinder.volume
    energy0 = (PROPS.rho * PROPS.cp * vol * T).sum()
    for _ in range(15):
        res = fs.step(st, T, PROPS, 0.1)
        T = T + 0.1 * res.q_adv / (PROPS.rho * PROPS.cp * np.where(vol > 0, vol, 1.0))
    assert np.abs(st.w).max() > 0.0  # it did move
    assert abs((PROPS.rho * PROPS.cp * vol * T).sum() / energy0 - 1.0) < 1e-12


def test_linear_temperature_advected_exactly_away_from_walls():
    """For div u = 0 and T = a x the flux form gives q = -rho cp V a (u_L + u_R) / 2."""
    gr = box_grid(12, 10, 10, 0.02)
    fs = FlowSolver(gr, poisson_tol=1e-12)
    st = _random_projected(fs, seed=4)
    a = 7.0
    xc = gr.cell_centers()[0]
    T = np.broadcast_to(330.0 + a * xc[:, None, None], gr.shape).copy()
    props = FlowProps(rho=1000.0, beta=0.0, nu=0.0, cp=4000.0, T_ref=330.0, g=0.0)
    res = fs.step(st, T, props, 1e-6)
    ubar = 0.5 * (st.u[:-1] + st.u[1:])
    expect = -props.rho * props.cp * gr.dx**3 * a * ubar
    inner = (slice(2, -2),) * 3
    scale = np.abs(expect[inner]).max()
    assert scale > 0
    np.testing.assert_allclose(res.q_adv[inner], expect[inner], atol=2e-5 * scale)


def test_tvd_scheme_creates_no_new_extrema():
    """Advecting a step profile with the limited scheme stays within the initial range and
    moves the front at the right speed (30 steps at CFL 0.5 = 15 cells)."""
    from thermosim.flow import kernels as K

    gr = box_grid(60, 1, 2, 0.01)
    fs = FlowSolver(gr)
    g = fs.grid
    u = np.zeros((61, 1, 2))
    u[1:-1] = 0.05  # prescribed (not divergence free at the end walls; only the flux is tested)
    T = np.zeros(gr.shape)
    T[:20] = 1.0
    Fx = np.zeros_like(u)
    dt = 0.5 * gr.dx / 0.05
    for _ in range(30):
        K.flux_x(Fx, T, u, g.mu, g.wx, g.liq, gr.dx**2, K.SCHEME_VANLEER)
        T = T + dt * (Fx[:-1] - Fx[1:]) / gr.dx**3
    assert T.min() >= -1e-12 and T.max() <= 1.0 + 1e-12
    front = int(np.flatnonzero(T[:, 0, 0] > 0.5).max()) + 1
    assert 33 <= front <= 37
    # sharper than first-order upwind, which smears over ~sqrt(30) * 2 cells more
    width = np.count_nonzero((T[:, 0, 0] > 0.05) & (T[:, 0, 0] < 0.95))
    assert width <= 6
