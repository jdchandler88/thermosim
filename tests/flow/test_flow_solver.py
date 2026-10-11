"""FlowSolver API, boundary conditions, subgrid model and buoyancy plume (WP21)."""

from __future__ import annotations

import numpy as np
import pytest
from flow_harness import box_grid

from thermosim.flow import FlowProps, FlowSolver, props_from_physics
from thermosim.flow import kernels as K
from thermosim.geometry.mesh import build_grid
from thermosim.testing.synthetic import box_domain

WATER = FlowProps(rho=1000.0, beta=4.0e-4, nu=5.0e-7, cp=4180.0, T_ref=330.0)


def test_accepts_domain_and_liquid_grid_and_state_shapes():
    dom = box_domain(6, 5, 4, 0.01, shell=False)
    fs = FlowSolver(dom)
    assert fs.grid.shape == (6, 5, 4)
    st = fs.initial_state()
    assert st.u.shape == (7, 5, 4) and st.v.shape == (6, 6, 4) and st.w.shape == (6, 5, 5)
    assert st.p.shape == (6, 5, 4)
    assert st.u.dtype == np.float64
    assert not st.u.any() and not st.p.any()
    fs32 = FlowSolver(dom.grid, dtype=np.float32)
    assert fs32.initial_state().w.dtype == np.float32
    with pytest.raises(ValueError):
        FlowSolver(dom, dtype=np.float16)
    with pytest.raises(ValueError):
        FlowSolver(dom, momentum_scheme="nope")


def test_hot_bottom_drives_an_upward_plume_and_closes_the_books():
    gr = box_grid(9, 9, 12, 0.01)
    fs = FlowSolver(gr)
    st = fs.initial_state()
    T = np.full(gr.shape, 330.0)
    T[3:6, 3:6, 0] += 2.0  # hot patch on the base
    res = None
    for _ in range(20):
        res = fs.step(st, T, WATER, 0.2)
    assert res is not None
    assert st.w[4, 4, 1:4].min() > 0.0  # rising above the patch
    assert res.max_divergence < 2e-6
    assert res.max_speed > 1e-4
    assert res.poisson_iterations > 0
    assert abs(res.q_adv.sum()) <= 1e-12 * np.abs(res.q_adv).sum()
    # the heat leaves the edge of the hot patch (negative source), and arrives above (positive)
    assert res.q_adv[3:6, 3:6, 0].min() < 0.0 and res.q_adv[:, :, 1:].max() > 0.0


def test_max_dt_limits():
    gr = box_grid(6, 6, 6, 0.01)
    fs = FlowSolver(gr)
    st = fs.initial_state()
    nu = 1.0e-4
    assert fs.max_dt(st, nu) == pytest.approx(0.01**2 / (12 * nu))  # viscous, at rest
    assert fs.max_dt(st, 0.0) == np.inf
    st.u[3, 3, 3] = 0.1
    # one face of 0.1 m/s gives the two cells it separates 0.05 m/s each
    assert fs.max_dt(st, nu) == pytest.approx(min(0.5 * 0.01 / 0.05, 0.01**2 / (12 * nu)))
    st.u[3, 3, 3] = 10.0
    assert fs.max_dt(st, nu) == pytest.approx(0.5 * 0.01 / 5.0)


def test_step_splits_into_substeps_of_at_most_max_dt():
    gr = box_grid(6, 6, 8, 0.01)
    fs = FlowSolver(gr)
    st = fs.initial_state()
    props = FlowProps(rho=1000.0, beta=4e-4, nu=1e-4, cp=4180.0, T_ref=330.0)
    T = np.full(gr.shape, 330.0)
    T[:, :, 0] += 1.0
    dt_max = 0.01**2 / (12 * 1e-4)
    res = fs.step(st, T, props, 10 * dt_max)
    assert res.substeps >= 10
    with pytest.raises(ValueError):
        fs.step(st, T, props, 0.0)


def test_slice_velocity_shape_dtype_and_values():
    gr = build_grid(0.05, 0.1, 0.01)
    fs = FlowSolver(gr)
    st = fs.initial_state()
    st.u[...] = 0.25 * fs.grid.mu
    st.w[...] = -0.5 * fs.grid.mw
    nx, ny, nz = gr.shape
    us, ws = fs.slice_velocity(st, ny // 2)
    assert us.shape == (nx, nz) and ws.shape == (nx, nz)
    assert us.dtype == np.float32 and ws.dtype == np.float32
    liq = gr.is_liquid[:, ny // 2, :]
    assert not us[~liq].any() and not ws[~liq].any()
    assert np.all(np.isfinite(us)) and np.all(np.isfinite(ws))
    # deep inside the liquid the centre value is the face value
    assert us[nx // 2, nz // 2] == pytest.approx(0.25)
    assert ws[nx // 2, nz // 2] == pytest.approx(-0.5)


def test_props_from_physics_water():
    p = props_from_physics(330.0, 1.000)
    assert 960 < p.rho < 990
    assert 3e-4 < p.beta < 6e-4
    assert 3e-7 < p.nu < 8e-7
    assert 4100 < p.cp < 4250
    assert p.T_ref == 330.0


def test_smagorinsky_eddy_viscosity_of_uniform_shear():
    """u = gamma z gives |S| = gamma, so nut = (Cs dx)^2 gamma in the interior."""
    gr = box_grid(8, 8, 8, 0.01)
    fs = FlowSolver(gr, smagorinsky_cs=0.17)
    g = fs.grid
    st = fs.initial_state()
    gamma = 3.0
    zc = (np.arange(8) + 0.5) * 0.01
    st.u[...] = gamma * zc[None, None, :] * g.mu
    K.smag_nut(
        fs._nut, st.u, st.v, st.w, g.mu, g.mv, g.mw, g.liq,
        (0.17 * 0.01) ** 2, 100.0, -1.0, -1.0, 1.0, 0.0, 0.0,
    )  # fmt: skip
    assert fs._nut[4, 4, 4] == pytest.approx((0.17 * 0.01) ** 2 * gamma, rel=1e-12)
    # at rest: zero
    st0 = fs.initial_state()
    K.smag_nut(
        fs._nut, st0.u, st0.v, st0.w, g.mu, g.mv, g.mw, g.liq,
        (0.17 * 0.01) ** 2, 100.0, -1.0, -1.0, 1.0, 0.0, 0.0,
    )  # fmt: skip
    assert not fs._nut.any()


def test_smagorinsky_returns_turbulent_conductivity_and_runs():
    gr = box_grid(10, 10, 12, 0.01)
    fs = FlowSolver(gr, smagorinsky_cs=0.15)
    st = fs.initial_state()
    T = np.full(gr.shape, 330.0)
    T[2:8, 2:8, 0] += 5.0
    res = None
    for _ in range(30):
        res = fs.step(st, T, WATER, 0.1)
    assert res is not None and res.k_turb is not None
    assert res.k_turb.shape == gr.shape and res.k_turb.min() >= 0.0
    assert res.k_turb.max() > 0.0
    assert st.nut_max > 0.0
    assert res.k_turb.max() == pytest.approx(
        WATER.rho * WATER.cp * st.nut_max / WATER.prandtl_t, rel=1e-12
    )
    assert abs(res.q_adv.sum()) <= 1e-12 * np.abs(res.q_adv).sum()
    assert fs.max_dt(st, WATER.nu) <= 0.01**2 / (12 * (WATER.nu + st.nut_max)) + 1e-18


def test_free_surface_has_no_shear_but_a_rigid_lid_does():
    """Viscous predictor of a plug flow in the top layer: free slip drags 3x less than no slip."""
    gr = box_grid(8, 5, 4, 0.01)
    results = {}
    for name, top in (("free", None), ("rigid", (0.0, 0.0))):
        fs = FlowSolver(gr, top_wall=top)
        g = fs.grid
        st = fs.initial_state()
        st.u[1:-1, :, -1] = 1.0  # plug flow in the top layer only
        us = np.zeros_like(st.u)
        nut = np.zeros(gr.shape)
        ftop, vou, _ = fs._top
        K.rhs_u(
            us, st.u, st.v, st.w, st.p, g.mu, nut, 100.0, 1e-9, 0.01, K.SCHEME_CENTRAL,
            -1.0, ftop, vou, False,
        )  # fmt: skip
        results[name] = (us[4, 2, -1] - 1.0) / 1e-9
    # z-diffusion: free slip -nu/dx^2; rigid lid -3 nu/dx^2 (ghost -c at the lid, 0 below).
    # In addition the y/x neighbours are equal to c so contribute nothing.
    assert results["rigid"] / results["free"] == pytest.approx(3.0, rel=2e-2)


def test_threads_do_not_change_the_result():
    import numba

    gr = build_grid(0.05, 0.1, 0.01)
    runs = []
    for nthreads in (1, 2):
        numba.set_num_threads(nthreads)
        try:
            fs = FlowSolver(gr)
            st = fs.initial_state()
            T = np.where(gr.is_liquid, 330.0, 0.0)
            T[gr.is_liquid & (np.arange(gr.shape[2])[None, None, :] == 0)] += 3.0
            for _ in range(5):
                res = fs.step(st, T, WATER, 0.1)
            runs.append((st.w.copy(), res.q_adv.copy()))
        finally:
            numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    np.testing.assert_array_equal(runs[0][1], runs[1][1])
