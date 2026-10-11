"""Pressure projection, Poisson solver and masks (WP21 validation case 1)."""

from __future__ import annotations

import numpy as np
import pytest
from flow_harness import box_grid

from thermosim.flow import FlowGrid, FlowProps, FlowSolver, PoissonSolver
from thermosim.geometry.mesh import build_grid


@pytest.fixture(scope="module")
def cylinder():
    # R = 0.1 m, 0.205 m fill (partial top layer), 1 cm cells: 20 x 20 x 21 cells
    return build_grid(0.1, 0.205, 0.01)


def _numpy_divergence(grid, u, v, w):
    """Independent divergence (1/s): sum of outward area flux over the cell volume scale dx^3."""
    dx = grid.dx
    ax = np.zeros_like(u)
    ay = np.zeros_like(v)
    az = np.zeros_like(w)
    ax[1:-1] = grid.area_x
    ay[:, 1:-1] = grid.area_y
    az[:, :, 1:-1] = grid.area_z
    flux = (
        (ax * u)[1:]
        - (ax * u)[:-1]
        + (ay * v)[:, 1:]
        - (ay * v)[:, :-1]
        + (az * w)[:, :, 1:]
        - (az * w)[:, :, :-1]
    )
    return np.where(grid.is_liquid, flux / dx**3, 0.0)


def _random_state(fs, seed=0):
    rng = np.random.default_rng(seed)
    st = fs.initial_state()
    g = fs.grid
    st.u[...] = rng.standard_normal(st.u.shape) * g.mu
    st.v[...] = rng.standard_normal(st.v.shape) * g.mv
    st.w[...] = rng.standard_normal(st.w.shape) * g.mw
    return st


def test_divergence_after_projection_cylinder(cylinder):
    """A random field in a cylinder mask becomes divergence free to <= 1e-8 (relative to |u|max)."""
    fs = FlowSolver(cylinder, poisson_tol=1e-10)
    st = _random_state(fs)
    umax = max(np.abs(st.u).max(), np.abs(st.v).max(), np.abs(st.w).max())
    before = np.abs(_numpy_divergence(cylinder, st.u, st.v, st.w)).max() * cylinder.dx / umax
    assert before > 0.1  # the field is far from divergence free
    fs.project(st)
    umax = max(np.abs(st.u).max(), np.abs(st.v).max(), np.abs(st.w).max())
    after = np.abs(_numpy_divergence(cylinder, st.u, st.v, st.w)).max() * cylinder.dx / umax
    assert after <= 1e-8
    assert fs.max_divergence(st) <= 1e-8
    # the solver's own divergence agrees with the independent one up to the dx^2 vs volume
    # normalisation (full cells: identical)
    assert np.abs(fs.divergence(st)).max() * cylinder.dx / umax <= 1e-7


def test_projection_keeps_closed_faces_at_zero(cylinder):
    fs = FlowSolver(cylinder, poisson_tol=1e-10)
    st = _random_state(fs)
    st.u[...] = np.random.default_rng(1).standard_normal(st.u.shape)  # garbage on closed faces
    fs.project(st)
    g = fs.grid
    assert not np.any(st.u[~g.mu])
    assert not np.any(st.v[~g.mv])
    assert not np.any(st.w[~g.mw])


def test_projection_is_idempotent_and_removes_only_gradient(cylinder):
    fs = FlowSolver(cylinder, poisson_tol=1e-11)
    st = _random_state(fs, seed=3)
    fs.project(st)
    u0, v0, w0 = st.u.copy(), st.v.copy(), st.w.copy()
    fs.project(st)
    scale = max(np.abs(u0).max(), np.abs(v0).max(), np.abs(w0).max())
    assert np.abs(st.u - u0).max() < 1e-9 * scale
    assert np.abs(st.w - w0).max() < 1e-9 * scale


def test_poisson_solves_to_tolerance_and_multigrid_converges_fast(cylinder):
    g = FlowGrid.from_liquid_grid(cylinder)
    ps = PoissonSolver(g.wx, g.wy, g.wz, g.liq, tol=1e-9)
    assert len(ps.levels) >= 2
    rng = np.random.default_rng(5)
    b = rng.standard_normal(g.shape) * g.liq
    b0 = b.copy()
    x = np.zeros(g.shape)
    n = ps.solve(b, x)
    assert n <= 25
    # residual of A x = b (b has its mean removed)
    from thermosim.flow import kernels as K

    r = np.zeros(g.shape)
    K.residual(r, b, x, g.wx, g.wy, g.wz)
    assert np.linalg.norm(r) <= 2e-9 * np.linalg.norm(b)
    assert abs(b[g.liq].mean()) < 1e-12
    np.testing.assert_allclose(b[g.liq], (b0 - b0[g.liq].mean())[g.liq], atol=1e-12)


def test_poisson_dense_direct_on_small_grid():
    gr = box_grid(6, 5, 4, 0.1)
    g = FlowGrid.from_liquid_grid(gr)
    ps = PoissonSolver(g.wx, g.wy, g.wz, g.liq)
    assert len(ps.levels) == 1  # small enough for the direct solve
    rng = np.random.default_rng(2)
    b = rng.standard_normal(g.shape)
    x = np.zeros(g.shape)
    assert ps.solve(b, x) <= 2


def test_masks_and_weights_follow_the_liquid_grid(cylinder):
    g = FlowGrid.from_liquid_grid(cylinder)
    nx, ny, nz = g.shape
    assert g.wx.shape == (nx + 1, ny, nz) and g.wy.shape == (nx, ny + 1, nz)
    assert g.wz.shape == (nx, ny, nz + 1)
    assert not g.mu[0].any() and not g.mu[-1].any()
    assert not g.mw[:, :, 0].any() and not g.mw[:, :, -1].any()  # base and free surface closed
    np.testing.assert_allclose(g.wx[1:nx] * cylinder.dx**2, cylinder.area_x)
    np.testing.assert_allclose(g.wz[:, :, 1:nz] * cylinder.dx**2, cylinder.area_z)
    # a face is open exactly when both cells are liquid
    both = cylinder.is_liquid[:-1] & cylinder.is_liquid[1:]
    assert np.array_equal(g.mu[1:nx], both)


def test_float32_projection(cylinder):
    fs = FlowSolver(cylinder, dtype=np.float32)
    st = _random_state(fs)
    assert st.u.dtype == np.float32
    fs.project(st)
    assert st.u.dtype == np.float32
    assert fs.max_divergence(st) <= 5e-4


def test_rest_state_is_a_fixed_point(cylinder):
    """A uniform temperature produces no flow and no advective source."""
    fs = FlowSolver(cylinder)
    st = fs.initial_state()
    props = FlowProps(rho=1000.0, beta=2e-4, nu=1e-6, cp=4180.0, T_ref=330.0)
    T = np.where(cylinder.is_liquid, 330.0, 0.0)
    for _ in range(3):
        res = fs.step(st, T, props, 0.5)
    assert np.abs(st.u).max() == 0.0 and np.abs(st.v).max() == 0.0 and np.abs(st.w).max() == 0.0
    assert np.abs(res.q_adv).max() == 0.0
    assert res.poisson_iterations == 0
    assert res.k_turb is None


def test_stable_stratification_stays_at_rest():
    """Hot over cold is hydrostatic: buoyancy is balanced by the pressure gradient."""
    gr = box_grid(6, 6, 10, 0.02)
    fs = FlowSolver(gr, poisson_tol=1e-11)
    st = fs.initial_state()
    props = FlowProps(rho=1000.0, beta=2e-4, nu=1e-6, cp=4180.0, T_ref=300.0)
    T = 300.0 + 20.0 * (np.arange(10) / 9.0 - 0.5)[None, None, :] * np.ones(gr.shape)
    for _ in range(5):
        fs.step(st, T, props, 0.2)
    assert np.abs(st.w).max() < 1e-9
    assert np.abs(st.u).max() < 1e-9
