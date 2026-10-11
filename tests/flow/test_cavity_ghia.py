"""Lid-driven cavity at Re = 100 against Ghia, Ghia & Shin (1982) (WP21 validation case 2).

Ghia, U., Ghia, K. N., Shin, C. T., "High-Re solutions for incompressible flow using the
Navier-Stokes equations and a multigrid method", J. Comput. Phys. 48 (1982) 387-411,
Table I (u along the vertical centreline x = 0.5) and Table II (v along the horizontal
centreline y = 0.5), Re = 100.

The cavity is a thin 3-D box (one cell in y, free slip on the y boundaries, so the flow is exactly
two-dimensional) with a no-slip lid moving at U = 1 (the solver's test hook ``top_wall``).
"""

from __future__ import annotations

import numpy as np
import pytest
from flow_harness import box_grid

from thermosim.flow import FlowProps, FlowSolver

# y, u(x = 0.5, y)
GHIA_Y = np.array(
    [
        0.0547,
        0.0625,
        0.0703,
        0.1016,
        0.1719,
        0.2813,
        0.4531,
        0.5,
        0.6172,
        0.7344,
        0.8516,
        0.9531,
        0.9609,
        0.9688,
        0.9766,
    ]
)
GHIA_U = np.array(
    [
        -0.03717,
        -0.04192,
        -0.04775,
        -0.06434,
        -0.10150,
        -0.15662,
        -0.21090,
        -0.20581,
        -0.13641,
        0.00332,
        0.23151,
        0.68717,
        0.73722,
        0.78871,
        0.84123,
    ]
)
# x, v(x, y = 0.5)
GHIA_X = np.array(
    [
        0.0625,
        0.0703,
        0.0781,
        0.0938,
        0.1563,
        0.2266,
        0.2344,
        0.5,
        0.8047,
        0.8594,
        0.9063,
        0.9453,
        0.9531,
        0.9609,
        0.9688,
    ]
)
GHIA_V = np.array(
    [
        0.09233,
        0.10091,
        0.10890,
        0.12317,
        0.16077,
        0.17507,
        0.17527,
        0.05454,
        -0.24533,
        -0.22445,
        -0.16914,
        -0.10313,
        -0.08864,
        -0.07391,
        -0.05906,
    ]
)


def run_cavity(n: int, scheme: str = "vanleer", t_end: float = 40.0, dtype=np.float64):
    g = box_grid(n, 1, n, 1.0 / n)
    fs = FlowSolver(
        g, momentum_scheme=scheme, top_wall=(1.0, 0.0), side_slip=(False, True), dtype=dtype
    )
    st = fs.initial_state()
    props = FlowProps(rho=1.0, beta=0.0, nu=0.01, cp=1.0, T_ref=0.0, g=0.0)  # Re = 1 * 1 / 0.01
    T = np.zeros(g.shape)
    t = 0.0
    n_steps = 0
    prev = st.u.copy()
    while t < t_end:
        dt = min(fs.max_dt(st, props.nu), 0.02)
        res = fs.step(st, T, props, dt)
        t += dt
        n_steps += 1
        assert res.max_divergence < (2e-6 if dtype == np.float64 else 1e-4)
        if n_steps % 400 == 0:
            if np.abs(st.u - prev).max() / (400 * dt) < 1e-6:  # steady
                break
            prev = st.u.copy()
    return st, t


def centreline_errors(st, n):
    z = (np.arange(n) + 0.5) / n
    u_c = np.concatenate([[0.0], st.u[n // 2, 0, :], [1.0]])
    u_i = np.interp(GHIA_Y, np.concatenate([[0.0], z, [1.0]]), u_c)
    # Ghia's v (vertical velocity) is our w; z = 0.5 is the z-face n/2 (n even)
    x = (np.arange(n) + 0.5) / n
    w_line = st.w[:, 0, n // 2]
    w_c = np.concatenate([[0.0], w_line, [0.0]])
    w_i = np.interp(GHIA_X, np.concatenate([[0.0], x, [1.0]]), w_c)
    return u_i - GHIA_U, w_i - GHIA_V


def test_ghia_re100_centreline_u_within_5_percent():
    n = 32
    st, t = run_cavity(n)
    err_u, err_v = centreline_errors(st, n)
    big = np.abs(GHIA_U) >= 0.05
    rel = np.abs(err_u[big]) / np.abs(GHIA_U[big])
    assert rel.max() <= 0.05, f"max relative error {rel.max():.3%}"
    assert np.abs(err_u).max() <= 0.01  # absolute, in units of the lid speed
    # horizontal centreline (vertical velocity), same standard
    bigv = np.abs(GHIA_V) >= 0.05
    relv = np.abs(err_v[bigv]) / np.abs(GHIA_V[bigv])
    assert relv.max() <= 0.05, f"max relative error {relv.max():.3%}"
    assert np.abs(err_v).max() <= 0.01
    assert t < 40.0  # reached steady state


@pytest.mark.slow
@pytest.mark.parametrize("scheme", ["central", "vanleer"])
def test_ghia_re100_refined(scheme):
    n = 48
    st, _ = run_cavity(n, scheme)
    err_u, err_v = centreline_errors(st, n)
    big = np.abs(GHIA_U) >= 0.05
    assert (np.abs(err_u[big]) / np.abs(GHIA_U[big])).max() <= 0.03
    assert np.abs(err_v).max() <= 0.01


def test_cavity_float32_matches_float64():
    n = 16
    s64, _ = run_cavity(n, t_end=3.0)
    s32, _ = run_cavity(n, t_end=3.0, dtype=np.float32)
    assert s32.u.dtype == np.float32
    scale = np.abs(s64.u).max()
    assert np.abs(s32.u - s64.u).max() < 1e-3 * scale
