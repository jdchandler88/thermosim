"""Differentially heated square cavity against de Vahl Davis (1983) (WP21 validation case 3).

de Vahl Davis, G., "Natural convection of air in a square cavity: a bench mark numerical
solution", Int. J. Numer. Meth. Fluids 3 (1983) 249-264. Pr = 0.71, hot left wall, cold right
wall, adiabatic top and bottom, no slip everywhere. Mean Nusselt number on the hot wall:
Ra = 1e3: 1.118, 1e4: 2.243, 1e5: 4.519. Peak velocities (in units of alpha / L, same paper):
Ra = 1e3: u 3.649, w 3.697; Ra = 1e4: u 16.178, w 19.617.

The cavity is a thin 3-D box (one cell in y with free-slip y boundaries: exactly 2-D) in
non-dimensional units (L = 1, g = beta = dT = 1, rho = cp = 1). The flow solver provides the
advective heat source; conduction with the fixed-temperature walls is integrated by the test.
"""

from __future__ import annotations

import numpy as np
import pytest
from flow_harness import box_grid, conduction_step

from thermosim.flow import FlowProps, FlowSolver

PR = 0.71


def run_cavity(n: int, ra: float, *, dtype=np.float64, max_steps: int = 40000):
    dx = 1.0 / n
    nu = np.sqrt(PR / ra)
    alpha = nu / PR
    g = box_grid(n, 1, n, dx)
    fs = FlowSolver(g, side_slip=(False, True), top_wall=(0.0, 0.0), dtype=dtype)
    st = fs.initial_state()
    props = FlowProps(rho=1.0, beta=1.0, nu=nu, cp=1.0, T_ref=0.5, g=1.0)
    T = np.full(g.shape, 0.5)
    walls = {"xlo": 1.0, "xhi": 0.0}
    last = None
    for step in range(1, max_steps + 1):
        dt = min(fs.max_dt(st, nu), 0.4 * dx * dx / (2 * alpha))
        res = fs.step(st, T, props, dt)
        T = conduction_step(T, res.q_adv, dt, alpha, dx, 1.0, walls)
        if step % 1000 == 0:
            nu_hot = nusselt_hot(T, dx)
            if last is not None and abs(nu_hot - last) < 1e-6:
                break
            last = nu_hot
    else:
        pytest.fail("no steady state")
    return st, T, alpha


def nusselt_hot(T: np.ndarray, dx: float) -> float:
    """Mean Nusselt number on the hot wall (x = 0): -dT/dx at the wall from the 3-point
    one-sided formula through T_wall = 1 and the first two cell values (L = dT = 1)."""
    grad = (-8.0 * 1.0 + 9.0 * T[0, 0, :] - T[1, 0, :]) / (3.0 * dx)
    return float(np.mean(-grad))


def nusselt_cold(T: np.ndarray, dx: float) -> float:
    grad = (8.0 * 0.0 - 9.0 * T[-1, 0, :] + T[-2, 0, :]) / (3.0 * dx)
    return float(np.mean(-grad))


@pytest.mark.parametrize(
    ("ra", "nu_ref", "umax_ref", "wmax_ref"),
    [(1e3, 1.118, 3.649, 3.697), (1e4, 2.243, 16.178, 19.617)],
)
def test_de_vahl_davis(ra, nu_ref, umax_ref, wmax_ref):
    n = 24
    st, T, alpha = run_cavity(n, ra)
    dx = 1.0 / n
    nu_hot, nu_cold = nusselt_hot(T, dx), nusselt_cold(T, dx)
    assert nu_hot == pytest.approx(nu_ref, rel=0.05)
    assert nu_cold == pytest.approx(nu_hot, rel=2e-2)  # heat in = heat out
    assert st.u.max() / alpha == pytest.approx(umax_ref, rel=0.05)
    assert st.w.max() / alpha == pytest.approx(wmax_ref, rel=0.05)


@pytest.mark.slow
def test_de_vahl_davis_ra1e5():
    st, T, alpha = run_cavity(48, 1e5, max_steps=80000)
    nu_hot = nusselt_hot(T, 1.0 / 48)
    assert nu_hot == pytest.approx(4.519, rel=0.05)
    assert st.w.max() / alpha == pytest.approx(68.59, rel=0.07)  # de Vahl Davis: 68.59


@pytest.mark.slow
def test_de_vahl_davis_converges_with_resolution():
    errs = []
    for n in (16, 32):
        _, T, _ = run_cavity(n, 1e4)
        errs.append(abs(nusselt_hot(T, 1.0 / n) / 2.243 - 1.0))
    assert errs[1] < errs[0]
    assert errs[1] < 0.02


def test_de_vahl_davis_float32_agrees():
    st64, T64, _ = run_cavity(16, 1e3)
    st32, T32, _ = run_cavity(16, 1e3, dtype=np.float32)
    assert nusselt_hot(T32, 1 / 16) == pytest.approx(nusselt_hot(T64, 1 / 16), rel=1e-3)
    assert float(st32.w.max()) == pytest.approx(float(st64.w.max()), rel=1e-2)
