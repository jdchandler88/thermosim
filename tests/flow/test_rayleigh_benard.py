"""Rayleigh-Benard onset (WP21 validation case 5).

A layer heated from below (rigid bottom and rigid lid, Pr = 1) is linearly unstable above
Ra_c = 1707.76 (Chandrasekhar 1961, Hydrodynamic and Hydromagnetic Stability, Table XIII; the
critical wavenumber is a_c = 3.117, wavelength 2.016 H). The box has free-slip, adiabatic side walls
at x = 0 and x = W (symmetry planes), so a mode cos(pi x / W) is an exact eigenmode of the
infinite layer when W is half a wavelength; W / H = 25 / 24 = 1.042 is within 3 % of 1.008 and
the neutral curve is flat near its minimum. The thin box (one cell in y with
free-slip y boundaries) makes the problem exactly 2-D.

The test measures the growth rate sigma of max|w| from a small perturbation: it decays well
below Ra_c, grows well above it, and (slow) the zero crossing interpolated from the growth rates
brackets Ra_c within 3 %. The steady state of the discrete problem does not depend on the time
step, so the interpolated Ra_c measures the spatial discretisation error.
"""

from __future__ import annotations

import numpy as np
import pytest
from flow_harness import box_grid, conduction_step

from thermosim.flow import FlowProps, FlowSolver

RA_C = 1707.76
PR = 1.0


def growth_rate(ra: float, *, nz: int = 24, nx: int = 25, t_end: float = 120.0, eps: float = 1e-4):
    """Slope of ln(max|w|) over the last 60 % of ``t_end`` (H = dT = g = beta = 1)."""
    dx = 1.0 / nz
    nu = np.sqrt(PR / ra)
    alpha = nu / PR
    g = box_grid(nx, 1, nz, dx)
    fs = FlowSolver(g, side_slip=(True, True), top_wall=(0.0, 0.0))
    st = fs.initial_state()
    props = FlowProps(rho=1.0, beta=1.0, nu=nu, cp=1.0, T_ref=0.5, g=1.0)
    x = (np.arange(nx) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dx
    pert = np.cos(np.pi * x / (nx * dx))[:, None, None] * np.sin(np.pi * z)[None, None, :]
    T = (1.0 - z)[None, None, :] + eps * pert
    walls = {"zlo": 1.0, "zhi": 0.0}
    t = 0.0
    ts: list[float] = []
    ws: list[float] = []
    n = 0
    while t < t_end:
        dt = min(fs.max_dt(st, nu), 0.4 * dx * dx / (2 * alpha))
        res = fs.step(st, T, props, dt)
        T = conduction_step(T, res.q_adv, dt, alpha, dx, 1.0, walls)
        t += dt
        n += 1
        if n % 50 == 0:
            ts.append(t)
            ws.append(float(np.abs(st.w).max()))
    tt, ww = np.array(ts), np.array(ws)
    sel = (tt > 0.4 * t_end) & (ww > 1e-300)
    return float(np.polyfit(tt[sel], np.log(ww[sel]), 1)[0]), float(ww[-1])


def test_still_below_and_convecting_above_onset():
    sig_lo, w_lo = growth_rate(1000.0, nz=16, nx=17, t_end=30.0)
    sig_hi, w_hi = growth_rate(3000.0, nz=16, nx=17, t_end=30.0)
    assert sig_lo < -0.05 and w_lo < 1e-6  # perturbation dies out: fluid stays still
    assert sig_hi > 0.05 and w_hi > 1e-3  # perturbation grows into convection


@pytest.mark.slow
def test_critical_rayleigh_number_from_growth_rates():
    ras = (1600.0, 1800.0)
    sig = [growth_rate(ra)[0] for ra in ras]
    assert sig[0] < 0.0 < sig[1]
    ra_c = ras[0] - sig[0] * (ras[1] - ras[0]) / (sig[1] - sig[0])
    assert ra_c == pytest.approx(RA_C, rel=0.03)
