"""The real solver against closed-form solutions (WP10 Part B).

An independent check of the solver tests in ``tests/solver``: the problems and references here
come from ``thermosim.validation.analytic`` (Carslaw & Jaeger; Incropera), and the domains from
``thermosim.testing.synthetic``. Temperatures in K, lengths in m.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from thermosim.core.types import Domain, StepCoefficients, ThermalState
from thermosim.solver import ThermalSolver
from thermosim.testing.synthetic import box_domain, uniform_coefficients
from thermosim.validation.analytic import (
    gaussian_pulse_3d,
    lumped_newton,
    semi_infinite_convection,
)

K, RHO_C = 0.6, 4.18e6
ALPHA = K / RHO_C


def _advance(
    solver: ThermalSolver,
    state: ThermalState,
    coeffs: StepCoefficients,
    t_end: float,
    margin: float = 0.9,
) -> int:
    """Step to ``t_end`` with equal steps at ``margin`` of the stable step; returns the count."""
    n = math.ceil(t_end / (margin * solver.stable_dt(coeffs)))
    dt = t_end / n
    for _ in range(n):
        solver.step(state, coeffs, dt)
    return n


def _column_error(nz: int, dx: float, h: float, t_end: float) -> float:
    """Max error / initial difference of a 1-D column heated through a convective surface."""
    domain = box_domain(1, 1, nz, dx, shell=False)
    solver = ThermalSolver(domain)
    coeffs = uniform_coefficients(domain, k=K, rho_c=RHO_C, h_surface=h, T_amb=400.0)
    state = solver.initial_state(300.0, 300.0, 400.0)
    _advance(solver, state, coeffs, t_end)
    depth = nz * dx - (np.arange(nz) + 0.5) * dx
    exact = semi_infinite_convection(depth, t_end, 300.0, 400.0, h, K, ALPHA)
    assert exact[0] < 300.0 + 1e-3  # the front has not reached the closed bottom
    return float(np.max(np.abs(state.T_liquid[0, 0, :] - exact)) / 100.0)


def test_convective_surface_matches_carslaw_jaeger() -> None:
    """Semi-infinite solid with a convective surface (C&J 2.7): the surface cell's half-cell
    resistance in series with 1/h makes the discrete boundary condition second-order."""
    err = _column_error(nz=160, dx=0.005, h=20.0, t_end=6000.0)
    assert err < 0.005


def test_convective_surface_converges_with_the_cell_size() -> None:
    coarse = _column_error(nz=80, dx=0.01, h=20.0, t_end=6000.0)
    fine = _column_error(nz=160, dx=0.005, h=20.0, t_end=6000.0)
    assert fine < coarse
    assert coarse / fine > 2.0  # at least first order in dx


def _gaussian_box() -> tuple[Domain, np.ndarray]:
    n, dx, sigma0 = 33, 0.01, 0.02
    domain = box_domain(n, n, n, dx, shell=False)
    c = (np.arange(n) + 0.5) * dx - 0.5 * n * dx
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    r2 = x**2 + y**2 + z**2
    unit = np.exp(-r2 / (2.0 * sigma0**2))
    energy = 1.0e5  # J, shared out over the cells in proportion to the Gaussian
    dT0 = energy * unit / (RHO_C * dx**3 * unit.sum())
    return domain, dT0


def test_gaussian_pulse_diffusion_in_an_insulated_box() -> None:
    """3-D diffusion of a Gaussian pulse (C&J 10.3): the width grows as 2 alpha t."""
    domain, dT0 = _gaussian_box()
    n, dx, sigma0 = 33, 0.01, 0.02
    solver = ThermalSolver(domain)
    coeffs = uniform_coefficients(domain, k=K, rho_c=RHO_C, T_amb=300.0)  # all losses zero
    state = solver.initial_state(300.0, 300.0, 300.0)
    state.T_liquid[:] = 300.0 + dT0
    e0 = float(np.sum(RHO_C * dx**3 * (state.T_liquid - 300.0)))
    t_end = (0.04**2 - sigma0**2) / (2.0 * ALPHA)  # sigma doubles
    _advance(solver, state, coeffs, t_end)
    c = (np.arange(n) + 0.5) * dx - 0.5 * n * dx
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    exact = gaussian_pulse_3d(x, y, z, t_end, e0, RHO_C, ALPHA, sigma0=sigma0, T_background=300.0)
    dT = state.T_liquid - 300.0
    assert np.max(np.abs(state.T_liquid - exact)) < 0.01 * float(exact.max() - 300.0)
    # energy is conserved to round-off (insulated box)
    assert float(np.sum(RHO_C * dx**3 * dT)) == pytest.approx(e0, rel=1e-12)
    # second moment: variance per axis = sigma0^2 + 2 alpha t
    var = float(np.sum(dT * x**2) / np.sum(dT))
    assert var == pytest.approx(sigma0**2 + 2.0 * ALPHA * t_end, rel=0.01)


def test_well_mixed_body_follows_newton_heating() -> None:
    """A well-mixed liquid with a uniform source and a convective top: T(t) of lumped Newton
    heating, C dT/dt = P - G (T - T_inf), with G from the surface conductance in series with
    the half-cell resistance (negligible for k = 1000)."""
    nx = ny = nz = 4
    dx, h, power = 0.05, 30.0, 25.0
    domain = box_domain(nx, ny, nz, dx, shell=False)
    solver = ThermalSolver(domain)
    t_inf = 293.15
    coeffs = uniform_coefficients(domain, k=1.0e3, rho_c=RHO_C, h_surface=h, T_amb=t_inf)
    n_cells = nx * ny * nz
    coeffs.q_cells[:] = power / n_cells
    area = nx * ny * dx**2
    g = 1.0 / (1.0 / (h * area) + (0.5 * dx / 1.0e3) / area)
    cap = RHO_C * n_cells * dx**3
    tau = cap / g
    state = solver.initial_state(t_inf, t_inf, t_inf)
    t = 0.0
    for target in (0.5 * tau, 1.0 * tau, 3.0 * tau):
        _advance(solver, state, coeffs, target - t)
        t = target
        exact = float(lumped_newton(t, t_inf, t_inf, g, cap, power))
        mean = float(state.T_liquid.mean())
        assert (mean - t_inf) == pytest.approx(exact - t_inf, rel=2e-3)
        # nearly uniform (the source and the loss are both spread/at the top; k is large)
        assert float(np.ptp(state.T_liquid)) < 0.01 * (exact - t_inf) + 1e-6


def test_steady_state_with_a_wall_is_the_series_conductance() -> None:
    """Heat through film -> wall -> ambient: at steady state the liquid sits P / G above ambient,
    with G the series conductance of the film and the outer wall conductance of every node."""
    domain = box_domain(5, 5, 5, 0.04)
    solver = ThermalSolver(domain)
    t_amb = 293.15
    h_film, h_out = 400.0, 8.0
    coeffs = uniform_coefficients(
        domain, k=300.0, rho_c=RHO_C, h_film=h_film, h_out=h_out, T_amb=t_amb
    )
    coeffs.h_dry[:] = 0.0  # inner dry wall and the top surface do not exchange heat
    power = 40.0
    coeffs.q_cells[:] = power / domain.grid.volume.size
    shell = domain.shell
    g_out = 1.0 / (
        shell.thickness / (2.0 * shell.k * shell.area) + 1.0 / (h_out * shell.outer_area)
    )
    g_film = 1.0 / (
        1.0 / (h_film * domain.shell.couple_area)
        + shell.thickness[shell.couple_shell]
        / (2.0 * shell.k[shell.couple_shell] * domain.shell.couple_area)
    )
    # per-node series path liquid -> node -> ambient (nodes with several couplings: sum of films)
    g_in = np.bincount(shell.couple_shell, weights=g_film, minlength=shell.n)
    g_tot = float(np.sum(g_in * g_out / (g_in + g_out)))
    state = solver.initial_state(t_amb, t_amb, t_amb)
    cap = float(np.sum(RHO_C * domain.grid.volume)) + float(
        np.sum(shell.rho_c * shell.thickness * shell.area)
    )
    t_end = 8.0 * cap / g_tot
    n = math.ceil(t_end / (0.5 * solver.stable_dt(coeffs)))
    dt = t_end / n
    for _ in range(n):
        solver.step(state, coeffs, dt)
    assert float(state.T_liquid.mean()) - t_amb == pytest.approx(power / g_tot, rel=2e-3)
