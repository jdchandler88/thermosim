"""The analytic reference solutions satisfy their PDEs, boundary conditions and limits."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy import special

from thermosim.validation.analytic import (
    gaussian_pulse_3d,
    lumped_newton,
    semi_infinite_convection,
    semi_infinite_step,
)

ALPHA = 1.4e-7  # water-like diffusivity, m^2/s
K = 0.6


def _heat_residual_1d(fn: Callable[[float, float], float], x: float, t: float) -> float:
    """|T_t - alpha T_xx| / (alpha |T_xx|) by central differences."""
    dx, dt = x * 1e-3, t * 1e-4
    T_t = (fn(x, t + dt) - fn(x, t - dt)) / (2 * dt)
    T_xx = (fn(x + dx, t) - 2 * fn(x, t) + fn(x - dx, t)) / dx**2
    return float(abs(T_t - ALPHA * T_xx) / (ALPHA * abs(T_xx)))


@pytest.mark.parametrize("x,t", [(1e-3, 50.0), (5e-3, 200.0), (2e-2, 3000.0)])
def test_step_solves_heat_equation(x: float, t: float) -> None:
    def f(xx: float, tt: float) -> float:
        return float(semi_infinite_step(xx, tt, 20.0, 100.0, ALPHA))

    assert _heat_residual_1d(f, x, t) < 1e-4


@pytest.mark.parametrize("x,t", [(1e-3, 50.0), (5e-3, 200.0), (2e-2, 3000.0)])
def test_convection_solves_heat_equation(x: float, t: float) -> None:
    def f(xx: float, tt: float) -> float:
        return float(semi_infinite_convection(xx, tt, 20.0, 100.0, 500.0, K, ALPHA))

    assert _heat_residual_1d(f, x, t) < 1e-4


def test_step_limits() -> None:
    assert float(semi_infinite_step(0.0, 10.0, 20.0, 100.0, ALPHA)) == pytest.approx(100.0)
    assert float(semi_infinite_step(1.0, 10.0, 20.0, 100.0, ALPHA)) == pytest.approx(20.0)
    assert float(semi_infinite_step(0.01, 0.0, 20.0, 100.0, ALPHA)) == pytest.approx(20.0)
    # Similarity: the value depends on x / sqrt(t) only.
    a = semi_infinite_step(0.002, 10.0, 0.0, 1.0, ALPHA)
    b = semi_infinite_step(0.004, 40.0, 0.0, 1.0, ALPHA)
    assert float(a) == pytest.approx(float(b), rel=1e-12)
    # Known value: eta = 0.5.
    x = 2 * 0.5 * np.sqrt(ALPHA * 100.0)
    assert float(semi_infinite_step(x, 100.0, 0.0, 1.0, ALPHA)) == pytest.approx(
        1.0 - float(special.erf(0.5))
    )


def test_convection_boundary_condition() -> None:
    h, t = 500.0, 300.0
    dx = 1e-7
    T0 = float(semi_infinite_convection(0.0, t, 20.0, 100.0, h, K, ALPHA))
    T1 = float(semi_infinite_convection(dx, t, 20.0, 100.0, h, K, ALPHA))
    flux_cond = -K * (T1 - T0) / dx  # conduction into the solid
    flux_conv = h * (100.0 - T0)  # convection from the fluid
    assert flux_cond == pytest.approx(flux_conv, rel=1e-4)


def test_convection_limits() -> None:
    x, t = 3e-3, 100.0
    # h -> infinity recovers the surface step; h -> 0 leaves the solid untouched.
    big = semi_infinite_convection(x, t, 20.0, 100.0, 1e9, K, ALPHA)
    step = semi_infinite_step(x, t, 20.0, 100.0, ALPHA)
    assert float(big) == pytest.approx(float(step), rel=1e-5)
    small = semi_infinite_convection(x, t, 20.0, 100.0, 1e-9, K, ALPHA)
    assert float(small) == pytest.approx(20.0, abs=1e-6)
    # t -> 0 and far field.
    assert float(semi_infinite_convection(x, 0.0, 20.0, 100.0, 500.0, K, ALPHA)) == 20.0
    far = semi_infinite_convection(10.0, t, 20.0, 100.0, 500.0, K, ALPHA)
    assert float(far) == pytest.approx(20.0)
    # No overflow for a large Biot number.
    assert np.isfinite(semi_infinite_convection(0.5, 1e5, 20.0, 100.0, 1e6, K, ALPHA))


def test_lumped_newton_limits_and_ode() -> None:
    C, G, P = 1.5e5, 12.0, 800.0
    assert float(lumped_newton(0.0, 20.0, 15.0, G, C, P)) == pytest.approx(20.0)
    assert float(lumped_newton(1e9, 20.0, 15.0, G, C, P)) == pytest.approx(15.0 + P / G)
    t = np.array([100.0, 1000.0, 5000.0])
    dt = 1e-3
    T = lumped_newton(t, 20.0, 15.0, G, C, P)
    up = lumped_newton(t + dt, 20.0, 15.0, G, C, P)
    dn = lumped_newton(t - dt, 20.0, 15.0, G, C, P)
    np.testing.assert_allclose(C * (up - dn) / (2 * dt), P - G * (T - 15.0), rtol=1e-6)
    # One time constant: 63.2 % of the way to the steady state.
    T_tau = float(lumped_newton(C / G, 20.0, 15.0, G, C, P))
    assert (T_tau - 20.0) / (15.0 + P / G - 20.0) == pytest.approx(1 - np.exp(-1.0))


def test_gaussian_pulse_pde_energy_and_limits() -> None:
    E, rc, sigma0 = 1.0e4, 4.18e6, 0.01
    x0, y0, z0, t0 = 0.013, -0.007, 0.02, 40.0

    def f(x: float, y: float, z: float, t: float) -> float:
        return float(gaussian_pulse_3d(x, y, z, t, E, rc, ALPHA, sigma0))

    h, dt = 1e-4, 1e-3
    T_t = (f(x0, y0, z0, t0 + dt) - f(x0, y0, z0, t0 - dt)) / (2 * dt)
    lap = (
        f(x0 + h, y0, z0, t0)
        + f(x0 - h, y0, z0, t0)
        + f(x0, y0 + h, z0, t0)
        + f(x0, y0 - h, z0, t0)
        + f(x0, y0, z0 + h, t0)
        + f(x0, y0, z0 - h, t0)
        - 6 * f(x0, y0, z0, t0)
    ) / h**2
    assert T_t == pytest.approx(ALPHA * lap, rel=1e-4)

    # Energy is conserved: the integral of rho c (T - Tb) dV equals E (midpoint rule).
    n, half = 61, 0.12
    g = (np.arange(n) + 0.5) * (2 * half / n) - half
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    T = gaussian_pulse_3d(X, Y, Z, 30.0, E, rc, ALPHA, sigma0, T_background=20.0)
    assert np.sum(rc * (T - 20.0)) * (2 * half / n) ** 3 == pytest.approx(E, rel=1e-3)

    # Point-source limit: peak E / (rho c (4 pi alpha t)^1.5); background is added.
    peak = float(gaussian_pulse_3d(0.0, 0.0, 0.0, 10.0, E, rc, ALPHA, 0.0, 5.0))
    assert peak == pytest.approx(5.0 + E / (rc * (4 * np.pi * ALPHA * 10.0) ** 1.5))
    far = float(gaussian_pulse_3d(1.0, 0.0, 0.0, 10.0, E, rc, ALPHA, 0.0, 5.0))
    assert far == pytest.approx(5.0)
