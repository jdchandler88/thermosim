"""Tests for thermosim.physics.correlations (WP03).

References
----------
The worked-example tests do NOT quote textbook example numbers (not reproduced from memory);
instead the Nusselt number is computed by hand in the test from the published correlation
(Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, 7th ed.) using the air properties
of Incropera Table A.4 at the film temperature, and compared with the module (which uses the
WP02 property fits, within ~2 % of Table A.4). Tolerance 5 % unless stated.

* Eq. 9.26 Churchill & Chu (vertical plate), Eqs. 9.30-9.32 (horizontal plates),
  Eq. 7.57 Churchill & Bernstein, Eqs. 7.30/7.38 (flat plate), Eq. 9.49 Globe & Dropkin.
* Evaporation sanity: ASHRAE HVAC Applications (2019) ch. 6, "Pool evaporation" equation
  w = A (p_w - p_a)(0.089 + 0.0782 V) / Y  (kg/s; A m^2, p kPa, V m/s, Y latent heat kJ/kg),
  quoted from memory; the test allows 25 % around it and requires 3-10 kW/m^2.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from thermosim.core.types import OuterFace
from thermosim.physics import correlations as cr
from thermosim.physics import properties as pr

P = 101325.0
G = 9.80665
SIGMA = 5.670374419e-8

# Incropera Table A.4, air at 1 atm, T = 350 K and 300 K
AIR350 = dict(k=0.030, nu=20.92e-6, alpha=29.9e-6, Pr=0.700)
AIR300 = dict(k=0.0263, nu=15.89e-6, alpha=22.5e-6, Pr=0.707)


def ra_hand(air: dict[str, float], Tf: float, dT: float, L: float) -> float:
    return G * (1.0 / Tf) * dT * L**3 / (air["nu"] * air["alpha"])


def test_vertical_churchill_chu_hand() -> None:
    Ra = ra_hand(AIR350, 350.0, 100.0, 0.5)
    Pr = AIR350["Pr"]
    Nu = (0.825 + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2
    ref = Nu * AIR350["k"] / 0.5
    assert cr.h_natural_vertical(400.0, 300.0, 0.5, P) == pytest.approx(ref, rel=0.05)
    # symmetric for a cold wall
    assert cr.h_natural_vertical(300.0, 400.0, 0.5, P) == pytest.approx(
        cr.h_natural_vertical(400.0, 300.0, 0.5, P), rel=1e-12
    )


@pytest.mark.parametrize("L,coef,expo", [(0.05, 0.54, 0.25), (0.5, 0.15, 1 / 3)])
def test_horizontal_up_hand(L: float, coef: float, expo: float) -> None:
    Ra = ra_hand(AIR350, 350.0, 100.0, L)
    assert (Ra <= 1e7) == (coef == 0.54)
    ref = coef * Ra**expo * AIR350["k"] / L
    assert cr.h_natural_horizontal_up(400.0, 300.0, L, P) == pytest.approx(ref, rel=0.05)


def test_horizontal_down_hand_and_swap() -> None:
    L = 0.2
    Ra = ra_hand(AIR350, 350.0, 100.0, L)
    ref_down = 0.27 * Ra**0.25 * AIR350["k"] / L
    assert cr.h_natural_horizontal_down(400.0, 300.0, L, P) == pytest.approx(ref_down, rel=0.05)
    # cold surfaces swap the correlations
    assert cr.h_natural_horizontal_up(300.0, 400.0, L, P) == pytest.approx(
        cr.h_natural_horizontal_down(400.0, 300.0, L, P), rel=1e-12
    )
    assert cr.h_natural_horizontal_down(300.0, 400.0, L, P) == pytest.approx(
        cr.h_natural_horizontal_up(400.0, 300.0, L, P), rel=1e-12
    )
    # hot-up transfers more than hot-down
    assert cr.h_natural_horizontal_up(400.0, 300.0, L, P) > cr.h_natural_horizontal_down(
        400.0, 300.0, L, P
    )


def test_small_dt_is_finite_positive() -> None:
    for fn in (cr.h_natural_vertical, cr.h_natural_horizontal_up, cr.h_natural_horizontal_down):
        for dT in (0.0, 1e-9, 5e-4, 1e-3, 2e-3, -1e-4):
            h = fn(300.0 + dT, 300.0, 0.3, P)
            assert np.isfinite(h) and h > 0.0
        # |dT| < 1 mK: floored at 1 W/m2K
        assert fn(300.0, 300.0, 0.3, P) >= 1.0


def test_forced_cylinder_hand() -> None:
    D, U, Tf = 0.3, 3.0, 300.0
    Re = U * D / AIR300["nu"]
    Pr = AIR300["Pr"]
    Nu = 0.3 + 0.62 * Re**0.5 * Pr ** (1 / 3) / (1 + (0.4 / Pr) ** (2 / 3)) ** 0.25 * (
        1 + (Re / 282000) ** (5 / 8)
    ) ** (4 / 5)
    ref = Nu * AIR300["k"] / D
    assert cr.h_forced_cylinder(Tf, Tf, D, U, P) == pytest.approx(ref, rel=0.05)
    assert cr.h_forced_cylinder(Tf, Tf, D, 0.0, P) > 0


def test_forced_plate_hand_laminar_turbulent() -> None:
    Pr3 = AIR300["Pr"] ** (1 / 3)
    L, U = 0.5, 3.0
    Re = U * L / AIR300["nu"]
    assert Re < 5e5
    ref = 0.664 * Re**0.5 * Pr3 * AIR300["k"] / L
    assert cr.h_forced_plate(300.0, 300.0, L, U, P) == pytest.approx(ref, rel=0.05)
    L, U = 2.0, 20.0
    Re = U * L / AIR300["nu"]
    assert Re > 5e5
    ref = (0.037 * Re**0.8 - 871.0) * Pr3 * AIR300["k"] / L
    assert cr.h_forced_plate(300.0, 300.0, L, U, P) == pytest.approx(ref, rel=0.05)
    # continuity at the transition Re = 5e5
    Uc = 5e5 * AIR300["nu"] / L
    lo = cr.h_forced_plate(300.0, 300.0, L, Uc * 0.999, P)
    hi = cr.h_forced_plate(300.0, 300.0, L, Uc * 1.001, P)
    assert hi == pytest.approx(lo, rel=0.02)


def test_mixed_and_radiation() -> None:
    assert cr.h_mixed(3.0, 4.0) == pytest.approx((27 + 64) ** (1 / 3))
    assert cr.h_mixed(5.0, 0.0) == pytest.approx(5.0)
    Ts, Ti, e = 363.15, 293.15, 0.85
    assert cr.h_radiation(Ts, Ti, e) == pytest.approx(e * SIGMA * (Ts**4 - Ti**4) / (Ts - Ti))
    assert cr.h_radiation(Ti, Ti, e) == pytest.approx(4 * e * SIGMA * Ti**3)
    assert 5.0 < cr.h_radiation(300.0, 300.0, 1.0) < 7.5


def test_h_outer_dispatch_and_sample_values() -> None:
    Ts, Ti, eps = 363.15, 293.15, 0.3
    side = cr.h_outer(Ts, Ti, OuterFace.SIDE, 0.4, 0.0, P, eps)
    rad = cr.h_radiation(Ts, Ti, eps)
    assert side == pytest.approx(cr.h_natural_vertical(Ts, Ti, 0.4, P) + rad, rel=1e-3)
    wind = cr.h_outer(Ts, Ti, OuterFace.SIDE, 0.4, 3.0, P, eps, diameter=0.32)
    assert wind > 1.5 * side
    mixed_ref = cr.h_mixed(
        cr.h_natural_vertical(Ts, Ti, 0.4, P), cr.h_forced_cylinder(Ts, Ti, 0.32, 3.0, P)
    )
    assert wind == pytest.approx(mixed_ref + rad, rel=1e-3)
    down = cr.h_outer(Ts, Ti, OuterFace.DOWN, 0.08, 0.0, P, eps)
    up = cr.h_outer(Ts, Ti, OuterFace.UP, 0.08, 0.0, P, eps)
    assert down == pytest.approx(cr.h_natural_horizontal_down(Ts, Ti, 0.08, P) + rad, rel=1e-3)
    assert up == pytest.approx(cr.h_natural_horizontal_up(Ts, Ti, 0.08, P) + rad, rel=1e-3)
    assert up > down
    assert 5.0 < side < 20.0


def test_h_outer_vectorized_matches_scalar() -> None:
    rng = np.random.default_rng(1)
    n = 50
    Ts = rng.uniform(290, 370, n)
    face = rng.integers(0, 3, n)
    L = rng.uniform(0.05, 0.5, n)
    eps = rng.uniform(0.05, 0.9, n)
    out = cr.h_outer(Ts, 293.15, face, L, 2.0, P, eps)
    assert isinstance(out, np.ndarray) and out.shape == (n,)
    for i in range(n):
        ref = cr.h_outer(Ts[i], 293.15, OuterFace(int(face[i])), L[i], 2.0, P, eps[i])
        assert out[i] == pytest.approx(ref, rel=2e-3)
    # large arrays use a 0.1 K film-temperature property grid
    big = cr.h_outer(np.full(5000, 350.0), 293.15, OuterFace.SIDE, 0.4, 0.0, P, 0.3)
    ref = cr.h_outer(350.0, 293.15, OuterFace.SIDE, 0.4, 0.0, P, 0.3)
    assert np.asarray(big)[0] == pytest.approx(ref, rel=1e-3)
    with pytest.raises(ValueError):
        cr.h_outer(350.0, 293.15, np.array([7]), 0.4, 0.0, P, 0.3)


def test_h_outer_performance_1e5_nodes() -> None:
    """Spec: 1e5 nodes in < 20 ms (about 16 ms on the dev box); asserted at 3x for CI noise."""
    rng = np.random.default_rng(0)
    n = 100_000
    Ts = rng.uniform(300, 370, n)
    face = rng.integers(0, 3, n)
    cr.h_outer(Ts, 293.15, face, 0.3, 1.0, P, 0.3)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        cr.h_outer(Ts, 293.15, face, 0.3, 1.0, P, 0.3)
        times.append(time.perf_counter() - t0)
    assert min(times) < 0.060


# ---------------------------------------------------------------- evaporation
def test_evaporation_sanity_95C() -> None:
    Ts, Ta, rh = 368.15, 293.15, 0.5
    h = cr.h_natural_horizontal_up(Ts, Ta, 0.30 / 4.0, P)  # D = 0.30 m, L = D/4
    q = cr.q_evaporation(Ts, Ta, rh, P, h)
    assert 3e3 < q < 10e3
    # ASHRAE pool-evaporation form for V = 0: w [kg/s/m2] = dp [kPa] * 0.089 / Y [kJ/kg]
    dp = (pr.saturation_pressure(Ts) - rh * pr.saturation_pressure(Ta)) / 1e3
    Y = pr.latent_heat_vaporization(Ts) / 1e3
    q_ashrae = dp * 0.089 / Y * Y * 1e3  # W/m2
    assert q == pytest.approx(q_ashrae, rel=0.25)
    m = cr.evaporation_mass_flux(Ts, Ta, rh, P, h)
    assert q == pytest.approx(m * pr.latent_heat_vaporization(Ts), rel=1e-12)


def test_evaporation_hand_formula() -> None:
    Ts, Ta, rh, h = 350.0, 293.15, 0.4, 7.0
    Tf = 0.5 * (Ts + Ta)
    a = pr.air_props(Tf, P)
    Le = a.diffusivity / pr.vapor_diffusivity_in_air(Tf, P)
    hm = h / (a.density * a.specific_heat * Le ** (2 / 3))
    Rv = 461.5
    drho = pr.saturation_pressure(Ts) / (Rv * Ts) - rh * pr.saturation_pressure(Ta) / (Rv * Ta)
    assert cr.evaporation_mass_flux(Ts, Ta, rh, P, h) == pytest.approx(hm * drho, rel=2e-3)
    assert 0.5 < Le < 1.2  # water vapour in air: Le ~ 0.85


def test_evaporation_floor_and_monotonic() -> None:
    assert cr.q_evaporation(293.15, 293.15, 1.0, P, 5.0) == pytest.approx(0.0, abs=1e-9)
    assert cr.q_evaporation(280.0, 293.15, 0.9, P, 5.0) == 0.0  # condensation floored
    T = np.linspace(300, 370, 8)
    q = np.asarray(cr.q_evaporation(T, 293.15, 0.5, P, 6.0))
    assert np.all(np.diff(q) > 0)
    assert cr.q_evaporation(350.0, 293.15, 0.2, P, 6.0) > cr.q_evaporation(
        350.0, 293.15, 0.8, P, 6.0
    )
    assert cr.q_evaporation(350.0, 293.15, 0.5, P, 12.0) > cr.q_evaporation(
        350.0, 293.15, 0.5, P, 6.0
    )


# ---------------------------------------------------------------- liquid side
def test_h_film_liquid_basics() -> None:
    base = cr.h_film_liquid(373.15, 363.15, OuterFace.UP, 0.1, 1.04)
    side = cr.h_film_liquid(373.15, 363.15, OuterFace.SIDE, 0.3, 1.04)
    assert 50.0 <= base <= 5000.0 and 50.0 <= side <= 5000.0
    # hand check of the unclipped base value: Nu = 0.15 Ra^(1/3) with wort properties at film T
    Tf = 368.15
    rho, cp = pr.wort_density(Tf, 1.04), pr.wort_specific_heat(Tf, 1.04)
    k, mu = pr.wort_conductivity(Tf, 1.04), pr.wort_viscosity(Tf, 1.04)
    beta = pr.wort_expansion(Tf, 1.04)
    Ra = G * beta * 10.0 * 0.1**3 * rho**2 * cp / (mu * k)
    assert base == pytest.approx(min(5000.0, max(50.0, 0.15 * Ra ** (1 / 3) * k / 0.1)), rel=1e-6)
    # colder wall facing up uses the weaker correlation
    assert cr.h_film_liquid(363.15, 373.15, OuterFace.UP, 0.1, 1.04) < base
    z = cr.h_film_liquid(350.0, 350.0, OuterFace.SIDE, 0.3, 1.04)
    assert np.isfinite(z) and z >= 50.0
    assert 50.0 <= cr.h_film_liquid(350.0, 350.0 + 1e-6, OuterFace.UP, 0.1, 1.04) <= 5000.0
    assert 50.0 <= cr.h_film_liquid(390.0, 280.0, OuterFace.UP, 1.0, 1.0) <= 5000.0
    arr = cr.h_film_liquid(
        np.array([373.0, 373.0, 360.0]),
        340.0,
        np.array([OuterFace.UP, OuterFace.SIDE, OuterFace.DOWN]),
        0.2,
        1.05,
    )
    assert np.asarray(arr).shape == (3,)


def test_h_headspace() -> None:
    h = cr.h_headspace(368.15, 360.0, 1.04, P)
    assert h > 20.0  # condensation dominates when the surface is hot
    assert cr.h_headspace(368.15, 350.0, 1.04, P) > 0
    assert cr.h_headspace(368.15, 360.0, 1.04, P) > 3 * cr.h_headspace(318.15, 310.0, 1.04, P)
    for Tl in (368.15, 370.0):  # dT = 0 and warmer lid
        v = cr.h_headspace(368.15, Tl, 1.04, P)
        assert np.isfinite(v) and v > 0
    assert np.asarray(cr.h_headspace(np.array([330.0, 360.0]), 320.0, 1.04, P)).shape == (2,)


# ---------------------------------------------------------------- effective conductivity
def ra_wort(Tb: float, dT: float, H: float, sg: float = 1.0) -> float:
    rho, cp = pr.wort_density(Tb, sg), pr.wort_specific_heat(Tb, sg)
    mu, beta = pr.wort_viscosity(Tb, sg), pr.wort_expansion(Tb, sg)
    k = pr.wort_conductivity(Tb, sg)
    return G * beta * dT * H**3 / ((mu / rho) * (k / (rho * cp)))


def test_k_effective() -> None:
    Tb = 333.15
    kmol = pr.wort_conductivity(Tb, 1.0)
    # 40 L water in a ~0.4 m pot: depth 0.32 m, hot wall 5 K above the bulk
    k = cr.k_effective(Tb, Tb + 5.0, 0.32, 1.0)
    assert k == pytest.approx(kmol * cr.K_EFF_C * ra_wort(Tb, 5.0, 0.32) ** (1 / 3), rel=1e-9)
    assert 20 * kmol < k < 1000 * kmol
    assert cr.K_EFF_C == 0.1
    assert cr.k_effective(Tb, Tb, 0.3, 1.0) == pytest.approx(kmol)
    assert cr.k_effective(Tb, Tb + 5, 0.3, 1.0, scale=0.0) == pytest.approx(kmol)
    assert cr.k_effective(Tb, Tb + 5, 0.3, 1.0, scale=0.5) == pytest.approx(
        kmol * 0.5 * cr.K_EFF_C * ra_wort(Tb, 5.0, 0.3) ** (1 / 3), rel=1e-9
    )
    arr = np.asarray(cr.k_effective(Tb, np.array([Tb, Tb + 1, Tb + 10]), 0.3, 1.05))
    assert np.all(np.diff(arr) >= 0) and arr[0] >= pr.wort_conductivity(Tb, 1.05)


# ---------------------------------------------------------------- robustness
@settings(max_examples=150, deadline=None)
@given(
    Ts=st.floats(275.0, 390.0),
    Ti=st.floats(275.0, 330.0),
    L=st.floats(0.01, 1.5),
    U=st.floats(0.0, 30.0),
    face=st.sampled_from([OuterFace.DOWN, OuterFace.SIDE, OuterFace.UP]),
    eps=st.floats(0.0, 1.0),
    rh=st.floats(0.0, 1.0),
    sg=st.floats(1.0, 1.13),
    p=st.floats(60e3, 105e3),
)
def test_never_nan_or_negative(
    Ts: float,
    Ti: float,
    L: float,
    U: float,
    face: OuterFace,
    eps: float,
    rh: float,
    sg: float,
    p: float,
) -> None:
    Tw = max(Ti, 275.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vals = [
            cr.h_natural_vertical(Ts, Ti, L, p),
            cr.h_natural_horizontal_up(Ts, Ti, L, p),
            cr.h_natural_horizontal_down(Ts, Ti, L, p),
            cr.h_forced_cylinder(Ts, Ti, L, U, p),
            cr.h_forced_plate(Ts, Ti, L, U, p),
            cr.h_radiation(Ts, Ti, eps),
            cr.h_outer(Ts, Ti, face, L, U, p, eps),
            cr.q_evaporation(Ts, Ti, rh, p, 5.0),
            cr.evaporation_mass_flux(Ts, Ti, rh, p, 5.0),
            cr.h_film_liquid(Ts, Tw, face, L, sg),
            cr.h_film_liquid(Tw, Ts, face, L, sg),
            cr.h_headspace(Ts, Tw, sg, p),
            cr.k_effective(Tw, Ts, L, sg),
        ]
    for v in vals:
        assert np.isfinite(v) and v >= 0.0
    assert cr.h_outer(Ts, Ti, face, L, U, p, eps) > 0.0
