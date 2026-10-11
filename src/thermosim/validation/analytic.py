"""Analytic reference solutions for conduction problems (pure functions, SI units).

Sources
-------
* H. S. Carslaw and J. C. Jaeger, *Conduction of Heat in Solids*, 2nd ed., Oxford, 1959:
  section 2.5 (semi-infinite solid, surface at constant temperature), section 2.7 (surface
  exchanging heat with a medium at constant temperature), section 10.3 (instantaneous
  point source in an infinite medium).
* F. P. Incropera et al., *Fundamentals of Heat and Mass Transfer*, 6th ed., Eqs. 5.57 and
  5.60 (the same two semi-infinite solutions) and Eq. 5.6 (lumped capacitance).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

F64 = NDArray[np.float64]


def semi_infinite_step(
    x: ArrayLike, t: ArrayLike, T_initial: float, T_surface: float, alpha: float
) -> F64:
    """Semi-infinite solid x >= 0 at T_initial, surface held at T_surface for t > 0.

    T(x, t) = T_s + (T_i - T_s) erf(x / (2 sqrt(alpha t)))   (Carslaw & Jaeger 2.5;
    Incropera Eq. 5.57). ``alpha`` is the thermal diffusivity in m^2/s. At t = 0 the
    solid is at T_initial for x > 0.
    """
    xa, ta = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(t, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(ta > 0.0, xa / (2.0 * np.sqrt(alpha * np.maximum(ta, 1e-300))), np.inf)
    return np.asarray(T_surface + (T_initial - T_surface) * special.erf(eta), dtype=np.float64)


def semi_infinite_convection(
    x: ArrayLike,
    t: ArrayLike,
    T_initial: float,
    T_fluid: float,
    h: float,
    k: float,
    alpha: float,
) -> F64:
    """Semi-infinite solid whose surface exchanges heat with a fluid at T_fluid.

    (T - T_i)/(T_f - T_i) = erfc(eta) - exp(h x/k + h^2 alpha t/k^2) erfc(eta + h sqrt(alpha t)/k),
    eta = x / (2 sqrt(alpha t))   (Carslaw & Jaeger 2.7; Incropera Eq. 5.60).
    Boundary condition: -k dT/dx = h (T - T_f) at x = 0. Evaluated with the scaled
    complementary error function so a large h x / k does not overflow:
    exp(2 eta b + b^2) erfc(eta + b) = erfcx(eta + b) exp(-eta^2), b = h sqrt(alpha t)/k.
    """
    xa, ta = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(t, dtype=float))
    pos = ta > 0.0
    sq = np.sqrt(alpha * np.maximum(ta, 1e-300))
    eta = np.where(pos, xa / (2.0 * sq), np.inf)
    b = h * sq / k
    with np.errstate(over="ignore", invalid="ignore"):
        second = special.erfcx(eta + b) * np.exp(-(eta**2))
    frac = np.where(pos, special.erfc(eta) - second, 0.0)
    return np.asarray(T_initial + (T_fluid - T_initial) * frac, dtype=np.float64)


def lumped_newton(t: ArrayLike, T0: float, T_inf: float, G: float, C: float, P: float = 0.0) -> F64:
    """Lumped body with capacity C (J/K), loss conductance G (W/K), constant source P (W).

    C dT/dt = P - G (T - T_inf)  =>  T(t) = T_inf + P/G + (T0 - T_inf - P/G) exp(-G t / C).
    (Newton's law of cooling; Incropera Eq. 5.6.) Requires G > 0.
    """
    ta = np.asarray(t, dtype=float)
    steady = T_inf + P / G
    return np.asarray(steady + (T0 - steady) * np.exp(-G * ta / C), dtype=np.float64)


def gaussian_pulse_3d(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: ArrayLike,
    energy: float,
    rho_c: float,
    alpha: float,
    sigma0: float = 0.0,
    T_background: float = 0.0,
) -> F64:
    """Temperature of an infinite medium after a Gaussian heat pulse, centred on the origin.

    The initial excess is a 3-D Gaussian of standard deviation ``sigma0`` (m) carrying
    ``energy`` joules; ``rho_c`` is the volumetric heat capacity (J/(m^3 K)). Diffusion
    widens it to sigma^2 = sigma0^2 + 2 alpha t:

        T - T_b = energy / (rho_c (2 pi sigma^2)^(3/2)) exp(-r^2 / (2 sigma^2))

    With sigma0 = 0 this is the instantaneous point source,
    energy / (rho_c (4 pi alpha t)^(3/2)) exp(-r^2 / (4 alpha t))   (Carslaw & Jaeger 10.3).
    Requires sigma0^2 + 2 alpha t > 0.
    """
    xa, ya, za, ta = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (x, y, z, t)))
    var = sigma0**2 + 2.0 * alpha * ta
    r2 = xa**2 + ya**2 + za**2
    amp = energy / (rho_c * (2.0 * np.pi * var) ** 1.5)
    return np.asarray(T_background + amp * np.exp(-r2 / (2.0 * var)), dtype=np.float64)
