"""Heat-transfer correlations and effective conductivity (CONTRACTS.md section 7, WP03).

Conventions
-----------
* SI units, temperatures in kelvin, pressures in Pa. Every function is vectorized (NumPy
  broadcasting; scalars in give ``float`` out).
* ``h_*`` return W/(m^2 K) and are always finite and positive; ``q_*`` return W/m^2 with
  decision D9: positive means heat LEAVING the surface. The linearized flux of a surface is
  therefore ``h * (T_surface - T_ref)``.
* Air properties are evaluated at the film temperature (T_s + T_inf)/2 via
  :func:`thermosim.physics.properties.air_props`; liquid (wort) properties at the film or bulk
  temperature as stated per function.
* Where the surface-to-fluid temperature difference is below ``DT_MIN`` (1 mK) the correlations
  are evaluated at ``DT_MIN`` and the result floored at ``H_FLOOR_SMALL_DT`` so that the limit
  dT -> 0 is a small, finite coefficient (never NaN or 0).

The correlations and their ranges are tabulated in ``docs/physics/correlations.md``.

Sources
-------
* Churchill, S.W., Chu, H.H.S., Int. J. Heat Mass Transfer 18 (1975) 1323 (Eq. 9.26, Incropera).
* Lloyd, J.R., Moran, W.R., J. Heat Transfer 96 (1974) 443 (Eqs. 9.30-9.32, Incropera; Cengel
  Eq. 9-25..9-27).
* Churchill, S.W., Bernstein, M., J. Heat Transfer 99 (1977) 300 (Eq. 7.57, Incropera).
* Flat plate, Eqs. 7.30 and 7.38 of Incropera (laminar; mixed boundary layer, Re_x,c = 5e5).
* Globe, S., Dropkin, D., J. Heat Transfer 81 (1959) 24 (Eq. 9.49, Incropera) for the
  horizontal enclosure.
* Chilton, T.H., Colburn, A.P., Ind. Eng. Chem. 26 (1934) 1183 (heat/mass transfer analogy,
  Incropera Eq. 6.60 / Cengel Eq. 14-xx).
* Rohsenow, W.M., Trans. ASME 74 (1952) 969 (nucleate pool boiling; Incropera Eq. 10.5, Table
  10.1 for C_sf) and Zuber, N., Trans. ASME 80 (1958) 711 (critical heat flux, Incropera
  Eq. 10.7), for the nucleate-boiling wall flux in :func:`h_film_liquid`.
* IAPWS Release on the Surface Tension of Ordinary Water Substance (1994).
* Effective conductivity: Rayleigh-Benard scaling Nu = C Ra^(1/3) (e.g. Kraichnan 1962;
  Grossmann & Lohse 2000); a modelling closure to be calibrated (WP23).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np

from thermosim.core.types import F64, OuterFace
from thermosim.physics import properties as pr

Num = float | F64

DT_MIN = 1e-3  # K, below this |dT| the natural-convection result is floored
H_FLOOR_SMALL_DT = 1.0  # W/(m^2 K), floor of a natural-convection h when |dT| < DT_MIN
H_FILM_MIN = 50.0  # W/(m^2 K), clip of the liquid-side film coefficient
H_FILM_MAX = 5000.0  # natural-convection part only; nucleate boiling can exceed it
_BISECT_ITER = 9  # surface-temperature bisection of boiling rows (0.08 K for a 40 K bracket)
_FILM_T_MIN, _FILM_T_MAX = 273.2, 392.9  # K, range of the wort property fits (with margin)
C_SF_WATER = 0.013  # Rohsenow surface-fluid constant, water on polished stainless/copper
K_EFF_C = 0.1  # Nu = C Ra^(1/3) coefficient of the effective conductivity closure
HEADSPACE_GAP_M = 0.10  # m, assumed liquid-surface-to-lid distance of the enclosed cavity
R_VAPOR = 8.314462618 / 0.01801528  # J/(kg K), water vapour gas constant
_LUT_MIN_SIZE = 2000  # arrays larger than this use the film-temperature lookup
_LUT_STEP = 0.1  # K
_G = pr.G
_SIGMA = pr.STEFAN_BOLTZMANN


def _a(x: Num) -> F64:
    return np.asarray(x, dtype=np.float64)


def _ret(x: F64) -> Num:
    return float(x) if x.ndim == 0 else x


class _Film:
    """Air film properties and driving temperature difference, broadcast to a common shape."""

    __slots__ = ("_idx", "_pr_grid", "alpha", "beta", "dT", "dT_signed", "k", "nu", "pr")

    def __init__(self, T_s: Num, T_inf: Num, p: Num) -> None:
        Ts, Ti = _a(T_s), _a(T_inf)
        Tf = 0.5 * (Ts + Ti)
        self.dT_signed = Ts - Ti
        self.dT = np.abs(self.dT_signed)
        self._idx: F64 | None = None
        self._pr_grid: F64 | None = None
        if Tf.size > _LUT_MIN_SIZE and _a(p).ndim == 0:
            # Large arrays: evaluate the properties on a 0.1 K grid of film temperatures and
            # gather (nearest node, error <= 0.05 K, <0.02 % in k) -- much cheaper than
            # interpolating the property tables for every node.
            lo, hi = float(Tf.min()), float(Tf.max())
            n = int(np.ceil((hi - lo) / _LUT_STEP)) + 1
            ap = pr.air_props(np.asarray(lo + _LUT_STEP * np.arange(n), dtype=np.float64), p)
            idx = np.rint((Tf - lo) / _LUT_STEP).astype(np.intp)
            self.k = _a(ap.conductivity)[idx]
            self.nu = _a(ap.kinematic_viscosity)[idx]
            self.alpha = _a(ap.diffusivity)[idx]
            self._pr_grid = _a(ap.prandtl)
            self._idx = idx  # type: ignore[assignment]
            self.pr = self._pr_grid[idx]
            self.beta = 1.0 / np.clip(Tf, pr.AIR_T_MIN_K, pr.AIR_T_MAX_K)
            return
        ap = pr.air_props(Tf, p)
        self.k = _a(ap.conductivity)
        self.nu = _a(ap.kinematic_viscosity)
        self.alpha = _a(ap.diffusivity)
        self.pr = _a(ap.prandtl)
        self.beta = _a(ap.expansion)

    def pr_fn(self, fn: Callable[[F64], F64]) -> F64:
        """fn(Pr), evaluated on the lookup grid when there is one (cheap for big arrays)."""
        if self._pr_grid is not None:
            return fn(self._pr_grid)[self._idx]  # type: ignore[index,no-any-return]
        return fn(self.pr)

    def rayleigh(self, L: Num) -> F64:
        dT = np.maximum(self.dT, DT_MIN)
        return _G * self.beta * dT * _a(L) ** 3 / (self.nu * self.alpha)  # type: ignore[no-any-return]


def _small_dt_floor(h: F64, dT: F64) -> F64:
    return np.where(dT < DT_MIN, np.maximum(h, H_FLOOR_SMALL_DT), h)


# ---------------------------------------------------------------- air-side natural convection
def _cc_den(Pr: F64) -> F64:
    return (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)  # type: ignore[no-any-return]


def _nu_churchill_chu(Ra: F64, den: F64) -> F64:
    """Churchill & Chu (1975), full-range vertical plate, Incropera Eq. 9.26."""
    return (  # type: ignore[no-any-return]
        0.825 + 0.387 * np.cbrt(np.sqrt(Ra)) / den
    ) ** 2


def _nu_hot_up(Ra: F64) -> F64:
    """Hot surface facing up (Incropera Eqs. 9.30/9.31, Lloyd & Moran 1974)."""
    return np.where(Ra <= 1e7, 0.54 * np.sqrt(np.sqrt(Ra)), 0.15 * np.cbrt(Ra))


def _nu_hot_down(Ra: F64) -> F64:
    """Hot surface facing down (Incropera Eq. 9.32, Lloyd & Moran 1974)."""
    return 0.27 * np.sqrt(np.sqrt(Ra))  # type: ignore[no-any-return]


def _h_vertical(f: _Film, height: Num) -> F64:
    h = _nu_churchill_chu(f.rayleigh(height), f.pr_fn(_cc_den)) * f.k / _a(height)
    return _small_dt_floor(h, f.dT)


def _h_up(f: _Film, L: Num) -> F64:
    """Horizontal plate whose exposed face looks UP. Hot -> 0.54/0.15, cold -> 0.27."""
    Ra = f.rayleigh(L)
    hot = f.dT_signed >= 0.0
    Nu = np.where(hot, _nu_hot_up(Ra), _nu_hot_down(Ra))
    return _small_dt_floor(Nu * f.k / _a(L), f.dT)  # type: ignore[no-any-return]


def _h_down(f: _Film, L: Num) -> F64:
    """Horizontal plate whose exposed face looks DOWN. Hot -> 0.27, cold -> 0.54/0.15."""
    Ra = f.rayleigh(L)
    hot = f.dT_signed >= 0.0
    Nu = np.where(hot, _nu_hot_down(Ra), _nu_hot_up(Ra))
    return _small_dt_floor(Nu * f.k / _a(L), f.dT)  # type: ignore[no-any-return]


def h_natural_vertical(T_s: Num, T_inf: Num, height: Num, p: Num) -> Num:
    """Churchill-Chu natural convection on a vertical wall, air side (Incropera Eq. 9.26).

    Nu = {0.825 + 0.387 Ra^(1/6) / [1 + (0.492/Pr)^(9/16)]^(8/27)}^2, valid for all Ra.
    T_s, T_inf in K, height (plate height) in m, p in Pa; air properties at the film
    temperature. Symmetric in the sign of T_s - T_inf. Returns W/(m^2 K).
    """
    return _ret(_h_vertical(_Film(T_s, T_inf, p), height))


def h_natural_horizontal_up(T_s: Num, T_inf: Num, L: Num, p: Num) -> Num:
    """Natural convection on a horizontal plate whose exposed face looks UP, air side.

    Hot surface (T_s > T_inf): Nu = 0.54 Ra^(1/4) for Ra <= 1e7, 0.15 Ra^(1/3) above
    (Incropera Eqs. 9.30, 9.31; Lloyd & Moran 1974). Cold surface: the face acts like a hot
    face looking down, Nu = 0.27 Ra^(1/4) (Eq. 9.32). L = A/P in m. Returns W/(m^2 K).
    """
    return _ret(_h_up(_Film(T_s, T_inf, p), L))


def h_natural_horizontal_down(T_s: Num, T_inf: Num, L: Num, p: Num) -> Num:
    """Natural convection on a horizontal plate whose exposed face looks DOWN, air side.

    Hot surface: Nu = 0.27 Ra^(1/4) (Incropera Eq. 9.32). Cold surface: acts like a hot face
    looking up, Nu = 0.54 Ra^(1/4) / 0.15 Ra^(1/3) (Eqs. 9.30, 9.31). L = A/P in m.
    Returns W/(m^2 K).
    """
    return _ret(_h_down(_Film(T_s, T_inf, p), L))


# ---------------------------------------------------------------- air-side forced convection
def _cb_pre(Pr: F64) -> F64:
    """Pr-dependent factor of Churchill-Bernstein: Pr^(1/3) / [1 + (0.4/Pr)^(2/3)]^(1/4)."""
    return np.cbrt(Pr) / (1.0 + (0.4 / Pr) ** (2.0 / 3.0)) ** 0.25  # type: ignore[no-any-return]


def _h_cylinder(f: _Film, D: Num, U: Num) -> F64:
    Re = _a(U) * _a(D) / f.nu
    pre = f.pr_fn(_cb_pre)
    Nu = 0.3 + (0.62 * np.sqrt(Re) * pre) * (1.0 + (Re / 282000.0) ** (5.0 / 8.0)) ** 0.8
    return Nu * f.k / _a(D)  # type: ignore[no-any-return]


def _h_plate(f: _Film, L: Num, U: Num) -> F64:
    Re = _a(U) * _a(L) / f.nu
    Pr3 = f.pr_fn(np.cbrt)
    lam = 0.664 * np.sqrt(Re) * Pr3
    mixed = (0.037 * np.maximum(Re, 1.0) ** 0.8 - 871.0) * Pr3
    Nu = np.where(Re <= 5e5, lam, mixed)
    return Nu * f.k / _a(L)  # type: ignore[no-any-return]


def h_forced_cylinder(T_s: Num, T_inf: Num, D: Num, U: Num, p: Num) -> Num:
    """Churchill-Bernstein (1977) crossflow over a cylinder (Incropera Eq. 7.57).

    Nu = 0.3 + 0.62 Re^(1/2) Pr^(1/3) / [1 + (0.4/Pr)^(2/3)]^(1/4) * [1 + (Re/282000)^(5/8)]^(4/5),
    valid for Re*Pr > 0.2, all Re. D in m, U in m/s (U = 0 gives the small Nu = 0.3 limit).
    Returns W/(m^2 K).
    """
    return _ret(_h_cylinder(_Film(T_s, T_inf, p), D, np.maximum(_a(U), 0.0)))


def h_forced_plate(T_s: Num, T_inf: Num, L: Num, U: Num, p: Num) -> Num:
    """Flat plate in parallel flow, average coefficient over length L (Incropera 7.30, 7.38).

    Re_L <= 5e5: Nu = 0.664 Re^(1/2) Pr^(1/3) (laminar). Re_L > 5e5: Nu = (0.037 Re^(4/5) - 871)
    Pr^(1/3) (mixed boundary layer, 0.6 < Pr < 60). L in m, U in m/s. Returns W/(m^2 K).
    """
    return _ret(_h_plate(_Film(T_s, T_inf, p), L, np.maximum(_a(U), 0.0)))


def h_mixed(h_nat: Num, h_forced: Num) -> Num:
    """Mixed natural/forced convection (Churchill 1977 cubic blending, n = 3):
    (h_n^3 + h_f^3)^(1/3). Both in W/(m^2 K)."""
    return _ret((_a(h_nat) ** 3 + _a(h_forced) ** 3) ** (1.0 / 3.0))


def h_radiation(T_s: Num, T_inf: Num, emissivity: Num) -> Num:
    """Linearized radiation coefficient to large surroundings at T_inf, W/(m^2 K):
    eps sigma (T_s^2 + T_inf^2)(T_s + T_inf), the exact factorization of
    eps sigma (T_s^4 - T_inf^4) / (T_s - T_inf). Temperatures in K."""
    Ts, Ti = _a(T_s), _a(T_inf)
    return _ret(_a(emissivity) * _SIGMA * (Ts * Ts + Ti * Ti) * (Ts + Ti))


def h_outer(
    T_s: Num,
    T_inf: Num,
    face: OuterFace | np.ndarray,
    char_length: Num,
    wind: Num,
    p: Num,
    emissivity: Num,
    diameter: Num | None = None,
) -> Num:
    """Outer-surface convection + radiation coefficient to ambient air, W/(m^2 K).

    ``face`` (scalar or per-node array of OuterFace) selects:

    * SIDE: Churchill-Chu vertical wall (height ``char_length``) mixed with Churchill-Bernstein
      crossflow over a cylinder of diameter ``diameter`` (m; falls back to ``char_length`` if
      None -- see the contract change request in docs/physics/correlations.md);
    * DOWN: hot-face-down plate mixed with parallel-flow plate, L = ``char_length``;
    * UP: hot-face-up plate mixed with parallel-flow plate, L = ``char_length``;

    plus :func:`h_radiation` with ``emissivity`` to surroundings at ``T_inf``. ``wind`` in m/s
    (0 = still air); mixing by :func:`h_mixed`.
    """
    shape = np.broadcast_shapes(
        _a(T_s).shape,
        _a(T_inf).shape,
        _a(face).shape,
        _a(char_length).shape,
        _a(wind).shape,
        _a(p).shape,
        _a(emissivity).shape,
        () if diameter is None else _a(diameter).shape,
    )

    def bc(x: Num) -> F64:
        return np.broadcast_to(_a(x), shape).ravel()

    Ts, Ti, fc = bc(T_s), bc(T_inf), bc(face).astype(np.int64)
    L, U, eps = bc(char_length), np.maximum(bc(wind), 0.0), bc(emissivity)
    pp = _a(p) if _a(p).ndim == 0 else bc(p)
    D = L if diameter is None else bc(diameter)
    codes = (int(OuterFace.SIDE), int(OuterFace.DOWN), int(OuterFace.UP))
    if (fc < min(codes)).any() or (fc > max(codes)).any():
        raise ValueError("face contains values that are not OuterFace members")
    # One film evaluation for all nodes; the three branches are cheap, so evaluate each on
    # every node and select (faster than boolean-mask gathers for large arrays).
    f = _Film(Ts, Ti, pp)
    side = fc == int(OuterFace.SIDE)
    Ra = f.rayleigh(L)
    hot = f.dT_signed >= 0.0
    up_like = (fc == int(OuterFace.UP)) == hot  # hot face up, or cold face down
    r4 = np.sqrt(np.sqrt(Ra))
    Nu_plate = np.where(up_like, np.where(Ra <= 1e7, 0.54 * r4, 0.15 * np.cbrt(Ra)), 0.27 * r4)
    Nu = np.where(side, _nu_churchill_chu(Ra, f.pr_fn(_cc_den)), Nu_plate)
    hn = _small_dt_floor(Nu * f.k / L, f.dT)
    if not U.any():  # still air: the forced term vanishes
        h = hn + _a(h_radiation(Ts, Ti, eps))
    else:
        hf = np.where(side, _h_cylinder(f, D, U), _h_plate(f, L, U))
        h = np.cbrt(hn * hn * hn + hf * hf * hf) + _a(h_radiation(Ts, Ti, eps))
    return _ret(h.reshape(shape))


# ---------------------------------------------------------------- evaporation
def _vapor_density(T: F64, rh: Num | F64 = 1.0) -> F64:
    """Water-vapour density (ideal gas) at saturation scaled by relative humidity, kg/m^3."""
    return _a(rh) * _a(pr.saturation_pressure(T)) / (R_VAPOR * T)  # type: ignore[no-any-return]


def _mass_transfer_coefficient(T_a: F64, T_b: F64, p: Num, h_conv: Num) -> F64:
    """h_m (m/s) from h_conv by the Chilton-Colburn analogy,
    h_m = h_conv / (rho cp Le^(2/3)), Le = alpha / D_AB, air properties at (T_a+T_b)/2."""
    Tf = 0.5 * (T_a + T_b)
    ap = pr.air_props(Tf, p)
    D_ab = _a(pr.vapor_diffusivity_in_air(Tf, p))
    le = _a(ap.diffusivity) / D_ab
    return _a(h_conv) / (_a(ap.density) * _a(ap.specific_heat) * le ** (2.0 / 3.0))  # type: ignore[no-any-return]


def evaporation_mass_flux(T_surf: Num, T_amb: Num, rh: Num, p: Num, h_conv: Num) -> Num:
    """Evaporative mass flux from a free water surface into humid air, kg/(m^2 s), >= 0.

    m = h_m (rho_v,s - rho_v,inf), with h_m = h_conv / (rho cp Le^(2/3)) (Chilton-Colburn,
    Incropera Eq. 6.60), Le = alpha/D_AB; vapour densities from the saturation pressure at
    T_surf and rh * p_sat(T_amb) via the ideal-gas law. T in K, rh in 0..1, p in Pa,
    h_conv in W/(m^2 K). Pure-water vapour pressure is used (the water-activity lowering of
    wort, ~1 %, is neglected). Condensation (negative flux) is floored to 0.
    """
    Ts, Ta = _a(T_surf), _a(T_amb)
    hm = _mass_transfer_coefficient(Ts, Ta, p, h_conv)
    m = hm * (_vapor_density(Ts) - _vapor_density(Ta, rh))
    return _ret(np.maximum(m, 0.0))


def q_evaporation(T_surf: Num, T_amb: Num, rh: Num, p: Num, h_conv: Num) -> Num:
    """Evaporative heat flux leaving the surface, W/m^2 (>= 0): q = m_dot * h_fg(T_surf).
    See :func:`evaporation_mass_flux` for the model and units."""
    m = _a(evaporation_mass_flux(T_surf, T_amb, rh, p, h_conv))
    return _ret(m * _a(pr.latent_heat_vaporization(_a(T_surf))))


# ---------------------------------------------------------------- liquid side
def _film_T(T_a: F64, T_b: F64) -> F64:
    """Film temperature of two temperatures that may lie outside the range of the property fits
    (a wall above 120 degC, a superheated cell): each is clipped before averaging, so the film
    shifts by a few kelvin at most while the temperature DIFFERENCE stays the true one."""
    return 0.5 * (np.clip(T_a, _FILM_T_MIN, _FILM_T_MAX) + np.clip(T_b, _FILM_T_MIN, _FILM_T_MAX))  # type: ignore[no-any-return]


@lru_cache(maxsize=8)
def _wort_lut(sg: float) -> tuple[F64, F64, F64, F64, F64]:
    """Wort (rho, cp, k, mu, beta) on a 0.1 K grid of film temperatures, for one gravity."""
    n = int(np.ceil((_FILM_T_MAX - _FILM_T_MIN) / _LUT_STEP)) + 1
    T = np.linspace(_FILM_T_MIN, _FILM_T_MIN + _LUT_STEP * (n - 1), n, dtype=np.float64)
    return (
        _a(pr.wort_density(T, sg)),
        _a(pr.wort_specific_heat(T, sg)),
        _a(pr.wort_conductivity(T, sg)),
        _a(pr.wort_viscosity(T, sg)),
        np.maximum(_a(pr.wort_expansion(T, sg)), 1e-6),  # water beta < 0 below 4 degC
    )


class _WortFilm:
    """Wort properties at film temperatures (already inside the validity range of the fits).

    Large arrays at one gravity are served from a 0.1 K lookup table (nearest node, error
    < 0.05 K in temperature), which is much cheaper than interpolating the property tables for
    every node.
    """

    __slots__ = ("beta", "cp", "k", "mu", "rho")

    def __init__(self, Tf: F64, sg: F64) -> None:
        if Tf.size > _LUT_MIN_SIZE and sg.ndim == 0:
            idx = np.rint((Tf - _FILM_T_MIN) / _LUT_STEP).astype(np.intp)
            self.rho, self.cp, self.k, self.mu, self.beta = (t[idx] for t in _wort_lut(float(sg)))
            return
        self.rho = _a(pr.wort_density(Tf, sg))
        self.cp = _a(pr.wort_specific_heat(Tf, sg))
        self.k = _a(pr.wort_conductivity(Tf, sg))
        self.mu = _a(pr.wort_viscosity(Tf, sg))
        self.beta = np.maximum(_a(pr.wort_expansion(Tf, sg)), 1e-6)  # water beta < 0 below 4 degC


def _h_natural_liquid(dT: F64, f: _WortFilm, face: F64, L: F64) -> F64:
    """Natural-convection film coefficient from the signed wall-bulk difference ``dT``."""
    pr_l = f.cp * f.mu / f.k
    Ra = _G * f.beta * np.maximum(np.abs(dT), DT_MIN) * L**3 * f.rho**2 * f.cp / (f.mu * f.k)
    hot = dT >= 0.0
    # cold wall facing DOWN behaves like hot facing UP; hot facing DOWN like cold facing UP
    is_up = face == int(OuterFace.UP)
    is_down = face == int(OuterFace.DOWN)
    enhanced = (is_up & hot) | (is_down & ~hot)
    Nu_enh = 0.15 * np.cbrt(Ra)
    Nu_weak = 0.27 * np.sqrt(np.sqrt(Ra))
    Nu_side = _nu_churchill_chu(Ra, _cc_den(pr_l))
    Nu = np.where(is_up | is_down, np.where(enhanced, Nu_enh, Nu_weak), Nu_side)
    return np.clip(Nu * f.k / L, H_FILM_MIN, H_FILM_MAX)  # type: ignore[no-any-return]


def _surface_tension_water(T: F64) -> F64:
    """Surface tension of water, N/m (IAPWS 1994): 0.2358 (1-Tr)^1.256 (1 - 0.625 (1-Tr))."""
    tau = 1.0 - np.asarray(T, dtype=np.float64) / 647.096
    return 0.2358 * tau**1.256 * (1.0 - 0.625 * tau)  # type: ignore[no-any-return]


def _rohsenow_constants(T_sat: F64, sg: F64) -> tuple[F64, F64]:
    """(K, q_chf) of the nucleate-boiling flux q = K (T_wall - T_sat)^3, capped at q_chf.

    Rohsenow's correlation, Incropera, Fundamentals of Heat and Mass Transfer, Eq. 10.5
    (Rohsenow 1952): q = mu_l h_fg [g (rho_l - rho_v) / sigma]^(1/2)
    [cp_l dT_e / (C_sf h_fg Pr_l^n)]^3 with dT_e = T_wall - T_sat, n = 1 (water),
    C_sf = ``C_SF_WATER`` (water on polished stainless steel or copper, Incropera Table 10.1).
    Saturated-liquid properties at the local boiling point ``T_sat`` of the wort, vapour
    density from the ideal-gas law. The cap is the critical heat flux (Zuber 1958, Incropera
    Eq. 10.7); film boiling is not modelled.
    """
    f = _WortFilm(_film_T(T_sat, T_sat), sg)
    h_fg = _a(pr.latent_heat_vaporization(np.clip(T_sat, _FILM_T_MIN, _FILM_T_MAX)))
    rho_v = _a(pr.saturation_pressure(np.clip(T_sat, _FILM_T_MIN, _FILM_T_MAX))) / (R_VAPOR * T_sat)
    sigma = _surface_tension_water(T_sat)
    pr_l = f.cp * f.mu / f.k
    K = (
        f.mu
        * h_fg
        * np.sqrt(_G * (f.rho - rho_v) / sigma)
        * (f.cp / (C_SF_WATER * h_fg * pr_l)) ** 3
    )
    q_chf = 0.131 * h_fg * np.sqrt(rho_v) * (sigma * _G * (f.rho - rho_v)) ** 0.25
    return K, q_chf  # type: ignore[return-value]


def _blend_boiling(h_nc: F64, Tw: F64, Tb: F64, T_sat: F64, K: F64, q_chf: F64) -> F64:
    """Smooth maximum (p = 4) of the natural-convection and the linearised boiling coefficient."""
    dT_e = np.maximum(Tw - T_sat, 0.0)
    q_b = np.minimum(K * dT_e * dT_e * dT_e, q_chf)
    h_b = np.where(Tw > Tb, q_b / np.maximum(Tw - Tb, DT_MIN), 0.0)
    h2, hb2 = h_nc * h_nc, h_b * h_b
    return np.sqrt(np.sqrt(h2 * h2 + hb2 * hb2))  # type: ignore[no-any-return]


def h_film_liquid(
    T_wall: Num,
    T_bulk: Num,
    orientation: OuterFace | np.ndarray,
    L: Num,
    sg: Num,
    p: Num = 101325.0,
) -> Num:
    """Wall -> liquid film coefficient, W/(m^2 K), q = h (T_wall - T_bulk).

    ``orientation`` is the direction the wall faces INTO the liquid: UP (base, heated from
    below), SIDE (vertical wall), DOWN. Wort properties at the film temperature
    (T_wall + T_bulk)/2 (each temperature clipped to 0-120 degC for the property fits only,
    the temperature difference is the true one); Ra = g beta |dT| L^3 rho^2 cp / (mu k).

    * hot wall facing UP / cold wall facing DOWN: Nu = 0.15 Ra^(1/3) (Incropera Eq. 9.31 form,
      turbulent horizontal plate, applied for all Ra);
    * hot wall facing DOWN / cold wall facing UP: Nu = 0.27 Ra^(1/4) (Eq. 9.32);
    * SIDE: Churchill-Chu with L = fill height.

    The natural-convection value is clipped to [50, 5000] W/(m^2 K).

    **Nucleate boiling at the wall.** Where T_wall exceeds the local boiling point T_sat(p, sg)
    (``p`` in Pa, default 101325), a hot wall drives nucleate (sub-cooled) boiling, which
    carries the flux at a superheat of only a few kelvin. The flux is Rohsenow's correlation
    (Incropera Eq. 10.5, C_sf = 0.013, n = 1; see `_rohsenow_constants`), linearised as
    h_b = q_b / (T_wall - T_bulk) and combined with the natural-convection value by a smooth
    maximum h = (h_nc^4 + h_b^4)^(1/4). Because q_b is proportional to (T_wall - T_sat)^3 the
    combination is continuous with a continuous slope at T_sat. ``h_b`` is capped by the
    critical heat flux. This only keeps heated walls from superheating unphysically; bulk
    boiling, vapour holdup and boil-off remain out of scope (REQUIREMENTS L4).
    """
    Tw, Tb, face, Ls = np.broadcast_arrays(
        _a(T_wall), _a(T_bulk), _a(orientation).astype(np.int64), _a(L)
    )
    s = _a(sg)
    f = _WortFilm(_film_T(Tw, Tb), s)
    h = _h_natural_liquid(Tw - Tb, f, face, Ls)
    T_sat = _a(pr.wort_boiling_point(sg, p))  # scalar when sg and p are scalars
    if bool(np.any(Tw > T_sat)):
        K, q_chf = _rohsenow_constants(T_sat, s)
        h = _blend_boiling(h, Tw, Tb, T_sat, K, q_chf)
    return _ret(h)


def h_film_through_wall(
    T_node: Num,
    T_bulk: Num,
    orientation: OuterFace | np.ndarray,
    L: Num,
    sg: Num,
    r_wall: Num,
    p: Num = 101325.0,
) -> Num:
    """Film coefficient of a wall node that conducts through a half-wall resistance, W/(m^2 K).

    ``T_node`` is the temperature of a shell node (the wall mid-plane) and ``r_wall`` =
    t / (2 k_wall) in m^2 K/W the conduction resistance between the node and the wetted
    surface. Where the node is below the boiling point this is just :func:`h_film_liquid` at
    the node temperature. Where it is above, nucleate boiling acts on the wetted surface,
    which is colder than the node by the conduction drop, so the surface temperature T_w is
    found from the series balance ``h(T_w) (T_w - T_bulk) = (T_node - T_w) / r_wall`` by
    bisection (the left side rises and the right side falls with T_w, so it converges
    unconditionally) with the film properties frozen at (T_sat + T_bulk)/2.
    """
    Tn, Tb, face, Ls, rw = np.broadcast_arrays(
        _a(T_node), _a(T_bulk), _a(orientation).astype(np.int64), _a(L), _a(r_wall)
    )
    s, pp = _a(sg), _a(p)
    h = np.array(h_film_liquid(Tn, Tb, face, Ls, s, pp), dtype=np.float64)
    T_sat = _a(pr.wort_boiling_point(s, pp))  # scalar unless sg or p is an array
    rows = np.flatnonzero(Tn > T_sat)
    if rows.size == 0:
        return _ret(h)
    Tn_h, Tb_h, face_h, L_h, rw_h = (x[rows] for x in (Tn, Tb, face, Ls, rw))
    s_h = s if s.ndim == 0 else np.broadcast_to(s, Tn.shape)[rows]
    ts_h = T_sat if T_sat.ndim == 0 else np.broadcast_to(T_sat, Tn.shape)[rows]
    f = _WortFilm(_film_T(ts_h, Tb_h), s_h)
    K, q_chf = _rohsenow_constants(ts_h, s_h)

    def film(Tw: F64) -> F64:
        return _blend_boiling(
            _h_natural_liquid(Tw - Tb_h, f, face_h, L_h), Tw, Tb_h, ts_h, K, q_chf
        )

    lo, hi = np.minimum(Tb_h, Tn_h), Tn_h.copy()
    for _ in range(_BISECT_ITER):
        Tw = 0.5 * (lo + hi)
        rising = film(Tw) * (Tw - Tb_h) < (Tn_h - Tw) / rw_h  # film carries less: T_w rises
        lo = np.where(rising, Tw, lo)
        hi = np.where(rising, hi, Tw)
    h[rows] = film(0.5 * (lo + hi))
    return _ret(h)


def h_headspace(
    T_liquid_surface: Num, T_lid: Num, sg: Num, p: Num, gap: Num = HEADSPACE_GAP_M
) -> Num:
    """Covered free surface <-> lid/headspace coefficient, W/(m^2 K), q = h (T_surf - T_lid).

    Sum of (a) enclosed-cavity natural convection of the humid-air layer (gap
    ``HEADSPACE_GAP_M``, horizontal layer heated from below): Globe & Dropkin (1959),
    Nu = max(1, 0.069 Ra^(1/3) Pr^0.074) (Incropera Eq. 9.49); Nu = 1 (conduction) when the
    lid is warmer than the surface; and (b) evaporation-condensation: net vapour flux
    h_m (a_w rho_v,sat(T_surf) - rho_v,sat(T_lid)) with h_m from the Chilton-Colburn analogy
    on the cavity convection coefficient, times h_fg(T_surf), divided by (T_surf - T_lid).
    The saturated headspace condenses on the lid; the water activity a_w of the wort comes
    from its boiling-point elevation. Air properties are those of dry air (humidity ignored).
    Assumptions: well-mixed saturated gas layer, lid at uniform T_lid, no non-condensables
    other than air, gap fixed (needs a contract change to pass the true headspace height).
    """
    Ts, Tl, s = np.broadcast_arrays(_a(T_liquid_surface), _a(T_lid), _a(sg))
    pp = _a(p)
    f = _Film(Ts, Tl, pp)
    L = np.maximum(np.asarray(gap, dtype=float), 1e-3)
    Ra = f.rayleigh(L)
    Nu = np.where(f.dT_signed >= 0.0, np.maximum(1.0, 0.069 * Ra ** (1.0 / 3.0) * f.pr**0.074), 1.0)
    h_cav = Nu * f.k / L
    hm = _mass_transfer_coefficient(Ts, Tl, pp, h_cav)
    # water activity of the wort from its boiling-point elevation: a_w = p / p_sat(T_boil)
    pb = np.clip(pp, 1e3, 1.9e5)
    aw = np.minimum(1.0, pb / _a(pr.saturation_pressure(_a(pr.wort_boiling_point(s, pb)))))
    small = np.abs(f.dT_signed) < DT_MIN
    Tl_eval = np.where(small, Ts - DT_MIN, Tl)
    dT_eff = Ts - Tl_eval
    m = hm * (aw * _vapor_density(Ts) - _vapor_density(Tl_eval))
    h_ev = np.maximum(m * _a(pr.latent_heat_vaporization(Ts)) / dT_eff, 0.0)
    return _ret(h_cav + h_ev)


# ---------------------------------------------------------------- effective conductivity
def k_effective(T_bulk: Num, T_hot: Num, depth: Num, sg: Num, scale: float = 1.0) -> Num:
    """Effective (turbulent-mixing) conductivity of the liquid, W/(m K).

    k_eff = k_mol * max(1, scale * C * Ra^(1/3)),   Ra = g beta |T_hot - T_bulk| H^3 / (nu alpha),
    C = ``K_EFF_C`` (0.1), wort properties at T_bulk, H = ``depth`` (m).

    MODELLING CLOSURE, not a measured property: it stands in for natural-convection mixing in
    the conduction-only Phase 1 model, using a Rayleigh-Benard-type scaling. It is to be
    calibrated against measurements in WP23 (Phase 2) through ``scale``. Applying ``scale`` to
    the Nusselt number (not to the molecular part) keeps k_eff >= k_mol for any scale >= 0.
    """
    Tb, Th, H, s = np.broadcast_arrays(_a(T_bulk), _a(T_hot), _a(depth), _a(sg))
    rho = _a(pr.wort_density(Tb, s))
    cp = _a(pr.wort_specific_heat(Tb, s))
    k = _a(pr.wort_conductivity(Tb, s))
    mu = _a(pr.wort_viscosity(Tb, s))
    beta = np.maximum(_a(pr.wort_expansion(Tb, s)), 1e-6)  # water beta < 0 below 4 degC
    nu = mu / rho
    alpha = k / (rho * cp)
    Ra = _G * beta * np.abs(Th - Tb) * H**3 / (nu * alpha)
    Nu = K_EFF_C * Ra ** (1.0 / 3.0)
    return _ret(k * np.maximum(1.0, max(float(scale), 0.0) * Nu))
