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
* Effective conductivity: Rayleigh-Benard scaling Nu = C Ra^(1/3) (e.g. Kraichnan 1962;
  Grossmann & Lohse 2000); a modelling closure to be calibrated (WP23).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from thermosim.core.types import F64, OuterFace
from thermosim.physics import properties as pr

Num = float | F64

DT_MIN = 1e-3  # K, below this |dT| the natural-convection result is floored
H_FLOOR_SMALL_DT = 1.0  # W/(m^2 K), floor of a natural-convection h when |dT| < DT_MIN
H_FILM_MIN = 50.0  # W/(m^2 K), clip of the liquid-side film coefficient
H_FILM_MAX = 5000.0
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
def _wort_film(T_wall: Num, T_bulk: Num, sg: Num) -> tuple[F64, F64, F64, F64, F64, F64]:
    Tw, Tb, s = np.broadcast_arrays(_a(T_wall), _a(T_bulk), _a(sg))
    Tf = 0.5 * (Tw + Tb)
    rho = _a(pr.wort_density(Tf, s))
    cp = _a(pr.wort_specific_heat(Tf, s))
    k = _a(pr.wort_conductivity(Tf, s))
    mu = _a(pr.wort_viscosity(Tf, s))
    beta = np.maximum(_a(pr.wort_expansion(Tf, s)), 1e-6)  # water beta < 0 below 4 degC
    return Tw - Tb, rho, cp, k, mu, beta


def h_film_liquid(
    T_wall: Num, T_bulk: Num, orientation: OuterFace | np.ndarray, L: Num, sg: Num
) -> Num:
    """Wall -> liquid natural-convection film coefficient (Phase 1), W/(m^2 K).

    ``orientation`` is the direction the wall faces INTO the liquid: UP (base, heated from
    below), SIDE (vertical wall), DOWN. Wort properties at the film temperature
    (T_wall + T_bulk)/2; Ra = g beta |dT| L^3 rho^2 cp / (mu k).

    * hot wall facing UP / cold wall facing DOWN: Nu = 0.15 Ra^(1/3) (Incropera Eq. 9.31 form,
      turbulent horizontal plate, applied for all Ra);
    * hot wall facing DOWN / cold wall facing UP: Nu = 0.27 Ra^(1/4) (Eq. 9.32);
    * SIDE: Churchill-Chu with L = fill height.

    Result clipped to [50, 5000] W/(m^2 K). Subcooled natural convection only: above about
    95 degC real coefficients increase through sub-cooled nucleate boiling (not modelled).
    """
    dT, rho, cp, k, mu, beta = _wort_film(T_wall, T_bulk, sg)
    pr_l = cp * mu / k
    Ra = _G * beta * np.maximum(np.abs(dT), DT_MIN) * _a(L) ** 3 * rho**2 * cp / (mu * k)
    face = _a(orientation).astype(np.int64)
    hot = dT >= 0.0
    # cold wall facing DOWN behaves like hot facing UP; hot facing DOWN like cold facing UP
    is_up = face == int(OuterFace.UP)
    is_down = face == int(OuterFace.DOWN)
    enhanced = (is_up & hot) | (is_down & ~hot)
    Nu_enh = 0.15 * Ra ** (1.0 / 3.0)
    Nu_weak = 0.27 * Ra**0.25
    Nu_side = _nu_churchill_chu(Ra, _cc_den(pr_l))
    Nu = np.where(is_up | is_down, np.where(enhanced, Nu_enh, Nu_weak), Nu_side)
    h = np.clip(Nu * k / _a(L), H_FILM_MIN, H_FILM_MAX)
    return _ret(h)


def h_headspace(T_liquid_surface: Num, T_lid: Num, sg: Num, p: Num) -> Num:
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
    L = HEADSPACE_GAP_M
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
