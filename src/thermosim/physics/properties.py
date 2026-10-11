"""Material and fluid properties (CONTRACTS.md section 6).

All functions are vectorized (NumPy, no per-element Python loops). Temperatures are in
kelvin, specific gravity ``sg`` is dimensionless (taken as 20/20 degC, the convention of the
Plato polynomial) and pressures are in Pa. A scalar input gives a scalar (``float``) result and an
array input gives an array of the broadcast shape.

Valid range: 273.15-393.15 K and SG 1.000-1.130 (air: 230-450 K). Inputs outside the range are
clipped and a single ``warnings.warn`` is emitted per call (never an exception).

Sources (also cited at the functions that use them)
---------------------------------------------------
* Water density: Kell, G.S., J. Chem. Eng. Data 20 (1975) 97-105 (1 atm, air-free water).
* Water cp, k, viscosity, latent heat: tables of IAPWS-95 / IAPWS R12-08 (viscosity) /
  IAPWS R15-11 (conductivity) / steam tables at 10 K spacing (NIST Chemistry WebBook,
  saturated or 1 atm liquid), interpolated with a cubic spline (log scale for viscosity).
* Saturation pressure and its inverse: IAPWS-IF97 Region 4 (Wagner et al. 2000).
* Sucrose-solution (wort) properties: Choi & Okos (1986) composition model for cp, k and
  density of the carbohydrate fraction; Genotelle (1978) viscosity correlation for sucrose
  solutions; Raoult's law with an osmotic coefficient for the boiling-point elevation.
* Air: Sutherland (1893) viscosity, polynomial fits to Incropera & DeWitt, Table A.4.
* Altitude: ICAO / ISO 2533 standard atmosphere, troposphere.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from thermosim.core.errors import ScenarioError
from thermosim.core.types import F64

Num = float | F64

PROPANE_LHV_J_PER_KG: float = 46.35e6
STEFAN_BOLTZMANN: float = 5.670374419e-8
G: float = 9.80665

# --------------------------------------------------------------------------------------
# Helpers: clipping with a single warning, scalar/array return
# --------------------------------------------------------------------------------------

T_MIN_K = 273.15
T_MAX_K = 393.15
SG_MIN = 1.000
SG_MAX = 1.130
AIR_T_MIN_K = 230.0
AIR_T_MAX_K = 450.0
ALT_MIN_M = -500.0
ALT_MAX_M = 11000.0
_T_REF_SG = 293.15  # SG reference temperature (20 degC)


def _arr(x: Num) -> F64:
    return np.asarray(x, dtype=np.float64)


def _clip(x: Num, lo: float, hi: float, name: str, unit: str, level: int) -> F64:
    """Clip x to [lo, hi]; warn once if anything was out of range. ``level`` is the stacklevel
    for warnings.warn relative to this function (3 = caller of the function that called us)."""
    a = _arr(x)
    if a.size:
        amin = float(a.min())
        amax = float(a.max())
        if amin < lo or amax > hi:
            warnings.warn(
                f"{name} outside valid range [{lo:g}, {hi:g}] {unit} "
                f"(got {amin:g}..{amax:g}); values clipped",
                RuntimeWarning,
                stacklevel=level,
            )
            a = np.clip(a, lo, hi)
    return a


def _ret(x: F64) -> Num:
    """Scalar in -> scalar out."""
    if x.ndim == 0:
        return float(x)
    return x


def _clip_T(T: Num, level: int = 4) -> F64:
    return _clip(T, T_MIN_K, T_MAX_K, "Temperature", "K", level)


def _clip_sg(sg: Num, level: int = 4) -> F64:
    return _clip(sg, SG_MIN, SG_MAX, "Specific gravity", "", level)


# --------------------------------------------------------------------------------------
# Solids
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SolidProps:
    name: str
    density: float  # kg/m^3
    specific_heat: float  # J/(kg K)
    conductivity: float  # W/(m K)
    emissivity: float  # -


# Room-temperature values. Sources: Incropera & DeWitt, Fundamentals of Heat and Mass Transfer,
# Tables A.1 (properties) and A.11 (total emissivity, oxidised / dull / commercial surfaces);
# ASM Handbook; Engineering ToolBox. Emissivities are for typical as-used (dull) surfaces.
SOLIDS: dict[str, SolidProps] = {
    "stainless_304": SolidProps("stainless_304", 8000.0, 500.0, 16.2, 0.35),
    "aluminum": SolidProps("aluminum", 2700.0, 900.0, 205.0, 0.10),
    "copper": SolidProps("copper", 8960.0, 385.0, 390.0, 0.07),
    "carbon_steel": SolidProps("carbon_steel", 7850.0, 490.0, 45.0, 0.80),
}


def solid_props(material: str | object) -> SolidProps:
    """Accepts a key of SOLIDS or a CustomMaterial-like object (duck-typed: ``density``,
    ``specific_heat``, ``conductivity``, ``emissivity`` attributes)."""
    if isinstance(material, str):
        try:
            return SOLIDS[material]
        except KeyError:
            raise ScenarioError(
                f"unknown material {material!r}; known: {', '.join(sorted(SOLIDS))}"
            ) from None
    if isinstance(material, SolidProps):
        return material
    try:
        return SolidProps(
            name=str(getattr(material, "name", "custom")),
            density=float(material.density),  # type: ignore[attr-defined]
            specific_heat=float(material.specific_heat),  # type: ignore[attr-defined]
            conductivity=float(material.conductivity),  # type: ignore[attr-defined]
            emissivity=float(material.emissivity),  # type: ignore[attr-defined]
        )
    except AttributeError as e:
        raise ScenarioError(f"object {material!r} is not a material: {e}") from None


# --------------------------------------------------------------------------------------
# Liquid water (private, operate on already-clipped arrays)
# --------------------------------------------------------------------------------------

# Reference table, 10 K spacing: T [degC], cp [J/(kg K)], k [W/(m K)], mu [uPa s], hfg [kJ/kg].
# 0-100 degC: liquid at 1 atm (IAPWS-95 via NIST WebBook; viscosity IAPWS 2008, conductivity
# IAPWS 2011). 110-120 degC: saturated liquid. hfg: IAPWS-95 steam tables.
_W_TABLE = np.array(
    [
        # T,    cp,     k,      mu,     hfg
        [0.0, 4219.9, 0.5610, 1791.4, 2500.9],
        [10.0, 4195.5, 0.5800, 1305.9, 2477.2],
        [20.0, 4184.4, 0.5984, 1001.6, 2453.5],
        [30.0, 4180.0, 0.6154, 797.2, 2429.8],
        [40.0, 4179.6, 0.6305, 652.7, 2406.0],
        [50.0, 4181.5, 0.6435, 546.5, 2382.0],
        [60.0, 4184.3, 0.6544, 466.0, 2357.7],
        [70.0, 4189.7, 0.6634, 404.0, 2333.0],
        [80.0, 4196.6, 0.6700, 354.8, 2308.0],
        [90.0, 4205.3, 0.6762, 314.9, 2282.5],
        [100.0, 4215.9, 0.6791, 282.2, 2256.4],
        [110.0, 4229.0, 0.6836, 254.6, 2229.7],
        [120.0, 4246.0, 0.6860, 232.1, 2202.1],
    ]
)
_GRID_T = np.linspace(T_MIN_K, T_MAX_K, 1201)  # 0.1 K dense grid for fast np.interp
_tc = _W_TABLE[:, 0] + 273.15
_GRID_CP = CubicSpline(_tc, _W_TABLE[:, 1])(_GRID_T)
_GRID_K = CubicSpline(_tc, _W_TABLE[:, 2])(_GRID_T)
_GRID_LNMU = CubicSpline(_tc, np.log(_W_TABLE[:, 3] * 1e-6))(_GRID_T)
_GRID_HFG = CubicSpline(_tc, _W_TABLE[:, 4] * 1e3)(_GRID_T)


def _rho_w_exact(T: F64) -> F64:
    """Kell (1975) density of air-free water at 1 atm, kg/m^3 (T in K)."""
    t = T - 273.15
    num = 999.83952 + t * (
        16.945176
        + t * (-7.9870401e-3 + t * (-46.170461e-6 + t * (105.56302e-9 + t * -280.54253e-12)))
    )
    return num / (1.0 + 16.879850e-3 * t)


def _vw_exact(T: F64) -> tuple[F64, F64]:
    """Specific volume 1/rho_w (m^3/kg) and its analytic T-derivative from the Kell formula."""
    t = T - 273.15
    num = 999.83952 + t * (
        16.945176
        + t * (-7.9870401e-3 + t * (-46.170461e-6 + t * (105.56302e-9 + t * -280.54253e-12)))
    )
    dnum = 16.945176 + t * (
        2 * -7.9870401e-3
        + t * (3 * -46.170461e-6 + t * (4 * 105.56302e-9 + t * 5 * -280.54253e-12))
    )
    den = 1.0 + 16.879850e-3 * t
    # v = den / num ; dv/dT = (den' num - den num') / num^2
    return den / num, (16.879850e-3 * num - den * dnum) / (num * num)


# Dense 0.1 K grids of the Kell specific volume and its derivative (linear interpolation error
# < 1e-8 relative); much faster than evaluating the rational polynomial on 1e6 points.
_GRID_VW, _GRID_DVW = _vw_exact(_GRID_T)


def _vw(T: F64) -> F64:
    return np.interp(T, _GRID_T, _GRID_VW)


def _cp_w(T: F64) -> F64:
    return np.interp(T, _GRID_T, _GRID_CP)


def _k_w(T: F64) -> F64:
    return np.interp(T, _GRID_T, _GRID_K)


def _mu_w(T: F64) -> F64:
    return np.exp(np.interp(T, _GRID_T, _GRID_LNMU))


# --------------------------------------------------------------------------------------
# Saturation line (IAPWS-IF97 Region 4)
# --------------------------------------------------------------------------------------

_N = (
    0.11670521452767e4,
    -0.72421316703206e6,
    -0.17073846940092e2,
    0.12020824702470e5,
    -0.32325550322333e7,
    0.14915108613530e2,
    -0.48232657361591e4,
    0.40511340542057e6,
    -0.23855557567849,
    0.65017534844798e3,
)


def _psat(T: F64) -> F64:
    """IF97 Region 4 saturation pressure, Pa (T in K, 273.15-647.096)."""
    n = _N
    th = T + n[8] / (T - n[9])
    a = th * th + n[0] * th + n[1]
    b = n[2] * th * th + n[3] * th + n[4]
    c = n[5] * th * th + n[6] * th + n[7]
    q = 2.0 * c / (-b + np.sqrt(b * b - 4.0 * a * c))
    q *= q
    return q * q * 1e6


def _tsat(p: F64) -> F64:
    """IF97 Region 4 backward equation: saturation temperature, K, for p in Pa."""
    n = _N
    be = np.sqrt(np.sqrt(p * 1e-6))
    e = be * be + n[2] * be + n[5]
    f = n[0] * be * be + n[3] * be + n[6]
    g = n[1] * be * be + n[4] * be + n[7]
    d = 2.0 * g / (-f - np.sqrt(f * f - 4.0 * e * g))
    s = n[9] + d
    return (s - np.sqrt(s * s - 4.0 * (n[8] + n[9] * d))) / 2.0


_P_MIN = float(_psat(np.asarray(T_MIN_K)))
_P_MAX = float(_psat(np.asarray(T_MAX_K)))
# Dense lookup grids (linear interpolation of ln p vs T and of T vs ln p; the interpolation
# error is < 1e-6 relative in p and < 1e-5 K in T) so that 1e6-element calls stay fast.
_GRID_LNP = np.log(_psat(_GRID_T))
_LNP_AXIS = np.linspace(np.log(_P_MIN), np.log(_P_MAX), 4001)
_TSAT_OF_LNP = _tsat(np.exp(_LNP_AXIS))


def saturation_pressure(T: Num) -> Num:
    """Saturation vapour pressure of water at T (K), Pa.

    IAPWS-IF97 Region 4 (Wagner et al., J. Eng. Gas Turbines Power 122 (2000) 150); agrees with
    the steam tables to better than 0.01 % over 0-120 degC.
    """
    return _ret(np.exp(np.interp(_clip_T(T, 3), _GRID_T, _GRID_LNP)))


def latent_heat_vaporization(T: Num) -> Num:
    """Latent heat of vaporization of water, J/kg (IAPWS-95 steam-table values, spline)."""
    return _ret(np.interp(_clip_T(T, 3), _GRID_T, _GRID_HFG))


# --------------------------------------------------------------------------------------
# Wort = sucrose solution at the same degrees Plato
# --------------------------------------------------------------------------------------


def sg_to_plato(sg: Num) -> Num:
    """Specific gravity to degrees Plato: P = -616.868 + 1111.14 SG - 630.272 SG^2 + 135.997 SG^3
    (the standard ASBC/MEBAK cubic)."""
    return _ret(_sg_to_plato(_arr(sg)))


def _sg_to_plato(sg: F64) -> F64:
    return -616.868 + sg * (1111.14 + sg * (-630.272 + sg * 135.997))


def plato_to_sg(plato: Num) -> Num:
    """Degrees Plato to specific gravity: numerical inverse (Newton iterations, vectorized) of
    the cubic used by sg_to_plato."""
    p = _arr(plato)
    sg = 1.0 + p / (258.6 - p * (227.1 / 258.2))  # classic approximation as the start value
    for _ in range(4):
        f = _sg_to_plato(sg) - p
        df = 1111.14 + sg * (-1260.544 + sg * 3 * 135.997)
        sg = sg - f / df
    return _ret(sg)


# Choi & Okos (1986), "Effects of temperature and composition on the thermal properties of
# foods", in Food Engineering and Process Applications vol 1: carbohydrate component, t in degC.
def _cp_carb(T: F64) -> F64:
    t = T - 273.15
    return 1e3 * (1.5488 + t * (1.9625e-3 + t * -5.9399e-6))


def _k_carb(T: F64) -> F64:
    t = T - 273.15
    return 0.20141 + t * (1.3874e-3 + t * -4.3312e-6)


def _rho_carb(T: F64) -> F64:
    return 1599.1 - 0.31046 * (T - 273.15)


def _wfrac_exact(sg: F64) -> F64:
    return np.maximum(_sg_to_plato(sg), 0.0) * 0.01


# Lookup grids over the valid SG range (1e-4 spacing, interpolation error < 1e-9 in w).
_SG_AXIS = np.linspace(SG_MIN, SG_MAX, 1301)
_GRID_W = _wfrac_exact(_SG_AXIS)


def _wfrac(sg: F64) -> F64:
    """Sucrose mass fraction w = degrees Plato / 100 (sg must already be clipped)."""
    return np.interp(sg, _SG_AXIS, _GRID_W)


_RHO_W20 = float(_rho_w_exact(np.asarray(_T_REF_SG)))
_RHO_C20 = 1599.1 - 0.31046 * (_T_REF_SG - 273.15)


def _wort_density(T: F64, sg: F64) -> F64:
    w = _wfrac(sg)
    v = w / _rho_carb(T) + (1.0 - w) * _vw(T)
    v20 = w / _RHO_C20 + (1.0 - w) / _RHO_W20
    # scale so that rho(20 degC) = sg * rho_water(20 degC) exactly (the definition of SG 20/20)
    return (sg * _RHO_W20) * v20 / v


def wort_density(T: Num, sg: Num) -> Num:
    """Liquid density, kg/m^3.

    Ideal volume mixing of sucrose (Choi-Okos carbohydrate density) and water (Kell), scaled by
    a constant factor so that the density at 20 degC equals ``sg * rho_water(20 degC)`` (SG is
    defined 20/20). The temperature dependence therefore comes from the mixture, and is within
    a few tens of percent (of the expansion coefficient) of real sucrose solutions at 10-20 degP.
    """
    T_, sg_ = _clip_T(T), _clip_sg(sg)
    return _ret(_wort_density(T_, sg_))


def wort_specific_heat(T: Num, sg: Num) -> Num:
    """Liquid specific heat, J/(kg K): mass-weighted mix of water (IAPWS-95 table) and the
    Choi-Okos (1986) carbohydrate cp. Gives 3.66 kJ/(kg K) for 20 % sucrose at 20 degC, in line
    with sugar-technology data (about 3.65-3.67)."""
    T_, sg_ = _clip_T(T), _clip_sg(sg)
    w = _wfrac(sg_)
    return _ret((1.0 - w) * _cp_w(T_) + w * _cp_carb(T_))


def wort_conductivity(T: Num, sg: Num) -> Num:
    """Liquid thermal conductivity, W/(m K): Choi-Okos (1986) parallel (volume-fraction) model,
    water from the IAPWS table and sucrose from the Choi-Okos carbohydrate correlation.
    Gives about 0.548 W/(m K) for 20 % sucrose at 20 degC (literature: 0.546)."""
    T_, sg_ = _clip_T(T), _clip_sg(sg)
    w = _wfrac(sg_)
    vc = w / _rho_carb(T_)
    xv = vc / (vc + (1.0 - w) * _vw(T_))
    return _ret((1.0 - xv) * _k_w(T_) + xv * _k_carb(T_))


def _genotelle_terms(sg: F64) -> tuple[F64, F64]:
    """Concentration terms of log10 eta (mPa s) of a sucrose solution, Genotelle (1978), as given
    in Bubnik, Kadlec, Urban, Bruhns, Sugar Technologists Manual (1995):
    log eta = 22.46 N - 0.114 + (30 - t)/(91 + t) * (1.1 + 43.1 N^1.25),  N = B/(1900 - 18 B),
    B = % sucrose w/w (taken equal to degrees Plato). Returns (22.46 N, 43.1 N^1.25); the
    concentration-free terms cancel in the ratio to water used by wort_viscosity."""
    b = np.maximum(_sg_to_plato(sg), 0.0)
    n = b / (1900.0 - 18.0 * b)
    return 22.46 * n, 43.1 * n * np.sqrt(np.sqrt(n))


_GEN_A, _GEN_B = _genotelle_terms(_SG_AXIS)


def wort_viscosity(T: Num, sg: Num) -> Num:
    """Liquid dynamic viscosity, Pa s.

    Water viscosity (IAPWS 2008 table) times the relative viscosity of a sucrose solution given
    by Genotelle (1978) (ratio of the correlation at the wort's degrees Plato to the same
    correlation at zero concentration):
    log10(mu_rel) = 22.46 N + (30 - t)/(91 + t) * 43.1 N^1.25. Reproduces 1.34 mPa s (10 %) and
    1.95 mPa s (20 %) at 20 degC to within about 5 %.
    """
    T_, sg_ = _clip_T(T), _clip_sg(sg)
    t = T_ - 273.15
    f = (30.0 - t) / (91.0 + t)
    a = np.interp(sg_, _SG_AXIS, _GEN_A)
    b = np.interp(sg_, _SG_AXIS, _GEN_B)
    rel = np.exp((a + b * f) * np.log(10.0))
    return _ret(_mu_w(T_) * rel)


def wort_expansion(T: Num, sg: Num) -> Num:
    """Volumetric thermal expansion coefficient, 1/K: -(1/rho) d rho/dT of the density model of
    wort_density, evaluated analytically. Equals the Kell-based water value for SG = 1.000."""
    T_, sg_ = _clip_T(T), _clip_sg(sg)
    w = _wfrac(sg_)
    rc_inv = 1.0 / _rho_carb(T_)
    v = w * rc_inv + (1.0 - w) * _vw(T_)
    dv = w * 0.31046 * rc_inv * rc_inv + (1.0 - w) * np.interp(T_, _GRID_T, _GRID_DVW)
    return _ret(dv / v)


_M_WATER = 0.018015  # kg/mol
_M_SUCROSE = 0.34230  # kg/mol


def wort_boiling_point(sg: Num, p: Num) -> Num:
    """Boiling temperature at pressure p (Pa), K.

    The wort boils when p_sat(T) * a_w = p, with the water activity of the sucrose solution
    ln a_w = -phi * m * M_w (m molality, M_w the molar mass of water) and an osmotic coefficient
    phi = 1 + 0.05 m + 0.04 m^2 fitted to sucrose osmotic-coefficient data (Robinson & Stokes;
    Starzak & Mathlouthi 2006) for m <= 2 mol/kg. T is then the IF97 saturation temperature at
    p * exp(phi m M_w) (evaluated through lookup grids, error < 1e-4 K). Pure water at 101325 Pa
    boils at 373.12 K; the elevation is about 0.2 K at 12 degP and 0.4 K at 20 degP.

    ``p`` is clipped to the saturation pressure range of 0-120 degC (0.61-198.7 kPa).
    """
    sg_ = _clip_sg(sg, 3)
    p_ = _clip(p, _P_MIN, _P_MAX, "Pressure", "Pa", 3)
    x = np.interp(sg_, _SG_AXIS, _BP_LNA)  # -ln a_w at this SG
    return _ret(np.interp(np.log(p_) + x, _LNP_AXIS, _TSAT_OF_LNP))


def _bp_lna(sg: F64) -> F64:
    w = _wfrac(sg)
    m = w / (1.0 - w) / _M_SUCROSE
    phi = 1.0 + m * (0.05 + 0.04 * m)
    return phi * m * _M_WATER


_BP_LNA = _bp_lna(_SG_AXIS)


def pressure_at_altitude(altitude_m: Num) -> Num:
    """Standard-atmosphere (ISA / ISO 2533) pressure in the troposphere, Pa:
    p = 101325 (1 - 2.25577e-5 h)^5.25588. Valid -500..11000 m (clipped outside)."""
    h = _clip(altitude_m, ALT_MIN_M, ALT_MAX_M, "Altitude", "m", 3)
    return _ret(101325.0 * (1.0 - 2.25577e-5 * h) ** 5.25588)


# --------------------------------------------------------------------------------------
# Air
# --------------------------------------------------------------------------------------

_R_AIR = 287.052  # J/(kg K), dry air

# Dry air at ~1 atm, Incropera & DeWitt Table A.4, 200-500 K: cp [J/(kg K)], k [W/(m K)].
_AIR_T = np.array([200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0])
_AIR_CP = np.array([1007.0, 1006.0, 1007.0, 1009.0, 1014.0, 1021.0, 1030.0])
_AIR_K = np.array([18.1e-3, 22.3e-3, 26.3e-3, 30.0e-3, 33.8e-3, 37.3e-3, 40.7e-3])
# Polynomial fits in x = (T - 300)/100 (computed once at import; deterministic).
_AIR_CP_COEF = np.polyfit((_AIR_T - 300.0) / 100.0, _AIR_CP, 3)
_AIR_K_COEF = np.polyfit((_AIR_T - 300.0) / 100.0, _AIR_K, 3)
_AIR_GRID_T = np.linspace(AIR_T_MIN_K, AIR_T_MAX_K, 2201)  # 0.1 K lookup grid of the fits
_AIR_GRID_CP = np.polyval(_AIR_CP_COEF, (_AIR_GRID_T - 300.0) / 100.0)
_AIR_GRID_K = np.polyval(_AIR_K_COEF, (_AIR_GRID_T - 300.0) / 100.0)


@dataclass(frozen=True)
class AirProps:
    density: Num  # kg/m^3
    specific_heat: Num  # J/(kg K)
    conductivity: Num  # W/(m K)
    viscosity: Num  # Pa s
    kinematic_viscosity: Num  # m^2/s
    diffusivity: Num  # m^2/s (thermal)
    prandtl: Num  # -
    expansion: Num  # 1/K


def air_props(T: Num, p: Num) -> AirProps:
    """Dry air properties at T (K) and p (Pa). Fields are arrays matching the broadcast shape
    of T and p (floats for scalar input).

    Ideal-gas density with R = 287.052 J/(kg K); viscosity by Sutherland's law (S = 110.4 K,
    mu0 = 1.716e-5 Pa s at 273.15 K); cp and k by cubic fits to Incropera & DeWitt Table A.4;
    expansion coefficient 1/T. Within 2 % of the standard tables over 250-400 K. The effect of
    humidity on these properties is neglected.
    """
    T_ = _clip(T, AIR_T_MIN_K, AIR_T_MAX_K, "Air temperature", "K", 3)
    p_ = _arr(p)
    shape = np.broadcast_shapes(T_.shape, p_.shape)
    cp = np.interp(T_, _AIR_GRID_T, _AIR_GRID_CP)
    k = np.interp(T_, _AIR_GRID_T, _AIR_GRID_K)
    x = T_ * (1.0 / 273.15)
    mu = (1.716e-5 * (273.15 + 110.4)) * x * np.sqrt(x) / (T_ + 110.4)
    rho = p_ / (_R_AIR * T_)

    def full(a: F64) -> Num:
        a = _arr(a)
        return _ret(a if a.shape == shape else np.broadcast_to(a, shape).copy())

    return AirProps(
        density=full(rho),
        specific_heat=full(cp),
        conductivity=full(k),
        viscosity=full(mu),
        kinematic_viscosity=full(mu / rho),
        diffusivity=full(k / (rho * cp)),
        prandtl=full(cp * mu / k),
        expansion=full(1.0 / T_),
    )


def vapor_diffusivity_in_air(T: Num, p: Num) -> Num:
    """Diffusivity of water vapour in air, m^2/s: D = 2.26e-5 (T/273.15)^1.81 (101325/p)
    (Schirmer 1938, as given in VDI Waermeatlas; about 2.6e-5 at 25 degC, 3.9e-5 at 100 degC)."""
    T_ = _clip(T, AIR_T_MIN_K, AIR_T_MAX_K, "Air temperature", "K", 3)
    return _ret(_arr(2.26e-5 * np.exp(1.81 * np.log(T_ / 273.15)) * (101325.0 / _arr(p))))
