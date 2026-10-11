"""Tests for thermosim.physics.properties (WP02).

Reference data embedded below (all at 1 atm unless noted):

* WATER_NIST: liquid water at 1 atm, IAPWS-95 (density, cp), IAPWS 2011 (conductivity) and
  IAPWS 2008 (viscosity) as tabulated in the NIST Chemistry WebBook (fluid properties,
  isobaric, 0.101325 MPa); T = 25 degC row is the usual handbook value.
* WATER_BETA: volumetric expansion coefficient at 1 atm, -(1/rho)(d rho/dT)_p from the
  IAPWS-95 density (NIST WebBook), rounded to three digits.
* WATER_INCROPERA: Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, Table A.6
  (saturated water), an independent cross-check with its own rounding.
* PSAT_TABLE: saturation pressure, steam tables (IAPWS-95 / Cengel & Boles Table A-4), kPa.
* IF97_VERIFY: IAPWS-IF97 Region 4 verification values (IAPWS R7-97(2012), Tables 35/36).
* HFG_TABLE: latent heat of vaporization, steam tables, kJ/kg.
* AIR_INCROPERA: Incropera & DeWitt Table A.4 (air at 1 atm; rho quoted at 1 bar).
* Sucrose-solution values at 20 degC (CRC Handbook of Chemistry and Physics, "Concentrative
  properties of aqueous solutions"; Sugar Technologists Manual).
"""

from __future__ import annotations

import time
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from thermosim.core.errors import ScenarioError
from thermosim.physics import properties as pr

C = 273.15

# T[degC], rho[kg/m3], cp[J/kg/K], k[W/m/K], mu[Pa s]
WATER_NIST = np.array(
    [
        [0.0, 999.84, 4219.9, 0.5610, 1791.4e-6],
        [10.0, 999.70, 4195.5, 0.5800, 1305.9e-6],
        [20.0, 998.21, 4184.4, 0.5984, 1001.6e-6],
        [25.0, 997.05, 4181.6, 0.6071, 890.2e-6],
        [30.0, 995.65, 4180.0, 0.6154, 797.2e-6],
        [40.0, 992.22, 4179.6, 0.6305, 652.7e-6],
        [50.0, 988.03, 4181.5, 0.6435, 546.5e-6],
        [60.0, 983.20, 4184.3, 0.6544, 466.0e-6],
        [70.0, 977.76, 4189.7, 0.6634, 404.0e-6],
        [80.0, 971.79, 4196.6, 0.6700, 354.8e-6],
        [90.0, 965.31, 4205.3, 0.6762, 314.9e-6],
        [100.0, 958.35, 4215.9, 0.6791, 282.2e-6],
    ]
)

# T[degC], beta[1/K]
WATER_BETA = np.array(
    [
        [5.0, 1.6e-5],
        [10.0, 8.8e-5],
        [20.0, 2.07e-4],
        [30.0, 3.03e-4],
        [40.0, 3.85e-4],
        [60.0, 5.23e-4],
        [80.0, 6.41e-4],
        [100.0, 7.50e-4],
    ]
)

# T[K], rho, cp[J/kg/K], mu[Pa s], k  (Incropera Table A.6; rho = 1/v_f)
WATER_INCROPERA = np.array(
    [
        [300.0, 997.0, 4179.0, 855e-6, 0.613],
        [320.0, 989.1, 4180.0, 577e-6, 0.640],
        [340.0, 979.4, 4188.0, 420e-6, 0.660],
        [360.0, 967.1, 4203.0, 324e-6, 0.674],
        [373.15, 957.9, 4217.0, 279e-6, 0.680],
    ]
)

# T[degC], psat[kPa]
PSAT_TABLE = np.array(
    [
        [0.01, 0.6117],
        [10.0, 1.2282],
        [20.0, 2.3392],
        [30.0, 4.2469],
        [40.0, 7.3849],
        [50.0, 12.352],
        [60.0, 19.947],
        [70.0, 31.201],
        [80.0, 47.416],
        [90.0, 70.182],
        [100.0, 101.42],
        [110.0, 143.38],
        [120.0, 198.67],
    ]
)

# T[degC], hfg[kJ/kg]
HFG_TABLE = np.array(
    [
        [0.01, 2500.9],
        [20.0, 2453.5],
        [40.0, 2406.0],
        [60.0, 2357.7],
        [80.0, 2308.0],
        [100.0, 2256.4],
        [120.0, 2202.1],
    ]
)

# T[K], cp[J/kg/K], mu[Pa s], k[W/m/K]; rho at 1 bar for the same rows
AIR_INCROPERA = np.array(
    [
        [250.0, 1006.0, 159.6e-7, 22.3e-3, 1.3947],
        [300.0, 1007.0, 184.6e-7, 26.3e-3, 1.1614],
        [350.0, 1009.0, 208.2e-7, 30.0e-3, 0.9950],
        [400.0, 1014.0, 230.1e-7, 33.8e-3, 0.8711],
    ]
)


def rel(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)) / np.abs(b)


# ------------------------------------------------------------------ water, IAPWS reference


def test_water_density_vs_iapws():
    T = WATER_NIST[:, 0] + C
    assert np.all(rel(pr.wort_density(T, 1.000), WATER_NIST[:, 1]) < 0.0005)


def test_water_cp_vs_iapws():
    T = WATER_NIST[:, 0] + C
    assert np.all(rel(pr.wort_specific_heat(T, 1.000), WATER_NIST[:, 2]) < 0.003)


def test_water_conductivity_vs_iapws():
    T = WATER_NIST[:, 0] + C
    assert np.all(rel(pr.wort_conductivity(T, 1.000), WATER_NIST[:, 3]) < 0.01)


def test_water_viscosity_vs_iapws():
    T = WATER_NIST[:, 0] + C
    assert np.all(rel(pr.wort_viscosity(T, 1.000), WATER_NIST[:, 4]) < 0.02)


def test_water_expansion_vs_reference():
    T = WATER_BETA[:, 0] + C
    got = pr.wort_expansion(T, 1.000)
    ref = WATER_BETA[:, 1]
    hot = WATER_BETA[:, 0] >= 10.0
    assert np.all(rel(got[hot], ref[hot]) < 0.03)
    assert np.all(np.abs(got[~hot] - ref[~hot]) < 0.2e-4)


def test_water_expansion_zero_near_4C():
    # density maximum at 3.98 degC: beta changes sign there
    assert pr.wort_expansion(C + 2.0, 1.0) < 0.0 < pr.wort_expansion(C + 6.0, 1.0)


def test_water_vs_incropera_cross_check():
    for T, rho, cp, mu, k in WATER_INCROPERA:
        assert rel(pr.wort_density(T, 1.0), rho) < 0.002
        assert rel(pr.wort_specific_heat(T, 1.0), cp) < 0.005
        assert rel(pr.wort_viscosity(T, 1.0), mu) < 0.03
        assert rel(pr.wort_conductivity(T, 1.0), k) < 0.01


def test_water_expansion_incropera_300K():
    assert rel(pr.wort_expansion(300.0, 1.0), 276.1e-6) < 0.03


# ------------------------------------------------------------------ saturation and latent heat


def test_saturation_pressure_table():
    got = pr.saturation_pressure(PSAT_TABLE[:, 0] + C) / 1e3
    assert np.all(rel(got, PSAT_TABLE[:, 1]) < 0.002)


def test_saturation_pressure_if97_verification_points():
    assert rel(pr.saturation_pressure(300.0), 0.353658941e-2 * 1e6) < 1e-5


def test_if97_backward_equation_verification_points():
    # IAPWS R7-97 Table 36: T_sat for p = 0.1, 1, 10 MPa
    for p_mpa, t_ref in ((0.1, 0.372755919e3), (1.0, 0.453035632e3), (10.0, 0.584149488e3)):
        assert abs(float(pr._tsat(np.asarray(p_mpa * 1e6))) - t_ref) < 1e-6


def test_latent_heat_table():
    got = pr.latent_heat_vaporization(HFG_TABLE[:, 0] + C) / 1e3
    assert np.all(rel(got, HFG_TABLE[:, 1]) < 0.005)


def test_latent_heat_between_table_points():
    # 105 degC: steam tables give 2243.0 kJ/kg (mean of neighbours within spline accuracy)
    assert rel(pr.latent_heat_vaporization(105.0 + C), 2243.0e3) < 0.002


# ------------------------------------------------------------------ boiling point


def test_boiling_point_pure_water_sea_level():
    assert abs(pr.wort_boiling_point(1.000, 101325.0) - 373.12) < 0.05


def test_boiling_point_at_altitude_1600m():
    p = pr.pressure_at_altitude(1600.0)
    assert abs(p - 83.5e3) < 0.3e3
    assert abs(pr.wort_boiling_point(1.000, p) - (273.15 + 95.0)) < 0.4  # IF97: 94.65 degC


def test_boiling_point_elevation_size():
    for plato, lo, hi in ((10.0, 0.1, 0.4), (12.0, 0.15, 0.5), (20.0, 0.3, 0.8)):
        sg = pr.plato_to_sg(plato)
        d = pr.wort_boiling_point(sg, 101325.0) - pr.wort_boiling_point(1.0, 101325.0)
        assert lo < d < hi, (plato, d)


def test_boiling_point_matches_saturation_for_water():
    p = np.array([5e3, 30e3, 101325.0, 190e3])
    T = pr.wort_boiling_point(1.0, p)
    assert np.all(rel(pr.saturation_pressure(T), p) < 1e-4)


# ------------------------------------------------------------------ Plato, altitude


def test_plato_reference_points():
    assert abs(pr.sg_to_plato(1.000)) < 0.01
    assert abs(pr.sg_to_plato(1.050) - 12.39) < 0.05  # ASBC table: 1.050 = 12.4 degP
    assert abs(pr.sg_to_plato(1.100) - 23.7) < 0.1  # ASBC table: 1.100 = 23.7 degP
    assert abs(pr.plato_to_sg(12.0) - 1.0484) < 0.0005


def test_plato_roundtrip():
    sg = np.linspace(1.0, 1.13, 500)
    assert np.max(np.abs(pr.plato_to_sg(pr.sg_to_plato(sg)) - sg)) < 1e-9
    pl = np.linspace(0.0, 30.0, 500)
    assert np.max(np.abs(pr.sg_to_plato(pr.plato_to_sg(pl)) - pl)) < 1e-8


def test_pressure_at_altitude():
    assert pr.pressure_at_altitude(0.0) == pytest.approx(101325.0)
    assert rel(pr.pressure_at_altitude(1000.0), 89874.6) < 1e-3  # ISA table
    assert rel(pr.pressure_at_altitude(5000.0), 54019.9) < 1e-3
    assert rel(pr.pressure_at_altitude(10000.0), 26436.3) < 2e-3


# ------------------------------------------------------------------ wort (sucrose solution)


def test_wort_20C_against_sucrose_data():
    sg20 = pr.plato_to_sg(20.0)
    assert rel(pr.wort_density(C + 20.0, sg20), sg20 * 998.21) < 1e-5  # NIST 998.21 vs Kell 998.204
    assert rel(pr.wort_specific_heat(C + 20.0, sg20), 3660.0) < 0.02
    assert rel(pr.wort_conductivity(C + 20.0, sg20), 0.546) < 0.03
    assert rel(pr.wort_viscosity(C + 20.0, sg20), 1.945e-3) < 0.06
    sg10 = pr.plato_to_sg(10.0)
    assert rel(pr.wort_viscosity(C + 20.0, sg10), 1.336e-3) < 0.06
    assert rel(pr.wort_specific_heat(C + 20.0, sg10), 3.89e3) < 0.02  # 10 %: 3.89 kJ/kg/K


def test_wort_trends_with_gravity():
    sg = np.linspace(1.0, 1.13, 200)
    for T in (C + 5.0, C + 20.0, C + 65.0, C + 99.0):
        assert np.all(np.diff(pr.wort_density(T, sg)) > 0)
        assert np.all(np.diff(pr.wort_specific_heat(T, sg)) < 0)
        assert np.all(np.diff(pr.wort_conductivity(T, sg)) < 0)
        assert np.all(np.diff(pr.wort_viscosity(T, sg)) > 0)
    for p in (30e3, 101325.0):
        assert np.all(np.diff(pr.wort_boiling_point(sg, p)) > 0)


def test_wort_trends_with_temperature():
    T = np.linspace(C + 15.0, C + 100.0, 200)
    for sg in (1.0, 1.048, 1.1):
        assert np.all(np.diff(pr.wort_density(T, sg)) < 0)
        assert np.all(np.diff(pr.wort_viscosity(T, sg)) < 0)
        assert np.all(np.diff(pr.wort_conductivity(T, sg)) > 0)
        assert np.all(pr.wort_expansion(T, sg) > 0)


def test_wort_expansion_consistent_with_density():
    T = np.array([290.0, 320.0, 350.0, 380.0])
    for sg in (1.0, 1.05, 1.1):
        h = 1e-3
        num = -(pr.wort_density(T + h, sg) - pr.wort_density(T - h, sg)) / (2 * h)
        num = num / pr.wort_density(T, sg)
        assert np.all(rel(pr.wort_expansion(T, sg), num) < 2e-4)


# ------------------------------------------------------------------ air


def test_air_vs_incropera():
    for T, cp, mu, k, rho in AIR_INCROPERA:
        a = pr.air_props(T, 1e5)  # Incropera's density corresponds to 1 bar
        assert rel(a.density, rho) < 0.005
        assert rel(a.specific_heat, cp) < 0.01
        assert rel(a.viscosity, mu) < 0.02
        assert rel(a.conductivity, k) < 0.02
        assert rel(a.kinematic_viscosity, mu / rho) < 0.02
        assert rel(a.diffusivity, k / (rho * cp)) < 0.02
        assert rel(a.prandtl, cp * mu / k) < 0.02
        assert a.expansion == pytest.approx(1.0 / T)


def test_air_ideal_gas_density_uses_pressure():
    assert rel(pr.air_props(273.15, 101325.0).density, 1.2922) < 1e-3
    d1 = pr.air_props(300.0, 101325.0).density
    d2 = pr.air_props(300.0, 50662.5).density
    assert d2 == pytest.approx(0.5 * d1)


def test_air_array_shapes_and_broadcast():
    T = np.linspace(250.0, 400.0, 6).reshape(2, 3)
    a = pr.air_props(T, 101325.0)
    for f in ("density", "specific_heat", "conductivity", "viscosity", "kinematic_viscosity"):
        assert getattr(a, f).shape == (2, 3)
    b = pr.air_props(300.0, np.array([1e5, 2e5]))
    assert b.viscosity.shape == (2,)
    assert b.density[1] == pytest.approx(2 * b.density[0])


def test_vapor_diffusivity_in_air():
    assert rel(pr.vapor_diffusivity_in_air(273.15, 101325.0), 2.26e-5) < 0.03
    assert rel(pr.vapor_diffusivity_in_air(298.15, 101325.0), 2.6e-5) < 0.04
    assert rel(pr.vapor_diffusivity_in_air(373.15, 101325.0), 3.9e-5) < 0.06
    assert pr.vapor_diffusivity_in_air(300.0, 50e3) > pr.vapor_diffusivity_in_air(300.0, 100e3)


# ------------------------------------------------------------------ solids


def test_solids_table():
    s = pr.SOLIDS
    assert set(s) == {"stainless_304", "aluminum", "copper", "carbon_steel"}
    p = pr.solid_props("stainless_304")
    assert (p.density, p.specific_heat, p.conductivity, p.emissivity) == (8000, 500, 16.2, 0.35)
    assert pr.solid_props("copper").conductivity == 390.0
    assert pr.solid_props("aluminum").emissivity == 0.10
    assert pr.solid_props("carbon_steel").emissivity == 0.80


def test_solid_props_duck_typed_and_errors():
    cm = SimpleNamespace(density=1000.0, specific_heat=800.0, conductivity=1.5, emissivity=0.9)
    p = pr.solid_props(cm)
    assert (p.density, p.specific_heat, p.conductivity, p.emissivity) == (1000, 800, 1.5, 0.9)
    assert pr.solid_props(p) is p
    with pytest.raises(ScenarioError):
        pr.solid_props("unobtainium")
    with pytest.raises(ScenarioError):
        pr.solid_props(SimpleNamespace(density=1.0))


# ------------------------------------------------------------------ shapes, clipping, warnings

ALL_TSG = [
    pr.wort_density,
    pr.wort_specific_heat,
    pr.wort_conductivity,
    pr.wort_viscosity,
    pr.wort_expansion,
]


@pytest.mark.parametrize("f", ALL_TSG)
def test_scalar_in_scalar_out(f):
    v = f(300.0, 1.05)
    assert isinstance(v, float)
    assert np.ndim(v) == 0
    a = f(np.full((2, 3), 300.0), 1.05)
    assert a.shape == (2, 3)
    assert f(300.0, np.array([1.0, 1.05, 1.1])).shape == (3,)
    assert f(np.array([[300.0]]), 1.05).shape == (1, 1)


def test_other_scalar_functions_return_floats():
    for v in (
        pr.saturation_pressure(350.0),
        pr.latent_heat_vaporization(350.0),
        pr.wort_boiling_point(1.05, 101325.0),
        pr.pressure_at_altitude(100.0),
        pr.sg_to_plato(1.05),
        pr.plato_to_sg(12.0),
        pr.vapor_diffusivity_in_air(300.0, 1e5),
    ):
        assert isinstance(v, float)
    a = pr.air_props(300.0, 1e5)
    assert all(isinstance(getattr(a, f), float) for f in a.__dataclass_fields__)


def test_out_of_range_clips_and_warns_once():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = pr.wort_specific_heat(np.array([200.0, 300.0, 500.0]), np.array([0.9, 1.05, 1.2]))
    assert len(w) == 2  # one for T, one for SG: a single warning per offending input
    assert all("clipped" in str(x.message) for x in w)
    assert all(x.filename == __file__ for x in w)  # stacklevel points at the caller
    ref = pr.wort_specific_heat(np.array([273.15, 300.0, 393.15]), np.array([1.0, 1.05, 1.13]))
    assert np.array_equal(v, ref)


def test_in_range_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pr.wort_density(np.linspace(273.15, 393.15, 50), np.linspace(1.0, 1.13, 50))
        pr.wort_boiling_point(1.05, 101325.0)
        pr.air_props(300.0, 101325.0)
        pr.saturation_pressure(C + 100.0)


def test_other_functions_clip_and_warn():
    for call, ref in (
        (lambda: pr.saturation_pressure(500.0), pr.saturation_pressure(393.15)),
        (lambda: pr.latent_heat_vaporization(200.0), pr.latent_heat_vaporization(273.15)),
        (lambda: pr.pressure_at_altitude(-2000.0), pr.pressure_at_altitude(-500.0)),
        (lambda: pr.wort_boiling_point(1.05, 300.0), pr.wort_boiling_point(1.05, pr._P_MIN)),
        (lambda: pr.air_props(100.0, 1e5).density, pr.air_props(230.0, 1e5).density),
    ):
        with pytest.warns(RuntimeWarning):
            assert call() == pytest.approx(ref)


def test_nan_does_not_raise():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(pr.wort_density(np.nan, 1.05))


# ------------------------------------------------------------------ performance


def _best_ms(f, *args, n=5):
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        f(*args)
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def test_performance_1e6_elements():
    n = 1_000_000
    T = np.linspace(280.0, 390.0, n)
    sg = np.linspace(1.0, 1.13, n)
    p = np.full(n, 95e3)
    calls = [
        (pr.wort_density, (T, sg)),
        (pr.wort_specific_heat, (T, sg)),
        (pr.wort_conductivity, (T, sg)),
        (pr.wort_viscosity, (T, sg)),
        (pr.wort_expansion, (T, sg)),
        (pr.wort_boiling_point, (sg, p)),
        (pr.saturation_pressure, (T,)),
        (pr.latent_heat_vaporization, (T,)),
        (pr.pressure_at_altitude, (T,)),
        (pr.sg_to_plato, (sg,)),
        (pr.plato_to_sg, (sg * 10 - 9.9,)),
        (pr.air_props, (T, p)),
        (pr.vapor_diffusivity_in_air, (T, p)),
    ]
    slow = {f.__name__: _best_ms(f, *a) for f, a in calls}
    # spec: < 50 ms; allow 2x headroom for loaded CI machines
    assert all(v < 100.0 for v in slow.values()), slow
