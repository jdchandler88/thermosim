"""Material and fluid properties (CONTRACTS.md section 6). WP00 stub, implemented by WP02.

All functions are vectorized. T in K, sg dimensionless, p in Pa.
"""

from __future__ import annotations

from dataclasses import dataclass

from thermosim.core.types import F64

Num = float | F64


@dataclass(frozen=True)
class SolidProps:
    name: str
    density: float  # kg/m^3
    specific_heat: float  # J/(kg K)
    conductivity: float  # W/(m K)
    emissivity: float  # -


# Populated by WP02 (keys: stainless_304, aluminum, copper, carbon_steel).
SOLIDS: dict[str, SolidProps] = {}


def solid_props(material: str | object) -> SolidProps:
    """Accepts a key of SOLIDS or a CustomMaterial-like object (duck-typed)."""
    raise NotImplementedError


def wort_density(T: Num, sg: Num) -> Num:
    """Liquid density, kg/m^3."""
    raise NotImplementedError


def wort_specific_heat(T: Num, sg: Num) -> Num:
    """Liquid specific heat, J/(kg K)."""
    raise NotImplementedError


def wort_conductivity(T: Num, sg: Num) -> Num:
    """Liquid thermal conductivity, W/(m K)."""
    raise NotImplementedError


def wort_viscosity(T: Num, sg: Num) -> Num:
    """Liquid dynamic viscosity, Pa s."""
    raise NotImplementedError


def wort_expansion(T: Num, sg: Num) -> Num:
    """Volumetric thermal expansion coefficient, 1/K."""
    raise NotImplementedError


def wort_boiling_point(sg: Num, p: Num) -> Num:
    """Boiling temperature at pressure p (Pa), K."""
    raise NotImplementedError


def latent_heat_vaporization(T: Num) -> Num:
    """Latent heat of vaporization of water, J/kg."""
    raise NotImplementedError


def saturation_pressure(T: Num) -> Num:
    """Saturation vapour pressure of water at T (K), Pa."""
    raise NotImplementedError


def pressure_at_altitude(altitude_m: Num) -> Num:
    """Standard-atmosphere pressure at altitude (m above sea level), Pa."""
    raise NotImplementedError


def sg_to_plato(sg: Num) -> Num:
    """Specific gravity to degrees Plato (dimensionless)."""
    raise NotImplementedError


def plato_to_sg(plato: Num) -> Num:
    """Degrees Plato to specific gravity (dimensionless)."""
    raise NotImplementedError


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
    """Dry air properties at T (K) and p (Pa). Fields are arrays matching T's shape."""
    raise NotImplementedError


def vapor_diffusivity_in_air(T: Num, p: Num) -> Num:
    """Diffusivity of water vapour in air, m^2/s."""
    raise NotImplementedError


PROPANE_LHV_J_PER_KG: float = 46.35e6
STEFAN_BOLTZMANN: float = 5.670374419e-8
G: float = 9.80665
