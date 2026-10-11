"""Domain builder (CONTRACTS.md section 8, WP04).

`build_domain` reads the scenario by attribute name only (duck typing), so it works with the
pydantic `Scenario` and with test doubles. Geometry conventions: see mesh.py and shell.py.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from thermosim.core.errors import GeometryError, ScenarioError
from thermosim.core.types import F32, F64, Domain
from thermosim.geometry.lid import build_lid
from thermosim.geometry.mesh import build_grid, cell_index_at
from thermosim.geometry.probes import make_probe
from thermosim.geometry.probes import probe_values as _probe_values
from thermosim.geometry.shell import build_shell
from thermosim.geometry.slice import build_slice, extract_slice_array
from thermosim.physics import properties

if TYPE_CHECKING:
    from thermosim.config.schema import Scenario

__all__ = [
    "build_domain",
    "cell_index_at",
    "extract_slice",
    "fill_height_from_volume",
    "probe_values",
]

_TOL = 1e-9


def build_domain(scenario: Scenario) -> Domain:
    """Build the full Domain from a scenario. Raises GeometryError / ScenarioError.

    Uses: vessel (inner_diameter_m, inner_height_m, wall_thickness_m, base_thickness_m,
    material, shape), liquid (volume_m3 | fill_height_m), lid, insulation, probes
    (name, position_m) and simulation.cell_size_m. Everything in SI units.
    """
    sc: Any = scenario
    v = sc.vessel
    if getattr(v, "shape", "cylinder") != "cylinder":
        raise ScenarioError(f"unsupported vessel shape {v.shape!r}")
    R = 0.5 * float(v.inner_diameter_m)
    H = float(v.inner_height_m)
    dx = float(sc.simulation.cell_size_m)
    if R <= 0 or H <= 0:
        raise ScenarioError("vessel dimensions must be positive")

    vol = sc.liquid.volume_m3
    hgt = sc.liquid.fill_height_m
    if (vol is None) == (hgt is None):
        raise ScenarioError("set exactly one of liquid.volume_m3 and liquid.fill_height_m")
    h = float(hgt) if hgt is not None else fill_height_from_volume(float(vol), R)
    if h <= 0:
        raise ScenarioError("fill height must be positive")
    if h > H * (1.0 + _TOL):
        raise GeometryError(f"liquid fill height {h:.4f} m exceeds the vessel height {H:.4f} m")
    h = min(h, H)

    grid = build_grid(R, h, dx)
    wall = properties.solid_props(v.material)
    shell, layout = build_shell(
        grid,
        R,
        H,
        h,
        float(v.wall_thickness_m),
        float(v.base_thickness_m),
        wall,
        sc.insulation,
    )
    lid = build_lid(sc, R, H, h)
    probes = tuple(make_probe(grid, p.name, p.position_m, h) for p in sc.probes)
    names = [p.name for p in probes]
    if len(set(names)) != len(names):
        raise ScenarioError("probe names must be unique")
    return Domain(
        grid=grid,
        shell=shell,
        lid=lid,
        probes=probes,
        slice=build_slice(grid, layout),
        inner_radius=R,
        inner_height=H,
        fill_height=h,
    )


def fill_height_from_volume(volume_m3: float, inner_radius: float) -> float:
    """Liquid fill height (m) of a cylinder of inner radius `inner_radius` (m) holding
    `volume_m3` (m^3): h = V / (pi R^2)."""
    if inner_radius <= 0:
        raise ValueError("inner_radius must be positive")
    return volume_m3 / (math.pi * inner_radius**2)


def extract_slice(domain: Domain, T_liquid: F64) -> F32:
    """Slice of the liquid temperature at plane y = 0, shape (nx, nz), K; NaN outside liquid."""
    return extract_slice_array(domain.grid, domain.slice.j, T_liquid)


def probe_values(domain: Domain, T_liquid: F64) -> dict[str, float]:
    """Trilinear-interpolated temperature (K) at every probe, keyed by probe name."""
    return _probe_values(domain.probes, T_liquid)
