"""Domain builder (CONTRACTS.md section 8). WP00 stub, implemented by WP04."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thermosim.core.types import F32, F64, Domain, LiquidGrid

if TYPE_CHECKING:
    from thermosim.config.schema import Scenario


def build_domain(scenario: Scenario) -> Domain:
    """Build the full Domain from a scenario. Raises GeometryError / ScenarioError."""
    raise NotImplementedError


def fill_height_from_volume(volume_m3: float, inner_radius: float) -> float:
    """Liquid fill height (m) of a cylinder of inner radius `inner_radius` (m) holding
    `volume_m3` (m^3)."""
    raise NotImplementedError


def extract_slice(domain: Domain, T_liquid: F64) -> F32:
    """Slice of the liquid temperature at plane y = 0, shape (nx, nz), K; NaN outside liquid."""
    raise NotImplementedError


def probe_values(domain: Domain, T_liquid: F64) -> dict[str, float]:
    """Trilinear-interpolated temperature (K) at every probe, keyed by probe name."""
    raise NotImplementedError


def cell_index_at(grid: LiquidGrid, point: F64) -> tuple[int, int, int] | None:
    """Index [i, j, k] of the cell containing `point` (m), or None if outside the grid."""
    raise NotImplementedError
