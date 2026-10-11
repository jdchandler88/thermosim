"""Heat sources: power output and where the heat is deposited (WP05).

A burner deposits its heat into the base shell nodes under its footprint. An element deposits
its heat into the liquid cells along its path. Both return a `SourcePower` for a command, an
actuator time and a step length; the delivered energy over a step is `delivered_W * dt`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from thermosim.core.errors import GeometryError
from thermosim.core.protocols import Actuator, SourcePower
from thermosim.core.types import F64, Domain, HeatDeposition, ShellKind
from thermosim.physics.properties import PROPANE_LHV_J_PER_KG

if TYPE_CHECKING:
    from thermosim.config.schema import Burner, Element

# Sub-samples per axis when integrating a base node's column square against the footprint.
_SUBSAMPLES = 8
# Element path sampling spacing, as a fraction of the cell edge (the spec asks for <= dx/4).
# dx/16 keeps the per-cell share within one sample of the exact path length.
_ELEMENT_SPACING_FRACTION = 1.0 / 16.0
_TOL = 1e-9
_J_PER_KWH = 3.6e6


class BurnerSource:
    """Propane burner: gross = rated * actuator output; delivered = efficiency * gross."""

    def __init__(
        self,
        rated_power_W: float,
        efficiency: float,
        actuator: Actuator,
        deposition: HeatDeposition,
    ) -> None:
        self.rated_power_W = float(rated_power_W)
        self.efficiency = float(efficiency)
        self.actuator = actuator
        self.deposition = deposition

    def power(self, command: float, t: float, dt: float) -> SourcePower:
        gross_W = self.rated_power_W * self.actuator.output(command, t, dt)
        return SourcePower(delivered_W=self.efficiency * gross_W, gross_W=gross_W)

    def fuel_kg(self, gross_J: float) -> float:
        return gross_J / PROPANE_LHV_J_PER_KG

    def electric_kwh(self, gross_J: float) -> float:
        return 0.0


class ElementSource:
    """Electric element: all of its electrical power goes into the liquid."""

    def __init__(
        self,
        rated_power_W: float,
        actuator: Actuator,
        deposition: HeatDeposition,
    ) -> None:
        self.rated_power_W = float(rated_power_W)
        self.actuator = actuator
        self.deposition = deposition

    def power(self, command: float, t: float, dt: float) -> SourcePower:
        power_W = self.rated_power_W * self.actuator.output(command, t, dt)
        return SourcePower(delivered_W=power_W, gross_W=power_W)

    def fuel_kg(self, gross_J: float) -> float:
        return 0.0

    def electric_kwh(self, gross_J: float) -> float:
        return gross_J / _J_PER_KWH


def _footprint_fraction(centers_xy: F64, dx: float, outer_r: float, inner_r: float) -> F64:
    """Fraction of each base node's column square (side dx) inside the annulus.

    Each square is sub-sampled on an 8 x 8 grid of midpoints; a point counts when its radius
    from the axis lies in [inner_r, outer_r]. A disk has inner_r = 0.
    """
    offsets = ((np.arange(_SUBSAMPLES) + 0.5) / _SUBSAMPLES - 0.5) * dx
    ox, oy = np.meshgrid(offsets, offsets, indexing="ij")
    px = centers_xy[:, 0:1] + ox.ravel()[np.newaxis, :]
    py = centers_xy[:, 1:2] + oy.ravel()[np.newaxis, :]
    r2 = px * px + py * py
    inside = (r2 <= outer_r * outer_r) & (r2 >= inner_r * inner_r)
    return inside.mean(axis=1)


def deposition_for_burner(cfg: Burner, domain: Domain) -> HeatDeposition:
    """Spread a burner's heat over the base shell nodes under its disk or ring footprint.

    Weights are the sub-sampled overlap of each base node's column square with the footprint,
    normalized to sum to 1. Raises GeometryError if no base node overlaps the footprint.
    """
    shell = domain.shell
    base = np.flatnonzero(shell.kind == ShellKind.BASE)
    outer_r = 0.5 * float(cfg.footprint_outer_diameter_m)
    inner_r = 0.5 * float(cfg.footprint_inner_diameter_m) if cfg.footprint == "ring" else 0.0
    fraction = _footprint_fraction(shell.position[base, :2], domain.grid.dx, outer_r, inner_r)
    total = float(fraction.sum())
    if total <= 0.0:
        raise GeometryError(
            f"burner footprint (outer diameter {cfg.footprint_outer_diameter_m:.3f} m) "
            "overlaps no base node"
        )
    shell_weights = np.zeros(shell.n, dtype=np.float64)
    shell_weights[base] = fraction / total
    return HeatDeposition(
        cell_weights=np.zeros(domain.grid.shape, dtype=np.float64),
        shell_weights=shell_weights,
    )


def deposition_for_element(cfg: Element, domain: Domain) -> HeatDeposition:
    """Spread an element's heat along its path over the liquid cells it passes through.

    The polyline is sampled at equal arc-length steps of at most dx/4. Each sample carries an
    equal share of the path length, which goes to the cell containing it. A sample whose cell
    lies outside the liquid footprint (near the wall) goes to the nearest liquid cell in its
    layer. Raises GeometryError if any sample is above the fill height (dry firing) or outside
    the vessel (radius > inner radius, or below the base).
    """
    grid = domain.grid
    dx = grid.dx
    pts = np.asarray(cfg.path_m, dtype=np.float64)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total <= 0.0:
        raise GeometryError("element path has zero length")

    n_samples = max(1, math.ceil(total / (_ELEMENT_SPACING_FRACTION * dx) - _TOL))
    s = (np.arange(n_samples) + 0.5) * (total / n_samples)
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg_len) - 1)
    safe_len = np.where(seg_len > 0.0, seg_len, 1.0)
    local = (s - cum[k]) / safe_len[k]
    pos = pts[k] + local[:, np.newaxis] * seg[k]

    fill = float(domain.fill_height)
    radius = float(domain.inner_radius)
    z = pos[:, 2]
    r = np.hypot(pos[:, 0], pos[:, 1])
    above = z > fill + _TOL
    if np.any(above):
        z_bad = float(z[int(np.argmax(above))])
        raise GeometryError(
            f"element segment at z={z_bad:.3f} m is above the liquid (fill {fill:.3f} m)"
        )
    outside = (r > radius + _TOL) | (z < -_TOL)
    if np.any(outside):
        bad_pos = pos[int(np.argmax(outside))]
        raise GeometryError(
            f"element segment at ({bad_pos[0]:.3f}, {bad_pos[1]:.3f}, {bad_pos[2]:.3f}) m "
            f"is outside the vessel (inner radius {radius:.3f} m)"
        )

    # Each sample goes to the liquid cell containing it. A sample whose cell lies outside the
    # staircase footprint (near the wall) goes to the nearest liquid cell in the same layer,
    # by horizontal distance between cell centres.
    ijk = np.floor((pos - grid.origin) / dx).astype(np.int64)
    shape = np.array(grid.shape, dtype=np.int64)
    cells = np.clip(ijk, 0, shape - 1)
    liquid = grid.is_liquid[cells[:, 0], cells[:, 1], cells[:, 2]]
    layer_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for idx in np.flatnonzero(~liquid):
        layer = int(cells[idx, 2])
        if layer not in layer_cache:
            ij = np.argwhere(grid.is_liquid[:, :, layer])
            centers = grid.origin[:2] + (ij + 0.5) * dx
            layer_cache[layer] = (ij, centers)
        ij, centers = layer_cache[layer]
        d2 = np.sum((centers - pos[idx, :2]) ** 2, axis=1)
        nearest = ij[int(np.argmin(d2))]
        cells[idx, 0] = nearest[0]
        cells[idx, 1] = nearest[1]

    cell_weights = np.zeros(grid.shape, dtype=np.float64)
    np.add.at(cell_weights, (cells[:, 0], cells[:, 1], cells[:, 2]), 1.0 / n_samples)
    return HeatDeposition(
        cell_weights=cell_weights,
        shell_weights=np.zeros(domain.shell.n, dtype=np.float64),
    )


__all__ = [
    "BurnerSource",
    "ElementSource",
    "deposition_for_burner",
    "deposition_for_element",
]
