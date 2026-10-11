"""Liquid voxel grid (WP04, D2/D3).

The liquid is a uniform Cartesian grid of cubic cells of edge dx, index order [i, j, k] =
[x, y, z], flat index C order. The origin of the coordinate system is the centre of the inner
base, so the grid is centred on the vessel axis and z = 0 is the bottom of the liquid.

Footprint correction (design note): a cell is liquid if its centre lies inside the circle of
radius R. The staircase footprint area A_cells = N_col * dx^2 is not exactly pi R^2, so every
cell volume and every HORIZONTAL face area (area_z, surface_area) is scaled by
s = pi R^2 / A_cells. This makes the total volume exactly pi R^2 h and the free-surface area
exactly pi R^2. Vertical face areas (area_x, area_y) are NOT scaled: in-plane conduction
between neighbouring columns is a property of the cell spacing, not of the footprint.
The top layer is partial: its cells have volume f * s * dx^3 and vertical faces f * dx^2,
where f = (h - (nz - 1) dx) / dx is in (0, 1].
"""

from __future__ import annotations

import math

import numpy as np

from thermosim.core.errors import GeometryError
from thermosim.core.types import F64, LiquidGrid

_EPS = 1e-9  # relative tolerance for ceil() of nearly-integer ratios


def ceil_cells(length: float, dx: float) -> int:
    """Number of cells of edge `dx` (m) needed to cover `length` (m), ignoring round-off."""
    return max(1, math.ceil(length / dx - _EPS))


def footprint_scale(n_columns: int, dx: float, inner_radius: float) -> float:
    """Horizontal area scale s = pi R^2 / (N_col dx^2) (dimensionless)."""
    return math.pi * inner_radius**2 / (n_columns * dx * dx)


def build_grid(inner_radius: float, fill_height: float, dx: float) -> LiquidGrid:
    """Build the liquid grid.

    Args:
        inner_radius: R, m (>= dx).
        fill_height: h, m (> 0).
        dx: cell edge, m (> 0).

    Returns:
        LiquidGrid with exact total volume pi R^2 h and surface area pi R^2.

    Raises:
        GeometryError: if the inputs are not positive or the vessel is thinner than one cell.
    """
    if dx <= 0 or fill_height <= 0 or inner_radius <= 0:
        raise GeometryError("radius, fill height and cell size must be positive")
    if inner_radius < dx:
        raise GeometryError(
            f"cell size {dx} m is too coarse for an inner radius of {inner_radius} m"
        )
    nx = ny = ceil_cells(2.0 * inner_radius, dx)
    nz = ceil_cells(fill_height, dx)
    origin = np.array([-nx * dx / 2.0, -ny * dx / 2.0, 0.0], dtype=np.float64)
    shape = (nx, ny, nz)

    xc = origin[0] + (np.arange(nx) + 0.5) * dx
    yc = origin[1] + (np.arange(ny) + 0.5) * dx
    footprint = (xc[:, None] ** 2 + yc[None, :] ** 2) <= inner_radius**2 * (1.0 + 1e-12)
    n_col = int(footprint.sum())
    if n_col == 0:
        raise GeometryError("no liquid cell found; reduce the cell size")
    s = footprint_scale(n_col, dx, inner_radius)

    f = min(1.0, (fill_height - (nz - 1) * dx) / dx)
    layer = np.ones(nz, dtype=np.float64)
    layer[-1] = f

    is_liquid = np.broadcast_to(footprint[:, :, None], shape).copy()
    volume = np.where(is_liquid, dx**3 * s * layer[None, None, :], 0.0)

    both_x = is_liquid[:-1] & is_liquid[1:]
    both_y = is_liquid[:, :-1] & is_liquid[:, 1:]
    both_z = is_liquid[:, :, :-1] & is_liquid[:, :, 1:]
    area_x = np.where(both_x, dx * dx * layer[None, None, :], 0.0)
    area_y = np.where(both_y, dx * dx * layer[None, None, :], 0.0)
    area_z = np.where(both_z, dx * dx * s, 0.0)

    top = np.zeros(shape, dtype=bool)
    top[:, :, nz - 1] = footprint
    surface_cells = np.flatnonzero(top.ravel()).astype(np.int64)
    surface_area = np.full(len(surface_cells), dx * dx * s, dtype=np.float64)

    return LiquidGrid(
        dx=dx,
        origin=origin,
        shape=shape,
        is_liquid=is_liquid,
        volume=volume,
        area_x=area_x,
        area_y=area_y,
        area_z=area_z,
        surface_cells=surface_cells,
        surface_area=surface_area,
    )


def cell_index_at(grid: LiquidGrid, point: F64) -> tuple[int, int, int] | None:
    """Index [i, j, k] of the grid cell containing `point` (m), or None if outside the grid.

    The cell may be a non-liquid one (check `grid.is_liquid`). Lower faces belong to the cell.
    """
    p = np.asarray(point, dtype=np.float64)
    idx = np.floor((p - grid.origin) / grid.dx).astype(np.int64)
    if np.any(idx < 0) or np.any(idx >= np.array(grid.shape)):
        return None
    return int(idx[0]), int(idx[1]), int(idx[2])
