"""MAC face masks and weights built from a :class:`~thermosim.core.types.LiquidGrid` (WP21).

Face ``i`` of the u array lies between cells ``i - 1`` and ``i``; faces on the grid boundary
(``i = 0`` and ``i = nx``) and faces next to a solid cell are closed. The open fraction of a face
is ``area / dx^2`` taken from the ``LiquidGrid`` (the staircase footprint correction and the
partial top layer make it differ from 1), so continuity uses the same face areas and cell
volumes as the thermal solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import DTypeLike

from thermosim.core.types import B, LiquidGrid


@dataclass(frozen=True, eq=False)
class FlowGrid:
    """Index tables of the staggered grid.

    Attributes:
        shape: ``(nx, ny, nz)`` cells.
        dx: cell edge, m.
        liq: ``(nx, ny, nz)`` liquid-cell mask.
        wx, wy, wz: open fractions ``area / dx^2`` of the faces, shapes ``(nx+1, ny, nz)``,
            ``(nx, ny+1, nz)``, ``(nx, ny, nz+1)``; 0 on closed faces.
        mu, mv, mw: boolean open-face masks (``w > 0``).
        volume: ``(nx, ny, nz)`` cell volume, m^3 (0 outside the liquid).
        n_liquid: number of liquid cells.
    """

    shape: tuple[int, int, int]
    dx: float
    liq: B
    wx: np.ndarray
    wy: np.ndarray
    wz: np.ndarray
    mu: B
    mv: B
    mw: B
    volume: np.ndarray
    n_liquid: int

    @staticmethod
    def from_liquid_grid(grid: LiquidGrid, dtype: DTypeLike = np.float64) -> FlowGrid:
        """Build the face tables of ``grid`` with weights stored as ``dtype``."""
        nx, ny, nz = grid.shape
        dx2 = grid.dx * grid.dx
        wx = np.zeros((nx + 1, ny, nz))
        wy = np.zeros((nx, ny + 1, nz))
        wz = np.zeros((nx, ny, nz + 1))
        wx[1:nx] = grid.area_x / dx2
        wy[:, 1:ny] = grid.area_y / dx2
        wz[:, :, 1:nz] = grid.area_z / dx2
        wxs, wys, wzs = (np.ascontiguousarray(a, dtype=dtype) for a in (wx, wy, wz))
        liq = np.ascontiguousarray(grid.is_liquid, dtype=bool)
        return FlowGrid(
            shape=(nx, ny, nz),
            dx=float(grid.dx),
            liq=liq,
            wx=wxs,
            wy=wys,
            wz=wzs,
            mu=wxs > 0,
            mv=wys > 0,
            mw=wzs > 0,
            volume=np.ascontiguousarray(grid.volume, dtype=np.float64),
            n_liquid=int(liq.sum()),
        )
