"""Helpers for the flow tests: box grids and a small explicit scalar (temperature) integrator.

The flow solver returns only the advective heat source ``q_adv``; the validation cases need
conduction with fixed-temperature walls too, which the Phase 1 shell model does not provide, so the
tests integrate the temperature with this plain NumPy step.
"""

from __future__ import annotations

import numpy as np

from thermosim.core.types import LiquidGrid


def box_grid(nx: int, ny: int, nz: int, dx: float) -> LiquidGrid:
    """A full box of liquid cells (no partial top layer)."""
    shape = (nx, ny, nz)
    flat = np.arange(nx * ny * nz, dtype=np.int64).reshape(shape)
    dx2 = dx * dx
    return LiquidGrid(
        dx=dx,
        origin=np.array([-nx * dx / 2, -ny * dx / 2, 0.0]),
        shape=shape,
        is_liquid=np.ones(shape, dtype=bool),
        volume=np.full(shape, dx**3),
        area_x=np.full((nx - 1, ny, nz), dx2),
        area_y=np.full((nx, ny - 1, nz), dx2),
        area_z=np.full((nx, ny, nz - 1), dx2),
        surface_cells=flat[:, :, -1].ravel().copy(),
        surface_area=np.full(nx * ny, dx2),
    )


def conduction_step(
    T: np.ndarray,
    q_adv: np.ndarray,
    dt: float,
    alpha: float,
    dx: float,
    rho_cp: float,
    walls: dict[str, float | None],
) -> np.ndarray:
    """Explicit step ``T += dt (q_adv / (rho cp dx^3) + alpha lap T)``.

    ``walls`` maps ``"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"`` to a fixed wall temperature
    (applied half a cell outside the boundary cells) or ``None`` (adiabatic).
    """
    out = T + dt * q_adv / (rho_cp * dx**3)
    lap = np.zeros_like(T)
    for axis, (lo, hi) in enumerate((("xlo", "xhi"), ("ylo", "yhi"), ("zlo", "zhi"))):
        pad_shape = list(T.shape)
        pad_shape[axis] = T.shape[axis] + 2
        ext = np.empty(pad_shape)
        sl = [slice(None)] * 3
        sl[axis] = slice(1, -1)
        ext[tuple(sl)] = T
        for side, name in ((0, lo), (-1, hi)):
            ghost = [slice(None)] * 3
            edge = [slice(None)] * 3
            ghost[axis] = slice(0, 1) if side == 0 else slice(-1, None)
            edge[axis] = slice(0, 1) if side == 0 else slice(-1, None)
            inner = [slice(None)] * 3
            inner[axis] = slice(0, 1) if side == 0 else slice(T.shape[axis] - 1, None)
            tw = walls.get(name)
            ext[tuple(ghost)] = T[tuple(inner)] if tw is None else 2.0 * tw - T[tuple(inner)]
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis] = slice(0, -2)
        b[axis] = slice(2, None)
        lap += ext[tuple(a)] + ext[tuple(b)] - 2.0 * T
    out += dt * alpha * lap / dx**2
    return out
