"""Temperature probes: trilinear interpolation over liquid cell centres (WP04).

Near a wall or the free surface some of the 8 surrounding cell centres are not liquid. Their
weights are set to 0 and the others renormalised, so the interpolation degenerates towards
nearest-liquid there. It reproduces linear fields exactly only where all 8 cells are liquid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from thermosim.core.errors import GeometryError
from thermosim.core.types import F64, LiquidGrid, Probe
from thermosim.geometry.mesh import cell_index_at


def make_probe(
    grid: LiquidGrid, name: str, position: Any, fill_height: float | None = None
) -> Probe:
    """Build the interpolation stencil of a probe.

    Args:
        grid: liquid grid.
        name: probe name.
        position: (x, y, z) in m (vessel coordinates, D2).
        fill_height: h, m. If given, points above it are rejected.

    Raises:
        GeometryError: if the point is outside the liquid or no surrounding cell is liquid.
    """
    p = np.asarray(position, dtype=np.float64).reshape(3)
    idx = cell_index_at(grid, p)
    if idx is None or not grid.is_liquid[idx] or (fill_height is not None and p[2] > fill_height):
        raise GeometryError(f"probe {name!r} at {tuple(p)} m is outside the liquid")
    shape = np.array(grid.shape)
    f = (p - grid.origin) / grid.dx - 0.5
    lo = np.floor(f).astype(np.int64)
    w_hi = f - lo
    cells = np.empty(8, dtype=np.int64)
    weights = np.empty(8)
    for corner in range(8):
        pick = np.array([(corner >> a) & 1 for a in range(3)])
        ijk = lo + pick
        w = float(np.prod(np.where(pick == 1, w_hi, 1.0 - w_hi)))
        inside = bool(np.all(ijk >= 0) and np.all(ijk < shape))
        if inside and grid.is_liquid[tuple(ijk)]:
            cells[corner] = np.ravel_multi_index(tuple(ijk), grid.shape)
            weights[corner] = w
        else:
            cells[corner] = np.ravel_multi_index(tuple(np.clip(ijk, 0, shape - 1)), grid.shape)
            weights[corner] = 0.0
    total = weights.sum()
    if total <= 0.0:
        raise GeometryError(f"probe {name!r} has no liquid cell around it")
    return Probe(name=name, position=p, cells=cells, weights=weights / total)


def probe_values(probes: tuple[Probe, ...], T_liquid: F64) -> dict[str, float]:
    """Interpolated temperature (K) of every probe, keyed by name."""
    flat = np.asarray(T_liquid).ravel()
    return {p.name: float(np.dot(p.weights, flat[p.cells])) for p in probes}
