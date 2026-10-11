"""Static index structures of the thermal network (built once per Domain).

Everything here depends on geometry only; the per-step coefficients come from
``StepCoefficients``. See ``thermosim.solver.thermal`` for the discretization.

Face order of the 6-neighbour table: -x, +x, -y, +y, -z, +z.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from thermosim.core.errors import SolverError
from thermosim.core.types import F64, I64, Domain

I32 = NDArray[np.int32]


@dataclass(frozen=True, eq=False)
class Network:
    """Compact liquid indexing plus shell/lid/surface connectivity.

    Attributes:
        n_liq: number of liquid cells N.
        liq_flat: (N,) compact index -> flat (C-order) index into (nx, ny, nz) arrays.
        volume: (N,) m^3 liquid volume of each compact cell.
        nbr: (N, 6) flat index of the liquid neighbour across each face, -1 if none.
        nbr_area: (N, 6) m^2 face areas matching ``nbr`` (0 where none).
        couple_flat: (C,) flat liquid index of each shell-liquid coupling row.
        couple_n: (C,) compact liquid index of each coupling row.
        surf_flat: (Ns,) flat index of each free-surface cell.
        surf_n: (Ns,) compact index of each free-surface cell.
        edge_sum: (n_shell,) W/K sum of in-plane edge conductances at each shell node.
        special: (S,) unique compact indices of cells with couplings or surface exposure.
    """

    n_liq: int
    liq_flat: I64
    volume: F64
    nbr: I32
    nbr_area: F64
    couple_flat: I64
    couple_n: I64
    surf_flat: I64
    surf_n: I64
    edge_sum: F64
    special: I64


def build_network(domain: Domain) -> Network:
    """Precompute the index arrays of ``domain`` (geometry only, no physics)."""
    grid = domain.grid
    nx, ny, nz = grid.shape
    total = nx * ny * nz
    liq_flat = np.flatnonzero(grid.is_liquid.reshape(-1)).astype(np.int64)
    n_liq = int(liq_flat.size)
    inv = np.full(total, -1, dtype=np.int64)
    inv[liq_flat] = np.arange(n_liq, dtype=np.int64)
    volume = np.ascontiguousarray(grid.volume.reshape(-1)[liq_flat], dtype=np.float64)

    i, j, k = np.unravel_index(liq_flat, grid.shape)
    coords = (i, j, k)
    strides = (ny * nz, nz, 1)
    areas = (grid.area_x, grid.area_y, grid.area_z)
    lengths = (nx, ny, nz)

    nbr = np.full((n_liq, 6), -1, dtype=np.int32)
    nbr_area = np.zeros((n_liq, 6), dtype=np.float64)
    for axis in range(3):
        c = coords[axis]
        area = areas[axis]
        if lengths[axis] < 2 or area.size == 0:
            continue
        for side in (0, 1):  # 0: lower neighbour, 1: upper neighbour
            valid = c > 0 if side == 0 else c < lengths[axis] - 1
            face = c - 1 if side == 0 else c
            idx = [i, j, k]
            idx[axis] = np.where(valid, face, 0)
            a = np.where(valid, area[tuple(idx)], 0.0)
            q = liq_flat + (-strides[axis] if side == 0 else strides[axis])
            ok = valid & (a > 0.0)
            q_safe = np.where(ok, q, 0)
            ok &= inv[q_safe] >= 0
            col = 2 * axis + side
            nbr[:, col] = np.where(ok, q, -1).astype(np.int32)
            nbr_area[:, col] = np.where(ok, a, 0.0)

    shell = domain.shell
    couple_flat = np.ascontiguousarray(shell.couple_cell, dtype=np.int64)
    couple_n = inv[couple_flat] if couple_flat.size else np.zeros(0, dtype=np.int64)
    if couple_n.size and couple_n.min() < 0:
        raise SolverError("a shell coupling row refers to a non-liquid cell")
    surf_flat = np.ascontiguousarray(grid.surface_cells, dtype=np.int64)
    surf_n = inv[surf_flat] if surf_flat.size else np.zeros(0, dtype=np.int64)
    if surf_n.size and surf_n.min() < 0:
        raise SolverError("a free-surface cell is not a liquid cell")

    edge_sum = np.zeros(shell.n, dtype=np.float64)
    if shell.edge_a.size:
        edge_sum += np.bincount(shell.edge_a, weights=shell.edge_conductance, minlength=shell.n)
        edge_sum += np.bincount(shell.edge_b, weights=shell.edge_conductance, minlength=shell.n)

    return Network(
        n_liq=n_liq,
        liq_flat=liq_flat,
        volume=volume,
        nbr=np.ascontiguousarray(nbr),
        nbr_area=np.ascontiguousarray(nbr_area),
        couple_flat=couple_flat,
        couple_n=np.ascontiguousarray(couple_n, dtype=np.int64),
        surf_flat=surf_flat,
        surf_n=np.ascontiguousarray(surf_n, dtype=np.int64),
        edge_sum=edge_sum,
        special=np.unique(np.concatenate([couple_n, surf_n])).astype(np.int64),
    )
