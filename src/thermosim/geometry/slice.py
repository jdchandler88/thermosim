"""Vertical slice y = 0 through the axis (WP04)."""

from __future__ import annotations

import numpy as np

from thermosim.core.types import F32, F64, LiquidGrid, SliceSpec
from thermosim.geometry.shell import ShellLayout


def build_slice(grid: LiquidGrid, layout: ShellLayout) -> SliceSpec:
    """Build the slice specification.

    j is the cell containing y = 0; when y = 0 lies on a cell boundary (even ny, always the
    case here) it is the cell centred at +dx/2, i.e. j = ny // 2. The wall lines are the side
    nodes whose bin centre angle is closest to pi (left, x < 0) and to 0 (right, x > 0); the
    ties between two equally close bins are broken towards y > 0, consistently with j.
    """
    j = grid.shape[1] // 2
    x_centers, _, z_centers = grid.cell_centers()
    n_theta = layout.n_theta
    a_left = (n_theta - 1) // 2  # bin centre at pi (odd n_theta) or the lower of the two
    a_right = 0
    rows = np.arange(layout.n_rows)
    wall_left = layout.side_offset + rows * n_theta + a_left
    wall_right = layout.side_offset + rows * n_theta + a_right
    column = layout.base_index[:, j]
    base_line = column[column >= 0]  # ascending i = ascending x
    return SliceSpec(
        j=j,
        x_centers=x_centers,
        z_centers=z_centers,
        wall_left=wall_left.astype(np.int64),
        wall_right=wall_right.astype(np.int64),
        base_line=base_line.astype(np.int64),
    )


def extract_slice_array(grid: LiquidGrid, j: int, T_liquid: F64) -> F32:
    """T_liquid[:, j, :] as float32 (nx, nz), NaN where the cell is not liquid."""
    out = np.asarray(T_liquid[:, j, :], dtype=np.float32).copy()
    out[~grid.is_liquid[:, j, :]] = np.nan
    return out
