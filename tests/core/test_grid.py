"""LiquidGrid helpers (CONTRACTS section 2)."""

from __future__ import annotations

import numpy as np
import pytest

from thermosim.core.types import Domain


def test_cell_centers_follow_origin_and_dx(box: Domain) -> None:
    grid = box.grid
    xs, ys, zs = grid.cell_centers()
    nx, ny, nz = grid.shape
    assert xs.shape == (nx,) and ys.shape == (ny,) and zs.shape == (nz,)
    dx = grid.dx
    assert xs[0] == pytest.approx(grid.origin[0] + 0.5 * dx)
    assert xs[-1] == pytest.approx(grid.origin[0] + (nx - 0.5) * dx)
    assert np.allclose(np.diff(xs), dx)
    assert np.allclose(np.diff(ys), dx)
    assert np.allclose(np.diff(zs), dx)
    # D2: the box is centred on x = y = 0, and the base is at z = 0.
    assert xs.mean() == pytest.approx(0.0, abs=1e-12)
    assert ys.mean() == pytest.approx(0.0, abs=1e-12)
    assert zs[0] == pytest.approx(0.5 * dx)
