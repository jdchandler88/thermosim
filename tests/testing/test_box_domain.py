"""Invariants of the synthetic box Domain (WP00 acceptance criteria)."""

from __future__ import annotations

import numpy as np
import pytest

from thermosim.core.types import Domain, OuterFace, ShellKind
from thermosim.testing.synthetic import box_domain


def test_volume_sums_for_full_box(box: Domain) -> None:
    nx, ny, nz = box.grid.shape
    dx = box.grid.dx
    assert box.grid.volume.sum() == pytest.approx(nx * ny * nz * dx**3, rel=1e-12)
    assert np.all(box.grid.is_liquid)


def test_volume_sums_with_partial_top_layer() -> None:
    d = box_domain(10, 8, 6, 0.01, fill_fraction_top=0.5)
    dx = d.grid.dx
    expected = 10 * 8 * (5 + 0.5) * dx**3
    assert d.grid.volume.sum() == pytest.approx(expected, rel=1e-12)
    assert d.fill_height == pytest.approx((5 + 0.5) * dx)
    # Top layer is partial, the layer below is full.
    assert np.allclose(d.grid.volume[:, :, -1], 0.5 * dx**3)
    assert np.allclose(d.grid.volume[:, :, :-1], dx**3)
    assert np.all(d.grid.volume > 0)


def test_surface_area_sums_to_plan_area(box: Domain) -> None:
    nx, ny, _ = box.grid.shape
    dx = box.grid.dx
    assert box.grid.surface_area.sum() == pytest.approx(nx * ny * dx**2, rel=1e-9)
    assert len(box.grid.surface_cells) == nx * ny
    # Every surface cell is in the top layer.
    k = np.unravel_index(box.grid.surface_cells, box.grid.shape)[2]
    assert np.all(k == box.grid.shape[2] - 1)


def test_shell_node_counts(box: Domain) -> None:
    nx, ny, nz = box.grid.shape
    shell = box.shell
    n_base = nx * ny
    n_side = 2 * ny * nz + 2 * nx * nz
    assert shell.n == n_base + n_side
    assert int(np.sum(shell.kind == ShellKind.BASE)) == n_base
    assert int(np.sum(shell.kind == ShellKind.SIDE)) == n_side
    assert np.all(shell.outer_face[shell.kind == ShellKind.BASE] == OuterFace.DOWN)
    assert np.all(shell.outer_face[shell.kind == ShellKind.SIDE] == OuterFace.SIDE)


def test_each_node_has_full_area_and_couples_to_one_cell(box: Domain) -> None:
    shell = box.shell
    dx = box.grid.dx
    assert np.allclose(shell.area, dx**2)
    assert len(shell.couple_shell) == shell.n
    assert np.array_equal(np.sort(shell.couple_shell), np.arange(shell.n))


def test_couple_area_per_node_does_not_exceed_area(box: Domain) -> None:
    shell = box.shell
    assert np.all(shell.wetted_area <= shell.area + 1e-15)
    assert np.all(shell.dry_area >= 0.0)
    assert np.allclose(shell.wetted_area + shell.dry_area, shell.area)
    summed = np.bincount(shell.couple_shell, weights=shell.couple_area, minlength=shell.n)
    assert np.allclose(summed, shell.wetted_area)


def test_partial_top_layer_wets_side_nodes_partially() -> None:
    d = box_domain(4, 4, 3, 0.01, fill_fraction_top=0.25)
    shell = d.shell
    k_of_cell = np.unravel_index(shell.couple_cell, d.grid.shape)[2]
    side_rows = shell.kind[shell.couple_shell] == ShellKind.SIDE
    top_rows = side_rows & (k_of_cell == 2)
    assert np.allclose(shell.couple_area[top_rows], 0.25 * d.grid.dx**2)
    assert np.allclose(shell.dry_area[shell.couple_shell[top_rows]], 0.75 * d.grid.dx**2)


def test_edges_are_ordered_unique_and_in_range(box: Domain) -> None:
    shell = box.shell
    assert len(shell.edge_a) == len(shell.edge_b) == len(shell.edge_conductance)
    assert np.all(shell.edge_a < shell.edge_b)
    pairs = set(zip(shell.edge_a.tolist(), shell.edge_b.tolist(), strict=True))
    assert len(pairs) == len(shell.edge_a)
    assert shell.edge_b.max() < shell.n
    assert shell.edge_a.min() >= 0
    assert np.all(shell.edge_conductance > 0)


def test_probe_weights_sum_to_one(box: Domain) -> None:
    assert len(box.probes) == 1
    probe = box.probes[0]
    assert probe.name == "center"
    assert probe.weights.shape == (8,) and probe.cells.shape == (8,)
    assert probe.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(probe.weights >= 0.0)
    assert np.all(probe.cells >= 0) and np.all(probe.cells < box.grid.volume.size)
    assert np.allclose(probe.position, [0.0, 0.0, box.grid.shape[2] * box.grid.dx / 2])


def test_probe_stencil_is_the_eight_cells_around_centre(box: Domain) -> None:
    probe = box.probes[0]
    i, j, k = np.unravel_index(probe.cells, box.grid.shape)
    nx, ny, nz = box.grid.shape
    assert set(i.tolist()) == {nx // 2 - 1, nx // 2}
    assert set(j.tolist()) == {ny // 2 - 1, ny // 2}
    assert set(k.tolist()) == {nz // 2 - 1, nz // 2}


def test_slice_spec(box: Domain) -> None:
    nx, ny, nz = box.grid.shape
    s = box.slice
    assert s.j == ny // 2
    assert s.x_centers.shape == (nx,) and s.z_centers.shape == (nz,)
    assert len(s.wall_left) == nz and len(s.wall_right) == nz
    assert len(s.base_line) == nx
    assert np.all(np.diff(box.shell.position[s.wall_left, 2]) > 0)
    assert np.all(box.shell.position[s.wall_left, 0] < 0)
    assert np.all(box.shell.position[s.wall_right, 0] > 0)
    assert np.all(np.diff(box.shell.position[s.base_line, 0]) > 0)


def test_lid_and_no_shell_variants() -> None:
    with_lid = box_domain(4, 4, 4, 0.01, lid=True)
    assert with_lid.lid.present
    assert with_lid.lid.area == pytest.approx(0.04**2)
    no_lid = box_domain(4, 4, 4, 0.01)
    assert not no_lid.lid.present

    bare = box_domain(4, 4, 4, 0.01, shell=False)
    assert bare.shell.n == 0
    assert len(bare.shell.edge_a) == 0 and len(bare.shell.couple_shell) == 0
    assert bare.probes[0].weights.sum() == pytest.approx(1.0)


def test_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        box_domain(4, 4, 4, 0.01, fill_fraction_top=0.0)
    with pytest.raises(ValueError):
        box_domain(4, 4, 4, 0.01, fill_fraction_top=1.5)
    with pytest.raises(ValueError):
        box_domain(0, 4, 4, 0.01)
