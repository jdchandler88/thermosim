"""WP04 acceptance tests for the geometry builder."""

from __future__ import annotations

import math
import time
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from thermosim.core.errors import GeometryError, ScenarioError
from thermosim.core.types import Domain, ShellKind
from thermosim.geometry import (
    build_domain,
    cell_index_at,
    extract_slice,
    fill_height_from_volume,
    probe_values,
)
from thermosim.physics.properties import SolidProps

STAINLESS = SolidProps("stainless_304", 8000.0, 500.0, 16.2, 0.35)  # as patched in conftest
DXS = [0.004, 0.005, 0.01, 0.02]
# (diameter, inner height, fill height)
VESSELS = [(0.30, 0.40, 0.31), (0.18, 0.25, 0.25), (0.40, 0.45, 0.3333)]


def _cases():
    for dx in DXS:
        for d, hv, h in VESSELS:
            yield pytest.param((d, hv, h, dx), id=f"D{d}-H{hv}-h{h}-dx{dx}")


@pytest.fixture(params=list(_cases()))
def domain_case(request, make_scenario) -> tuple[Domain, float, float, float]:
    d, hv, h, dx = request.param
    sc = make_scenario(diameter=d, height=hv, fill_height=h, dx=dx)
    return build_domain(sc), d / 2, hv, h


def test_volume_and_surface(domain_case):
    dom, R, _, h = domain_case
    g = dom.grid
    assert g.volume.sum() == pytest.approx(math.pi * R**2 * h, rel=1e-12)
    assert g.surface_area.sum() == pytest.approx(math.pi * R**2, rel=1e-9)
    assert np.all(g.volume[~g.is_liquid] == 0)
    assert np.all(g.volume[g.is_liquid] > 0)
    # top layer is partial only by the fill fraction; surface cells are the top layer
    assert len(g.surface_cells) == g.is_liquid[:, :, -1].sum()


def test_wetted_and_side_areas(domain_case):
    dom, R, hv, h = domain_case
    sh = dom.shell
    base = sh.kind == int(ShellKind.BASE)
    side = ~base
    tw = 0.002
    assert sh.wetted_area[base].sum() == pytest.approx(math.pi * R**2, rel=1e-9)
    assert sh.wetted_area[side].sum() == pytest.approx(2 * math.pi * R * h, rel=1e-9)
    assert sh.area[side].sum() == pytest.approx(2 * math.pi * R * hv, rel=1e-9)
    assert sh.outer_area[side].sum() == pytest.approx(2 * math.pi * (R + tw) * hv, rel=1e-9)
    assert sh.outer_area[base].sum() == pytest.approx(math.pi * (R + tw) ** 2, rel=1e-9)
    assert np.all(sh.dry_area >= 0)
    assert np.all(sh.wetted_area <= sh.area * (1 + 1e-12))
    np.testing.assert_allclose(sh.dry_area, np.maximum(sh.area - sh.wetted_area, 0), atol=1e-15)


def test_shell_graph(domain_case):
    dom, *_ = domain_case
    sh = dom.shell
    assert np.all(sh.edge_a < sh.edge_b)  # no self-loops, each edge once, ordered
    keys = sh.edge_a * sh.n + sh.edge_b
    assert len(np.unique(keys)) == len(keys)
    assert np.all(sh.edge_conductance > 0) and np.all(np.isfinite(sh.edge_conductance))
    m = coo_matrix((np.ones(len(keys)), (sh.edge_a, sh.edge_b)), shape=(sh.n, sh.n))
    ncomp, _ = connected_components(m, directed=False)
    assert ncomp == 1


def test_couplings(domain_case):
    dom, *_ = domain_case
    g, sh = dom.grid, dom.shell
    assert np.all(g.is_liquid.ravel()[sh.couple_cell])
    assert np.all(sh.couple_area > 0)
    # staircase cells (non-liquid x/y neighbour) all have a coupling row
    pad = np.pad(g.is_liquid, ((1, 1), (1, 1), (0, 0)), constant_values=False)
    interior = pad[1:-1, 1:-1]
    exposed = g.is_liquid & ~(pad[2:, 1:-1] & pad[:-2, 1:-1] & pad[1:-1, 2:] & pad[1:-1, :-2])
    coupled = np.zeros(g.is_liquid.size, dtype=bool)
    side_rows = sh.kind[sh.couple_shell] == int(ShellKind.SIDE)
    coupled[sh.couple_cell[side_rows]] = True
    assert interior.shape == g.shape
    assert np.all(coupled[exposed.ravel()])
    assert not np.any(coupled[(g.is_liquid & ~exposed).ravel()])
    # every base column has exactly one base coupling, onto its k = 0 cell
    base_rows = ~side_rows
    assert base_rows.sum() == g.is_liquid[:, :, 0].sum()
    assert len(np.unique(sh.couple_shell[base_rows])) == base_rows.sum()
    ijk = np.unravel_index(sh.couple_cell[base_rows], g.shape)
    assert np.all(ijk[2] == 0)
    np.testing.assert_allclose(sh.couple_area[base_rows], sh.area[sh.couple_shell[base_rows]])
    # (shell node, cell) pairs are unique
    pair = sh.couple_shell * g.is_liquid.size + sh.couple_cell
    assert len(np.unique(pair)) == len(pair)
    # per-node sum matches wetted_area
    np.testing.assert_allclose(
        np.bincount(sh.couple_shell, weights=sh.couple_area, minlength=sh.n), sh.wetted_area
    )


def test_every_base_perimeter_column_touches_a_side_node(domain_case):
    dom, *_ = domain_case
    sh = dom.shell
    base = sh.kind == int(ShellKind.BASE)
    foot = dom.grid.is_liquid[:, :, 0]
    pad = np.pad(foot, 1, constant_values=False)
    perim = foot & ~(pad[2:, 1:-1] & pad[:-2, 1:-1] & pad[1:-1, 2:] & pad[1:-1, :-2])
    n_base = int(base.sum())
    cross = base[sh.edge_a] & ~base[sh.edge_b]
    linked = np.zeros(n_base, dtype=bool)
    linked[sh.edge_a[cross]] = True
    ids = np.full(foot.shape, -1)
    ids[foot] = np.arange(n_base)
    assert np.all(linked[ids[perim]])


def test_node_properties(make_scenario):
    dom = build_domain(make_scenario())
    sh = dom.shell
    R, tw, tb = 0.15, 0.002, 0.003
    assert np.all(sh.rho_c == STAINLESS.density * STAINLESS.specific_heat)
    assert np.all(sh.k == STAINLESS.conductivity)
    assert np.all(sh.emissivity == STAINLESS.emissivity)
    assert np.all(sh.insulation_r == 0)
    base = sh.kind == int(ShellKind.BASE)
    assert np.all(sh.outer_face[base] == 0) and np.all(sh.outer_face[~base] == 1)
    np.testing.assert_allclose(sh.thickness[base], tb)
    np.testing.assert_allclose(sh.thickness[~base], tw)
    np.testing.assert_allclose(sh.position[base, 2], -tb / 2)
    np.testing.assert_allclose(sh.char_length[base], 2 * (R + tw) / 4)
    np.testing.assert_allclose(sh.char_length[~base], 0.40 + tb)
    assert sh.outer_diameter == pytest.approx(2 * (R + tw))
    r = np.hypot(sh.position[~base, 0], sh.position[~base, 1])
    np.testing.assert_allclose(r, R + tw / 2)
    # base in-plane conductance is k * t_b
    both_base = base[sh.edge_a] & base[sh.edge_b]
    np.testing.assert_allclose(sh.edge_conductance[both_base], STAINLESS.conductivity * tb)


def test_insulation_and_custom_material(make_scenario):
    ins = SimpleNamespace(
        thickness_m=0.02, conductivity_W_mK=0.04, emissivity=0.9, covers={"side", "lid"}
    )
    custom = SimpleNamespace(
        density=2700.0, specific_heat=900.0, conductivity=167.0, emissivity=0.1
    )
    dom = build_domain(make_scenario(insulation=ins, material=custom, lid_mode="full"))
    sh = dom.shell
    base = sh.kind == int(ShellKind.BASE)
    assert np.all(sh.insulation_r[~base] == pytest.approx(0.5))
    assert np.all(sh.emissivity[~base] == 0.9)
    assert np.all(sh.insulation_r[base] == 0)
    assert np.all(sh.emissivity[base] == 0.1)
    assert np.all(sh.k == 167.0)
    assert sh.outer_diameter == pytest.approx(2 * (0.15 + 0.002 + 0.02))
    assert dom.lid.insulation_r == pytest.approx(0.5)
    assert dom.lid.emissivity == 0.9
    ins_base = SimpleNamespace(
        thickness_m=0.01, conductivity_W_mK=0.05, emissivity=0.8, covers={"base"}
    )
    dom2 = build_domain(make_scenario(insulation=ins_base))
    b2 = dom2.shell.kind == int(ShellKind.BASE)
    assert np.all(dom2.shell.insulation_r[b2] == pytest.approx(0.2))
    assert np.all(dom2.shell.insulation_r[~b2] == 0)
    assert dom2.shell.outer_diameter == pytest.approx(2 * 0.152)


def test_lid(make_scenario):
    R, hv, h = 0.15, 0.40, 0.32
    none = build_domain(make_scenario(height=hv, fill_height=h)).lid
    assert not none.present and none.area == 0 and none.headspace_volume == 0
    full = build_domain(make_scenario(height=hv, fill_height=h, lid_mode="full")).lid
    assert full.present and full.covered_fraction == 1.0
    assert full.area == pytest.approx(math.pi * R**2)
    assert full.heat_capacity == pytest.approx(8000 * 500 * 0.001 * math.pi * R**2)
    assert full.emissivity == STAINLESS.emissivity
    assert full.insulation_r == 0
    assert full.headspace_volume == pytest.approx(math.pi * R**2 * (hv - h))
    part = build_domain(
        make_scenario(height=hv, fill_height=h, lid_mode="partial", lid_fraction=0.4)
    ).lid
    assert part.covered_fraction == 0.4
    assert part.area == pytest.approx(0.4 * math.pi * R**2)
    assert part.headspace_volume == pytest.approx(0.4 * math.pi * R**2 * (hv - h))
    with pytest.raises(ScenarioError):
        build_domain(make_scenario(lid_mode="partial", lid_fraction=1.5))


def test_probe_reproduces_linear_field(make_scenario):
    pts = [("a", (0.02, -0.03, 0.1)), ("b", (-0.05, 0.04, 0.2)), ("c", (0.0, 0.0, 0.0833))]
    dom = build_domain(make_scenario(dx=0.01, probes=pts))
    g = dom.grid
    xc, yc, zc = g.cell_centers()
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    a, b, c, d = 3.0, -2.0, 7.5, 300.0
    T = a * X + b * Y + c * Z + d
    vals = probe_values(dom, T)
    for name, (x, y, z) in pts:
        assert vals[name] == pytest.approx(a * x + b * y + c * z + d, abs=1e-12)
    for p in dom.probes:
        assert p.weights.sum() == pytest.approx(1.0)
        assert p.cells.shape == (8,) and p.weights.shape == (8,)
        assert np.all(g.is_liquid.ravel()[p.cells[p.weights > 0]])


def test_probe_near_wall_renormalises(make_scenario):
    # within half a cell of the wall: some stencil cells are not liquid
    dom = build_domain(make_scenario(dx=0.01, probes=[("w", (0.148, 0.0, 0.1))]))
    p = dom.probes[0]
    assert (p.weights == 0).any()
    assert p.weights.sum() == pytest.approx(1.0)
    T = np.full(dom.grid.shape, 330.0)
    assert probe_values(dom, T)["w"] == pytest.approx(330.0)


def test_probe_errors(make_scenario):
    for pos in [(0.2, 0.0, 0.1), (0.0, 0.0, -0.01), (0.0, 0.0, 0.36), (0.0, 0.0, 0.5)]:
        with pytest.raises(GeometryError):
            build_domain(make_scenario(fill_height=0.32, probes=[("x", pos)]))


def test_extract_slice_of_z_field(make_scenario):
    dom = build_domain(make_scenario(dx=0.01))
    g = dom.grid
    xc, _yc, zc = g.cell_centers()
    T = np.broadcast_to(zc[None, None, :], g.shape).copy()
    sl = extract_slice(dom, T)
    assert sl.dtype == np.float32
    assert sl.shape == (g.shape[0], g.shape[2])
    live = g.is_liquid[:, dom.slice.j, :]
    assert np.all(np.isnan(sl[~live]))
    for i in range(g.shape[0]):
        if live[i].any():
            np.testing.assert_allclose(sl[i, live[i]], zc[live[i]], rtol=1e-6)
    np.testing.assert_array_equal(dom.slice.z_centers, zc)
    np.testing.assert_array_equal(dom.slice.x_centers, xc)


def test_slice_spec(make_scenario):
    for dx in (0.01, 0.0125):
        dom = build_domain(make_scenario(dx=dx))
        sl, sh, g = dom.slice, dom.shell, dom.grid
        assert sl.j == g.shape[1] // 2
        assert yc_of(g, sl.j) == pytest.approx(g.dx / 2)
        nrow = len(sl.wall_left)
        assert nrow == len(sl.wall_right) >= g.shape[2]
        assert np.all(np.diff(sh.position[sl.wall_left, 2]) > 0)
        assert np.all(np.diff(sh.position[sl.wall_right, 2]) > 0)
        assert np.all(sh.position[sl.wall_left, 0] < 0) and np.all(
            sh.position[sl.wall_right, 0] > 0
        )
        assert np.all(sh.kind[sl.wall_left] == int(ShellKind.SIDE))
        # nodes are the ones closest to y = 0
        assert np.all(np.abs(sh.position[sl.wall_left, 1]) < 0.2 * 0.15 + 1e-12)
        assert np.all(np.abs(sh.position[sl.wall_right, 1]) < 0.2 * 0.15 + 1e-12)
        assert np.all(sh.kind[sl.base_line] == int(ShellKind.BASE))
        assert np.all(np.diff(sh.position[sl.base_line, 0]) > 0)
        assert np.all(sh.position[sl.base_line, 1] == yc_of(g, sl.j))
        assert len(sl.base_line) == g.is_liquid[:, sl.j, 0].sum()


def yc_of(g, j):
    return g.origin[1] + (j + 0.5) * g.dx


def test_cell_index_at_and_fill_height(make_scenario):
    dom = build_domain(make_scenario(dx=0.01))
    g = dom.grid
    assert cell_index_at(g, np.array([0.001, 0.001, 0.001])) == (
        g.shape[0] // 2,
        g.shape[1] // 2,
        0,
    )
    assert cell_index_at(g, np.array([1.0, 0.0, 0.0])) is None
    assert cell_index_at(g, np.array([0.0, 0.0, -1e-3])) is None
    assert fill_height_from_volume(0.05, 0.2) == pytest.approx(0.05 / (math.pi * 0.04))


def test_volume_input_and_scenario_errors(make_scenario):
    dom = build_domain(make_scenario(diameter=0.4, fill_height=None, volume=0.05, height=0.6))
    assert dom.fill_height == pytest.approx(0.05 / (math.pi * 0.04))
    assert dom.grid.volume.sum() == pytest.approx(0.05, rel=1e-12)
    with pytest.raises(GeometryError):
        build_domain(make_scenario(height=0.3, fill_height=0.35))
    sc = make_scenario()
    sc.liquid.volume_m3 = 0.01  # both set
    with pytest.raises(ScenarioError):
        build_domain(sc)
    sc = make_scenario()
    sc.liquid.fill_height_m = None
    with pytest.raises(ScenarioError):
        build_domain(sc)
    dup = make_scenario(probes=[("a", (0, 0, 0.1)), ("a", (0.01, 0, 0.1))])
    with pytest.raises(ScenarioError):
        build_domain(dup)


def test_top_layer_partial_and_full_vessel(make_scenario):
    dom = build_domain(make_scenario(height=0.3, fill_height=0.3, dx=0.01))  # exactly full
    assert dom.grid.shape[2] == 30
    assert dom.shell.dry_area[dom.shell.kind == int(ShellKind.SIDE)].max() < 1e-12
    dom = build_domain(make_scenario(height=0.3, fill_height=0.255, dx=0.01))
    g = dom.grid
    assert g.shape[2] == 26
    f = 0.5
    live = g.is_liquid[:, :, 0]
    ratio = g.volume[:, :, -1][live] / g.volume[:, :, 0][live]
    np.testing.assert_allclose(ratio, f)
    assert np.all(g.area_z[..., -1][g.is_liquid[:, :, 0][..., None].squeeze()] > 0)
    # vertical faces in the top layer are scaled by f, horizontal ones are not
    assert g.area_x[:, :, -1].max() == pytest.approx(f * g.dx**2)
    assert g.area_x[:, :, 0].max() == pytest.approx(g.dx**2)


def test_face_areas_zero_next_to_non_liquid(make_scenario):
    g = build_domain(make_scenario(dx=0.01)).grid
    lx = g.is_liquid[:-1] & g.is_liquid[1:]
    ly = g.is_liquid[:, :-1] & g.is_liquid[:, 1:]
    lz = g.is_liquid[:, :, :-1] & g.is_liquid[:, :, 1:]
    assert np.all((g.area_x > 0) == lx)
    assert np.all((g.area_y > 0) == ly)
    assert np.all((g.area_z > 0) == lz)


def test_grid_is_centred(make_scenario):
    g = build_domain(make_scenario(dx=0.01)).grid
    np.testing.assert_allclose(g.origin[:2], -np.array(g.shape[:2]) * g.dx / 2)
    assert g.origin[2] == 0
    live = g.is_liquid
    np.testing.assert_array_equal(live, live[::-1])
    np.testing.assert_array_equal(live, live[:, ::-1])


def test_build_time_and_memory_50_litre_kettle(make_scenario):
    # 50 L: D = 0.40 m, fill 0.398 m; vessel 0.5 m tall; dx = 5 mm (80 x 80 x 80 cells)
    sc = make_scenario(
        diameter=0.40, height=0.50, volume=0.05, dx=0.005, probes=[("mid", (0.0, 0.0, 0.2))]
    )
    build_domain(make_scenario(diameter=0.1, height=0.1, fill_height=0.05, dx=0.01))  # warm up
    t0 = time.perf_counter()
    dom = build_domain(sc)
    elapsed = time.perf_counter() - t0
    assert dom.grid.shape[0] == 80
    assert elapsed < 3.0, f"build took {elapsed:.2f} s"
    tracemalloc.start()
    build_domain(sc)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 500e6, f"peak {peak / 1e6:.0f} MB"


def test_debug_plot(make_scenario):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from thermosim.geometry.debug import plot_domain

    ax = plot_domain(build_domain(make_scenario(dx=0.01)))
    assert ax is not None
