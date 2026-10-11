"""Synthetic test fixtures: a box-shaped Domain, uniform coefficients and constant properties.

Used by tests across all work packages. Not used by production code.

Geometry (D2): the liquid is a box of nx*dx by ny*dx by nz*dx. Its origin is the centre of
the inner base, so x and y run from -L/2 to +L/2 and z from 0 up. The slice is at j = ny // 2.

Shell: one node per boundary face of the boundary cells. The base has nx*ny nodes
(BASE, outer DOWN). Each side has its own nodes (SIDE, outer SIDE): ny*nz on each x face and
nx*nz on each y face. Every node has area dx^2. In-plane edges join neighbours on the same
face, the base-to-wall junction, and the vertical corners of the walls.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from thermosim.core.types import (
    F64,
    I8,
    I64,
    Domain,
    LidSpec,
    LiquidGrid,
    OuterFace,
    Probe,
    Shell,
    ShellKind,
    SliceSpec,
    StepCoefficients,
)
from thermosim.physics.properties import AirProps, SolidProps

# Stainless-304-like wall used by default. Values are rounded and only for tests.
TEST_WALL = SolidProps(
    name="stainless_304",
    density=8000.0,
    specific_heat=500.0,
    conductivity=16.0,
    emissivity=0.3,
)


def box_domain(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    *,
    shell: bool = True,
    wall_thickness: float = 0.002,
    wall: SolidProps = TEST_WALL,
    lid: bool = False,
    fill_fraction_top: float = 1.0,
) -> Domain:
    """A box-shaped Domain with every liquid cell present.

    Args:
        nx, ny, nz: number of cells along x, y, z (all >= 1).
        dx: cube edge length, m.
        shell: if True, build the wall and base shell nodes. If False, the shell is empty.
        wall_thickness: wall and base thickness, m.
        wall: material of the wall (density kg/m^3, specific_heat J/(kg K),
            conductivity W/(m K), emissivity -).
        lid: if True, add a lumped lid covering the whole top with its headspace.
        fill_fraction_top: liquid fraction (0, 1] of the top layer. The free surface stays
            flat at (nz - 1 + fraction) * dx.

    Returns:
        A Domain whose invariants are tested in tests/testing/test_box_domain.py.
    """
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("nx, ny and nz must be >= 1")
    if not 0.0 < fill_fraction_top <= 1.0:
        raise ValueError("fill_fraction_top must be in (0, 1]")

    f = fill_fraction_top
    shape = (nx, ny, nz)
    lx, ly = nx * dx, ny * dx
    origin = np.array([-lx / 2.0, -ly / 2.0, 0.0], dtype=np.float64)
    t = wall_thickness
    dx2 = dx * dx

    # Per-layer liquid fraction: only the top layer is partial.
    layer_fill = np.ones(nz, dtype=np.float64)
    layer_fill[-1] = f
    fill3 = layer_fill[np.newaxis, np.newaxis, :]

    flat = np.arange(nx * ny * nz, dtype=np.int64).reshape(shape)
    grid = LiquidGrid(
        dx=dx,
        origin=origin,
        shape=shape,
        is_liquid=np.ones(shape, dtype=bool),
        volume=np.ones(shape, dtype=np.float64) * dx**3 * fill3,
        area_x=np.ones((max(nx - 1, 0), ny, nz), dtype=np.float64) * dx2 * fill3,
        area_y=np.ones((nx, max(ny - 1, 0), nz), dtype=np.float64) * dx2 * fill3,
        area_z=np.full((nx, ny, max(nz - 1, 0)), dx2, dtype=np.float64),
        surface_cells=flat[:, :, -1].ravel().copy(),
        surface_area=np.full(nx * ny, dx2, dtype=np.float64),
    )

    shell_obj, node_index = _build_shell(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        origin=origin,
        wall_thickness=t,
        wall=wall,
        layer_fill=layer_fill,
        flat=flat,
        enabled=shell,
    )

    if lid:
        lid_area = lx * ly
        lid_spec = LidSpec(
            present=True,
            covered_fraction=1.0,
            area=lid_area,
            heat_capacity=wall.density * wall.specific_heat * lid_area * 1e-3,
            emissivity=wall.emissivity,
            insulation_r=0.0,
            headspace_volume=lid_area * dx,
        )
    else:
        lid_spec = LidSpec(
            present=False,
            covered_fraction=0.0,
            area=0.0,
            heat_capacity=0.0,
            emissivity=0.0,
            insulation_r=0.0,
            headspace_volume=0.0,
        )

    centre = np.array([0.0, 0.0, nz * dx / 2.0], dtype=np.float64)
    probe = _make_probe("center", centre, grid, flat)

    j_slice = ny // 2
    # Slice lines: the x-walls at grid row j_slice (ordered by z), and the base row (by x).
    wall_left = np.zeros(0, dtype=np.int64)
    wall_right = np.zeros(0, dtype=np.int64)
    base_line = np.zeros(0, dtype=np.int64)
    if shell:
        wall_left = np.array([node_index[("X-", j_slice, k)] for k in range(nz)], dtype=np.int64)
        wall_right = np.array([node_index[("X+", j_slice, k)] for k in range(nz)], dtype=np.int64)
        base_line = np.array([node_index[("B", i, j_slice)] for i in range(nx)], dtype=np.int64)
    centers_x, _, centers_z = grid.cell_centers()
    slice_spec = SliceSpec(
        j=j_slice,
        x_centers=centers_x,
        z_centers=centers_z,
        wall_left=wall_left,
        wall_right=wall_right,
        base_line=base_line,
    )

    return Domain(
        grid=grid,
        shell=shell_obj,
        lid=lid_spec,
        probes=(probe,),
        slice=slice_spec,
        inner_radius=0.5 * max(lx, ly),
        inner_height=nz * dx,
        fill_height=(nz - 1 + f) * dx,
    )


def _build_shell(
    *,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    origin: F64,
    wall_thickness: float,
    wall: SolidProps,
    layer_fill: F64,
    flat: I64,
    enabled: bool,
) -> tuple[Shell, dict[tuple[str, int, int], int]]:
    """Create the shell nodes, their in-plane edges and their coupling to the liquid.

    Returns the Shell and a map from (face, a, b) keys to node indices: ("B", i, j) for base
    nodes, and (face, j, k) or (face, i, k) for side nodes, where face is X-, X+, Y- or Y+.
    """
    node_index: dict[tuple[str, int, int], int] = {}
    kind: list[int] = []
    outer_face: list[int] = []
    position: list[tuple[float, float, float]] = []
    couple_shell: list[int] = []
    couple_cell: list[int] = []
    couple_area: list[float] = []
    edges: set[tuple[int, int]] = set()

    t = wall_thickness
    lx, ly = nx * dx, ny * dx
    xs = origin[0] + (np.arange(nx) + 0.5) * dx
    ys = origin[1] + (np.arange(ny) + 0.5) * dx
    zs = (np.arange(nz) + 0.5) * dx

    def add_node(
        key: tuple[str, int, int],
        k_kind: ShellKind,
        face: OuterFace,
        pos: tuple[float, float, float],
    ) -> int:
        idx = len(kind)
        node_index[key] = idx
        kind.append(int(k_kind))
        outer_face.append(int(face))
        position.append(pos)
        return idx

    def couple(node: int, cell_flat: int, area: float) -> None:
        couple_shell.append(node)
        couple_cell.append(cell_flat)
        couple_area.append(area)

    def edge(a: int, b: int) -> None:
        edges.add((min(a, b), max(a, b)))

    if enabled:
        # Base nodes: one per bottom cell, centred on the mid-surface (z = -t/2).
        for i in range(nx):
            for j in range(ny):
                node = add_node(
                    ("B", i, j),
                    ShellKind.BASE,
                    OuterFace.DOWN,
                    (float(xs[i]), float(ys[j]), -t / 2.0),
                )
                couple(node, int(flat[i, j, 0]), dx * dx)

        # Side nodes: one per boundary face of the boundary cells, outside the inner surface.
        for j in range(ny):
            for k in range(nz):
                fill = float(layer_fill[k])
                node = add_node(
                    ("X-", j, k),
                    ShellKind.SIDE,
                    OuterFace.SIDE,
                    (-(lx / 2.0 + t / 2.0), float(ys[j]), float(zs[k])),
                )
                couple(node, int(flat[0, j, k]), dx * dx * fill)
                node = add_node(
                    ("X+", j, k),
                    ShellKind.SIDE,
                    OuterFace.SIDE,
                    (lx / 2.0 + t / 2.0, float(ys[j]), float(zs[k])),
                )
                couple(node, int(flat[nx - 1, j, k]), dx * dx * fill)
        for i in range(nx):
            for k in range(nz):
                fill = float(layer_fill[k])
                node = add_node(
                    ("Y-", i, k),
                    ShellKind.SIDE,
                    OuterFace.SIDE,
                    (float(xs[i]), -(ly / 2.0 + t / 2.0), float(zs[k])),
                )
                couple(node, int(flat[i, 0, k]), dx * dx * fill)
                node = add_node(
                    ("Y+", i, k),
                    ShellKind.SIDE,
                    OuterFace.SIDE,
                    (float(xs[i]), ly / 2.0 + t / 2.0, float(zs[k])),
                )
                couple(node, int(flat[i, ny - 1, k]), dx * dx * fill)

        # In-plane edges.
        def ix(key: tuple[str, int, int]) -> int:
            return node_index[key]

        for i in range(nx):
            for j in range(ny):
                if i + 1 < nx:
                    edge(ix(("B", i, j)), ix(("B", i + 1, j)))
                if j + 1 < ny:
                    edge(ix(("B", i, j)), ix(("B", i, j + 1)))
        for face_a in ("X-", "X+"):
            for j in range(ny):
                for k in range(nz):
                    if k + 1 < nz:
                        edge(ix((face_a, j, k)), ix((face_a, j, k + 1)))
                    if j + 1 < ny:
                        edge(ix((face_a, j, k)), ix((face_a, j + 1, k)))
        for face_b in ("Y-", "Y+"):
            for i in range(nx):
                for k in range(nz):
                    if k + 1 < nz:
                        edge(ix((face_b, i, k)), ix((face_b, i, k + 1)))
                    if i + 1 < nx:
                        edge(ix((face_b, i, k)), ix((face_b, i + 1, k)))
        # Vertical corners, where an X wall meets a Y wall.
        for k in range(nz):
            edge(ix(("X+", ny - 1, k)), ix(("Y+", nx - 1, k)))
            edge(ix(("X+", 0, k)), ix(("Y-", nx - 1, k)))
            edge(ix(("X-", ny - 1, k)), ix(("Y+", 0, k)))
            edge(ix(("X-", 0, k)), ix(("Y-", 0, k)))
        # Base-to-wall junction along the bottom perimeter.
        for j in range(ny):
            edge(ix(("B", nx - 1, j)), ix(("X+", j, 0)))
            edge(ix(("B", 0, j)), ix(("X-", j, 0)))
        for i in range(nx):
            edge(ix(("B", i, ny - 1)), ix(("Y+", i, 0)))
            edge(ix(("B", i, 0)), ix(("Y-", i, 0)))

    n = len(kind)
    area = np.full(n, dx * dx, dtype=np.float64)
    couple_shell_arr = np.array(couple_shell, dtype=np.int64)
    couple_cell_arr = np.array(couple_cell, dtype=np.int64)
    couple_area_arr = np.array(couple_area, dtype=np.float64)
    wetted = np.bincount(couple_shell_arr, weights=couple_area_arr, minlength=n).astype(np.float64)
    dry = np.maximum(area - wetted, 0.0)

    edge_list = sorted(edges)
    edge_a = np.array([a for a, _ in edge_list], dtype=np.int64)
    edge_b = np.array([b for _, b in edge_list], dtype=np.int64)
    # Each edge is a neighbour step of length dx, with centre distance dx along the surface.
    # Conductance = k * t * (shared length) / (distance) = k * t.
    k_arr = np.full(n, wall.conductivity, dtype=np.float64)
    t_arr = np.full(n, wall_thickness, dtype=np.float64)
    # Shared edge length / centre distance = dx / dx = 1, so G = k_edge * t_edge.
    edge_cond = 0.5 * (k_arr[edge_a] + k_arr[edge_b]) * 0.5 * (t_arr[edge_a] + t_arr[edge_b])

    kind_arr: I8 = np.array(kind, dtype=np.int8)
    face_arr: I8 = np.array(outer_face, dtype=np.int8)
    base_char = lx * ly / (2.0 * (lx + ly))  # A/P of the base rectangle
    side_char = nz * dx  # height of the vessel
    char_length = np.where(kind_arr == int(ShellKind.BASE), base_char, side_char).astype(np.float64)

    return Shell(
        n=n,
        kind=kind_arr,
        position=np.array(position, dtype=np.float64).reshape(n, 3),
        area=area,
        thickness=t_arr,
        rho_c=np.full(n, wall.density * wall.specific_heat, dtype=np.float64),
        k=k_arr,
        emissivity=np.full(n, wall.emissivity, dtype=np.float64),
        edge_a=edge_a,
        edge_b=edge_b,
        edge_conductance=edge_cond,
        couple_shell=couple_shell_arr,
        couple_cell=couple_cell_arr,
        couple_area=couple_area_arr,
        wetted_area=wetted,
        dry_area=dry,
        outer_face=face_arr,
        outer_area=np.full(n, dx * dx, dtype=np.float64),
        insulation_r=np.zeros(n, dtype=np.float64),
        char_length=char_length,
        outer_diameter=max(lx, ly) + 2.0 * wall_thickness,
    ), node_index


def _make_probe(name: str, position: F64, grid: LiquidGrid, flat: I64) -> Probe:
    """Trilinear interpolation stencil of the 8 cells around `position`.

    Cell centres sit at origin + (i + 0.5) * dx. Points on the outer edge are clamped.
    """
    shape = grid.shape
    lows: list[int] = []
    wts_hi: list[float] = []
    for a in range(3):
        n = shape[a]
        f = (position[a] - grid.origin[a]) / grid.dx - 0.5
        lo = int(min(max(math.floor(f), 0), n - 1))
        hi = min(lo + 1, n - 1)
        w_hi = min(max(f - lo, 0.0), 1.0) if hi > lo else 0.0
        lows.append(lo)
        wts_hi.append(w_hi)
    hi_idx = [min(lows[a] + 1, shape[a] - 1) for a in range(3)]

    cells: list[int] = []
    weights: list[float] = []
    for corner in range(8):
        pick = [(corner >> a) & 1 for a in range(3)]
        idx = tuple(hi_idx[a] if pick[a] else lows[a] for a in range(3))
        w = 1.0
        for a in range(3):
            w *= wts_hi[a] if pick[a] else 1.0 - wts_hi[a]
        cells.append(int(flat[idx]))
        weights.append(w)
    return Probe(
        name=name,
        position=np.asarray(position, dtype=np.float64),
        cells=np.array(cells, dtype=np.int64),
        weights=np.array(weights, dtype=np.float64),
    )


def uniform_coefficients(
    domain: Domain,
    *,
    k: float = 0.6,
    rho_c: float = 4.18e6,
    h_film: float = 0.0,
    h_out: float = 0.0,
    h_surface: float = 0.0,
    T_amb: float = 293.15,
) -> StepCoefficients:
    """Full-shape coefficients with every source zero and every loss coefficient uniform.

    Mapping: h_film -> h_film (every coupling row); h_out -> h_out_shell and h_dry (shell
    outer and inner dry wall) and h_lid_outer; h_surface -> h_surface_open,
    h_surface_covered and h_lid_inner. Evaporation is zero.

    Args:
        domain: the Domain the coefficients are for.
        k: liquid effective conductivity, W/(m K).
        rho_c: liquid volumetric heat capacity, J/(m^3 K).
        h_film: film coefficient, W/(m^2 K).
        h_out: outer loss coefficient, W/(m^2 K).
        h_surface: free-surface coefficient, W/(m^2 K).
        T_amb: ambient temperature, K.
    """
    shape = domain.grid.shape
    n = domain.shell.n
    couplings = len(domain.shell.couple_shell)
    ns = len(domain.grid.surface_cells)
    return StepCoefficients(
        T_amb=T_amb,
        k_liquid=np.full(shape, k, dtype=np.float64),
        rho_c_liquid=np.full(shape, rho_c, dtype=np.float64),
        h_film=np.full(couplings, h_film, dtype=np.float64),
        h_out_shell=np.full(n, h_out, dtype=np.float64),
        h_dry=np.full(n, h_out, dtype=np.float64),
        h_surface_open=np.full(ns, h_surface, dtype=np.float64),
        q_evap_open=np.zeros(ns, dtype=np.float64),
        h_surface_covered=np.full(ns, h_surface, dtype=np.float64),
        h_lid_inner=float(h_surface),
        h_lid_outer=float(h_out),
        q_cells=np.zeros(shape, dtype=np.float64),
        q_shell=np.zeros(n, dtype=np.float64),
    )


def _const(value: float, x: ArrayLike) -> float | F64:
    """Scalar for scalar input, otherwise an array of the same shape filled with `value`."""
    shape = np.shape(x)
    if shape == ():
        return float(value)
    return np.full(shape, value, dtype=np.float64)


class ConstantProps:
    """Constant water (liquid) and air properties at 20 C, with the same function names as
    thermosim.physics.properties. Returns scalars for scalar inputs and arrays otherwise.

    Solid properties are not included; use TEST_WALL in this module.
    """

    WATER_DENSITY = 998.2  # kg/m^3
    WATER_CP = 4182.0  # J/(kg K)
    WATER_K = 0.598  # W/(m K)
    WATER_MU = 1.002e-3  # Pa s
    WATER_EXPANSION = 2.07e-4  # 1/K
    WATER_BOILING_K = 373.15  # K at 101325 Pa
    WATER_LATENT = 2.454e6  # J/kg
    WATER_PSAT = 2339.0  # Pa
    STD_PRESSURE = 101325.0  # Pa

    @staticmethod
    def wort_density(T: ArrayLike, sg: ArrayLike) -> float | F64:
        return _const(ConstantProps.WATER_DENSITY, T)

    @staticmethod
    def wort_specific_heat(T: ArrayLike, sg: ArrayLike) -> float | F64:
        return _const(ConstantProps.WATER_CP, T)

    @staticmethod
    def wort_conductivity(T: ArrayLike, sg: ArrayLike) -> float | F64:
        return _const(ConstantProps.WATER_K, T)

    @staticmethod
    def wort_viscosity(T: ArrayLike, sg: ArrayLike) -> float | F64:
        return _const(ConstantProps.WATER_MU, T)

    @staticmethod
    def wort_expansion(T: ArrayLike, sg: ArrayLike) -> float | F64:
        return _const(ConstantProps.WATER_EXPANSION, T)

    @staticmethod
    def wort_boiling_point(sg: ArrayLike, p: ArrayLike) -> float | F64:
        return _const(ConstantProps.WATER_BOILING_K, p)

    @staticmethod
    def latent_heat_vaporization(T: ArrayLike) -> float | F64:
        return _const(ConstantProps.WATER_LATENT, T)

    @staticmethod
    def saturation_pressure(T: ArrayLike) -> float | F64:
        return _const(ConstantProps.WATER_PSAT, T)

    @staticmethod
    def pressure_at_altitude(altitude_m: ArrayLike) -> float | F64:
        return _const(ConstantProps.STD_PRESSURE, altitude_m)

    @staticmethod
    def sg_to_plato(sg: ArrayLike) -> float | F64:
        arr = np.asarray(sg, dtype=np.float64)
        out = 259.0 - 259.0 / arr
        return float(out) if out.ndim == 0 else out

    @staticmethod
    def plato_to_sg(plato: ArrayLike) -> float | F64:
        arr = np.asarray(plato, dtype=np.float64)
        out = 259.0 / (259.0 - arr)
        return float(out) if out.ndim == 0 else out

    @staticmethod
    def air_props(T: ArrayLike, p: ArrayLike) -> AirProps:
        rho, cp, k, mu = 1.204, 1005.0, 0.0257, 1.81e-5
        nu = mu / rho
        alpha = k / (rho * cp)
        return AirProps(
            density=_const(rho, T),
            specific_heat=_const(cp, T),
            conductivity=_const(k, T),
            viscosity=_const(mu, T),
            kinematic_viscosity=_const(nu, T),
            diffusivity=_const(alpha, T),
            prandtl=_const(nu / alpha, T),
            expansion=_const(1.0 / 293.15, T),
        )

    @staticmethod
    def vapor_diffusivity_in_air(T: ArrayLike, p: ArrayLike) -> float | F64:
        return _const(2.5e-5, T)
