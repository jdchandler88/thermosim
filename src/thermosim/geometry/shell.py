"""Vessel shell mesh: base disk + unrolled cylinder wall (WP04, D4).

Node layout (indices into the Shell arrays):
  * base nodes   0 .. N_col-1, one per liquid column (C order of the column mask);
  * side nodes   N_col + b * n_theta + a, row b (bottom = 0), angular bin a.

Design notes
------------
Rows of the side mesh have height dx (the top row may be shorter, H_v - (nzs-1) dx) and start
at z = 0, so they coincide with the liquid layers. A staircase face of liquid layer k therefore
always lies inside row k, so the "face straddles two rows" case of the specification cannot
occur and the row of a face is simply its layer index.

Wall <-> liquid coupling. Every exposed (staircase) face of the liquid footprint is mapped to
the angular bins of the side mesh it overlaps (overlap of the angular interval of the face as
seen from the axis with the bin intervals). The overlap matrix W[bin, face] is then balanced
by iterative proportional fitting so that every face carries equal weight (raw face area is
dx * layer height, identical for all faces of a layer) and, as the final step, every bin
sums to exactly its target. The result is a pattern P[bin, column] whose rows sum to 1 and
that is the same for every layer; the coupled area of a (bin, column) pair in layer k is
P * R * dtheta * H_k, with H_k the liquid height of layer k. Consequently:
  * sum of side wetted area per layer is exactly 2 pi R H_k (the true wetted wall area);
  * every side node below the liquid line is wetted over exactly its full area (the capping
    step of the specification is satisfied by construction, no excess has to be spread);
  * the top row is wetted over R dtheta f dx; the rest of it is dry.
The same pattern P gives the junction between the base and the bottom ring of the wall:
conductance k_avg t_avg (R dtheta P) / (distance between node centres).

All couple areas are for the INNER wall surface at radius R. Side `area` uses radius R,
`outer_area` the radius R + t_w.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from thermosim.core.errors import GeometryError
from thermosim.core.types import F64, I64, LiquidGrid, OuterFace, Shell, ShellKind
from thermosim.geometry.mesh import ceil_cells, footprint_scale
from thermosim.physics.properties import SolidProps

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True, eq=False)
class ShellLayout:
    """Index bookkeeping of a shell built by `build_shell` (used by the slice builder)."""

    base_index: I64  # (nx, ny) base node of column (i, j), -1 where not liquid
    n_theta: int
    n_rows: int
    side_offset: int  # index of side node (a=0, b=0)
    theta: F64  # (n_theta,) rad, bin centre angles
    row_z: F64  # (n_rows,) m, row centre heights

    def side_node(self, a: int, b: int) -> int:
        return self.side_offset + b * self.n_theta + a


def _exposed_faces(
    grid: LiquidGrid, col_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Staircase faces of the liquid footprint.

    Returns (col, x1, y1, x2, y2): column id and the two end points (m) of every face.
    """
    dx = grid.dx
    nx, ny, _ = grid.shape
    foot = col_id >= 0
    pad = np.pad(foot, 1, constant_values=False)
    xs = grid.origin[0] + (np.arange(nx) + 0.5) * dx
    ys = grid.origin[1] + (np.arange(ny) + 0.5) * dx
    xc, yc = np.meshgrid(xs, ys, indexing="ij")
    h = dx / 2.0
    cols, x1s, y1s, x2s, y2s = [], [], [], [], []
    # direction: neighbour mask, then end points of the face
    specs = (
        (pad[2:, 1:-1], lambda: (xc + h, yc - h, xc + h, yc + h)),  # +x
        (pad[:-2, 1:-1], lambda: (xc - h, yc - h, xc - h, yc + h)),  # -x
        (pad[1:-1, 2:], lambda: (xc - h, yc + h, xc + h, yc + h)),  # +y
        (pad[1:-1, :-2], lambda: (xc - h, yc - h, xc + h, yc - h)),  # -y
    )
    for neighbour, ends in specs:
        m = foot & ~neighbour
        x1, y1, x2, y2 = ends()
        cols.append(col_id[m])
        x1s.append(x1[m])
        y1s.append(y1[m])
        x2s.append(x2[m])
        y2s.append(y2[m])
    return (
        np.concatenate(cols),
        np.concatenate(x1s),
        np.concatenate(y1s),
        np.concatenate(x2s),
        np.concatenate(y2s),
    )


def coupling_pattern(
    grid: LiquidGrid, col_id: np.ndarray, n_theta: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wall-to-liquid coupling pattern P[bin, column] (rows sum to 1), as sparse triplets.

    Returns:
        (bins, columns, weights): unique (bin, column) pairs and their weights; the weights of
        every bin sum to exactly 1.
    """
    n_col = int(col_id.max()) + 1
    fcol, x1, y1, x2, y2 = _exposed_faces(grid, col_id)
    n_face = len(fcol)
    dth = _TWO_PI / n_theta
    t1 = np.arctan2(y1, x1)
    t2 = np.arctan2(y2, x2)
    d = np.mod(t2 - t1 + math.pi, _TWO_PI) - math.pi
    start = np.mod(np.minimum(t1, t1 + d), _TWO_PI)
    end = start + np.abs(d)
    m0 = np.floor(start / dth).astype(np.int64)
    m1 = np.floor(end / dth).astype(np.int64)
    span = int((m1 - m0).max()) + 1

    f_idx, b_idx, w_ov = [], [], []
    faces = np.arange(n_face)
    for r in range(span):
        m = m0 + r
        ov = np.minimum(end, (m + 1) * dth) - np.maximum(start, m * dth)
        keep = (m <= m1) & (ov > 1e-14)
        f_idx.append(faces[keep])
        b_idx.append(np.mod(m[keep], n_theta))
        w_ov.append(ov[keep])
    face = np.concatenate(f_idx)
    bins = np.concatenate(b_idx)
    w = np.concatenate(w_ov)

    # Bins that no face overlaps (only possible for very small vessels): attach the face whose
    # mid-angle is closest.
    covered = np.zeros(n_theta, dtype=bool)
    covered[bins] = True
    if not covered.all():
        mid = np.mod(start + 0.5 * np.abs(d), _TWO_PI)
        extra_b = np.flatnonzero(~covered)
        theta_b = (extra_b + 0.5) * dth
        dist = np.abs(np.mod(mid[None, :] - theta_b[:, None] + math.pi, _TWO_PI) - math.pi)
        extra_f = np.argmin(dist, axis=1)
        face = np.concatenate([face, extra_f])
        bins = np.concatenate([bins, extra_b])
        w = np.concatenate([w, np.full(len(extra_b), 1e-3 * float(w.mean()))])

    # Iterative proportional fitting: faces carry equal weight, bins carry equal weight.
    # The last step is always the bin normalisation, so bins sum to exactly 1.
    face_target = 1.0 / n_face
    bin_target = 1.0 / n_theta
    for _ in range(200):
        fs = np.bincount(face, weights=w, minlength=n_face)
        w = w * (face_target / fs[face])
        bs = np.bincount(bins, weights=w, minlength=n_theta)
        w = w * (bin_target / bs[bins])
        if np.max(np.abs(np.bincount(face, weights=w, minlength=n_face) / face_target - 1)) < 1e-9:
            break
    bs = np.bincount(bins, weights=w, minlength=n_theta)
    w = w / bs[bins]

    # Merge faces of the same column that hit the same bin.
    key = bins * n_col + fcol[face]
    uniq, inv = np.unique(key, return_inverse=True)
    weights = np.bincount(inv, weights=w)
    return (uniq // n_col).astype(np.int64), (uniq % n_col).astype(np.int64), weights


def _insulation_parts(insulation: Any, part: str) -> tuple[float, float, float]:
    """(thickness m, resistance m^2 K/W, emissivity) of the insulation on `part`, or zeros."""
    if insulation is None or part not in insulation.covers:
        return 0.0, 0.0, 0.0
    return (
        float(insulation.thickness_m),
        float(insulation.thickness_m) / float(insulation.conductivity_W_mK),
        float(insulation.emissivity),
    )


def build_shell(
    grid: LiquidGrid,
    inner_radius: float,
    inner_height: float,
    fill_height: float,
    wall_thickness: float,
    base_thickness: float,
    wall: SolidProps,
    insulation: Any = None,
) -> tuple[Shell, ShellLayout]:
    """Build the shell mesh and its coupling to the liquid.

    Args:
        grid: the liquid grid (from `build_grid`).
        inner_radius: R, m.
        inner_height: H_v, m (inner height of the vessel, >= fill_height).
        fill_height: h, m.
        wall_thickness: t_w, m.
        base_thickness: t_b, m.
        wall: vessel material (density kg/m^3, specific_heat J/(kg K), conductivity W/(m K),
            emissivity -).
        insulation: None or an object with thickness_m (m), conductivity_W_mK (W/(m K)),
            emissivity (-) and covers (set of "side"/"base"/"lid").

    Returns:
        (Shell, ShellLayout).

    Raises:
        GeometryError: for non-positive thicknesses.
    """
    if wall_thickness <= 0 or base_thickness <= 0:
        raise GeometryError("wall and base thickness must be positive")
    R = inner_radius
    dx = grid.dx
    nx, ny, nz = grid.shape
    tw, tb = wall_thickness, base_thickness
    k_w = wall.conductivity
    rho_c = wall.density * wall.specific_heat

    foot = grid.is_liquid[:, :, 0]
    n_col = int(foot.sum())
    col_id = np.full((nx, ny), -1, dtype=np.int64)
    col_id[foot] = np.arange(n_col)
    cols_ij = np.argwhere(foot)  # (n_col, 2), same C order as col_id
    xs = grid.origin[0] + (np.arange(nx) + 0.5) * dx
    ys = grid.origin[1] + (np.arange(ny) + 0.5) * dx
    s = footprint_scale(n_col, dx, R)

    # ----- side mesh geometry
    n_theta = max(12, round(_TWO_PI * R / dx))
    dth = _TWO_PI / n_theta
    n_rows = ceil_cells(inner_height, dx)
    dz = np.full(n_rows, dx)
    dz[-1] = inner_height - (n_rows - 1) * dx
    z_lo = np.arange(n_rows) * dx
    z_row = z_lo + 0.5 * dz
    theta = (np.arange(n_theta, dtype=np.float64) + 0.5) * dth
    n_side = n_theta * n_rows
    n = n_col + n_side
    off = n_col

    # liquid height of each layer (all dx except the partial top layer)
    layer_h = np.full(nz, dx)
    layer_h[-1] = fill_height - (nz - 1) * dx
    if n_rows < nz:
        raise GeometryError("fill height exceeds inner height")

    # ----- node arrays
    kind = np.empty(n, dtype=np.int8)
    kind[:n_col] = ShellKind.BASE
    kind[n_col:] = ShellKind.SIDE
    position = np.empty((n, 3))
    position[:n_col, 0] = xs[cols_ij[:, 0]]
    position[:n_col, 1] = ys[cols_ij[:, 1]]
    position[:n_col, 2] = -tb / 2.0
    r_mid = R + tw / 2.0
    position[n_col:, 0] = np.tile(r_mid * np.cos(theta), n_rows)
    position[n_col:, 1] = np.tile(r_mid * np.sin(theta), n_rows)
    position[n_col:, 2] = np.repeat(z_row, n_theta)

    area = np.empty(n)
    area[:n_col] = dx * dx * s
    area[n_col:] = np.repeat(R * dth * dz, n_theta)
    thickness = np.empty(n)
    thickness[:n_col] = tb
    thickness[n_col:] = tw

    outer_face = np.empty(n, dtype=np.int8)
    outer_face[:n_col] = OuterFace.DOWN
    outer_face[n_col:] = OuterFace.SIDE
    outer_area = np.empty(n)
    outer_area[:n_col] = area[:n_col] * ((R + tw) / R) ** 2
    outer_area[n_col:] = np.repeat((R + tw) * dth * dz, n_theta)
    char_length = np.empty(n)
    char_length[:n_col] = 2.0 * (R + tw) / 4.0
    char_length[n_col:] = inner_height + tb

    emissivity = np.full(n, wall.emissivity)
    insulation_r = np.zeros(n)
    _, r_side, e_side = _insulation_parts(insulation, "side")
    _, r_base, e_base = _insulation_parts(insulation, "base")
    t_ins_side = _insulation_parts(insulation, "side")[0]
    if r_side > 0:
        insulation_r[n_col:] = r_side
        emissivity[n_col:] = e_side
    if r_base > 0:
        insulation_r[:n_col] = r_base
        emissivity[:n_col] = e_base

    # ----- in-plane edges
    ea: list[np.ndarray] = []
    eb: list[np.ndarray] = []
    eg: list[np.ndarray] = []

    def add_edges(a: np.ndarray, b: np.ndarray, g: np.ndarray) -> None:
        ea.append(np.minimum(a, b))
        eb.append(np.maximum(a, b))
        eg.append(g)

    # base: 4-neighbour columns, G = k t_b * dx / dx
    mx = foot[:-1, :] & foot[1:, :]
    a = col_id[:-1, :][mx]
    add_edges(a, col_id[1:, :][mx], np.full(len(a), k_w * tb))
    my = foot[:, :-1] & foot[:, 1:]
    a = col_id[:, :-1][my]
    add_edges(a, col_id[:, 1:][my], np.full(len(a), k_w * tb))

    # side: circumferential (periodic) and vertical
    a_idx = np.arange(n_theta)
    for b in range(n_rows):
        n1 = off + b * n_theta + a_idx
        n2 = off + b * n_theta + (a_idx + 1) % n_theta
        add_edges(n1, n2, np.full(n_theta, k_w * tw * dz[b] / (R * dth)))
    for b in range(n_rows - 1):
        n1 = off + b * n_theta + a_idx
        n2 = off + (b + 1) * n_theta + a_idx
        dist = 0.5 * (dz[b] + dz[b + 1])
        add_edges(n1, n2, np.full(n_theta, k_w * tw * R * dth / dist))

    # ----- wall <-> liquid coupling pattern, junction base <-> bottom ring
    p_bin, p_col, p_w = coupling_pattern(grid, col_id, n_theta)
    base_pos = position[p_col]
    side_pos = position[off + p_bin]
    dist = np.linalg.norm(base_pos - side_pos, axis=1)
    k_avg = k_w  # one wall material for base and side
    t_avg = 0.5 * (tb + tw)
    add_edges(p_col, off + p_bin, k_avg * t_avg * (R * dth * p_w) / dist)

    # side coupling rows: one block per liquid layer
    ci, cj = cols_ij[p_col, 0], cols_ij[p_col, 1]
    flat_ij = (ci * ny + cj) * nz
    cs_list = [np.arange(n_col, dtype=np.int64)]  # base: node c couples to cell (i, j, 0)
    cc_list = [(cols_ij[:, 0] * ny + cols_ij[:, 1]) * nz]
    ca_list = [area[:n_col].copy()]
    for k in range(nz):
        cs_list.append(off + k * n_theta + p_bin)
        cc_list.append(flat_ij + k)
        ca_list.append(p_w * (R * dth * layer_h[k]))
    couple_shell = np.concatenate(cs_list).astype(np.int64)
    couple_cell = np.concatenate(cc_list).astype(np.int64)
    couple_area = np.concatenate(ca_list)

    wetted = np.bincount(couple_shell, weights=couple_area, minlength=n).astype(np.float64)
    dry = area - wetted
    if np.any(dry < -1e-12 * area):  # pragma: no cover - guarded by construction
        raise GeometryError("internal error: wetted area exceeds node area")
    dry = np.maximum(dry, 0.0)

    edge_a = np.concatenate(ea).astype(np.int64)
    edge_b = np.concatenate(eb).astype(np.int64)
    edge_g = np.concatenate(eg)

    shell = Shell(
        n=n,
        kind=kind,
        position=position,
        area=area,
        thickness=thickness,
        rho_c=np.full(n, rho_c),
        k=np.full(n, k_w),
        emissivity=emissivity,
        edge_a=edge_a,
        edge_b=edge_b,
        edge_conductance=edge_g,
        couple_shell=couple_shell,
        couple_cell=couple_cell,
        couple_area=couple_area,
        wetted_area=wetted,
        dry_area=dry,
        outer_face=outer_face,
        outer_area=outer_area,
        insulation_r=insulation_r,
        char_length=char_length,
        outer_diameter=2.0 * (R + tw + t_ins_side),
    )
    layout = ShellLayout(
        base_index=col_id,
        n_theta=n_theta,
        n_rows=n_rows,
        side_offset=off,
        theta=theta,
        row_z=z_row,
    )
    return shell, layout
