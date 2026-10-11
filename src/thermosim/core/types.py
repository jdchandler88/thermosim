"""Shared dataclasses and enums (CONTRACTS.md section 2). Frozen after WP00.

Coordinates (D2): origin at the centre of the inner base, z up, slice plane y = 0.
Array index order is [i, j, k] = [x, y, z] (D3). Temperatures are in kelvin (D1).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum, StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
F32 = NDArray[np.float32]
I64 = NDArray[np.int64]
I8 = NDArray[np.int8]
B = NDArray[np.bool_]


class ShellKind(IntEnum):
    BASE = 0
    SIDE = 1


class OuterFace(IntEnum):
    """Orientation of a surface's OUTSIDE, used to choose a correlation."""

    DOWN = 0  # underside of the base (hot surface facing down)
    SIDE = 1  # vertical wall
    UP = 2  # top of lid / free liquid surface


class RunMode(StrEnum):
    FAST = "fast"
    REALTIME = "realtime"


# ---------------------------------------------------------------- geometry
@dataclass(frozen=True, eq=False)
class LiquidGrid:
    """Uniform Cartesian grid of the liquid. Index order [i, j, k] = [x, y, z]."""

    dx: float  # m, cube edge length
    origin: F64  # (3,) m, coordinates of the LOWER CORNER of cell (0,0,0)
    shape: tuple[int, int, int]  # (nx, ny, nz)
    is_liquid: B  # (nx, ny, nz)
    volume: F64  # (nx, ny, nz) m^3 of liquid in the cell (0 if not liquid;
    #   top layer may be partial: 0 < volume < dx^3)
    area_x: F64  # (nx-1, ny, nz) m^2 face area between [i,j,k] and [i+1,j,k]
    #   (0 unless both are liquid; scaled by fill fraction in top layer)
    area_y: F64  # (nx, ny-1, nz)
    area_z: F64  # (nx, ny, nz-1) face between [i,j,k] and [i,j,k+1]
    surface_cells: I64  # (Ns,) flat indices of cells whose top face is the free surface
    surface_area: F64  # (Ns,) m^2 free-surface area of each (sum == pi*R^2 within 1e-9 rel)

    def cell_centers(self) -> tuple[F64, F64, F64]:
        """1-D arrays of cell-center coordinates along x, y, z (m)."""
        nx, ny, nz = self.shape
        xs = self.origin[0] + (np.arange(nx, dtype=np.float64) + 0.5) * self.dx
        ys = self.origin[1] + (np.arange(ny, dtype=np.float64) + 0.5) * self.dx
        zs = self.origin[2] + (np.arange(nz, dtype=np.float64) + 0.5) * self.dx
        return xs, ys, zs


@dataclass(frozen=True, eq=False)
class Shell:
    """Thin vessel wall + base as a 2-D surface mesh. One temperature per node."""

    n: int
    kind: I8  # (n,) ShellKind
    position: F64  # (n, 3) m, node center on the wall mid-surface
    area: F64  # (n,) m^2 inner-surface area represented by node
    thickness: F64  # (n,) m
    rho_c: F64  # (n,) J/(m^3 K) volumetric heat capacity of wall material
    k: F64  # (n,) W/(m K) wall conductivity (through- and in-plane)
    emissivity: F64  # (n,) outer-surface emissivity (insulated nodes: insulation's)
    # in-plane conduction graph, undirected, each edge once (a < b)
    edge_a: I64  # (E,)
    edge_b: I64  # (E,)
    edge_conductance: F64  # (E,) W/K  (k * t * shared_edge_length / center_distance)
    # coupling to liquid (inner side). One row per (shell node, liquid cell) contact.
    couple_shell: I64  # (C,) shell node index
    couple_cell: I64  # (C,) flat liquid cell index
    couple_area: F64  # (C,) m^2 contact area; sum over a node <= area[node]
    wetted_area: F64  # (n,) m^2 = sum of couple_area per node
    dry_area: F64  # (n,) m^2 = area - wetted_area (inner side facing headspace)
    # outer side
    outer_face: I8  # (n,) OuterFace
    outer_area: F64  # (n,) m^2 (side nodes: scaled by (R+t)/R for curvature)
    insulation_r: F64  # (n,) m^2 K/W insulation resistance on outer side (0 = none)
    char_length: F64  # (n,) m characteristic length for outer correlations
    #   (side: vessel outer height; base: outer diameter / 4 = A/P)
    outer_diameter: float  # m, 2*(R + wall thickness [+ insulation]) for cylinder crossflow


@dataclass(frozen=True)
class LidSpec:
    """Lumped lid and headspace. If present is False, the headspace is ambient air."""

    present: bool
    covered_fraction: float  # 0..1 (1 = full lid); ignored if not present
    area: float  # m^2 lid area (= covered_fraction * pi*R^2)
    heat_capacity: float  # J/K lid thermal mass
    emissivity: float
    insulation_r: float  # m^2 K/W
    headspace_volume: float  # m^3 (for headspace air heat capacity)


@dataclass(frozen=True)
class Probe:
    name: str
    position: F64  # (3,) m
    cells: I64  # (8,) flat liquid cell indices for trilinear interpolation
    weights: F64  # (8,) sum to 1; weights on non-liquid cells are 0 and renormalized


@dataclass(frozen=True)
class SliceSpec:
    """Vertical plane y = 0 through the axis."""

    j: int  # grid j index of the slice (cell containing y=0; if y=0 is a
    #   cell boundary, the cell with center at +dx/2)
    x_centers: F64  # (nx,) m
    z_centers: F64  # (nz,) m
    wall_left: I64  # shell node indices on the x<0 side wall, ordered by z ascending
    wall_right: I64  # same, x>0 side
    base_line: I64  # base shell nodes along y≈0, ordered by x ascending


@dataclass(frozen=True)
class Domain:
    grid: LiquidGrid
    shell: Shell
    lid: LidSpec
    probes: tuple[Probe, ...]
    slice: SliceSpec
    inner_radius: float  # m
    inner_height: float  # m
    fill_height: float  # m


@dataclass(frozen=True, eq=False)
class HeatDeposition:
    """Where a source's delivered power goes.

    Exactly one of the arrays is non-zero; it sums to 1.
    """

    cell_weights: F64  # (nx, ny, nz)
    shell_weights: F64  # (n_shell,)


# ---------------------------------------------------------------- solver I/O
@dataclass(eq=False)
class ThermalState:
    t: float  # s, simulated time
    T_liquid: F64  # (nx, ny, nz) K; value outside liquid is undefined (keep 0)
    T_shell: F64  # (n,) K
    T_lid: float  # K (== T_amb and unused if no lid)
    T_headspace: float  # K (== T_amb and unused if no lid)


@dataclass(eq=False)
class StepCoefficients:
    """Everything the solver needs that depends on physics. Linearized: q = h * (T - T_ref)."""

    T_amb: float  # K
    k_liquid: F64  # (nx, ny, nz) W/(m K) effective liquid conductivity
    rho_c_liquid: F64  # (nx, ny, nz) J/(m^3 K)
    h_film: F64  # (C,) W/(m^2 K) liquid-side film coefficient per coupling row
    h_out_shell: (
        F64  # (n,) W/(m^2 K) outer conv+rad to T_amb (insulation added in series by solver)
    )
    h_dry: F64  # (n,) W/(m^2 K) inner dry wall <-> headspace (or ambient if no lid)
    h_surface_open: F64  # (Ns,) W/(m^2 K) uncovered free surface conv+rad to T_amb
    q_evap_open: F64  # (Ns,) W/m^2 evaporation heat flux from uncovered surface (explicit, >=0)
    h_surface_covered: (
        F64  # (Ns,) W/(m^2 K) covered free surface <-> headspace (incl. evap/condense)
    )
    h_lid_inner: float  # W/(m^2 K) headspace <-> lid
    h_lid_outer: float  # W/(m^2 K) lid outer conv+rad to T_amb (insulation in series by solver)
    q_cells: F64  # (nx, ny, nz) W heat source into liquid cells
    q_shell: F64  # (n,) W heat source into shell nodes


@dataclass
class EnergyFlows:
    """Energy in J transferred during one step (or accumulated).

    All >= 0 except stored_* (signed).
    """

    source_liquid: float = 0.0
    source_shell: float = 0.0
    loss_shell_outer: float = 0.0  # side + base outer surfaces to ambient
    loss_surface_convrad: float = 0.0  # uncovered free surface conv + rad
    loss_surface_evap: float = 0.0  # uncovered free surface evaporation
    loss_lid_outer: float = 0.0  # lid to ambient
    loss_dry_wall_open: float = 0.0  # dry inner wall to ambient (no-lid case only)
    stored_liquid: float = 0.0  # change in liquid internal energy
    stored_shell: float = 0.0
    stored_lid_headspace: float = 0.0

    def add(self, other: EnergyFlows) -> None:  # in-place accumulate
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def total_in(self) -> float:  # source_liquid + source_shell
        return self.source_liquid + self.source_shell

    def total_loss(self) -> float:  # sum of loss_*
        return (
            self.loss_shell_outer
            + self.loss_surface_convrad
            + self.loss_surface_evap
            + self.loss_lid_outer
            + self.loss_dry_wall_open
        )

    def total_stored(self) -> float:  # sum of stored_*
        return self.stored_liquid + self.stored_shell + self.stored_lid_headspace

    def balance_error(self) -> float:  # total_in - total_loss - total_stored
        return self.total_in() - self.total_loss() - self.total_stored()


# ---------------------------------------------------------------- outputs
@dataclass(frozen=True)
class Sample:
    """One row of the time series."""

    t: float  # s
    probes: dict[str, float]  # K
    T_mean: float  # K, volume-weighted liquid mean
    T_min: float
    T_max: float
    T_shell_mean: float  # K, area-weighted
    command: float  # 0..1 requested power fraction
    power_delivered: float  # W into vessel/liquid
    power_gross: float  # W consumed (fuel energy rate or electrical)
    cumulative: EnergyFlows  # J since start
    fuel_kg: float  # propane used (0 for electric)
    electric_kwh: float  # electricity used (0 for burner)


@dataclass(frozen=True)
class Snapshot:
    """Data needed to draw the slice at one instant. float32 to keep memory small."""

    t: float
    slice_T: F32  # (nx, nz) K; NaN where not liquid. Index [i, k].
    x_centers: F64  # (nx,) m
    z_centers: F64  # (nz,) m
    wall_left_z: F64  # (m,) m
    wall_left_T: F32  # (m,) K
    wall_right_z: F64
    wall_right_T: F32
    base_x: F64  # (b,) m
    base_T: F32  # (b,) K
    lid_T: float | None  # K or None if no lid
    probes: dict[str, float]  # K
    T_mean: float  # K
    power_delivered: float  # W
    velocity_u: F32 | None = None  # Phase 2: (nx, nz) m/s x-velocity in slice
    velocity_w: F32 | None = None  # Phase 2: (nx, nz) m/s z-velocity in slice


class StopReason(StrEnum):
    TARGET_REACHED = "target_reached"
    PLATEAU = "plateau"  # levelled off below target (losses == input)
    MAX_TIME = "max_time"
    USER_STOP = "user_stop"
    ERROR = "error"


@dataclass(frozen=True)
class Summary:
    stop_reason: StopReason
    sim_time_s: float
    wall_time_s: float
    time_to_target_s: float | None
    target_temp_K: float
    target_probe: str  # probe name or "mean"
    final_probes: dict[str, float]
    final_T_mean: float
    energy: EnergyFlows  # cumulative J
    energy_balance_rel_error: float  # |balance_error| / max(total_in, 1)
    fuel_kg: float
    electric_kwh: float
    heating_efficiency: float  # stored_liquid / (gross energy)


@dataclass(frozen=True)
class RunResult:
    summary: Summary
    samples: list[Sample]
    snapshots: SnapshotStoreLike
    scenario_toml: str  # the exact scenario used, serialized


class SnapshotStoreLike(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Snapshot: ...
    def times(self) -> list[float]: ...
    def nearest(self, t: float) -> Snapshot: ...
