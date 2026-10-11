# ThermoSim Interface Contracts

These are the shared types and signatures that let the work packages proceed
in parallel. WP00 commits §2–§4 *verbatim* as code under `src/thermosim/core/`.
After that they are frozen; changes go through the coordinator (README §6).

The other sections (§5–§11) are the public APIs each work package must
provide. The owning package implements them exactly; consumers rely on
nothing else.

---

## 1. Dependencies (pinned by WP00 in `pyproject.toml`)

| Package | Min version | Used by |
|---|---|---|
| python | 3.12 | all |
| numpy | 2.0 | all |
| numba | 0.60 | WP06, WP04 (optional) |
| scipy | 1.13 | WP02 (interpolation), WP10 (reference solutions) |
| pydantic | 2.7 | WP01 |
| tomli-w | 1.0 | WP01 (writing TOML; reading uses stdlib `tomllib`) |
| PySide6 | 6.7 | WP08, WP11 |
| pyqtgraph | 0.13.7 | WP08, WP11 |
| **dev:** pytest, pytest-qt, ruff, mypy, hypothesis | latest | all |

## 2. `thermosim/core/types.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

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
    DOWN = 0   # underside of the base (hot surface facing down)
    SIDE = 1   # vertical wall
    UP = 2     # top of lid / free liquid surface


class RunMode(StrEnum):
    FAST = "fast"
    REALTIME = "realtime"


# ---------------------------------------------------------------- geometry
@dataclass(frozen=True, eq=False)
class LiquidGrid:
    """Uniform Cartesian grid of the liquid. Index order [i, j, k] = [x, y, z]."""
    dx: float                       # m, cube edge length
    origin: F64                     # (3,) m, coordinates of the LOWER CORNER of cell (0,0,0)
    shape: tuple[int, int, int]     # (nx, ny, nz)
    is_liquid: B                    # (nx, ny, nz)
    volume: F64                     # (nx, ny, nz) m^3 of liquid in the cell (0 if not liquid;
                                    #   top layer may be partial: 0 < volume < dx^3)
    area_x: F64                     # (nx-1, ny, nz) m^2 face area between [i,j,k] and [i+1,j,k]
                                    #   (0 unless both are liquid; scaled by fill fraction in top layer)
    area_y: F64                     # (nx, ny-1, nz)
    area_z: F64                     # (nx, ny, nz-1) face between [i,j,k] and [i,j,k+1]
    surface_cells: I64              # (Ns,) flat indices of cells whose top face is the free surface
    surface_area: F64               # (Ns,) m^2 free-surface area of each (sum == pi*R^2 within 1e-9 rel)

    def cell_centers(self) -> tuple[F64, F64, F64]:
        """1-D arrays of cell-center coordinates along x, y, z (m)."""
        n = self.shape
        return tuple(self.origin[a] + (np.arange(n[a]) + 0.5) * self.dx for a in range(3))  # type: ignore[return-value]


@dataclass(frozen=True, eq=False)
class Shell:
    """Thin vessel wall + base as a 2-D surface mesh. One temperature per node."""
    n: int
    kind: I8                        # (n,) ShellKind
    position: F64                   # (n, 3) m, node center on the wall mid-surface
    area: F64                       # (n,) m^2 inner-surface area represented by node
    thickness: F64                  # (n,) m
    rho_c: F64                      # (n,) J/(m^3 K) volumetric heat capacity of wall material
    k: F64                          # (n,) W/(m K) wall conductivity (through- and in-plane)
    emissivity: F64                 # (n,) outer-surface emissivity (insulated nodes: insulation's)
    # in-plane conduction graph, undirected, each edge once (a < b)
    edge_a: I64                     # (E,)
    edge_b: I64                     # (E,)
    edge_conductance: F64           # (E,) W/K  (k * t * shared_edge_length / center_distance)
    # coupling to liquid (inner side). One row per (shell node, liquid cell) contact.
    couple_shell: I64               # (C,) shell node index
    couple_cell: I64                # (C,) flat liquid cell index
    couple_area: F64                # (C,) m^2 contact area; sum over a node <= area[node]
    wetted_area: F64                # (n,) m^2 = sum of couple_area per node
    dry_area: F64                   # (n,) m^2 = area - wetted_area (inner side facing headspace)
    # outer side
    outer_face: I8                  # (n,) OuterFace
    outer_area: F64                 # (n,) m^2 (side nodes: scaled by (R+t)/R for curvature)
    insulation_r: F64               # (n,) m^2 K/W insulation resistance on outer side (0 = none)
    char_length: F64                # (n,) m characteristic length for outer correlations
                                    #   (side: vessel outer height; base: outer diameter / 4 = A/P)
    outer_diameter: float           # m, 2*(R + wall thickness [+ insulation]) for cylinder crossflow


@dataclass(frozen=True)
class LidSpec:
    """Lumped lid and headspace. If present is False, the headspace is ambient air."""
    present: bool
    covered_fraction: float         # 0..1 (1 = full lid); ignored if not present
    area: float                     # m^2 lid area (= covered_fraction * pi*R^2)
    heat_capacity: float            # J/K lid thermal mass; must be > 0 when present (all fields 0 when not present)
    emissivity: float
    insulation_r: float             # m^2 K/W
    headspace_volume: float         # m^3 (for headspace air heat capacity)


@dataclass(frozen=True)
class Probe:
    name: str
    position: F64                   # (3,) m
    cells: I64                      # (8,) flat liquid cell indices for trilinear interpolation
    weights: F64                    # (8,) sum to 1; weights on non-liquid cells are 0 and renormalized


@dataclass(frozen=True)
class SliceSpec:
    """Vertical plane y = 0 through the axis."""
    j: int                          # grid j index of the slice (cell containing y=0; if y=0 is a
                                    #   cell boundary, the cell with center at +dx/2)
    x_centers: F64                  # (nx,) m
    z_centers: F64                  # (nz,) m
    wall_left: I64                  # shell node indices on the x<0 side wall, ordered by z ascending
    wall_right: I64                 # same, x>0 side
    base_line: I64                  # base shell nodes along y≈0, ordered by x ascending


@dataclass(frozen=True)
class Domain:
    grid: LiquidGrid
    shell: Shell
    lid: LidSpec
    probes: tuple[Probe, ...]
    slice: SliceSpec
    inner_radius: float             # m
    inner_height: float             # m
    fill_height: float              # m


@dataclass(frozen=True, eq=False)
class HeatDeposition:
    """Where a source's delivered power goes. Exactly one of the arrays is non-zero; it sums to 1."""
    cell_weights: F64               # (nx, ny, nz)
    shell_weights: F64              # (n_shell,)


# ---------------------------------------------------------------- solver I/O
@dataclass(eq=False)
class ThermalState:
    t: float                        # s, simulated time
    T_liquid: F64                   # (nx, ny, nz) K; value outside liquid is undefined (keep 0)
    T_shell: F64                    # (n,) K
    T_lid: float                    # K (== T_amb and unused if no lid)
    T_headspace: float              # K (== T_amb and unused if no lid)


@dataclass(eq=False)
class StepCoefficients:
    """Everything the solver needs that depends on physics. Linearized: q = h * (T - T_ref)."""
    T_amb: float                    # K
    k_liquid: F64                   # (nx, ny, nz) W/(m K) effective liquid conductivity
    rho_c_liquid: F64               # (nx, ny, nz) J/(m^3 K)
    h_film: F64                     # (C,) W/(m^2 K) liquid-side film coefficient per coupling row
    h_out_shell: F64                # (n,) W/(m^2 K) outer conv+rad to T_amb (insulation added in series by solver)
    h_dry: F64                      # (n,) W/(m^2 K) inner dry wall <-> headspace (or ambient if no lid)
    h_surface_open: F64             # (Ns,) W/(m^2 K) uncovered free surface conv+rad to T_amb
    q_evap_open: F64                # (Ns,) W/m^2 evaporation heat flux from uncovered surface (explicit, >=0)
    h_surface_covered: F64          # (Ns,) W/(m^2 K) covered free surface <-> headspace (incl. evap/condense)
    h_lid_inner: float              # W/(m^2 K) headspace <-> lid
    h_lid_outer: float              # W/(m^2 K) lid outer conv+rad to T_amb (insulation in series by solver)
    q_cells: F64                    # (nx, ny, nz) W heat source into liquid cells
    q_shell: F64                    # (n,) W heat source into shell nodes


@dataclass
class EnergyFlows:
    """Energy in J transferred during one step (or accumulated). All >= 0 except stored_* (signed)."""
    source_liquid: float = 0.0
    source_shell: float = 0.0
    loss_shell_outer: float = 0.0       # side + base outer surfaces to ambient
    loss_surface_convrad: float = 0.0   # uncovered free surface conv + rad
    loss_surface_evap: float = 0.0      # uncovered free surface evaporation
    loss_lid_outer: float = 0.0         # lid to ambient
    loss_dry_wall_open: float = 0.0     # dry inner wall to ambient (no-lid case only)
    stored_liquid: float = 0.0          # change in liquid internal energy
    stored_shell: float = 0.0
    stored_lid_headspace: float = 0.0

    def add(self, other: EnergyFlows) -> None: ...      # in-place accumulate
    def total_in(self) -> float: ...                     # source_liquid + source_shell
    def total_loss(self) -> float: ...                   # sum of loss_*
    def total_stored(self) -> float: ...                 # sum of stored_*
    def balance_error(self) -> float: ...                # total_in - total_loss - total_stored


# ---------------------------------------------------------------- outputs
@dataclass(frozen=True)
class Sample:
    """One row of the time series."""
    t: float                         # s
    probes: dict[str, float]         # K
    T_mean: float                    # K, volume-weighted liquid mean
    T_min: float
    T_max: float
    T_shell_mean: float              # K, area-weighted
    command: float                   # 0..1 requested power fraction
    power_delivered: float           # W into vessel/liquid
    power_gross: float               # W consumed (fuel energy rate or electrical)
    cumulative: EnergyFlows          # J since start
    fuel_kg: float                   # propane used (0 for electric)
    electric_kwh: float              # electricity used (0 for burner)


@dataclass(frozen=True)
class Snapshot:
    """Data needed to draw the slice at one instant. float32 to keep memory small."""
    t: float
    slice_T: F32                     # (nx, nz) K; NaN where not liquid. Index [i, k].
    x_centers: F64                   # (nx,) m
    z_centers: F64                   # (nz,) m
    wall_left_z: F64                 # (m,) m
    wall_left_T: F32                 # (m,) K
    wall_right_z: F64
    wall_right_T: F32
    base_x: F64                      # (b,) m
    base_T: F32                      # (b,) K
    lid_T: float | None              # K or None if no lid
    probes: dict[str, float]         # K
    T_mean: float                    # K
    power_delivered: float           # W
    velocity_u: F32 | None = None    # Phase 2: (nx, nz) m/s x-velocity in slice
    velocity_w: F32 | None = None    # Phase 2: (nx, nz) m/s z-velocity in slice


class StopReason(StrEnum):
    TARGET_REACHED = "target_reached"
    PLATEAU = "plateau"               # levelled off below target (losses == input)
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
    target_probe: str                 # probe name or "mean"
    final_probes: dict[str, float]
    final_T_mean: float
    energy: EnergyFlows               # cumulative J
    energy_balance_rel_error: float   # |balance_error| / max(total_in, 1)
    fuel_kg: float
    electric_kwh: float
    heating_efficiency: float         # stored_liquid / (gross energy)


@dataclass(frozen=True)
class RunResult:
    summary: Summary
    samples: list[Sample]
    snapshots: SnapshotStoreLike
    scenario_toml: str                # the exact scenario used, serialized


class SnapshotStoreLike(Protocol):    # (import Protocol from typing at top of module)
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Snapshot: ...
    def times(self) -> list[float]: ...
    def nearest(self, t: float) -> Snapshot: ...
```

## 3. `thermosim/core/errors.py`

```python
class ThermoSimError(Exception): ...
class ScenarioError(ThermoSimError): ...      # invalid / inconsistent scenario
class GeometryError(ThermoSimError): ...      # e.g. element above liquid line, probe outside liquid
class ModeError(ThermoSimError): ...          # e.g. set_power_command in FAST mode
class SolverError(ThermoSimError): ...        # NaN / instability detected
```

## 4. `thermosim/core/protocols.py`

```python
from __future__ import annotations
from enum import StrEnum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from .types import HeatDeposition, RunMode, RunResult, Sample, Snapshot, SnapshotStoreLike


class Actuator(Protocol):
    def output(self, command: float, t: float, dt: float) -> float:
        """Average fraction 0..1 of max power actually delivered over [t, t+dt]
        for a requested command 0..1. May keep internal state (SSR cycle, valve lag)."""
    def reset(self) -> None: ...


@dataclass(frozen=True)
class SourcePower:
    delivered_W: float     # into the vessel/liquid (after burner efficiency)
    gross_W: float         # fuel energy rate (burner, LHV basis) or electrical power


class HeatSource(Protocol):
    deposition: HeatDeposition
    rated_power_W: float   # gross rating
    def power(self, command: float, t: float, dt: float) -> SourcePower: ...
    def fuel_kg(self, gross_J: float) -> float: ...       # 0 for electric
    def electric_kwh(self, gross_J: float) -> float: ...  # 0 for burner


class CommandSource(Protocol):
    """Gives the requested power fraction 0..1 at time t (schedule, user, PID)."""
    def command(self, t: float, state_probe_K: float) -> float: ...


class RunnerState(StrEnum):
    IDLE = "idle"; RUNNING = "running"; PAUSED = "paused"
    FINISHED = "finished"; STOPPED = "stopped"; FAILED = "failed"


@dataclass(frozen=True)
class SnapshotEvent:
    snapshot: Snapshot

@dataclass(frozen=True)
class SampleEvent:
    sample: Sample

@dataclass(frozen=True)
class ProgressEvent:
    sim_time_s: float
    wall_time_s: float
    max_time_s: float
    realtime_lag_s: float          # REALTIME: how far behind the wall clock the sim is (0 in FAST)

@dataclass(frozen=True)
class FinishedEvent:
    result: RunResult

@dataclass(frozen=True)
class ErrorEvent:
    message: str
    traceback: str

RunnerEvent = SnapshotEvent | SampleEvent | ProgressEvent | FinishedEvent | ErrorEvent


@runtime_checkable
class Runner(Protocol):
    mode: RunMode
    state: RunnerState
    def start(self) -> None: ...            # non-blocking; runs in a worker thread
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...             # graceful; emits FinishedEvent with USER_STOP
    def set_power_command(self, value: float) -> None: ...  # REALTIME only, else ModeError
    def set_speed(self, factor: float) -> None: ...         # REALTIME only, factor >= 0.1
    def poll_events(self, max_events: int = 1000) -> list[RunnerEvent]: ...  # thread-safe, non-blocking
    @property
    def result(self) -> RunResult | None: ...
```

---

## 5. WP01: `thermosim.config`

```python
# config/schema.py — pydantic v2 models. All numeric fields:
#   Field(..., title="Inner diameter", description=..., ge=..., le=...,
#         json_schema_extra={"quantity": "<Quantity>"})
Quantity = Literal["temperature", "temperature_delta", "temperature_rate", "length", "altitude",
                   "area", "volume", "power", "speed", "time", "fraction", "specific_gravity",
                   "conductivity", "density", "specific_heat", "thermal_resistance", "dimensionless"]
# Display units: time is shown in seconds in both systems; altitude in m / ft;
# temperature_rate in °C/min / °F/min. Scenario(...) constructors may raise pydantic
# ValidationError for field-level errors; config.io converts everything to ScenarioError.

class CustomMaterial(BaseModel): density; specific_heat; conductivity; emissivity
class Vessel(BaseModel):
    shape: Literal["cylinder"] = "cylinder"
    inner_diameter_m: float; inner_height_m: float
    wall_thickness_m: float; base_thickness_m: float
    material: str | CustomMaterial = "stainless_304"     # str = key in physics.properties.SOLIDS
class Lid(BaseModel):
    mode: Literal["none", "full", "partial"] = "none"
    covered_fraction: float = 1.0         # used when mode == "partial" (0 < f < 1)
    material: str | CustomMaterial = "stainless_304"; thickness_m: float = 0.001
class Insulation(BaseModel):
    thickness_m: float; conductivity_W_mK: float; emissivity: float = 0.9
    covers: set[Literal["side", "base", "lid"]] = {"side"}
class Liquid(BaseModel):
    volume_m3: float | None = None; fill_height_m: float | None = None   # exactly one set
    initial_temp_K: float; specific_gravity: float = 1.000
class OnOffActuator(BaseModel):    type: Literal["on_off"] = "on_off"; cycle_period_s: float = 2.0
class VariableActuator(BaseModel): type: Literal["variable"] = "variable"
class GasValveActuator(BaseModel): type: Literal["gas_valve"] = "gas_valve"
                                   min_fraction: float = 0.2; response_time_s: float = 2.0
Actuator = Annotated[OnOffActuator | VariableActuator | GasValveActuator, Field(discriminator="type")]
class Burner(BaseModel):
    type: Literal["burner"] = "burner"; rated_power_W: float; efficiency: float = 0.4
    footprint: Literal["disk", "ring"] = "disk"
    footprint_outer_diameter_m: float; footprint_inner_diameter_m: float = 0.0
    actuator: Actuator = GasValveActuator()
class Element(BaseModel):
    type: Literal["element"] = "element"; rated_power_W: float
    path_m: list[tuple[float, float, float]]   # >= 2 points, vessel coordinates (D2)
    diameter_m: float = 0.01
    actuator: Actuator = OnOffActuator()
HeatSourceCfg = Annotated[Burner | Element, Field(discriminator="type")]
class Atmosphere(BaseModel):
    ambient_temp_K: float; wind_speed_m_s: float = 0.0
    relative_humidity: float = 0.5; altitude_m: float = 0.0
class ProbeCfg(BaseModel): name: str; position_m: tuple[float, float, float]
class Target(BaseModel): temp_K: float; probe: str = "mean"    # probe name or "mean"
class PowerStep(BaseModel): t_s: float; command: float          # 0..1
class SimSettings(BaseModel):
    cell_size_m: float = 0.005; max_sim_time_s: float = 4 * 3600
    mode: RunMode = RunMode.FAST; realtime_speed: float = 1.0
    sample_interval_s: float = 1.0; snapshot_interval_s: float = 5.0
    coeff_refresh_s: float = 1.0
    plateau_window_s: float = 600.0; plateau_rate_K_per_min: float = 0.05
class ModelSettings(BaseModel):
    liquid_transport: Literal["effective_conductivity", "flow"] = "effective_conductivity"
    k_eff_scale: float = 1.0
class Scenario(BaseModel):
    schema_version: Literal[1] = 1
    name: str; vessel: Vessel; lid: Lid = Lid(); insulation: Insulation | None = None
    liquid: Liquid; heat_source: HeatSourceCfg; atmosphere: Atmosphere
    probes: list[ProbeCfg]; target: Target
    power_schedule: list[PowerStep] = [PowerStep(t_s=0, command=1.0)]
    simulation: SimSettings = SimSettings(); model: ModelSettings = ModelSettings()
    # model validators: see WP01 for the full cross-field rule list

# config/io.py
def load_scenario(path: Path) -> Scenario: ...        # raises ScenarioError with readable message
def save_scenario(scenario: Scenario, path: Path) -> None: ...
def scenario_to_toml(scenario: Scenario) -> str: ...
def scenario_from_toml(text: str) -> Scenario: ...
def example_scenarios() -> dict[str, Scenario]: ...    # loads package data scenarios/*.toml

# config/units.py
UnitSystem = Literal["metric", "us"]
def to_display(value: float | np.ndarray, quantity: Quantity, system: UnitSystem) -> float | np.ndarray
def from_display(value, quantity, system) -> float | np.ndarray
def unit_label(quantity: Quantity, system: UnitSystem) -> str        # e.g. "°F", "gal", "BTU/hr"
def quantity_of(model: type[BaseModel], field_name: str) -> Quantity | None
```

## 6. WP02: `thermosim.physics.properties`

All functions are vectorized. `T` in K, `sg` dimensionless, `p` in Pa.

```python
@dataclass(frozen=True)
class SolidProps:
    name: str; density: float; specific_heat: float; conductivity: float; emissivity: float
SOLIDS: dict[str, SolidProps]   # keys: stainless_304, aluminum, copper, carbon_steel
def solid_props(material: str | object) -> SolidProps   # accepts key or CustomMaterial-like (duck-typed)

def wort_density(T, sg) -> ...              # kg/m^3
def wort_specific_heat(T, sg) -> ...        # J/(kg K)
def wort_conductivity(T, sg) -> ...         # W/(m K)
def wort_viscosity(T, sg) -> ...            # Pa s (dynamic)
def wort_expansion(T, sg) -> ...            # 1/K volumetric thermal expansion coefficient
def wort_boiling_point(sg, p) -> ...        # K
def latent_heat_vaporization(T) -> ...      # J/kg (water)
def saturation_pressure(T) -> ...           # Pa (water)
def pressure_at_altitude(altitude_m) -> ... # Pa (standard atmosphere)
def sg_to_plato(sg) -> ...; def plato_to_sg(plato) -> ...

@dataclass(frozen=True)
class AirProps: density; specific_heat; conductivity; viscosity; kinematic_viscosity; diffusivity; prandtl; expansion
def air_props(T, p) -> AirProps             # fields are arrays matching T's shape
def vapor_diffusivity_in_air(T, p) -> ...   # m^2/s

PROPANE_LHV_J_PER_KG: float = 46.35e6
STEFAN_BOLTZMANN: float = 5.670374419e-8
G: float = 9.80665
```

## 7. WP03: `thermosim.physics.correlations`

Every function returns W/m² K (`h_*`) or W/m² (`q_*`, positive = leaving the
surface), and is vectorized.

```python
def h_natural_vertical(T_s, T_inf, height, p) -> ...            # Churchill–Chu, air side
def h_natural_horizontal_up(T_s, T_inf, L, p) -> ...            # hot-up/cold-down plate
def h_natural_horizontal_down(T_s, T_inf, L, p) -> ...
def h_forced_cylinder(T_s, T_inf, D, U, p) -> ...               # Churchill–Bernstein
def h_forced_plate(T_s, T_inf, L, U, p) -> ...                  # laminar/turbulent flat plate
def h_mixed(h_nat, h_forced) -> ...                             # (h_n^3 + h_f^3)^(1/3)
def h_radiation(T_s, T_inf, emissivity) -> ...                  # exact linearization eps*sigma*(Ts^2+Ti^2)(Ts+Ti)
def h_outer(T_s, T_inf, face: OuterFace | array, char_length, wind, p, emissivity, diameter=None) -> ...
                                                                # diameter: cylinder crossflow D (pass Shell.outer_diameter); defaults to char_length
                                                                # convenience: picks + mixes + adds radiation
def q_evaporation(T_surf, T_amb, rh, p, h_conv) -> ...          # W/m^2 via Chilton–Colburn analogy; 0 if negative
def evaporation_mass_flux(T_surf, T_amb, rh, p, h_conv) -> ...  # kg/(m^2 s)
def h_film_liquid(T_wall, T_bulk, orientation: OuterFace|array, L, sg) -> ...
    # liquid-side natural convection film coefficient for wall->liquid (Phase 1). orientation is
    # the direction the WALL faces into the liquid: base → UP (heated plate facing up), side → SIDE.
def h_headspace(T_liquid_surface, T_lid, sg, p, gap=0.10) -> ...  # covered surface <-> headspace incl. condensation; gap = H_v - fill (m)
def k_effective(T_bulk, T_hot, depth, sg, scale=1.0) -> ...     # W/(m K) = k_mol * max(1, scale * C * Ra^(1/3)); always >= k_mol
```

## 8. WP04: `thermosim.geometry`

```python
def build_domain(scenario: Scenario) -> Domain                 # raises GeometryError / ScenarioError
def fill_height_from_volume(volume_m3: float, inner_radius: float) -> float
def extract_slice(domain: Domain, T_liquid: F64) -> F32        # (nx, nz) with NaN outside liquid
def probe_values(domain: Domain, T_liquid: F64) -> dict[str, float]
def cell_index_at(grid: LiquidGrid, point: F64) -> tuple[int, int, int] | None
```

## 9. WP05: `thermosim.sources`

```python
def build_heat_source(scenario: Scenario, domain: Domain) -> HeatSource
def build_actuator(cfg) -> Actuator
def deposition_for_burner(cfg: Burner, domain: Domain) -> HeatDeposition
def deposition_for_element(cfg: Element, domain: Domain) -> HeatDeposition   # GeometryError if any path above liquid
class ScheduleCommand(CommandSource): ...          # piecewise-constant from Scenario.power_schedule
class ManualCommand(CommandSource):                # thread-safe; used in REALTIME
    def set(self, value: float) -> None: ...
```

## 10. WP06: `thermosim.solver`

```python
class ThermalSolver:
    def __init__(self, domain: Domain) -> None: ...         # precomputes index arrays; triggers Numba compile
    def initial_state(self, T0_liquid: float, T0_shell: float, T_amb: float) -> ThermalState: ...  # lid & headspace start at T_amb
    def stable_dt(self, coeffs: StepCoefficients) -> float: ...   # s, liquid stability limit * 0.9
    def step(self, state: ThermalState, coeffs: StepCoefficients, dt: float) -> EnergyFlows: ...
        # advances state IN PLACE by dt (state.t += dt). Shell/lid/headspace may substep internally.
        # Raises SolverError if dt > stable_dt or any NaN appears.
    def internal_energy(self, state: ThermalState, coeffs: StepCoefficients) -> float: ...  # J, reference 0 K
```

## 11. WP07: `thermosim.recording`

```python
class Recorder:
    def add(self, sample: Sample) -> None: ...
    @property
    def samples(self) -> list[Sample]: ...
    def to_arrays(self) -> dict[str, F64]: ...   # columns: t, T_mean, ..., probe_<name>, energy_<field>
class SnapshotStore(SnapshotStoreLike):
    def __init__(self, max_in_memory: int | None = None, spill_dir: Path | None = None): ...
    def add(self, snap: Snapshot) -> None: ...
    def save(self, path: Path) -> None: ...      # .npz
    @classmethod
    def load(cls, path: Path) -> SnapshotStore: ...
def summarize(samples: list[Sample], target_temp_K: float, target_probe: str,
              stop_reason: StopReason, wall_time_s: float) -> Summary
def export_csv(samples: list[Sample], path: Path, system: UnitSystem = "metric") -> None
def save_run(result: RunResult, directory: Path) -> None   # summary.json, timeseries.csv, snapshots.npz, scenario.toml
def load_run(directory: Path) -> RunResult
@dataclass(frozen=True)
class MeasuredSeries: t: F64; T: F64; label: str          # SI
def load_measured_csv(path: Path, time_col: str | None = None, temp_col: str | None = None,
                      temp_unit: Literal["C", "F", "K"] = "C", time_unit: Literal["s", "min"] = "s") -> MeasuredSeries
@dataclass(frozen=True)
class Comparison: rmse_K: float; max_abs_K: float; bias_K: float
                  time_to_target_sim_s: float | None; time_to_target_meas_s: float | None
                  time_to_target_error_rel: float | None
def compare(samples: list[Sample], probe: str, measured: MeasuredSeries, target_temp_K: float | None) -> Comparison
    # probe: a probe name or "mean" (T_mean)
```

## 12. WP09: `thermosim.sim`

```python
class Model:
    def __init__(self, scenario: Scenario, command: CommandSource) -> None: ...
    def advance(self, until_t: float) -> list[Sample | Snapshot]: ...   # steps until state.t >= until_t
    @property
    def stop_reason(self) -> StopReason | None: ...
    state: ThermalState; domain: Domain
class SimulationRunner(Runner):
    def __init__(self, scenario: Scenario, mode: RunMode | None = None) -> None: ...
def run_headless(scenario: Scenario, progress: Callable[[ProgressEvent], None] | None = None) -> RunResult
```
