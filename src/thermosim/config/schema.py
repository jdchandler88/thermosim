"""Scenario schema (CONTRACTS.md section 5).

A `Scenario` is the complete user setup: vessel, lid, insulation, liquid, heat source,
atmosphere, probes, target, power schedule and simulation settings. All values are SI
(D1): metres, seconds, kilograms, watts, kelvin. Display units are handled by
`thermosim.config.units`.

Every numeric field carries `json_schema_extra={"quantity": ...}`, so the GUI can build
its forms and unit conversions from the schema. Field order is form order.

Validation has two layers:

* Field-level checks (types, `ge`/`le`/`gt`, lengths) and the material key check raise
  pydantic's `ValidationError` when a model is constructed.
* Cross-field rules are model validators that raise `ScenarioError` directly. The
  message names the field path. `thermosim.config.io` turns a `ValidationError` into a
  `ScenarioError` that also names the file.

Coordinates (D2): origin at the centre of the inner base, z up. Element and probe
positions are given in these coordinates.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from thermosim.core.errors import ScenarioError
from thermosim.core.types import RunMode

Quantity = Literal[
    "temperature",
    "temperature_delta",
    "temperature_rate",
    "length",
    "altitude",
    "area",
    "volume",
    "power",
    "speed",
    "time",
    "fraction",
    "specific_gravity",
    "conductivity",
    "density",
    "specific_heat",
    "thermal_resistance",
    "dimensionless",
]

# Material keys known to physics.properties.SOLIDS. Hard-coded on purpose so that config
# does not import physics (WP01 spec, requirement 2). Keep in sync with WP02.
MATERIAL_KEYS: tuple[str, ...] = ("stainless_304", "aluminum", "copper", "carbon_steel")

# A coordinate in vessel space (m). The bounds are plausibility limits for the schema;
# the real limits come from the vessel dimensions, checked in Scenario.
Coord = Annotated[float, Field(ge=-5.0, le=5.0)]


def _num(
    quantity: Quantity,
    title: str,
    description: str,
    default: Any = ...,
    **constraints: Any,
) -> Any:
    """A numeric field with the metadata the GUI needs. `default=...` means required."""
    return Field(
        default=default,
        title=title,
        description=description,
        json_schema_extra={"quantity": quantity},
        **constraints,
    )


# Typed default factories. Lambdas here confuse mypy's pydantic Field overloads.
def _default_covers() -> set[Literal["side", "base", "lid"]]:
    return {"side"}


def _default_gas_valve() -> GasValveActuator:
    return GasValveActuator()


def _default_on_off() -> OnOffActuator:
    return OnOffActuator()


def _default_lid() -> Lid:
    return Lid()


def _default_sim() -> SimSettings:
    return SimSettings()


def _default_model() -> ModelSettings:
    return ModelSettings()


def _default_schedule() -> list[PowerStep]:
    return [PowerStep(t_s=0.0, command=1.0)]


def _disk_area(diameter_m: float) -> float:
    return math.pi * (diameter_m / 2.0) ** 2


# --------------------------------------------------------------------------- materials
class CustomMaterial(BaseModel):
    """User-defined wall material, used instead of a preset key."""

    model_config = ConfigDict(extra="forbid")

    density: float = _num("density", "Density", "Solid density in kg/m^3.", gt=0.0, le=25000.0)
    specific_heat: float = _num(
        "specific_heat",
        "Specific heat",
        "Specific heat capacity in J/(kg K).",
        gt=0.0,
        le=10000.0,
    )
    conductivity: float = _num(
        "conductivity",
        "Thermal conductivity",
        "Thermal conductivity in W/(m K).",
        gt=0.0,
        le=1000.0,
    )
    emissivity: float = _num(
        "fraction",
        "Emissivity",
        "Surface emissivity (0 to 1) used for radiation.",
        ge=0.0,
        le=1.0,
    )


# --------------------------------------------------------------------------- vessel
class Vessel(BaseModel):
    """Cylindrical vessel. The origin is the centre of the inner base (D2)."""

    model_config = ConfigDict(extra="forbid")

    shape: Literal["cylinder"] = Field(
        default="cylinder", title="Shape", description="Vessel shape. Only cylinders are supported."
    )
    inner_diameter_m: float = _num(
        "length",
        "Inner diameter",
        "Inside diameter of the vessel in metres.",
        gt=0.0,
        le=3.0,
    )
    inner_height_m: float = _num(
        "length",
        "Inner height",
        "Inside height of the vessel, from the inner base to the inside of the top, in metres.",
        gt=0.0,
        le=3.0,
    )
    wall_thickness_m: float = _num(
        "length",
        "Wall thickness",
        "Thickness of the side wall in metres.",
        gt=0.0,
        le=0.02,
    )
    base_thickness_m: float = _num(
        "length",
        "Base thickness",
        "Thickness of the base in metres.",
        gt=0.0,
        le=0.05,
    )
    material: str | CustomMaterial = Field(
        default="stainless_304",
        title="Material",
        description=(
            "Wall material: a preset key (stainless_304, aluminum, copper, carbon_steel) "
            "or a custom material table."
        ),
    )

    @field_validator("material")
    @classmethod
    def _known_material(cls, value: str | CustomMaterial) -> str | CustomMaterial:
        if isinstance(value, str) and value not in MATERIAL_KEYS:
            raise ValueError(
                f"unknown material {value!r}; expected one of {', '.join(MATERIAL_KEYS)} "
                "or a custom material table"
            )
        return value


# --------------------------------------------------------------------------- lid
class Lid(BaseModel):
    """Lid over the liquid. `partial` covers a fraction of the top."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["none", "full", "partial"] = Field(
        default="none", title="Lid", description="No lid, a full lid, or a partial lid."
    )
    covered_fraction: float = _num(
        "fraction",
        "Covered fraction",
        "Fraction of the vessel top covered by the lid. Used only when mode is 'partial' "
        "(0 < value < 1). Forced to 1 for 'full' and ignored for 'none'.",
        1.0,
        ge=0.0,
        le=1.0,
    )
    material: str | CustomMaterial = Field(
        default="stainless_304",
        title="Lid material",
        description="Lid material: a preset key or a custom material table.",
    )
    thickness_m: float = _num(
        "length",
        "Lid thickness",
        "Lid thickness in metres.",
        0.001,
        gt=0.0,
        le=0.02,
    )

    @field_validator("material")
    @classmethod
    def _known_material(cls, value: str | CustomMaterial) -> str | CustomMaterial:
        if isinstance(value, str) and value not in MATERIAL_KEYS:
            raise ValueError(
                f"unknown material {value!r}; expected one of {', '.join(MATERIAL_KEYS)} "
                "or a custom material table"
            )
        return value

    @model_validator(mode="after")
    def _check_mode(self) -> Self:
        if self.mode == "partial" and not (0.0 < self.covered_fraction < 1.0):
            raise ScenarioError(
                "lid.covered_fraction: must satisfy 0 < covered_fraction < 1 when "
                f"lid.mode is 'partial' (got {self.covered_fraction})"
            )
        if self.mode == "full":
            self.covered_fraction = 1.0
        return self


# --------------------------------------------------------------------------- insulation
class Insulation(BaseModel):
    """Optional insulation on the outside of the vessel."""

    model_config = ConfigDict(extra="forbid")

    thickness_m: float = _num(
        "length",
        "Insulation thickness",
        "Thickness of the insulation layer in metres.",
        gt=0.0,
        le=0.5,
    )
    conductivity_W_mK: float = _num(
        "conductivity",
        "Insulation conductivity",
        "Thermal conductivity of the insulation in W/(m K).",
        gt=0.0,
        le=10.0,
    )
    emissivity: float = _num(
        "fraction",
        "Insulation emissivity",
        "Outer-surface emissivity of the insulation (0 to 1).",
        0.9,
        ge=0.0,
        le=1.0,
    )
    covers: set[Literal["side", "base", "lid"]] = Field(
        default_factory=_default_covers,
        title="Covers",
        description="Which surfaces are insulated: any of side, base, lid. 'lid' requires a lid.",
    )


# --------------------------------------------------------------------------- liquid
class Liquid(BaseModel):
    """The liquid in the vessel. Give exactly one of volume and fill height."""

    model_config = ConfigDict(extra="forbid")

    volume_m3: float | None = _num(
        "volume",
        "Volume",
        "Liquid volume in m^3. Set this or fill height, not both.",
        None,
        gt=0.0,
        le=10.0,
    )
    fill_height_m: float | None = _num(
        "length",
        "Fill height",
        "Liquid depth above the inner base in metres. Set this or volume, not both.",
        None,
        gt=0.0,
        le=3.0,
    )
    initial_temp_K: float = _num(
        "temperature",
        "Initial temperature",
        "Starting temperature of the liquid in kelvin.",
        ge=273.15,
        le=373.15,
    )
    specific_gravity: float = _num(
        "specific_gravity",
        "Specific gravity",
        "Specific gravity of the liquid. Water is 1.000; wort is typically 1.000 to 1.120.",
        1.0,
        ge=0.9,
        le=1.3,
    )

    @model_validator(mode="after")
    def _exactly_one_fill(self) -> Self:
        if (self.volume_m3 is None) == (self.fill_height_m is None):
            raise ScenarioError(
                "liquid: set exactly one of volume_m3 and fill_height_m "
                f"(got volume_m3={self.volume_m3}, fill_height_m={self.fill_height_m})"
            )
        return self


# --------------------------------------------------------------------------- heat sources
class OnOffActuator(BaseModel):
    """On/off switching (SSR) with time-proportioning over a fixed cycle."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["on_off"] = Field(default="on_off", title="Type", description="Actuator type.")
    cycle_period_s: float = _num(
        "time",
        "Cycle period",
        "Length of one on/off cycle in seconds. The command sets the on-share of each cycle.",
        2.0,
        gt=0.0,
        le=60.0,
    )


class VariableActuator(BaseModel):
    """Power follows the command directly."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["variable"] = Field(
        default="variable", title="Type", description="Actuator type."
    )


class GasValveActuator(BaseModel):
    """Modulated gas valve with a minimum turndown and a first-order response lag."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["gas_valve"] = Field(
        default="gas_valve", title="Type", description="Actuator type."
    )
    min_fraction: float = _num(
        "fraction",
        "Minimum fraction",
        "Lowest fraction of rated power the valve can hold while on (turndown limit).",
        0.2,
        ge=0.0,
        le=1.0,
    )
    response_time_s: float = _num(
        "time",
        "Response time",
        "First-order time constant of the valve in seconds.",
        2.0,
        ge=0.0,
        le=60.0,
    )


Actuator = Annotated[
    OnOffActuator | VariableActuator | GasValveActuator, Field(discriminator="type")
]


class Burner(BaseModel):
    """Propane burner under the base. Heat enters the base over a disk or ring footprint."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["burner"] = Field(default="burner", title="Type", description="Heat source type.")
    rated_power_W: float = _num(
        "power",
        "Rated power",
        "Burner rating in watts (gross, fuel energy basis).",
        gt=0.0,
        le=1.0e6,
    )
    efficiency: float = _num(
        "fraction",
        "Burner efficiency",
        "Fraction of the burner's heat that reaches the vessel.",
        0.4,
        gt=0.0,
        le=1.0,
    )
    footprint: Literal["disk", "ring"] = Field(
        default="disk",
        title="Footprint",
        description="Where the flame heats the base: disk or ring.",
    )
    footprint_outer_diameter_m: float = _num(
        "length",
        "Footprint outer diameter",
        "Outer diameter of the heated footprint on the base, in metres. Must not exceed the "
        "outer diameter of the vessel.",
        gt=0.0,
        le=3.0,
    )
    footprint_inner_diameter_m: float = _num(
        "length",
        "Footprint inner diameter",
        "Inner diameter of a ring footprint in metres. 0 for a disk. Must be less than the "
        "outer diameter.",
        0.0,
        ge=0.0,
        le=3.0,
    )
    actuator: Actuator = Field(
        default_factory=_default_gas_valve,
        title="Actuator",
        description="How the command is turned into fuel flow.",
    )


class Element(BaseModel):
    """Electric element inside the vessel, given as a path of points with a diameter."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["element"] = Field(
        default="element", title="Type", description="Heat source type."
    )
    rated_power_W: float = _num(
        "power",
        "Rated power",
        "Element wattage in watts.",
        gt=0.0,
        le=1.0e5,
    )
    path_m: list[tuple[Coord, Coord, Coord]] = Field(
        ...,
        min_length=2,
        title="Path",
        description="Centre-line points of the element in vessel coordinates (metres): x, y, z "
        "with the origin at the centre of the inner base. At least two points. Every point "
        "must lie inside the vessel.",
        json_schema_extra={"quantity": "length"},
    )
    diameter_m: float = _num(
        "length",
        "Element diameter",
        "Diameter of the element tube in metres.",
        0.01,
        gt=0.0,
        le=0.1,
    )
    actuator: Actuator = Field(
        default_factory=_default_on_off,
        title="Actuator",
        description="How the command is turned into electrical power.",
    )


HeatSourceCfg = Annotated[Burner | Element, Field(discriminator="type")]


# --------------------------------------------------------------------------- atmosphere
class Atmosphere(BaseModel):
    """Conditions outside the vessel."""

    model_config = ConfigDict(extra="forbid")

    ambient_temp_K: float = _num(
        "temperature",
        "Ambient temperature",
        "Air temperature around the vessel in kelvin.",
        ge=233.15,
        le=333.15,
    )
    wind_speed_m_s: float = _num(
        "speed",
        "Wind speed",
        "Wind speed at the vessel in m/s. Drives forced convection on the outside.",
        0.0,
        ge=0.0,
        le=60.0,
    )
    relative_humidity: float = _num(
        "fraction",
        "Relative humidity",
        "Relative humidity of the ambient air (0 to 1). Sets evaporation from open liquid.",
        0.5,
        ge=0.0,
        le=1.0,
    )
    altitude_m: float = _num(
        "altitude",
        "Altitude",
        "Altitude above sea level in metres. Sets air pressure, boiling point and evaporation.",
        0.0,
        ge=-500.0,
        le=9000.0,
    )


# --------------------------------------------------------------------------- probes & target
class ProbeCfg(BaseModel):
    """A named temperature probe inside the liquid."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ..., min_length=1, title="Name", description="Probe name, unique, and not 'mean'."
    )
    position_m: tuple[Coord, Coord, Coord] = Field(
        ...,
        title="Position",
        description="Probe position in vessel coordinates (metres). Must be inside the liquid.",
        json_schema_extra={"quantity": "length"},
    )


class Target(BaseModel):
    """When the run stops: the temperature to reach, at a probe or the volume mean."""

    model_config = ConfigDict(extra="forbid")

    temp_K: float = _num(
        "temperature",
        "Target temperature",
        "Temperature to reach, in kelvin. The run stops when it is reached.",
        ge=273.15,
        le=373.15,
    )
    probe: str = Field(
        default="mean",
        title="Target probe",
        description="Probe name whose temperature is compared with the target, or 'mean' for "
        "the volume-weighted liquid mean.",
    )


# --------------------------------------------------------------------------- schedule & settings
class PowerStep(BaseModel):
    """One step of the power schedule: the command holds from t_s until the next step."""

    model_config = ConfigDict(extra="forbid")

    t_s: float = _num(
        "time",
        "Time",
        "Simulated time at which this command starts, in seconds. The first step is at 0.",
        ge=0.0,
    )
    command: float = _num(
        "fraction",
        "Command",
        "Requested power as a fraction of rated power (0 to 1).",
        ge=0.0,
        le=1.0,
    )


class SimSettings(BaseModel):
    """How the simulation is run. Defaults suit the standard 5 mm cell size."""

    model_config = ConfigDict(extra="forbid")

    cell_size_m: float = _num(
        "length",
        "Cell size",
        "Edge length of the liquid grid cells in metres. 2 mm to 50 mm; 5 mm is standard, "
        "10 mm is a quick preview.",
        0.005,
        ge=0.002,
        le=0.05,
    )
    max_sim_time_s: float = _num(
        "time",
        "Maximum simulated time",
        "Stop after this much simulated time, in seconds, if the target is not reached.",
        4 * 3600.0,
        gt=0.0,
        le=7 * 24 * 3600.0,
    )
    mode: RunMode = Field(
        default=RunMode.FAST,
        title="Run mode",
        description="fast: as fast as possible, following the schedule. realtime: in step with "
        "the wall clock, with live power control.",
    )
    realtime_speed: float = _num(
        "dimensionless",
        "Real-time speed",
        "Speed factor for real-time mode: 1 is wall-clock speed, 2 is twice as fast.",
        1.0,
        ge=0.1,
        le=100.0,
    )
    sample_interval_s: float = _num(
        "time",
        "Sample interval",
        "Simulated time between time-series samples, in seconds.",
        1.0,
        gt=0.0,
        le=3600.0,
    )
    snapshot_interval_s: float = _num(
        "time",
        "Snapshot interval",
        "Simulated time between slice snapshots, in seconds.",
        5.0,
        gt=0.0,
        le=3600.0,
    )
    coeff_refresh_s: float = _num(
        "time",
        "Coefficient refresh interval",
        "Simulated time between recalculations of property and heat-transfer coefficients.",
        1.0,
        gt=0.0,
        le=3600.0,
    )
    plateau_window_s: float = _num(
        "time",
        "Plateau window",
        "Length of the window over which a levelling-off is detected, in seconds.",
        600.0,
        gt=0.0,
        le=86400.0,
    )
    plateau_rate_K_per_min: float = _num(
        "temperature_rate",
        "Plateau rate",
        "Heating rate below which the liquid counts as levelled off. Stored in K per minute; "
        "shown as \u00b0C/min or \u00b0F/min.",
        0.05,
        ge=0.0,
        le=10.0,
    )


class ModelSettings(BaseModel):
    """Physics options."""

    model_config = ConfigDict(extra="forbid")

    liquid_transport: Literal["effective_conductivity", "flow"] = Field(
        default="effective_conductivity",
        title="Liquid transport",
        description="How heat moves in the liquid. 'flow' (buoyancy-driven flow) arrives in "
        "Phase 2.",
    )
    k_eff_scale: float = _num(
        "dimensionless",
        "Effective conductivity factor",
        "Calibration factor multiplying the effective conductivity of the liquid.",
        1.0,
        gt=0.0,
        le=10.0,
    )

    @model_validator(mode="after")
    def _flow_not_available(self) -> Self:
        if self.liquid_transport == "flow":
            raise ScenarioError("flow mode is not available until Phase 2")
        return self


# --------------------------------------------------------------------------- scenario
class Scenario(BaseModel):
    """A complete, validated simulation setup. Field order is the GUI form order."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = Field(
        default=1, title="Schema version", description="Version of the scenario file format."
    )
    name: str = Field(..., min_length=1, title="Name", description="Name of the setup.")
    vessel: Vessel = Field(..., title="Vessel", description="The vessel.")
    lid: Lid = Field(default_factory=_default_lid, title="Lid", description="Lid over the liquid.")
    insulation: Insulation | None = Field(
        default=None,
        title="Insulation",
        description="Optional insulation on the outside of the vessel.",
    )
    liquid: Liquid = Field(..., title="Liquid", description="The liquid in the vessel.")
    heat_source: HeatSourceCfg = Field(
        ..., title="Heat source", description="A propane burner or an electric element."
    )
    atmosphere: Atmosphere = Field(
        ..., title="Atmosphere", description="Ambient conditions around the vessel."
    )
    probes: list[ProbeCfg] = Field(
        default_factory=list, title="Probes", description="Temperature probes in the liquid."
    )
    target: Target = Field(..., title="Target", description="When the run stops.")
    power_schedule: list[PowerStep] = Field(
        default_factory=_default_schedule,
        min_length=1,
        title="Power schedule",
        description="Piecewise-constant power command. The first step is at t = 0 and the "
        "times are strictly increasing.",
    )
    simulation: SimSettings = Field(
        default_factory=_default_sim, title="Simulation", description="Run settings."
    )
    model: ModelSettings = Field(
        default_factory=_default_model, title="Model", description="Physics options."
    )

    @model_validator(mode="after")
    def _check_cross_field(self) -> Self:
        vessel = self.vessel
        radius = vessel.inner_diameter_m / 2.0
        outer_diameter = vessel.inner_diameter_m + 2.0 * vessel.wall_thickness_m

        # Liquid: fill height within the vessel, and at least six cells across.
        liquid = self.liquid
        if liquid.fill_height_m is not None:
            fill = liquid.fill_height_m
        else:
            assert liquid.volume_m3 is not None  # guaranteed by Liquid validator
            fill = liquid.volume_m3 / _disk_area(vessel.inner_diameter_m)
        if fill > vessel.inner_height_m:
            raise ScenarioError(
                f"liquid: fill height {fill:.4g} m exceeds vessel.inner_height_m "
                f"{vessel.inner_height_m:.4g} m"
            )
        cell = self.simulation.cell_size_m
        if vessel.inner_diameter_m < 6.0 * cell * (1.0 - 1e-9):  # tolerate float rounding
            raise ScenarioError(
                f"vessel.inner_diameter_m: {vessel.inner_diameter_m:.4g} m is less than 6 "
                f"cells at simulation.cell_size_m {cell:.4g} m"
            )

        # Heat source geometry.
        source = self.heat_source
        if isinstance(source, Burner):
            if source.footprint_outer_diameter_m > outer_diameter:
                raise ScenarioError(
                    f"heat_source.footprint_outer_diameter_m: "
                    f"{source.footprint_outer_diameter_m:.4g} m exceeds the vessel outer "
                    f"diameter {outer_diameter:.4g} m"
                )
            if source.footprint_inner_diameter_m >= source.footprint_outer_diameter_m:
                raise ScenarioError(
                    "heat_source.footprint_inner_diameter_m: must be less than "
                    "heat_source.footprint_outer_diameter_m"
                )
        else:
            clearance = radius - source.diameter_m / 2.0
            for idx, (x, y, z) in enumerate(source.path_m):
                if math.hypot(x, y) > clearance or z < 0.0 or z > vessel.inner_height_m:
                    raise ScenarioError(
                        f"heat_source.path_m[{idx}] = ({x:.4g}, {y:.4g}, {z:.4g}): element "
                        "point must lie inside the vessel (radius <= inner radius minus "
                        "half the element diameter; 0 <= z <= inner height)"
                    )

        # Probes: unique names, inside the liquid.
        names = [p.name for p in self.probes]
        if len(set(names)) != len(names):
            raise ScenarioError("probes: probe names must be unique")
        for probe in self.probes:
            if probe.name == "mean":
                raise ScenarioError("probes: 'mean' is reserved and cannot be a probe name")
            x, y, z = probe.position_m
            if math.hypot(x, y) > radius or z < 0.0 or z >= fill:
                raise ScenarioError(
                    f"probes[{probe.name!r}]: position ({x:.4g}, {y:.4g}, {z:.4g}) m must be "
                    f"inside the vessel and below the fill height {fill:.4g} m"
                )
        if self.target.probe != "mean" and self.target.probe not in names:
            raise ScenarioError(
                f"target.probe: {self.target.probe!r} is not 'mean' or a probe name"
            )

        # Power schedule: first step at t = 0, strictly increasing times.
        schedule = self.power_schedule
        if schedule[0].t_s != 0.0:
            raise ScenarioError("power_schedule: the first step must be at t_s = 0")
        for prev, nxt in pairwise(schedule):
            if nxt.t_s <= prev.t_s:
                raise ScenarioError(
                    "power_schedule: times must be strictly increasing "
                    f"(t_s {nxt.t_s:g} follows {prev.t_s:g})"
                )

        # Insulation covering the lid needs a lid.
        if self.insulation is not None and "lid" in self.insulation.covers:
            if self.lid.mode == "none":
                raise ScenarioError("insulation.covers: 'lid' requires a lid (lid.mode is 'none')")
        return self


__all__ = [
    "MATERIAL_KEYS",
    "Actuator",
    "Atmosphere",
    "Burner",
    "CustomMaterial",
    "Element",
    "GasValveActuator",
    "HeatSourceCfg",
    "Insulation",
    "Lid",
    "Liquid",
    "ModelSettings",
    "OnOffActuator",
    "PowerStep",
    "ProbeCfg",
    "Quantity",
    "Scenario",
    "SimSettings",
    "Target",
    "VariableActuator",
    "Vessel",
]
