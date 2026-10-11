"""Scenario schema (CONTRACTS.md section 5). WP00 stub: field shapes only.

WP01 adds the Field(...) metadata (title, description, ge/le, quantity tag) to every
numeric field and the cross-field validators.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from thermosim.core.types import RunMode

Quantity = Literal[
    "temperature",
    "temperature_delta",
    "length",
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


class CustomMaterial(BaseModel):
    density: float
    specific_heat: float
    conductivity: float
    emissivity: float


class Vessel(BaseModel):
    shape: Literal["cylinder"] = "cylinder"
    inner_diameter_m: float
    inner_height_m: float
    wall_thickness_m: float
    base_thickness_m: float
    material: str | CustomMaterial = "stainless_304"  # str = key in physics.properties.SOLIDS


class Lid(BaseModel):
    mode: Literal["none", "full", "partial"] = "none"
    covered_fraction: float = 1.0  # used when mode == "partial" (0 < f < 1)
    material: str | CustomMaterial = "stainless_304"
    thickness_m: float = 0.001


class Insulation(BaseModel):
    thickness_m: float
    conductivity_W_mK: float
    emissivity: float = 0.9
    covers: set[Literal["side", "base", "lid"]] = {"side"}


class Liquid(BaseModel):
    volume_m3: float | None = None
    fill_height_m: float | None = None  # exactly one set
    initial_temp_K: float
    specific_gravity: float = 1.000


class OnOffActuator(BaseModel):
    type: Literal["on_off"] = "on_off"
    cycle_period_s: float = 2.0


class VariableActuator(BaseModel):
    type: Literal["variable"] = "variable"


class GasValveActuator(BaseModel):
    type: Literal["gas_valve"] = "gas_valve"
    min_fraction: float = 0.2
    response_time_s: float = 2.0


Actuator = Annotated[
    OnOffActuator | VariableActuator | GasValveActuator, Field(discriminator="type")
]


class Burner(BaseModel):
    type: Literal["burner"] = "burner"
    rated_power_W: float
    efficiency: float = 0.4
    footprint: Literal["disk", "ring"] = "disk"
    footprint_outer_diameter_m: float
    footprint_inner_diameter_m: float = 0.0
    actuator: Actuator = GasValveActuator()


class Element(BaseModel):
    type: Literal["element"] = "element"
    rated_power_W: float
    path_m: list[tuple[float, float, float]]  # >= 2 points, vessel coordinates (D2)
    diameter_m: float = 0.01
    actuator: Actuator = OnOffActuator()


HeatSourceCfg = Annotated[Burner | Element, Field(discriminator="type")]


class Atmosphere(BaseModel):
    ambient_temp_K: float
    wind_speed_m_s: float = 0.0
    relative_humidity: float = 0.5
    altitude_m: float = 0.0


class ProbeCfg(BaseModel):
    name: str
    position_m: tuple[float, float, float]


class Target(BaseModel):
    temp_K: float
    probe: str = "mean"  # probe name or "mean"


class PowerStep(BaseModel):
    t_s: float
    command: float  # 0..1


class SimSettings(BaseModel):
    cell_size_m: float = 0.005
    max_sim_time_s: float = 4 * 3600
    mode: RunMode = RunMode.FAST
    realtime_speed: float = 1.0
    sample_interval_s: float = 1.0
    snapshot_interval_s: float = 5.0
    coeff_refresh_s: float = 1.0
    plateau_window_s: float = 600.0
    plateau_rate_K_per_min: float = 0.05


class ModelSettings(BaseModel):
    liquid_transport: Literal["effective_conductivity", "flow"] = "effective_conductivity"
    k_eff_scale: float = 1.0


class Scenario(BaseModel):
    schema_version: Literal[1] = 1
    name: str
    vessel: Vessel
    lid: Lid = Lid()
    insulation: Insulation | None = None
    liquid: Liquid
    heat_source: HeatSourceCfg
    atmosphere: Atmosphere
    probes: list[ProbeCfg]
    target: Target
    power_schedule: list[PowerStep] = [PowerStep(t_s=0, command=1.0)]
    simulation: SimSettings = SimSettings()
    model: ModelSettings = ModelSettings()
    # model validators: see WP01 for the full cross-field rule list
