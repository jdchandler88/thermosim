"""Heat-source construction and power commands (CONTRACTS.md section 9, WP05).

Dispatch from the scenario to the runtime objects. The implementations live in
`heat_sources.py`, `actuators.py` and `commands.py`; they are re-exported here so that
`thermosim.sources.build` stays the single entry point.
"""

from __future__ import annotations

from thermosim.config import schema
from thermosim.core.errors import ScenarioError
from thermosim.core.protocols import Actuator, HeatSource
from thermosim.core.types import Domain
from thermosim.sources.actuators import GasValveActuator, OnOffActuator, VariableActuator
from thermosim.sources.commands import ManualCommand, ScheduleCommand
from thermosim.sources.heat_sources import (
    BurnerSource,
    ElementSource,
    deposition_for_burner,
    deposition_for_element,
)

__all__ = [
    "ManualCommand",
    "ScheduleCommand",
    "build_actuator",
    "build_heat_source",
    "deposition_for_burner",
    "deposition_for_element",
]


def build_heat_source(scenario: schema.Scenario, domain: Domain) -> HeatSource:
    """Create the heat source (burner or element) described by the scenario.

    Raises GeometryError if the source cannot be placed in the domain (no base node under a
    burner footprint, or an element segment above the liquid).
    """
    cfg = scenario.heat_source
    if isinstance(cfg, schema.Burner):
        return BurnerSource(
            rated_power_W=cfg.rated_power_W,
            efficiency=cfg.efficiency,
            actuator=build_actuator(cfg.actuator),
            deposition=deposition_for_burner(cfg, domain),
        )
    if isinstance(cfg, schema.Element):
        return ElementSource(
            rated_power_W=cfg.rated_power_W,
            actuator=build_actuator(cfg.actuator),
            deposition=deposition_for_element(cfg, domain),
        )
    raise ScenarioError(f"heat_source.type: unknown heat source {cfg.type!r}")


def build_actuator(
    cfg: schema.OnOffActuator | schema.VariableActuator | schema.GasValveActuator,
) -> Actuator:
    """Create the actuator for an OnOff / Variable / GasValve config."""
    if isinstance(cfg, schema.OnOffActuator):
        return OnOffActuator(cfg.cycle_period_s)
    if isinstance(cfg, schema.VariableActuator):
        return VariableActuator()
    if isinstance(cfg, schema.GasValveActuator):
        return GasValveActuator(cfg.min_fraction, cfg.response_time_s)
    raise ScenarioError(f"actuator.type: unknown actuator {cfg.type!r}")
