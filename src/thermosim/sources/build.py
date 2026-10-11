"""Heat-source construction and power commands (CONTRACTS.md section 9).

WP00 stub, implemented by WP05. ScheduleCommand and ManualCommand live here in the
stub; WP05 may move them into sources/commands.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from thermosim.core.protocols import Actuator, CommandSource, HeatSource
from thermosim.core.types import Domain, HeatDeposition

if TYPE_CHECKING:
    from thermosim.config.schema import (
        Burner,
        Element,
        GasValveActuator,
        OnOffActuator,
        Scenario,
        VariableActuator,
    )


def build_heat_source(scenario: Scenario, domain: Domain) -> HeatSource:
    """Create the heat source (burner or element) described by the scenario."""
    raise NotImplementedError


def build_actuator(cfg: OnOffActuator | VariableActuator | GasValveActuator) -> Actuator:
    """Create the actuator for an OnOff / Variable / GasValve config."""
    raise NotImplementedError


def deposition_for_burner(cfg: Burner, domain: Domain) -> HeatDeposition:
    """Where a burner's heat enters: the base cells / base shell nodes under its footprint."""
    raise NotImplementedError


def deposition_for_element(cfg: Element, domain: Domain) -> HeatDeposition:
    """Where an element's heat enters, along its path. GeometryError if any path point is
    above the liquid line."""
    raise NotImplementedError


class ScheduleCommand(CommandSource):
    """Piecewise-constant command from Scenario.power_schedule."""

    def __init__(self, scenario: Scenario) -> None:
        raise NotImplementedError

    def command(self, t: float, state_probe_K: float) -> float:
        raise NotImplementedError


class ManualCommand(CommandSource):
    """Thread-safe command set by the user; used in REALTIME mode."""

    def set(self, value: float) -> None:
        """Set the requested power fraction, 0..1."""
        raise NotImplementedError

    def command(self, t: float, state_probe_K: float) -> float:
        raise NotImplementedError
