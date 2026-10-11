"""Dispatch from scenario configs to runtime sources, actuators and commands (WP05)."""

from __future__ import annotations

import pytest

import thermosim.sources as sources
import thermosim.sources.build as build_mod
from thermosim.config import schema
from thermosim.config.io import example_scenarios
from thermosim.geometry.build import build_domain
from thermosim.sources.actuators import GasValveActuator, OnOffActuator, VariableActuator
from thermosim.sources.build import (
    ManualCommand,
    ScheduleCommand,
    build_actuator,
    build_heat_source,
    deposition_for_burner,
    deposition_for_element,
)
from thermosim.sources.heat_sources import BurnerSource, ElementSource


def _cell_scenario(name: str, cell: float = 0.02) -> schema.Scenario:
    scenario = example_scenarios()[name]
    return scenario.model_copy(
        update={
            "simulation": scenario.simulation.model_copy(update={"cell_size_m": cell}),
        }
    )


def test_build_heat_source_burner_from_example() -> None:
    scenario = _cell_scenario("kettle_propane_40L")
    domain = build_domain(scenario)
    src = build_heat_source(scenario, domain)
    assert isinstance(src, BurnerSource)
    assert src.rated_power_W == pytest.approx(17585.0)
    assert src.efficiency == pytest.approx(0.35)
    assert isinstance(src.actuator, GasValveActuator)
    assert src.deposition.shell_weights.sum() == pytest.approx(1.0, abs=1e-12)

    # Steady state: the gas valve settles at full output, so delivered = efficiency * rated.
    for k in range(40):
        power = src.power(1.0, 100.0 * k, 100.0)
    assert power.gross_W == pytest.approx(17585.0, rel=1e-9)
    assert power.delivered_W == pytest.approx(0.35 * 17585.0, rel=1e-9)


def test_build_heat_source_element_from_example() -> None:
    scenario = _cell_scenario("kettle_electric_insulated")
    domain = build_domain(scenario)
    src = build_heat_source(scenario, domain)
    assert isinstance(src, ElementSource)
    assert src.rated_power_W == pytest.approx(5500.0)
    assert isinstance(src.actuator, OnOffActuator)
    assert src.deposition.cell_weights.sum() == pytest.approx(1.0, abs=1e-12)
    # Half command on the SSR gives half of the wattage on average over whole cycles.
    power = src.power(0.5, 0.0, 200.0)
    assert power.delivered_W == pytest.approx(0.5 * 5500.0, rel=1e-9)


def test_build_heat_source_routes_to_the_same_deposition_functions() -> None:
    scenario = _cell_scenario("kettle_propane_40L")
    domain = build_domain(scenario)
    src = build_heat_source(scenario, domain)
    ref = deposition_for_burner(scenario.heat_source, domain)  # type: ignore[arg-type]
    assert (src.deposition.shell_weights == ref.shell_weights).all()


def test_build_actuator_dispatches_on_type() -> None:
    assert isinstance(build_actuator(schema.VariableActuator()), VariableActuator)
    on_off = build_actuator(schema.OnOffActuator(cycle_period_s=3.0))
    assert isinstance(on_off, OnOffActuator)
    valve = build_actuator(schema.GasValveActuator(min_fraction=0.3, response_time_s=0.0))
    assert isinstance(valve, GasValveActuator)
    assert valve.output(0.2, 0.0, 1.0) == pytest.approx(0.3)  # turndown clamp


def test_public_names_are_importable_from_both_modules() -> None:
    for name in ("ScheduleCommand", "ManualCommand", "build_actuator", "build_heat_source"):
        assert getattr(sources, name) is getattr(build_mod, name)
    assert sources.deposition_for_burner is deposition_for_burner
    assert sources.deposition_for_element is deposition_for_element
    assert build_mod.ScheduleCommand is ScheduleCommand
    assert build_mod.ManualCommand is ManualCommand


def test_schedule_command_from_example_is_usable_by_the_model() -> None:
    scenario = _cell_scenario("kettle_propane_40L")
    cmd = ScheduleCommand(scenario)
    assert cmd.command(0.0, 288.0) == pytest.approx(1.0)
