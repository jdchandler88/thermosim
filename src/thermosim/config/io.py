"""Scenario TOML I/O (CONTRACTS.md section 5). WP00 stub, implemented by WP01."""

from __future__ import annotations

from pathlib import Path

from thermosim.config.schema import Scenario


def load_scenario(path: Path) -> Scenario:
    """Read a scenario TOML file. Raises ScenarioError with a readable message on failure."""
    raise NotImplementedError


def save_scenario(scenario: Scenario, path: Path) -> None:
    """Write a scenario to a TOML file."""
    raise NotImplementedError


def scenario_to_toml(scenario: Scenario) -> str:
    """Serialize a scenario to TOML text."""
    raise NotImplementedError


def scenario_from_toml(text: str) -> Scenario:
    """Parse TOML text into a validated scenario. Raises ScenarioError on failure."""
    raise NotImplementedError


def example_scenarios() -> dict[str, Scenario]:
    """Load the example scenarios shipped as package data (thermosim/scenarios/*.toml)."""
    raise NotImplementedError
