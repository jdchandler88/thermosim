"""Scenario TOML I/O (CONTRACTS.md section 5).

Files are written with `tomli_w` and read with the standard library `tomllib`. Every
read goes through `Scenario` validation. All errors are raised as `ScenarioError`, and
`load_scenario` names the file in the message.

Example scenarios are package data in `thermosim/scenarios/*.toml`.
"""

from __future__ import annotations

import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import ValidationError

from thermosim.config.schema import Scenario
from thermosim.core.errors import ScenarioError

_HEADER = (
    "# ThermoSim scenario file.\n"
    "# Units are SI: metres, seconds, kilograms, watts, pascals.\n"
    "# Temperatures are in kelvin (K), not Celsius or Fahrenheit.\n"
)


def _format_validation_error(exc: ValidationError) -> str:
    """Turn pydantic errors into 'field.path: problem' lines."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(scenario)"
        msg = err["msg"]
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts)


def scenario_from_toml(text: str) -> Scenario:
    """Parse TOML text into a validated scenario. Raises ScenarioError on failure."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ScenarioError(f"invalid TOML: {exc}") from exc
    try:
        return Scenario.model_validate(data)
    except ValidationError as exc:
        raise ScenarioError(_format_validation_error(exc)) from exc


def load_scenario(path: Path) -> Scenario:
    """Read a scenario TOML file. Raises ScenarioError with a readable message on failure."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioError(f"{path}: cannot read file: {exc.strerror or exc}") from exc
    try:
        return scenario_from_toml(text)
    except ScenarioError as exc:
        raise ScenarioError(f"{path}: {exc}") from exc


def scenario_to_toml(scenario: Scenario) -> str:
    """Serialize a scenario to TOML text.

    Tuples are written as arrays, sets as sorted arrays, and None-valued optional fields
    are omitted. The file starts with a comment header that states the units.
    """
    data: dict[str, Any] = scenario.model_dump(mode="json", exclude_none=True)
    if "insulation" in data:
        data["insulation"]["covers"] = sorted(data["insulation"]["covers"])
    return _HEADER + "\n" + tomli_w.dumps(data)


def save_scenario(scenario: Scenario, path: Path) -> None:
    """Write a scenario to a TOML file (UTF-8)."""
    path.write_text(scenario_to_toml(scenario), encoding="utf-8")


def example_scenarios() -> dict[str, Scenario]:
    """Load the example scenarios shipped as package data (thermosim/scenarios/*.toml).

    Keys are the file names without the .toml extension, in alphabetical order.
    """
    folder = files("thermosim") / "scenarios"
    examples: dict[str, Scenario] = {}
    entries = sorted(
        (entry for entry in folder.iterdir() if entry.name.endswith(".toml")),
        key=lambda entry: entry.name,
    )
    for entry in entries:
        name = entry.name[: -len(".toml")]
        try:
            examples[name] = scenario_from_toml(entry.read_text(encoding="utf-8"))
        except ScenarioError as exc:
            raise ScenarioError(f"scenarios/{entry.name}: {exc}") from exc
    return examples
