"""The default scenario, used when the GUI or CLI starts without a setup file."""

from __future__ import annotations

from thermosim.config.io import example_scenarios
from thermosim.config.schema import Scenario

DEFAULT_EXAMPLE = "kettle_propane_40L"


def default_scenario() -> Scenario:
    """The first example: a 40 L propane kettle (kettle_propane_40L.toml)."""
    return example_scenarios()[DEFAULT_EXAMPLE]
