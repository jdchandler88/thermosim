"""Fixtures for the simulation-assembly tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from thermosim.config.io import example_scenarios
from thermosim.config.schema import Scenario


def with_sim(scenario: Scenario, **sim: Any) -> Scenario:
    """Copy of `scenario` with simulation settings replaced."""
    return scenario.model_copy(update={"simulation": scenario.simulation.model_copy(update=sim)})


@pytest.fixture
def example() -> Callable[..., Scenario]:
    """example('small_pot_stove', cell_size_m=0.02, ...) -> Scenario."""

    def make(name: str, **sim: Any) -> Scenario:
        sim.setdefault("cell_size_m", 0.02)
        return with_sim(example_scenarios()[name], **sim)

    return make
