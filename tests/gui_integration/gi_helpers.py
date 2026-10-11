"""Helpers for the GUI integration tests."""

from __future__ import annotations

from thermosim.config.io import example_scenarios
from thermosim.config.schema import Scenario


def quick_scenario(name: str = "small_pot_stove", dx: float = 0.02, **sim: float) -> Scenario:
    """An example on a coarse grid, so real runs take seconds."""
    base = example_scenarios()[name]
    update = {"cell_size_m": dx, "sample_interval_s": 2.0, "snapshot_interval_s": 10.0, **sim}
    return base.model_copy(update={"simulation": base.simulation.model_copy(update=update)})
