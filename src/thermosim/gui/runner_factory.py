"""Factory seam: the only place that decides which runner the GUI uses."""

from __future__ import annotations

from thermosim.config.schema import Scenario
from thermosim.core.protocols import Runner
from thermosim.core.types import RunMode
from thermosim.gui.fake_runner import FakeRunner
from thermosim.sim.runner import SimulationRunner


def make_runner(scenario: Scenario, mode: RunMode) -> Runner:
    """Create the real `SimulationRunner` for `scenario`."""
    return SimulationRunner(scenario, mode)


def make_fake_runner(scenario: Scenario, mode: RunMode) -> Runner:
    """Create the synthetic `FakeRunner` (for UI development, `--fake`)."""
    return FakeRunner(scenario, mode)
