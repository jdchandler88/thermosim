"""Factory seam for WP11: the only place that decides which runner the GUI uses."""

from __future__ import annotations

from thermosim.config.schema import Scenario
from thermosim.core.protocols import Runner
from thermosim.core.types import RunMode
from thermosim.gui.fake_runner import FakeRunner


def make_runner(scenario: Scenario, mode: RunMode) -> Runner:
    """Create a runner for `scenario`. WP11 will return the real SimulationRunner here."""
    return FakeRunner(scenario, mode)
