"""Helpers shared by the GUI tests."""

from __future__ import annotations

from thermosim.config.io import example_scenarios
from thermosim.config.schema import Scenario
from thermosim.gui.fake_runner import FakeRunner


def short_scenario(max_time_s: float = 60.0) -> Scenario:
    """The small-pot example, with a coarse grid and a short maximum time."""
    base = example_scenarios()["small_pot_stove"]
    sim = base.simulation.model_copy(
        update={
            "cell_size_m": 0.01,
            "max_sim_time_s": max_time_s,
            "sample_interval_s": 1.0,
            "snapshot_interval_s": 5.0,
        }
    )
    return base.model_copy(update={"simulation": sim})


class RecordingRunner(FakeRunner):
    """FakeRunner that records the commands it receives."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.power_calls: list[float] = []
        self.speed_calls: list[float] = []

    def set_power_command(self, value: float) -> None:
        super().set_power_command(value)
        self.power_calls.append(value)

    def set_speed(self, factor: float) -> None:
        super().set_speed(factor)
        self.speed_calls.append(factor)
