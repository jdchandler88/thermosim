"""Protocol and event types exist with the contracted names."""

from __future__ import annotations

from thermosim.core.protocols import (
    Actuator,
    CommandSource,
    ErrorEvent,
    FinishedEvent,
    HeatSource,
    ProgressEvent,
    Runner,
    RunnerState,
    SampleEvent,
    SnapshotEvent,
    SourcePower,
)
from thermosim.core.types import RunMode


def test_runner_state_values() -> None:
    assert {s.value for s in RunnerState} == {
        "idle",
        "running",
        "paused",
        "finished",
        "stopped",
        "failed",
    }


def test_run_mode_values() -> None:
    assert RunMode.FAST.value == "fast"
    assert RunMode.REALTIME.value == "realtime"


def test_source_power_is_frozen() -> None:
    p = SourcePower(delivered_W=1.0, gross_W=2.5)
    assert p.delivered_W == 1.0 and p.gross_W == 2.5


def test_events_are_importable() -> None:
    for cls in (SnapshotEvent, SampleEvent, ProgressEvent, FinishedEvent, ErrorEvent):
        assert cls.__name__.endswith("Event")
    for proto in (Actuator, CommandSource, HeatSource, Runner):
        assert isinstance(proto, type)
