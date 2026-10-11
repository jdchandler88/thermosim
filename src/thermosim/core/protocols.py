"""Protocols and event types (CONTRACTS.md section 4). Frozen after WP00."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .types import HeatDeposition, RunMode, RunResult, Sample, Snapshot


class Actuator(Protocol):
    def output(self, command: float, t: float, dt: float) -> float:
        """Average fraction 0..1 of max power actually delivered over [t, t+dt]
        for a requested command 0..1. May keep internal state (SSR cycle, valve lag)."""

    def reset(self) -> None: ...


@dataclass(frozen=True)
class SourcePower:
    delivered_W: float  # into the vessel/liquid (after burner efficiency)
    gross_W: float  # fuel energy rate (burner, LHV basis) or electrical power


class HeatSource(Protocol):
    deposition: HeatDeposition
    rated_power_W: float  # gross rating

    def power(self, command: float, t: float, dt: float) -> SourcePower: ...
    def fuel_kg(self, gross_J: float) -> float: ...  # 0 for electric
    def electric_kwh(self, gross_J: float) -> float: ...  # 0 for burner


class CommandSource(Protocol):
    """Gives the requested power fraction 0..1 at time t (schedule, user, PID)."""

    def command(self, t: float, state_probe_K: float) -> float: ...


class RunnerState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class SnapshotEvent:
    snapshot: Snapshot


@dataclass(frozen=True)
class SampleEvent:
    sample: Sample


@dataclass(frozen=True)
class ProgressEvent:
    sim_time_s: float
    wall_time_s: float
    max_time_s: float
    realtime_lag_s: float  # REALTIME: how far behind the wall clock the sim is (0 in FAST)


@dataclass(frozen=True)
class FinishedEvent:
    result: RunResult


@dataclass(frozen=True)
class ErrorEvent:
    message: str
    traceback: str


RunnerEvent = SnapshotEvent | SampleEvent | ProgressEvent | FinishedEvent | ErrorEvent


@runtime_checkable
class Runner(Protocol):
    mode: RunMode
    state: RunnerState

    def start(self) -> None: ...  # non-blocking; runs in a worker thread
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...  # graceful; emits FinishedEvent with USER_STOP
    def set_power_command(self, value: float) -> None: ...  # REALTIME only, else ModeError
    def set_speed(self, factor: float) -> None: ...  # REALTIME only, factor >= 0.1
    def poll_events(
        self, max_events: int = 1000
    ) -> list[RunnerEvent]: ...  # thread-safe, non-blocking
    @property
    def result(self) -> RunResult | None: ...
