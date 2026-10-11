"""Threaded runner implementing the Runner protocol, and the headless entry point
(CONTRACTS section 12).

`SimulationRunner` runs a `Model` on a daemon worker thread and talks to the caller through a
`queue.SimpleQueue` of events (`poll_events`).

* FAST mode advances in chunks sized to about 10 ms of wall time, checks the pause and stop flags
  at least every 50 ms, and emits a `ProgressEvent` about every 100 ms.
* REALTIME mode follows a `Pacer` (simulated time = integral of speed over wall time) on a
  20 ms tick. If the simulation cannot keep up it runs flat out and `realtime_lag_s` > 0.
  Power commands go through a `ManualCommand`.

The model is built on the worker thread, so a slow build (or a scenario error) never blocks the
caller; a failed build produces an `ErrorEvent` and the state FAILED.

State transitions: IDLE -> RUNNING <-> PAUSED -> FINISHED | STOPPED | FAILED. A call that is not
valid in the current state is a no-op with a logged warning.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from collections.abc import Callable

from thermosim.config.schema import Scenario
from thermosim.core.errors import ModeError
from thermosim.core.protocols import (
    ErrorEvent,
    FinishedEvent,
    ProgressEvent,
    Runner,
    RunnerEvent,
    RunnerState,
    SampleEvent,
    SnapshotEvent,
)
from thermosim.core.types import RunMode, RunResult, Sample, Snapshot, StopReason
from thermosim.sim.model import Model
from thermosim.sim.pacing import MIN_SPEED, Pacer
from thermosim.sources.commands import ManualCommand, ScheduleCommand

log = logging.getLogger(__name__)

_CHUNK_WALL_S = 0.010
_FAST_SLICE_S = 0.050
_PROGRESS_S = 0.100
_TICK_S = 0.020
_REALTIME_BURST_S = 0.100


class _Chunker:
    """Calls `Model.advance` in chunks of simulated time that take about `_CHUNK_WALL_S`."""

    def __init__(self) -> None:
        self.chunk = 1.0

    def advance(self, model: Model) -> list[Sample | Snapshot]:
        t0 = time.perf_counter()
        out = model.advance(model.state.t + self.chunk)
        wall = max(time.perf_counter() - t0, 1e-5)
        ratio = min(2.0, max(0.5, _CHUNK_WALL_S / wall))
        self.chunk = min(3600.0, max(0.05, self.chunk * ratio))
        return out


class SimulationRunner(Runner):
    mode: RunMode
    state: RunnerState

    def __init__(self, scenario: Scenario, mode: RunMode | None = None) -> None:
        self.scenario = scenario
        self.mode = mode if mode is not None else scenario.simulation.mode
        self.state = RunnerState.IDLE
        self._queue: queue.SimpleQueue[RunnerEvent] = queue.SimpleQueue()
        self._lock = threading.RLock()
        self._run_gate = threading.Event()  # set while running (not paused)
        self._stop_flag = threading.Event()
        self._manual = ManualCommand(float(scenario.power_schedule[0].command))
        self._speed = max(float(scenario.simulation.realtime_speed), MIN_SPEED)
        self._pacer: Pacer | None = None
        self._thread: threading.Thread | None = None
        self._result: RunResult | None = None

    # ------------------------------------------------------------------ Runner protocol
    def start(self) -> None:
        with self._lock:
            if self.state != RunnerState.IDLE:
                log.warning("start() ignored in state %s", self.state)
                return
            self.state = RunnerState.RUNNING
            self._run_gate.set()
            if self.mode == RunMode.REALTIME:
                self._pacer = Pacer(self._speed)
            self._thread = threading.Thread(
                target=self._worker, name="SimulationRunner", daemon=True
            )
            self._thread.start()

    def pause(self) -> None:
        with self._lock:
            if self.state != RunnerState.RUNNING:
                log.warning("pause() ignored in state %s", self.state)
                return
            self.state = RunnerState.PAUSED
            self._run_gate.clear()
            if self._pacer is not None:
                self._pacer.pause()

    def resume(self) -> None:
        with self._lock:
            if self.state != RunnerState.PAUSED:
                log.warning("resume() ignored in state %s", self.state)
                return
            self.state = RunnerState.RUNNING
            if self._pacer is not None:
                self._pacer.resume()
            self._run_gate.set()

    def stop(self) -> None:
        with self._lock:
            if self.state not in (RunnerState.RUNNING, RunnerState.PAUSED):
                log.warning("stop() ignored in state %s", self.state)
                return
            self._stop_flag.set()
            self._run_gate.set()

    def set_power_command(self, value: float) -> None:
        if self.mode != RunMode.REALTIME:
            raise ModeError("set_power_command is only available in REALTIME mode")
        self._manual.set(value)

    def set_speed(self, factor: float) -> None:
        if self.mode != RunMode.REALTIME:
            raise ModeError("set_speed is only available in REALTIME mode")
        with self._lock:
            self._speed = max(float(factor), MIN_SPEED)
            if self._pacer is not None:
                self._pacer.set_speed(self._speed)

    def poll_events(self, max_events: int = 1000) -> list[RunnerEvent]:
        out: list[RunnerEvent] = []
        while len(out) < max_events:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out

    @property
    def result(self) -> RunResult | None:
        return self._result

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker thread to end (for tests and scripts)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # ------------------------------------------------------------------ worker
    def _emit_records(self, records: list[Sample | Snapshot]) -> None:
        for r in records:
            self._queue.put(SampleEvent(r) if isinstance(r, Sample) else SnapshotEvent(r))

    def _worker(self) -> None:
        wall0 = time.perf_counter()
        model: Model | None = None
        try:
            command = (
                self._manual if self.mode == RunMode.REALTIME else ScheduleCommand(self.scenario)
            )
            model = Model(self.scenario, command)
            self._emit_records(model.advance(0.0))
            if self.mode == RunMode.REALTIME:
                self._loop_realtime(model, wall0)
            else:
                self._loop_fast(model, wall0)
            self._finish(model, wall0)
        except Exception as exc:
            self._queue.put(ErrorEvent(str(exc), traceback.format_exc()))
            self.state = RunnerState.FAILED
            if model is not None:
                self._result = model.build_result(model.wall_s)

    def _progress(self, model: Model, wall0: float, lag: float = 0.0) -> None:
        self._queue.put(
            ProgressEvent(
                sim_time_s=model.state.t,
                wall_time_s=time.perf_counter() - wall0,
                max_time_s=float(self.scenario.simulation.max_sim_time_s),
                realtime_lag_s=lag,
            )
        )

    def _loop_fast(self, model: Model, wall0: float) -> None:
        chunker = _Chunker()
        last_progress = 0.0
        while model.stop_reason is None and not self._stop_flag.is_set():
            if not self._run_gate.is_set():
                self._run_gate.wait(0.05)
                continue
            slice_start = time.perf_counter()
            while (
                time.perf_counter() - slice_start < _FAST_SLICE_S
                and model.stop_reason is None
                and not self._stop_flag.is_set()
            ):
                self._emit_records(chunker.advance(model))
            now = time.perf_counter()
            if now - last_progress >= _PROGRESS_S:
                self._progress(model, wall0)
                last_progress = now

    def _loop_realtime(self, model: Model, wall0: float) -> None:
        pacer = self._pacer
        assert pacer is not None
        last_progress = 0.0
        while model.stop_reason is None and not self._stop_flag.is_set():
            if not self._run_gate.is_set():
                self._run_gate.wait(0.05)
                continue
            t0 = time.perf_counter()
            with self._lock:
                target = pacer.target()
            while (
                model.state.t < target - 1e-9
                and time.perf_counter() - t0 < _REALTIME_BURST_S
                and model.stop_reason is None
                and not self._stop_flag.is_set()
            ):
                self._emit_records(model.advance(min(target, model.state.t + 1.0)))
            with self._lock:
                lag = pacer.lag_wall_s(model.state.t)
            now = time.perf_counter()
            if now - last_progress >= _PROGRESS_S:
                self._progress(model, wall0, lag)
                last_progress = now
            spare = _TICK_S - (time.perf_counter() - t0)
            if spare > 0 and model.stop_reason is None:
                self._stop_flag.wait(spare)

    def _finish(self, model: Model, wall0: float) -> None:
        if model.stop_reason is None:
            self._emit_records(model.finish(StopReason.USER_STOP))
        if model.stop_reason == StopReason.ERROR:
            err = model.error
            tb = "".join(traceback.format_exception(err)) if err is not None else ""
            self._queue.put(ErrorEvent(str(err), tb))
            self._result = model.build_result(model.wall_s)
            self.state = RunnerState.FAILED
            return
        self._progress(model, wall0)
        self._result = model.build_result(model.wall_s)
        self.state = (
            RunnerState.STOPPED
            if model.stop_reason == StopReason.USER_STOP
            else RunnerState.FINISHED
        )
        self._queue.put(FinishedEvent(self._result))


def run_headless(
    scenario: Scenario, progress: Callable[[ProgressEvent], None] | None = None
) -> RunResult:
    """Run a scenario to completion without a GUI or thread (FAST mode, schedule command).

    Args:
        scenario: the setup.
        progress: called about every 100 ms of wall time, and once at the end.

    Raises:
        SolverError: if the solver fails (the partial run is lost).
        ScenarioError / GeometryError: if the scenario cannot be built.
    """
    model = Model(scenario, ScheduleCommand(scenario))
    wall0 = time.perf_counter()
    max_t = float(scenario.simulation.max_sim_time_s)

    def report() -> None:
        if progress is not None:
            progress(ProgressEvent(model.state.t, time.perf_counter() - wall0, max_t, 0.0))

    chunker = _Chunker()
    model.advance(0.0)
    last = time.perf_counter()
    while model.stop_reason is None:
        chunker.advance(model)
        now = time.perf_counter()
        if now - last >= _PROGRESS_S:
            report()
            last = now
    if model.error is not None:
        raise model.error
    report()
    return model.build_result(model.wall_s)
