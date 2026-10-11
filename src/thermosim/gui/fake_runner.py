"""A fake `Runner` that produces plausible synthetic data in a background thread.

Used by the GUI until WP11 plugs in the real simulation. It honors the full `Runner`
protocol (pause, resume, stop, speed, power command, event polling).

The synthetic physics is deliberately simple:

* The mean temperature relaxes toward ``T0 + (T_target - T0) * command`` with time
  constant ``tau``; at full power that is ``T_target - (T_target - T0) exp(-t/tau)``.
* Probes follow the same law with their own time constants (near the wall or low in the
  liquid they respond faster).
* The slice field is the mean plus a zero-mean spatial profile (hot near the base for a
  burner, hot around the path for an element) whose amplitude scales with the remaining
  temperature difference, so it fades as the liquid levels out.
* Energy flows are made up but balance exactly (``balance_error() == 0``).
"""

from __future__ import annotations

import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass

import numpy as np

from thermosim.config.io import scenario_to_toml
from thermosim.config.schema import Burner, Element, Scenario
from thermosim.core.errors import ModeError
from thermosim.core.protocols import (
    ErrorEvent,
    FinishedEvent,
    ProgressEvent,
    RunnerEvent,
    RunnerState,
    SampleEvent,
    SnapshotEvent,
)
from thermosim.core.types import (
    EnergyFlows,
    RunMode,
    RunResult,
    Sample,
    Snapshot,
    StopReason,
    Summary,
)
from thermosim.gui.snapshot_list import ListSnapshotStore

PROPANE_LHV_J_PER_KG = 46.35e6
_J_PER_KWH = 3.6e6
_TARGET_TOLERANCE_K = 0.5


@dataclass
class _Geometry:
    radius: float
    fill: float
    dx: float
    nx: int
    nz: int
    x: np.ndarray
    z: np.ndarray
    liquid: np.ndarray  # (nx, nz) bool


def slice_dimensions(scenario: Scenario) -> tuple[float, float, float, int, int]:
    """(radius, fill height, dx, nx, nz) of the slice: nx = ceil(D/dx), nz = ceil(h/dx)."""
    v = scenario.vessel
    radius = v.inner_diameter_m / 2.0
    liquid = scenario.liquid
    if liquid.fill_height_m is not None:
        fill = liquid.fill_height_m
    else:
        assert liquid.volume_m3 is not None
        fill = liquid.volume_m3 / (math.pi * radius**2)
    fill = min(fill, v.inner_height_m)
    dx = scenario.simulation.cell_size_m
    nx = math.ceil(v.inner_diameter_m / dx - 1e-9)
    nz = math.ceil(fill / dx - 1e-9)
    return radius, fill, dx, max(nx, 1), max(nz, 1)


def _geometry(scenario: Scenario) -> _Geometry:
    radius, fill, dx, nx, nz = slice_dimensions(scenario)
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    z = (np.arange(nz) + 0.5) * dx
    liquid = (np.abs(x)[:, None] <= radius + 1e-9) & ((z - dx / 2.0)[None, :] < fill - 1e-9)
    return _Geometry(radius, fill, dx, nx, nz, x, z, liquid)


class FakeRunner:
    """Synthetic runner implementing the `Runner` protocol."""

    def __init__(
        self,
        scenario: Scenario,
        mode: RunMode | None = None,
        *,
        fast_rate: float = 600.0,
        tick_s: float = 0.02,
        speed: float | None = None,
    ) -> None:
        """
        Args:
            scenario: the setup to pretend to simulate.
            mode: FAST (default from the scenario) or REALTIME.
            fast_rate: simulated seconds per wall second in FAST mode (so the GUI shows
                progress instead of finishing instantly).
            tick_s: wall-clock seconds between worker iterations.
            speed: initial real-time speed factor (default: scenario.simulation.realtime_speed).
        """
        self.scenario = scenario
        self.mode: RunMode = mode if mode is not None else scenario.simulation.mode
        self.state: RunnerState = RunnerState.IDLE
        self._fast_rate = fast_rate
        self._tick_s = tick_s
        self._speed = speed if speed is not None else scenario.simulation.realtime_speed
        self._queue: queue.Queue[RunnerEvent] = queue.Queue()
        self._lock = threading.Lock()
        self._command = float(scenario.power_schedule[0].command)
        self._paused = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: RunResult | None = None

    # ------------------------------------------------------------------ Runner protocol
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("runner already started")
        self.state = RunnerState.RUNNING
        self._thread = threading.Thread(target=self._worker, name="FakeRunner", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        if self.state == RunnerState.RUNNING:
            self._paused.set()
            self.state = RunnerState.PAUSED

    def resume(self) -> None:
        if self.state == RunnerState.PAUSED:
            self._paused.clear()
            self.state = RunnerState.RUNNING

    def stop(self) -> None:
        self._stop.set()

    def set_power_command(self, value: float) -> None:
        if self.mode != RunMode.REALTIME:
            raise ModeError("set_power_command is only available in REALTIME mode")
        with self._lock:
            self._command = min(max(float(value), 0.0), 1.0)

    def set_speed(self, factor: float) -> None:
        if self.mode != RunMode.REALTIME:
            raise ModeError("set_speed is only available in REALTIME mode")
        with self._lock:
            self._speed = max(float(factor), 0.1)

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
        """Wait for the worker thread (for tests and shutdown)."""
        if self._thread is not None:
            self._thread.join(timeout)

    # ------------------------------------------------------------------ worker
    def _worker(self) -> None:
        try:
            self._run()
        except Exception as exc:  # pragma: no cover - defensive
            self.state = RunnerState.FAILED
            self._queue.put(ErrorEvent(str(exc), traceback.format_exc()))

    def _run(self) -> None:
        sc = self.scenario
        sim = sc.simulation
        geo = _geometry(sc)
        t0_K = sc.liquid.initial_temp_K
        amb_K = sc.atmosphere.ambient_temp_K
        target_K = sc.target.temp_K
        max_t = sim.max_sim_time_s
        tau = max(min(0.25 * max_t, 600.0), 5.0)
        src = sc.heat_source
        rated = src.rated_power_W
        eff = src.efficiency if isinstance(src, Burner) else 1.0
        profile = self._profile(geo)
        pmin = float(profile[geo.liquid].min()) if geo.liquid.any() else 0.0
        pmax = float(profile[geo.liquid].max()) if geo.liquid.any() else 0.0
        vol = max(float(np.count_nonzero(geo.liquid)) * geo.dx**3, 1e-6)
        heat_cap = vol * 4.0e6  # J/K, made-up liquid heat capacity

        probe_tau = {}
        for p in sc.probes:
            px, _, pz = p.position_m
            near = abs(px) > 0.7 * geo.radius or pz < 0.25 * geo.fill
            probe_tau[p.name] = tau * (0.7 if near else 1.25)

        t = 0.0
        t_mean = t0_K
        t_probe = {p.name: t0_K for p in sc.probes}
        src_j = 0.0
        gross_j = 0.0
        samples: list[Sample] = []
        store = ListSnapshotStore()
        next_sample = 0.0
        next_snap = 0.0
        wall_run = 0.0
        time_to_target: float | None = None
        reason: StopReason | None = None
        last_wall = time.perf_counter()

        def command_at(tt: float) -> float:
            if self.mode == RunMode.REALTIME:
                with self._lock:
                    return self._command
            cmd = sc.power_schedule[0].command
            for step in sc.power_schedule:
                if step.t_s <= tt:
                    cmd = step.command
            return float(cmd)

        def spread_of(mean: float, tinf: float) -> float:
            return 0.8 * max(tinf - mean, 0.0) + 0.1

        def make_sample(cmd: float) -> Sample:
            tinf = t0_K + (target_K - t0_K) * cmd
            spread = spread_of(t_mean, tinf)
            gross = rated * cmd
            deliv = gross * eff
            stored = min(heat_cap * (t_mean - t0_K), 0.85 * src_j)
            stored_shell = 0.03 * stored
            loss = max(src_j - stored - stored_shell, 0.0)
            flows = EnergyFlows(
                source_liquid=0.9 * src_j,
                source_shell=0.1 * src_j,
                loss_shell_outer=0.45 * loss,
                loss_surface_convrad=0.2 * loss,
                loss_surface_evap=0.15 * loss,
                loss_lid_outer=0.1 * loss,
                loss_dry_wall_open=0.1 * loss,
                stored_liquid=stored,
                stored_shell=stored_shell,
                stored_lid_headspace=0.0,
            )
            # Make the balance exact whatever the rounding.
            flows.stored_lid_headspace = flows.balance_error()
            return Sample(
                t=t,
                probes=dict(t_probe),
                T_mean=t_mean,
                T_min=t_mean + spread * pmin,
                T_max=t_mean + spread * pmax,
                T_shell_mean=t_mean + 0.5 * spread,
                command=cmd,
                power_delivered=deliv,
                power_gross=gross,
                cumulative=flows,
                fuel_kg=gross_j / PROPANE_LHV_J_PER_KG if isinstance(src, Burner) else 0.0,
                electric_kwh=gross_j / _J_PER_KWH if isinstance(src, Element) else 0.0,
            )

        def make_snapshot(cmd: float) -> Snapshot:
            tinf = t0_K + (target_K - t0_K) * cmd
            spread = spread_of(t_mean, tinf)
            field = t_mean + spread * profile
            slice_t = np.where(geo.liquid, field, np.nan).astype(np.float32)
            left = field[0, :] + 0.4 * spread
            right = field[-1, :] + 0.4 * spread
            hot = np.exp(-((geo.x / max(geo.radius * 0.6, 1e-6)) ** 2))
            base = field[:, 0] + 0.8 * spread * hot
            lid_T = None
            if sc.lid.mode != "none":
                lid_T = float(amb_K + 0.5 * (t_mean - amb_K))
            return Snapshot(
                t=t,
                slice_T=slice_t,
                x_centers=geo.x.copy(),
                z_centers=geo.z.copy(),
                wall_left_z=geo.z.copy(),
                wall_left_T=left.astype(np.float32),
                wall_right_z=geo.z.copy(),
                wall_right_T=right.astype(np.float32),
                base_x=geo.x.copy(),
                base_T=base.astype(np.float32),
                lid_T=lid_T,
                probes=dict(t_probe),
                T_mean=t_mean,
                power_delivered=rated * cmd * eff,
            )

        def advance_to(t_new: float) -> None:
            nonlocal t, t_mean, src_j, gross_j
            dt = t_new - t
            if dt <= 0:
                return
            cmd = command_at(t)
            tinf = t0_K + (target_K - t0_K) * cmd
            t_mean = tinf + (t_mean - tinf) * math.exp(-dt / tau)
            for name in t_probe:
                t_probe[name] = tinf + (t_probe[name] - tinf) * math.exp(-dt / probe_tau[name])
            src_j += rated * cmd * eff * dt
            gross_j += rated * cmd * dt
            t = t_new

        def target_value() -> float:
            if sc.target.probe in t_probe:
                return t_probe[sc.target.probe]
            return t_mean

        def emit_progress(lag: float) -> None:
            self._queue.put(ProgressEvent(t, wall_run, max_t, lag))

        def emit_sample_and_snapshot(force: bool = False) -> None:
            nonlocal next_sample, next_snap
            cmd = command_at(t)
            if force or t >= next_sample - 1e-9:
                s = make_sample(cmd)
                samples.append(s)
                self._queue.put(SampleEvent(s))
                next_sample = (math.floor(t / sim.sample_interval_s + 1e-9) + 1) * (
                    sim.sample_interval_s
                )
            if force or t >= next_snap - 1e-9:
                snap = make_snapshot(cmd)
                store.add(snap)
                self._queue.put(SnapshotEvent(snap))
                next_snap = (math.floor(t / sim.snapshot_interval_s + 1e-9) + 1) * (
                    sim.snapshot_interval_s
                )

        emit_sample_and_snapshot()
        while reason is None:
            if self._stop.is_set():
                reason = StopReason.USER_STOP
                break
            now = time.perf_counter()
            wall_dt = now - last_wall
            last_wall = now
            if self._paused.is_set():
                time.sleep(self._tick_s)
                continue
            wall_run += wall_dt
            if self.mode == RunMode.REALTIME:
                with self._lock:
                    adv = wall_dt * self._speed
            else:
                adv = wall_dt * self._fast_rate
            t_goal = min(t + adv, max_t)
            while t < t_goal - 1e-12 and reason is None:
                t_next = min(t_goal, next_sample, next_snap)
                if t_next <= t:
                    t_next = t_goal
                advance_to(t_next)
                emit_sample_and_snapshot()
                if target_value() >= target_K - _TARGET_TOLERANCE_K:
                    reason = StopReason.TARGET_REACHED
                    time_to_target = t
                elif t >= max_t - 1e-9:
                    reason = StopReason.MAX_TIME
            emit_progress(0.0)
            if reason is None:
                time.sleep(self._tick_s)

        # Final sample and snapshot so the results end exactly where the run did.
        if not samples or samples[-1].t < t - 1e-9:
            emit_sample_and_snapshot(force=True)
        final = samples[-1]
        flows = final.cumulative
        gross_energy = gross_j
        summary = Summary(
            stop_reason=reason,
            sim_time_s=t,
            wall_time_s=wall_run,
            time_to_target_s=time_to_target,
            target_temp_K=target_K,
            target_probe=sc.target.probe,
            final_probes=dict(final.probes),
            final_T_mean=final.T_mean,
            energy=flows,
            energy_balance_rel_error=abs(flows.balance_error()) / max(flows.total_in(), 1.0),
            fuel_kg=final.fuel_kg,
            electric_kwh=final.electric_kwh,
            heating_efficiency=flows.stored_liquid / gross_energy if gross_energy > 0 else 0.0,
        )
        self._result = RunResult(
            summary=summary,
            samples=samples,
            snapshots=store,
            scenario_toml=scenario_to_toml(sc),
        )
        self.state = RunnerState.STOPPED if reason == StopReason.USER_STOP else RunnerState.FINISHED
        self._queue.put(FinishedEvent(self._result))

    def _profile(self, geo: _Geometry) -> np.ndarray:
        """Zero-mean spatial profile in [-1, 1] over the liquid (nx, nz)."""
        xx, zz = np.meshgrid(geo.x, geo.z, indexing="ij")
        src = self.scenario.heat_source
        if isinstance(src, Element):
            prof = np.zeros_like(xx)
            width = max(0.2 * geo.fill, 2 * geo.dx)
            for px, _, pz in src.path_m:
                prof += np.exp(-(((xx - px) ** 2 + (zz - pz) ** 2) / (2 * width**2)))
        else:
            prof = np.exp(-zz / max(0.25 * geo.fill, geo.dx)) * (
                1.0 - 0.4 * (xx / max(geo.radius, 1e-6)) ** 2
            )
        if geo.liquid.any():
            prof = prof - prof[geo.liquid].mean()
            peak = float(np.abs(prof[geo.liquid]).max())
            if peak > 0:
                prof = prof / peak
        return prof
