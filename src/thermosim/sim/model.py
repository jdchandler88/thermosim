"""One simulation step: orchestration of physics, sources and solver (CONTRACTS section 12).

`Model` owns the domain, the solver, the heat source and the state. `advance(until_t)` steps
forward, takes samples and snapshots, and checks the stop criteria. Samples and the stop
checks happen at sample times (every ``sample_interval_s``), so the time to target is
resolved to that interval (and refined by interpolation in `recording.summarize`).

Initial state (decided): the liquid starts at ``liquid.initial_temp_K``; the vessel shell, lid
and headspace start at ambient temperature.

Time step: ``dt = min(stable_dt, 1 s, time to the next sample / snapshot / coefficient refresh,
time to max time, time to `until_t`)``. The stable time step is evaluated once per coefficient
refresh (the liquid-side coefficients that set it only change then).
"""

from __future__ import annotations

import bisect
import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from thermosim.config.io import scenario_to_toml
from thermosim.config.schema import Scenario
from thermosim.core.errors import SolverError
from thermosim.core.protocols import CommandSource, HeatSource, SourcePower
from thermosim.core.types import (
    Domain,
    EnergyFlows,
    RunResult,
    Sample,
    Snapshot,
    StepCoefficients,
    StopReason,
    ThermalState,
)
from thermosim.geometry.build import build_domain, extract_slice, probe_values
from thermosim.physics.properties import solid_props
from thermosim.recording import Recorder, SnapshotStore, summarize
from thermosim.sim.coefficients import (
    CoefficientCache,
    apply_source,
    build_coefficients,
    make_cache,
    mean_liquid_T,
)
from thermosim.solver.thermal import ThermalSolver
from thermosim.sources.build import build_heat_source

MAX_DT_S = 1.0
_EPS = 1e-9


class Model:
    """The assembled simulation."""

    state: ThermalState
    domain: Domain

    def __init__(
        self,
        scenario: Scenario,
        command: CommandSource,
        *,
        coeff_hook: Callable[[StepCoefficients], None] | None = None,
    ) -> None:
        """Build the domain, solver and heat source, and the initial state.

        Args:
            scenario: the setup.
            command: gives the requested power fraction (schedule, manual, PID).
            coeff_hook: test-only. Called with the `StepCoefficients` after every coefficient
                refresh, e.g. to disable the losses for a lumped-model comparison.
        """
        self.scenario = scenario
        self.command = command
        self._coeff_hook = coeff_hook
        sim = scenario.simulation
        self.domain = build_domain(scenario)
        self.solver = ThermalSolver(self.domain)
        self.source: HeatSource = build_heat_source(scenario, self.domain)
        T_amb = float(scenario.atmosphere.ambient_temp_K)
        self.state = self.solver.initial_state(float(scenario.liquid.initial_temp_K), T_amb, T_amb)
        self._cache: CoefficientCache = make_cache(
            self.domain,
            scenario,
            self.source.deposition,
            solid_props(scenario.vessel.material).emissivity,
        )
        self.recorder = Recorder()
        self.snapshots = SnapshotStore()
        self.cumulative = EnergyFlows()
        self.gross_J = 0.0
        self.step_count = 0
        self.refresh_wall_s = 0.0
        self.wall_s = 0.0
        self.error: Exception | None = None
        self._stop: StopReason | None = None
        self._finalized = False
        self._started = False
        self._probe_names = [p.name for p in self.domain.probes]
        self._target_probe = scenario.target.probe
        self._target_K = float(scenario.target.temp_K)
        self._sample_dt = float(sim.sample_interval_s)
        self._snap_dt = float(sim.snapshot_interval_s)
        self._refresh_dt = float(sim.coeff_refresh_s)
        self._max_t = float(sim.max_sim_time_s)
        self._next_sample = 0.0
        self._next_snap = 0.0
        self._next_refresh = 0.0
        self._stable_dt = float("inf")
        self._last_cmd = 0.0
        self._last_sp = SourcePower(0.0, 0.0)
        self._liq_flat = np.flatnonzero(self.domain.grid.is_liquid.reshape(-1))
        self._hist_t: list[float] = []
        self._hist_v: list[float] = []
        self._hist_cmd: list[float] = []

    # ------------------------------------------------------------------ public API
    @property
    def stop_reason(self) -> StopReason | None:
        return self._stop

    @property
    def refresh_fraction(self) -> float:
        """Share of the stepping wall time spent refreshing coefficients."""
        return self.refresh_wall_s / self.wall_s if self.wall_s > 0 else 0.0

    def scenario_toml(self) -> str:
        return scenario_to_toml(self.scenario)

    def advance(self, until_t: float, max_steps: int | None = None) -> list[Sample | Snapshot]:
        """Step until ``state.t >= until_t`` or a stop criterion fires.

        Returns the samples and snapshots produced, in time order. On a `SolverError` the
        model stops with `StopReason.ERROR`, stores the exception in `error` and returns what
        it has.
        """
        out: list[Sample | Snapshot] = []
        t0 = time.perf_counter()
        try:
            self._advance(until_t, max_steps, out)
        except SolverError as exc:
            self.error = exc
            self._stop = StopReason.ERROR
        self.wall_s += time.perf_counter() - t0
        return out

    def finish(self, reason: StopReason) -> list[Sample | Snapshot]:
        """Stop with `reason` (USER_STOP from the runner), taking a final sample and snapshot."""
        out: list[Sample | Snapshot] = []
        if self._stop is None:
            self._stop = reason
        self._final_records(out)
        return out

    # ------------------------------------------------------------------ internals
    def _advance(self, until_t: float, max_steps: int | None, out: list[Sample | Snapshot]) -> None:
        st = self.state
        if self._stop is not None:
            return
        if not self._started:
            self._started = True
            self._refresh(st.t)
            self._record(out, sample=True, snap=True)
            self._next_sample += self._sample_dt
            self._next_snap += self._snap_dt
        steps = 0
        while self._stop is None and st.t < until_t - _EPS:
            if max_steps is not None and steps >= max_steps:
                break
            if st.t >= self._next_refresh - _EPS:
                self._refresh(st.t)
            dt = min(
                self._stable_dt,
                MAX_DT_S,
                self._next_sample - st.t,
                self._next_snap - st.t,
                self._next_refresh - st.t,
                self._max_t - st.t,
                until_t - st.t,
            )
            if dt < _EPS:
                dt = _EPS * 10
            cmd = self.command.command(st.t, self._hist_v[-1])
            sp = self.source.power(cmd, st.t, dt)
            apply_source(self._coeffs, self._cache, sp)
            flows = self.solver.step(st, self._coeffs, dt)
            self.cumulative.add(flows)
            self.gross_J += sp.gross_W * dt
            self._last_cmd, self._last_sp = cmd, sp
            self.step_count += 1
            steps += 1
            self._after_step(out)

    @property
    def _coeffs(self) -> StepCoefficients:
        coeffs = self._cache.coeffs
        assert coeffs is not None
        return coeffs

    def _refresh(self, t: float) -> None:
        t0 = time.perf_counter()
        build_coefficients(self.domain, self.state, self.scenario, self._last_sp, self._cache)
        if self._coeff_hook is not None:
            self._coeff_hook(self._coeffs)
        self._stable_dt = self.solver.stable_dt(self._coeffs)
        self._next_refresh = t + self._refresh_dt
        self.refresh_wall_s += time.perf_counter() - t0

    def _after_step(self, out: list[Sample | Snapshot]) -> None:
        st = self.state
        sample_due = st.t >= self._next_sample - _EPS
        snap_due = st.t >= self._next_snap - _EPS
        if sample_due or snap_due:
            self._record(out, sample=sample_due, snap=snap_due)
        if sample_due:
            while self._next_sample <= st.t + _EPS:
                self._next_sample += self._sample_dt
        if snap_due:
            while self._next_snap <= st.t + _EPS:
                self._next_snap += self._snap_dt
        if sample_due:
            reason = self._check_stop()
            if reason is not None:
                self._stop = reason
                self._final_records(out)
                return
        if st.t >= self._max_t - _EPS:
            self._stop = StopReason.MAX_TIME
            self._final_records(out)

    def _check_stop(self) -> StopReason | None:
        t = self._hist_t[-1]
        v = self._hist_v[-1]
        if v >= self._target_K:
            return StopReason.TARGET_REACHED
        sim = self.scenario.simulation
        window = float(sim.plateau_window_s)
        if t >= window - _EPS:
            i = bisect.bisect_left(self._hist_t, t - window - _EPS)
            rise = v - self._hist_v[i]
            limit = float(sim.plateau_rate_K_per_min) * window / 60.0
            mean_cmd = float(np.mean(self._hist_cmd[i:]))
            if rise < limit and mean_cmd > 0.0:
                return StopReason.PLATEAU
        return None

    def _final_records(self, out: list[Sample | Snapshot]) -> None:
        if self._finalized:
            return
        self._finalized = True
        samples = self.recorder.samples if self._hist_t else []
        last_t = samples[-1].t if samples else -1.0
        snap_t = self.snapshots[len(self.snapshots) - 1].t if len(self.snapshots) else -1.0
        t = self.state.t
        self._record(out, sample=t > last_t + _EPS, snap=t > snap_t + _EPS)

    def _record(self, out: list[Sample | Snapshot], *, sample: bool, snap: bool) -> None:
        st, dom = self.state, self.domain
        probes = probe_values(dom, st.T_liquid)
        t_mean = mean_liquid_T(dom, st.T_liquid)
        sp = self._last_sp
        if sample:
            vals = np.take(st.T_liquid.reshape(-1), self._liq_flat)
            area = dom.shell.area
            sample_obj = Sample(
                t=st.t,
                probes=probes,
                T_mean=t_mean,
                T_min=float(vals.min()),
                T_max=float(vals.max()),
                T_shell_mean=float(np.sum(st.T_shell * area) / np.sum(area)),
                command=self._last_cmd,
                power_delivered=sp.delivered_W,
                power_gross=sp.gross_W,
                cumulative=replace(self.cumulative),
                fuel_kg=self.source.fuel_kg(self.gross_J),
                electric_kwh=self.source.electric_kwh(self.gross_J),
            )
            self.recorder.add(sample_obj)
            self._hist_t.append(st.t)
            self._hist_v.append(
                t_mean if self._target_probe == "mean" else probes[self._target_probe]
            )
            self._hist_cmd.append(self._last_cmd)
            out.append(sample_obj)
        if snap:
            shell = dom.shell
            sl = dom.slice
            snapshot = Snapshot(
                t=st.t,
                slice_T=extract_slice(dom, st.T_liquid),
                x_centers=sl.x_centers,
                z_centers=sl.z_centers,
                wall_left_z=shell.position[sl.wall_left, 2].astype(np.float64),
                wall_left_T=st.T_shell[sl.wall_left].astype(np.float32),
                wall_right_z=shell.position[sl.wall_right, 2].astype(np.float64),
                wall_right_T=st.T_shell[sl.wall_right].astype(np.float32),
                base_x=shell.position[sl.base_line, 0].astype(np.float64),
                base_T=st.T_shell[sl.base_line].astype(np.float32),
                lid_T=float(st.T_lid) if dom.lid.present else None,
                probes=dict(probes),
                T_mean=t_mean,
                power_delivered=sp.delivered_W,
            )
            self.snapshots.add(snapshot)
            out.append(snapshot)

    def build_result(self, wall_time_s: float) -> RunResult:
        """Summary and `RunResult` for the run so far."""
        reason = self._stop if self._stop is not None else StopReason.USER_STOP
        samples = self.recorder.samples
        summary = summarize(samples, self._target_K, self._target_probe, reason, wall_time_s)
        return RunResult(
            summary=summary,
            samples=samples,
            snapshots=self.snapshots,
            scenario_toml=self.scenario_toml(),
        )
