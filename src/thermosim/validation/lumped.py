"""Lumped (0-D) reference model: one liquid temperature, one shell temperature, one lid.

A deliberately simple counterpart of the 3-D simulator, used to cross-check it and to give the
GUI an instant "quick estimate". It uses the same property (WP02) and correlation (WP03)
functions and the same physics choices as the WP09 coefficient table:

* shell-to-liquid film: ``h_film_liquid`` (base: UP, L = D/4; side: SIDE, L = fill height);
* outer surface: ``h_outer`` (base DOWN, L = outer D/4; side SIDE, L = outer height, cylinder
  diameter = outer diameter) in series with the insulation, using one fixed-point estimate of
  the outer-surface temperature; the **base outer loss is zero while the burner fires**
  (the burner efficiency already accounts for the losses around the flame);
* open free surface: ``h_outer(UP, D/4, eps = 0.95)`` plus ``q_evaporation`` with the
  convective part of that coefficient; covered free surface and lid: ``h_headspace`` with
  ``gap = H_v - h``; lid outer: ``h_outer(UP)`` with the lid emissivity; dry inner wall:
  ``h_headspace`` (lid) or vertical natural convection + half the radiation (no lid);
* the headspace has no heat capacity: its temperature is the conductance-weighted mean of the
  surface, the dry wall, the lid and (partial lid) the ambient, as in the 3-D solver;
* the vessel (shell, lid) starts at ambient temperature, the liquid at its initial temperature.

Differences from the 3-D model (all documented limits of a 0-D model): one shell temperature
(no in-plane conduction or wall half-thickness resistance), the surface temperature is the
liquid mean, the probe named by the target is ignored (the target is the liquid mean), and
the liquid is perfectly mixed (the 3-D model approaches this for a large ``k_eff_scale``).

Heat source: the power is ``rated x efficiency x actuator output`` for the command of the
power schedule, with the actuator at its steady value: variable and on/off actuators deliver
the command as an average fraction, a gas valve applies its turndown rule
(``GasValveActuator.target``). The valve lag (a few seconds) and the SSR cycling are not
modelled; the schedule is applied as piecewise-constant segments. This replicates
``thermosim.sources`` without building a domain; a burner heats the shell, an element heats
the liquid.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from thermosim.config import schema
from thermosim.core.types import F64, OuterFace, StopReason
from thermosim.physics import correlations as cor
from thermosim.physics import properties as pr
from thermosim.sources.actuators import GasValveActuator
from thermosim.sources.build import build_actuator

_SURFACE_EMISSIVITY = 0.95
_BIG = float("inf")


@dataclass(frozen=True)
class LumpedResult:
    """Outcome of a lumped run. Temperatures in K, power in W."""

    time_to_target_s: float | None
    stop_reason: StopReason
    t: F64
    T_liquid: F64
    T_shell: F64
    power_W: F64  # delivered power
    T_lid: F64 | None = field(default=None)


@dataclass(frozen=True)
class _Coeffs:
    """Conductances (W/K) and the evaporative loss (W) at one instant."""

    G_film: float
    G_out: float  # shell outer surface to ambient
    G_dry: float  # dry inner wall to headspace (lid) or ambient (no lid)
    G_open: float  # open free surface (conv + rad) to ambient
    Q_evap: float  # W, open-surface evaporation
    G_cov: float  # covered free surface to headspace
    G_lid_in: float  # headspace to lid
    G_lid_out: float  # lid to ambient
    G_opening: float  # headspace to ambient through a partial lid's opening


class LumpedModel:
    """0-D model of a scenario. ``run()`` integrates to the target, a plateau or the max time.

    Args:
        scenario: the scenario (the same object the 3-D model takes).
        refresh_s: if set, coefficients are frozen between refreshes of this many simulated
            seconds (as in the 3-D model); if None they follow the state continuously.
        rtol: relative tolerance of the RK45 integration.
        sample_interval_s: spacing of the returned time series.
    """

    def __init__(
        self,
        scenario: schema.Scenario,
        *,
        refresh_s: float | None = None,
        rtol: float = 1e-8,
        sample_interval_s: float = 1.0,
    ) -> None:
        self.scenario = scenario
        self.refresh_s = refresh_s
        self.rtol = rtol
        self.sample_interval_s = sample_interval_s
        # Test hook: with a zero shell capacity the shell is algebraic (no-lid scenarios only).
        self.shell_quasi_steady = False

        v, liq, atm = scenario.vessel, scenario.liquid, scenario.atmosphere
        self.R = 0.5 * v.inner_diameter_m
        self.H_v = v.inner_height_m
        self.t_w = v.wall_thickness_m
        self.t_b = v.base_thickness_m
        self.fill = (
            liq.fill_height_m
            if liq.fill_height_m is not None
            else float(liq.volume_m3) / (math.pi * self.R**2)  # type: ignore[arg-type]
        )
        self.volume = math.pi * self.R**2 * self.fill
        self.sg = liq.specific_gravity
        self.T_amb = atm.ambient_temp_K
        self.p = float(pr.pressure_at_altitude(atm.altitude_m))
        self.wind = atm.wind_speed_m_s
        self.rh = atm.relative_humidity
        self.mass = float(pr.wort_density(liq.initial_temp_K, self.sg)) * self.volume

        wall = pr.solid_props(v.material)
        self.wall = wall
        self.C_shell = (
            wall.density
            * wall.specific_heat
            * (2.0 * math.pi * self.R * self.H_v * self.t_w + math.pi * self.R**2 * self.t_b)
        )
        ins = scenario.insulation

        def ins_parts(part: str) -> tuple[float, float, float]:
            if ins is None or part not in ins.covers:
                return 0.0, 0.0, 0.0
            return ins.thickness_m, ins.thickness_m / ins.conductivity_W_mK, ins.emissivity

        self.t_ins_side, self.R_ins_side, e_side = ins_parts("side")
        _, self.R_ins_base, e_base = ins_parts("base")
        _, self.R_ins_lid, e_lid = ins_parts("lid")
        self.eps_side = e_side if self.R_ins_side > 0 else wall.emissivity
        self.eps_base = e_base if self.R_ins_base > 0 else wall.emissivity
        Ro = self.R + self.t_w
        self.A_base_out = math.pi * Ro**2
        self.A_side_out = 2.0 * math.pi * Ro * self.H_v
        self.D_out = 2.0 * (Ro + self.t_ins_side)

        lid = scenario.lid
        self.has_lid = lid.mode != "none"
        self.f_cover = (
            0.0 if not self.has_lid else (1.0 if lid.mode == "full" else lid.covered_fraction)
        )
        self.A_disk = math.pi * self.R**2
        self.A_lid = self.f_cover * self.A_disk
        if self.has_lid:
            lm = pr.solid_props(lid.material)
            self.C_lid = lm.density * lm.specific_heat * lid.thickness_m * self.A_lid
            self.eps_lid = e_lid if self.R_ins_lid > 0 else lm.emissivity
        else:
            self.C_lid = 0.0
            self.eps_lid = 0.0

        self._cfg = scenario.heat_source
        self._actuator = build_actuator(self._cfg.actuator)
        self._burner = isinstance(self._cfg, schema.Burner)
        self._eff = self._cfg.efficiency if isinstance(self._cfg, schema.Burner) else 1.0
        steps = sorted(scenario.power_schedule, key=lambda s: s.t_s)
        self._sched_t = [float(s.t_s) for s in steps]
        self._sched_c = [min(1.0, max(0.0, float(s.command))) for s in steps]

    # ------------------------------------------------------------------ hooks
    def _C_liquid(self, T_liquid: float) -> float:
        return self.mass * float(pr.wort_specific_heat(T_liquid, self.sg))

    def _C_shell(self) -> float:
        return self.C_shell

    def _output_fraction(self, command: float) -> float:
        """Steady actuator output (average fraction of rated power) for a command."""
        if isinstance(self._actuator, GasValveActuator):
            return self._actuator.target(command)
        return command

    def power(self, command: float) -> float:
        """Delivered power (W) for a command."""
        return self._cfg.rated_power_W * self._eff * self._output_fraction(command)

    def _command_at(self, t: float) -> float:
        i = bisect.bisect_right(self._sched_t, t) - 1
        return self._sched_c[max(i, 0)]

    def _outer_G(
        self,
        T_s: float,
        area: float,
        face: OuterFace,
        L: float,
        eps: float,
        R_ins: float,
        diameter: float | None,
    ) -> float:
        """Conductance of a wall's outer side to ambient, insulation in series (W/K)."""
        h = float(cor.h_outer(T_s, self.T_amb, face, L, self.wind, self.p, eps, diameter=diameter))
        if R_ins > 0.0:  # one fixed-point estimate of the outer-surface temperature
            R_out = 1.0 / h
            T_o = self.T_amb + (T_s - self.T_amb) * R_out / (R_ins + R_out)
            h = float(
                cor.h_outer(T_o, self.T_amb, face, L, self.wind, self.p, eps, diameter=diameter)
            )
        return area / (R_ins + 1.0 / h)

    def _coefficients(self, T_liquid: float, T_shell: float, T_lid: float, firing: bool) -> _Coeffs:
        R, p, sg, Ta = self.R, self.p, self.sg, self.T_amb
        h_base = float(cor.h_film_liquid(T_shell, T_liquid, OuterFace.UP, R / 2.0, sg, p))
        h_side = float(cor.h_film_liquid(T_shell, T_liquid, OuterFace.SIDE, self.fill, sg, p))
        G_film = h_base * self.A_disk + h_side * 2.0 * math.pi * R * self.fill

        G_side = self._outer_G(
            T_shell,
            self.A_side_out,
            OuterFace.SIDE,
            self.H_v + self.t_b,
            self.eps_side,
            self.R_ins_side,
            self.D_out,
        )
        G_base = 0.0
        if not firing or not self._burner:
            G_base = self._outer_G(
                T_shell,
                self.A_base_out,
                OuterFace.DOWN,
                (self.R + self.t_w) / 2.0,
                self.eps_base,
                self.R_ins_base,
                None,
            )

        A_dry = 2.0 * math.pi * R * (self.H_v - self.fill)
        gap = self.H_v - self.fill
        A_open = (1.0 - self.f_cover) * self.A_disk
        h_open = float(
            cor.h_outer(T_liquid, Ta, OuterFace.UP, R / 2.0, self.wind, p, _SURFACE_EMISSIVITY)
        )
        G_open = h_open * A_open
        Q_evap = 0.0
        if A_open > 0.0:
            h_rad = float(cor.h_radiation(T_liquid, Ta, _SURFACE_EMISSIVITY))
            Q_evap = A_open * float(
                cor.q_evaporation(T_liquid, Ta, self.rh, p, max(h_open - h_rad, 0.0))
            )

        if self.has_lid:
            h_hs = float(cor.h_headspace(T_liquid, T_lid, sg, p, gap))
            h_dry = float(cor.h_headspace(T_shell, T_lid, sg, p, gap))
            G_dry = h_dry * A_dry
            G_cov = h_hs * self.f_cover * self.A_disk
            G_lid_in = h_hs * self.A_lid
            G_opening = h_hs * (1.0 - self.f_cover) * self.A_disk
            G_lid_out = self._outer_G(
                T_lid, self.A_lid, OuterFace.UP, R / 2.0, self.eps_lid, self.R_ins_lid, None
            )
        else:
            h_dry = float(cor.h_natural_vertical(T_shell, Ta, gap, p)) + 0.5 * float(
                cor.h_radiation(T_shell, Ta, self.wall.emissivity)
            )
            G_dry = h_dry * A_dry
            G_cov = G_lid_in = G_opening = G_lid_out = 0.0
        return _Coeffs(
            G_film, G_side + G_base, G_dry, G_open, Q_evap, G_cov, G_lid_in, G_lid_out, G_opening
        )

    # ------------------------------------------------------------------ ODE
    def _unpack(self, y: F64, co: _Coeffs | None, Q_s: float) -> tuple[float, float, float]:
        T_L = float(y[0])
        if self.shell_quasi_steady:
            assert co is not None and not self.has_lid
            den = co.G_film + co.G_out + co.G_dry
            T_s = (Q_s + co.G_film * T_L + (co.G_out + co.G_dry) * self.T_amb) / den
            return T_L, T_s, self.T_amb
        return T_L, float(y[1]), float(y[2]) if self.has_lid else self.T_amb

    def _rhs(
        self,
        y: F64,
        command_power: tuple[float, float],
        firing: bool,
        frozen: _Coeffs | None,
    ) -> F64:
        Q_L, Q_s = command_power
        Ta = self.T_amb
        if self.shell_quasi_steady:
            co0 = frozen or self._coefficients(float(y[0]), Ta, Ta, firing)
            T_L, T_s, T_lid = self._unpack(y, co0, Q_s)
            co = frozen or self._coefficients(T_L, T_s, T_lid, firing)
            T_s = self._unpack(y, co, Q_s)[1]
        else:
            T_L, T_s, T_lid = self._unpack(y, None, Q_s)
            co = frozen or self._coefficients(T_L, T_s, T_lid, firing)
        if self.has_lid:
            den = co.G_cov + co.G_dry + co.G_lid_in + co.G_opening
            T_hs = (
                (co.G_cov * T_L + co.G_dry * T_s + co.G_lid_in * T_lid + co.G_opening * Ta) / den
                if den > 0.0
                else Ta
            )
            T_dry_ref = T_hs
        else:
            T_hs = Ta
            T_dry_ref = Ta
        q_film = co.G_film * (T_s - T_L)
        dT_L = (
            Q_L + q_film - co.G_open * (T_L - Ta) - co.Q_evap - co.G_cov * (T_L - T_hs)
        ) / self._C_liquid(T_L)
        out = [dT_L]
        if not self.shell_quasi_steady:
            dT_s = (
                Q_s - q_film - co.G_out * (T_s - Ta) - co.G_dry * (T_s - T_dry_ref)
            ) / self._C_shell()
            out.append(dT_s)
            if self.has_lid:
                out.append(
                    (co.G_lid_in * (T_hs - T_lid) - co.G_lid_out * (T_lid - Ta)) / self.C_lid
                )
        elif self.has_lid:
            raise NotImplementedError("quasi-steady shell supports no-lid scenarios only")
        return np.asarray(out)

    def _shell_temperature(self, y: F64, Q_s: float, firing: bool) -> float:
        if not self.shell_quasi_steady:
            return float(y[1])
        co = self._coefficients(float(y[0]), self.T_amb, self.T_amb, firing)
        return self._unpack(y, co, Q_s)[1]

    # ------------------------------------------------------------------ run
    def run(self) -> LumpedResult:
        """Integrate from t = 0 until the target, a plateau, or the maximum time."""
        sim = self.scenario.simulation
        target = self.scenario.target.temp_K
        t_max = sim.max_sim_time_s
        Ta = self.T_amb
        y: F64 = np.array([self.scenario.liquid.initial_temp_K], dtype=float)
        if not self.shell_quasi_steady:
            y = np.append(y, Ta)
            if self.has_lid:
                y = np.append(y, Ta)

        ts: list[float] = [0.0]
        TL: list[float] = [float(y[0])]
        Ts: list[float] = [self._shell_temperature(y, 0.0, False)]
        TLid: list[float] = [Ta]
        Pw: list[float] = []
        t = 0.0
        chunk = self.refresh_s if self.refresh_s else 300.0
        bounds = [b for b in self._sched_t if b > 0.0]
        window = sim.plateau_window_s
        reason = StopReason.MAX_TIME
        t_hit: float | None = None

        if y[0] >= target:
            reason, t_hit = StopReason.TARGET_REACHED, 0.0
        dt_s = self.sample_interval_s

        while t_hit is None and t < t_max - 1e-9:
            cmd = self._command_at(t)
            P = self.power(cmd)
            Q = (P, 0.0) if not self._burner else (0.0, P)
            firing = P > 0.0
            nxt = min([b for b in bounds if b > t + 1e-12] + [t_max])
            t_end = min(t + chunk, nxt)
            frozen = None
            if self.refresh_s:
                T_L0, T_s0, T_lid0 = (
                    self._unpack(y, None, Q[1])
                    if not self.shell_quasi_steady
                    else (float(y[0]), Ta, Ta)
                )
                frozen = self._coefficients(T_L0, T_s0, T_lid0, firing)

            def hit(_t: float, yy: F64) -> float:
                return float(yy[0] - target)

            hit.terminal = True  # type: ignore[attr-defined]
            hit.direction = 1.0  # type: ignore[attr-defined]
            sol = solve_ivp(
                lambda _t, yy, Q=Q, firing=firing, frozen=frozen: self._rhs(yy, Q, firing, frozen),
                (t, t_end),
                y,
                method="RK45",
                rtol=self.rtol,
                atol=1e-9,
                events=hit,
                dense_output=True,
            )
            t_stop = float(sol.t[-1])
            k0 = math.floor(t / dt_s + 1e-9) + 1
            grid = [k * dt_s for k in range(k0, int(t_stop / dt_s + 1e-9) + 1)]
            if sol.t_events[0].size:
                t_hit = float(sol.t_events[0][0])
                grid.append(t_hit)
            elif not grid or grid[-1] < t_stop - 1e-9:
                grid.append(t_stop)
            if grid:
                ys = sol.sol(np.asarray(grid))
                for j, tg in enumerate(grid):
                    ts.append(tg)
                    TL.append(float(ys[0, j]))
                    Pw.append(P)
                    Ts.append(self._shell_temperature(ys[:, j], Q[1], firing))
                    TLid.append(
                        float(ys[2, j]) if (self.has_lid and not self.shell_quasi_steady) else Ta
                    )
            y = np.asarray(sol.y[:, -1])
            t = t_stop
            if t_hit is not None:
                reason = StopReason.TARGET_REACHED
                break
            if t >= window and len(ts) > 2 and self._plateau(ts, TL, Pw, window, sim):
                reason = StopReason.PLATEAU
                break
        if len(Pw) < len(ts):  # t = 0 sample carries the first segment's power
            Pw.insert(0, self.power(self._command_at(0.0)))
        return LumpedResult(
            time_to_target_s=t_hit,
            stop_reason=reason,
            t=np.asarray(ts),
            T_liquid=np.asarray(TL),
            T_shell=np.asarray(Ts),
            power_W=np.asarray(Pw),
            T_lid=np.asarray(TLid) if self.has_lid else None,
        )

    @staticmethod
    def _plateau(
        ts: list[float], TL: list[float], Pw: list[float], window: float, sim: schema.SimSettings
    ) -> bool:
        t_arr = np.asarray(ts)
        t_now = t_arr[-1]
        if t_now < window:
            return False
        T_then = float(np.interp(t_now - window, t_arr, np.asarray(TL)))
        rise = TL[-1] - T_then
        p_arr = np.asarray(Pw)
        tp = t_arr[1 : 1 + len(p_arr)]
        in_win = tp >= t_now - window
        mean_p = float(p_arr[in_win].mean()) if in_win.any() else 0.0
        return mean_p > 0.0 and rise < sim.plateau_rate_K_per_min * window / 60.0


def quick_estimate(scenario: schema.Scenario) -> float | None:
    """Fast lumped estimate of the time to reach the target, seconds (None if not reached).

    Uses the lumped model with coefficients refreshed every 30 s and a loose tolerance, so it
    runs in well under a second. Intended for the GUI's "quick estimate".
    """
    result = LumpedModel(scenario, refresh_s=30.0, rtol=1e-4, sample_interval_s=60.0).run()
    return result.time_to_target_s
