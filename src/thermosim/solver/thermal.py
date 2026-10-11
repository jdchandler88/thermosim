"""Finite-volume explicit thermal solver (CONTRACTS.md section 10, WP06).

Design
------
The liquid is a Cartesian grid of finite volumes, the vessel wall a 2-D network of shell
nodes, and the optional lid a single lumped node over a zero-capacity headspace. All nodes
advance with forward Euler. The solver contains no physics correlations: it receives
linearised coefficients (``StepCoefficients``) and returns the energy that flowed, by
category (``EnergyFlows``).

Conductances (C in J/K, G in W/K, temperatures in K):

* liquid-liquid: ``G = A_face * k_h / dx`` with the harmonic mean ``k_h = 2 k_p k_q / (k_p + k_q)``;
* shell-liquid: ``G = A / (1/h_film + t/(2 k_s))``. The film coefficient replaces the
  unresolved boundary layer, so no extra half-cell liquid resistance is added;
* free surface of cell p (area A_s, covered fraction f, top-layer half depth
  ``delta = V_p / (2 A_s)``): open part to ambient ``A_s (1-f) / (delta/k_p + 1/h_open)``,
  covered part to the headspace ``A_s f / (delta/k_p + 1/h_cov)``; the open part also loses
  ``q_evap_open * A_s * (1-f)``;
* shell outer side ``1 / (t/(2 k A) + R_ins/A_out + 1/(h_out A_out))``; dry inner side
  ``A_dry / (1/h_dry + t/(2k))`` to the headspace (lid) or ambient (no lid);
* lid: headspace to lid ``h_lid_inner * A_lid``; lid to ambient
  ``1 / (R_ins/A_lid + 1/(h_lid_outer A_lid))``; partial lid: opening conductance
  ``h_lid_inner (1-f) pi R^2`` from the headspace to ambient.

The headspace temperature is the conductance-weighted mean of everything connected to it
(covered surface cells, dry walls, lid, opening), recomputed at every (sub)step.

Time stepping: ``stable_dt`` is 0.9 * min_p C_p / sum(G_p) over liquid cells. If ``dt``
exceeds the same limit for the shell or lid nodes, those nodes take ``ceil(dt / dt_shell)``
equal substeps with the liquid held at its start-of-step temperatures; the energy
exchanged with each liquid cell is accumulated and applied once at the end of the step.
Stored energy is ``C * dT`` per node, so ``EnergyFlows.balance_error()`` is round-off only.
``step`` leaves the state untouched when it raises.

Threading. The liquid kernels run serially on small grids (below ``SERIAL_BELOW_CELLS``
liquid cells) and otherwise in parallel with an automatically tuned thread count: every
``_RETUNE_S`` seconds of stepping the solver times a few real steps with 1, half and all of
the available threads and keeps the fastest. When other processes compete for the CPU a
parallel region waits for its slowest thread, which can make the full thread count many
times slower than a single thread; the tuner then falls back to fewer threads. The
environment variable ``THERMOSIM_THREADS`` (or the ``threads`` argument) fixes the count
instead. Neither changes the results (the reductions use fixed blocks).

Known limitations: the solver instance owns scratch buffers, so one instance must not be
stepped from two threads at once. A lid needs a positive heat capacity.
"""

from __future__ import annotations

import math
import os
import time
from functools import cache
from typing import Any, NamedTuple

import numba
import numpy as np

from thermosim.core.errors import SolverError
from thermosim.core.types import F64, I64, Domain, EnergyFlows, StepCoefficients, ThermalState

from . import kernels as kn
from .network import Network, build_network

_STABILITY_MARGIN = 0.9
_DT_TOLERANCE = 1.0001
SERIAL_BELOW_CELLS = 40_000  # liquid cells; smaller grids never use the threading layer
_RETUNE_S = 8.0  # wall seconds of stepping between thread-count trials
_TRIAL_STEPS = 4  # steps per candidate in a trial (the first is discarded)


def _f64(a: object) -> F64:
    return np.ascontiguousarray(a, dtype=np.float64)


class _Layout(NamedTuple):
    """Which cells the interior fast path and the general path handle (see `kernels`)."""

    gen_n: I64  # compact indices of the cells updated through the neighbour tables
    run_start: I64  # flat index of the first cell of each interior run
    run_comp: I64  # compact index of the first cell of each interior run
    run_len: I64
    blk: I64  # (N_INTERIOR_BLOCKS + 1,) run index range of each block
    g: tuple[float, float, float]  # face conductance of an interior cell per axis (uniform k)
    sg6: float  # sum of its six face conductances


class _ThreadTuner:
    """Chooses the Numba thread count of the liquid kernels from the measured step times."""

    def __init__(self, n_max: int, fixed: int | None, n_cells: int) -> None:
        self.n_max = max(1, n_max)
        self.best = 1
        self._queue: list[int] = []
        self._samples: dict[int, list[float]] = {}
        self._next_tune = math.inf
        if fixed is not None:
            self.best = max(1, min(fixed, self.n_max))
        elif n_cells >= SERIAL_BELOW_CELLS and self.n_max > 1:
            self.best = self.n_max
            self._next_tune = 0.0  # tune on the first steps
        cands = {1, self.n_max}
        if self.n_max >= 4:
            cands.add(self.n_max // 2)
        self._cands = sorted(cands)

    def pick(self) -> int:
        """Thread count for the next liquid step."""
        if self._queue:
            return self._queue[0]
        if self._next_tune < math.inf and time.perf_counter() >= self._next_tune:
            self._queue = [c for c in self._cands for _ in range(_TRIAL_STEPS)]
            self._samples = {c: [] for c in self._cands}
            return self._queue[0]
        return self.best

    def record(self, n: int, seconds: float) -> None:
        """Report the duration of a step run with ``n`` threads."""
        if not self._queue or n != self._queue[0]:
            return
        self._queue.pop(0)
        self._samples[n].append(seconds)
        if not self._queue:
            med = {c: sorted(v[1:])[len(v[1:]) // 2] for c, v in self._samples.items()}
            self.best = min(med, key=lambda c: med[c] * (1.0 if c == self.n_max else 1.1))
            self._next_tune = time.perf_counter() + _RETUNE_S


def _thread_limit() -> int | None:
    """Fixed thread count from ``THERMOSIM_THREADS`` (None if unset or invalid)."""
    raw = os.environ.get("THERMOSIM_THREADS", "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


class ThermalSolver:
    """Explicit finite-volume solver for liquid, shell, lid and headspace temperatures."""

    def __init__(self, domain: Domain, threads: int | None = None) -> None:
        """Precompute index arrays and trigger (or load from cache) the Numba compile.

        Args:
            domain: the geometry. Must satisfy ``lid.heat_capacity > 0`` when a lid is present.
            threads: fixed number of threads for the liquid kernels (1 = serial). Default:
                ``THERMOSIM_THREADS`` if set, otherwise serial for small grids and
                automatically tuned for large ones (see the module docstring).
        """
        self.domain = domain
        self.net: Network = build_network(domain)
        grid, shell, lid = domain.grid, domain.shell, domain.lid
        self._shape = grid.shape
        self._dx = float(grid.dx)
        self._lid = lid
        self._f_cover = float(lid.covered_fraction) if lid.present else 0.0
        if lid.present and not lid.heat_capacity > 0.0:
            raise SolverError("lid heat capacity must be positive")
        self._opening_area = math.pi * domain.inner_radius**2

        n_sh = shell.n
        n_c = len(shell.couple_shell)
        n_s = len(self.net.surf_n)
        n_liq = self.net.n_liq
        self._extra = np.zeros((n_liq, 2), dtype=np.float64)
        self._out: F64 | None = None  # spare full-size buffer the step writes into
        self._part = np.zeros((kn.N_BLOCKS, 3), dtype=np.float64)
        self._cs = np.zeros(n_sh)
        self._g_out = np.zeros(n_sh)
        self._g_dry = np.zeros(n_sh)
        self._g_sum = np.zeros(n_sh)
        self._g_c = np.zeros(n_c)
        self._net = np.zeros(n_sh)
        self._g_open = np.zeros(n_s)
        self._g_cov = np.zeros(n_s)
        self._surf_area = _f64(grid.surface_area)
        self._couple_shell = np.ascontiguousarray(shell.couple_shell, dtype=np.int64)
        self._edge_a = np.ascontiguousarray(shell.edge_a, dtype=np.int64)
        self._edge_b = np.ascontiguousarray(shell.edge_b, dtype=np.int64)
        self._edge_g = _f64(shell.edge_conductance)
        self._shell_static = tuple(
            _f64(a)
            for a in (
                shell.area,
                shell.thickness,
                shell.rho_c,
                shell.k,
                shell.outer_area,
                shell.insulation_r,
                shell.dry_area,
                shell.couple_area,
            )
        )
        #: Number of shell/lid substeps used by the last call to ``step`` (for diagnostics).
        self.last_substeps = 1
        self._cap = np.zeros(n_liq, dtype=np.float64)
        self._build_layout()
        fixed = threads if threads is not None else _thread_limit()
        pool = int(numba.config.NUMBA_NUM_THREADS)  # type: ignore[attr-defined]
        self._tuner = _ThreadTuner(pool, fixed, n_liq)
        self._prep: tuple[object, ...] | None = None
        _warmup()
        if self._tuner.best > 1:
            _warmup_parallel()

    # ------------------------------------------------------------------ public API
    def initial_state(self, T0_liquid: float, T0_shell: float, T_amb: float) -> ThermalState:
        """Uniform initial temperatures.

        Args:
            T0_liquid: initial liquid temperature, K. Cells outside the liquid hold 0.
            T0_shell: initial shell temperature, K.
            T_amb: ambient temperature, K. The lid and headspace start at ambient.

        Returns:
            A fresh ``ThermalState`` at t = 0.
        """
        t_liq = np.where(self.domain.grid.is_liquid, float(T0_liquid), 0.0).astype(np.float64)
        return ThermalState(
            t=0.0,
            T_liquid=t_liq,
            T_shell=np.full(self.domain.shell.n, float(T0_shell), dtype=np.float64),
            T_lid=float(T_amb),
            T_headspace=float(T_amb),
        )

    def stable_dt(self, coeffs: StepCoefficients) -> float:
        """Liquid stability limit times 0.9, in s (``inf`` if no liquid cell has any loss path).

        Args:
            coeffs: the coefficients that will be used for the step.
        """
        self._prepare(coeffs)
        lay = self._layout
        args = (
            _f64(coeffs.k_liquid).reshape(-1),
            self._cap,
            self.net.liq_flat,
            self.net.nbr,
            self.net.nbr_area,
            self._extra,
            lay.gen_n,
            lay.run_comp,
            lay.run_len,
            lay.blk,
            lay.sg6,
            self._dx,
            self._part,
        )
        if self._tuner.best > 1:
            numba.set_num_threads(self._tuner.best)
            kn.liquid_max_rate_par(*args)
        else:
            kn.liquid_max_rate_ser(*args)
        return self._dt_from_rate(float(self._part[:, 2].max()))

    def step(
        self,
        state: ThermalState,
        coeffs: StepCoefficients,
        dt: float,
        *,
        reuse_conductances: bool = False,
        swap_buffers: bool = False,
    ) -> EnergyFlows:
        """Advance ``state`` in place by ``dt`` seconds (``state.t += dt``).

        Args:
            state: temperatures, K. Left untouched if an error is raised.
            coeffs: physics coefficients for this step.
            dt: time step, s, > 0 and at most ``stable_dt(coeffs) * 1.0001``.
            reuse_conductances: the caller promises that, apart from the source terms, the
                coefficients have not changed since the last ``stable_dt`` or ``step`` call
                (the model refreshes them at fixed intervals), so the conductances are not
                rebuilt. Ignored (they are rebuilt) when ``coeffs`` holds a different set of
                arrays than at the last preparation.
            swap_buffers: commit the new liquid temperatures by replacing
                ``state.T_liquid`` with the solver's spare array (and keeping the old one as the
                spare) instead of copying into the existing array. Saves a pass over the grid, but
                anything holding a reference to the old ``state.T_liquid`` sees stale values, so
                it is off by default; the model uses it. Cells outside the liquid keep their
                initial values (zero).

        Returns:
            The energy in J that flowed during the step, by category.

        Raises:
            SolverError: if ``dt`` is too large for stability or a non-finite value appears.
        """
        if not dt > 0.0:
            raise SolverError(f"dt must be positive, got {dt}")
        net, lid = self.net, self._lid
        t_liq = state.T_liquid
        if t_liq.dtype != np.float64 or not t_liq.flags.c_contiguous:
            raise SolverError("state.T_liquid must be a C-contiguous float64 array")
        t_flat = t_liq.reshape(-1)
        k = _f64(coeffs.k_liquid).reshape(-1)
        q_cells = _f64(coeffs.q_cells).reshape(-1)
        q_shell = _f64(coeffs.q_shell)
        t_amb = float(coeffs.T_amb)
        flows = np.zeros(kn.N_FLOWS)

        if reuse_conductances and self._prep is not None and self._prep[0] == self._ids(coeffs):
            kn.reset_extra_energy(self._extra, net.special)
            dt_shell, g_hl, g_lo, g_opn = self._prep[1]  # type: ignore[misc]
        else:
            dt_shell, g_hl, g_lo, g_opn = self._prepare(coeffs)
        kn.surface_open_energy(
            net.surf_n,
            net.surf_flat,
            self._surf_area,
            t_flat,
            self._g_open,
            _f64(coeffs.q_evap_open),
            self._f_cover,
            t_amb,
            dt,
            self._extra,
            flows,
        )

        # Shell, lid and headspace, with the liquid held at its start-of-step temperatures.
        dt_lim = _STABILITY_MARGIN * dt_shell
        if lid.present:
            g_tot = g_hl + g_lo
            if g_tot > 0.0:
                dt_lim = min(dt_lim, _STABILITY_MARGIN * lid.heat_capacity / g_tot)
        nsub = 1 if not math.isfinite(dt_lim) or dt <= dt_lim else math.ceil(dt / dt_lim)
        self.last_substeps = nsub
        ts_work = state.T_shell.copy()
        ts_start = state.T_shell
        t_lid, t_hs = kn.shell_substeps(
            nsub,
            dt / nsub,
            ts_work,
            float(state.T_lid),
            float(state.T_headspace),
            bool(lid.present),
            t_amb,
            self._cs,
            self._g_out,
            self._g_dry,
            self._g_c,
            q_shell,
            self._edge_a,
            self._edge_b,
            self._edge_g,
            self._couple_shell,
            net.couple_n,
            t_flat,
            net.couple_flat,
            net.surf_n,
            self._g_cov,
            net.surf_flat,
            float(lid.heat_capacity),
            g_hl,
            g_lo,
            g_opn,
            self._net,
            self._extra,
            flows,
        )

        # Everything so far works on copies; stop before touching the state if it went bad.
        d_shell, shell_finite = kn.shell_energy_change(self._cs, ts_work, ts_start)
        if not (
            shell_finite and np.isfinite(flows).all() and math.isfinite(t_lid + t_hs + d_shell)
        ):
            self._raise_non_finite(ts_work, t_lid, t_hs)

        # Liquid: update into the spare buffer; committed below only if both checks pass.
        out = self._spare(t_liq)
        out_flat = out.reshape(-1)
        lay = self._layout
        n_thr = self._tuner.pick()
        t_start = time.perf_counter()
        args = (
            t_flat,
            k,
            self._cap,
            q_cells,
            net.liq_flat,
            net.nbr,
            net.nbr_area,
            self._extra,
            lay.gen_n,
            lay.run_start,
            lay.run_comp,
            lay.run_len,
            lay.blk,
            *lay.g,
            lay.sg6,
            self._sx,
            self._sy,
            self._dx,
            dt,
            _STABILITY_MARGIN * _DT_TOLERANCE,
            out_flat,
            self._part,
        )
        if n_thr > 1:
            numba.set_num_threads(n_thr)
            ok = kn.liquid_step_par(*args)
        else:
            ok = kn.liquid_step_ser(*args)
        self._tuner.record(n_thr, time.perf_counter() - t_start)
        if not ok:
            if not np.isfinite(out_flat[net.liq_flat]).all():
                self._raise_non_finite(ts_work, t_lid, t_hs, out_flat)
            stable = self._dt_from_rate(float(self._part[:, 2].max()))
            raise SolverError(f"dt = {dt:g} s exceeds the stable time step {stable:g} s")
        flows[kn.F_ST_LIQ] = float(self._part[:, 0].sum())
        flows[kn.F_SRC_LIQ] = float(self._part[:, 1].sum())
        flows[kn.F_SRC_SHELL] = float(q_shell.sum()) * dt
        flows[kn.F_ST_SHELL] = d_shell
        flows[kn.F_ST_LID] = lid.heat_capacity * (t_lid - state.T_lid) if lid.present else 0.0

        # Commit: liquid, then the small parts.
        if swap_buffers:
            self._out = t_liq
            state.T_liquid = out
        else:
            kn.copy_liquid(t_flat, out_flat, net.liq_flat)
        state.T_shell[:] = ts_work
        state.T_lid = float(t_lid)
        state.T_headspace = float(t_hs)
        state.t += dt
        return EnergyFlows(
            source_liquid=float(flows[kn.F_SRC_LIQ]),
            source_shell=float(flows[kn.F_SRC_SHELL]),
            loss_shell_outer=float(flows[kn.F_OUT]),
            loss_surface_convrad=float(flows[kn.F_CONV]),
            loss_surface_evap=float(flows[kn.F_EVAP]),
            loss_lid_outer=float(flows[kn.F_LID]),
            loss_dry_wall_open=float(flows[kn.F_DRY]),
            stored_liquid=float(flows[kn.F_ST_LIQ]),
            stored_shell=float(flows[kn.F_ST_SHELL]),
            stored_lid_headspace=float(flows[kn.F_ST_LID]),
        )

    def internal_energy(self, state: ThermalState, coeffs: StepCoefficients) -> float:
        """Total internal energy in J relative to 0 K (liquid + shell + lid; headspace is 0).

        Args:
            state: temperatures, K.
            coeffs: supplies the liquid volumetric heat capacity.
        """
        net = self.net
        t_flat = state.T_liquid.reshape(-1)[net.liq_flat]
        rc = _f64(coeffs.rho_c_liquid).reshape(-1)[net.liq_flat]
        energy = float(np.sum(rc * net.volume * t_flat))
        shell = self.domain.shell
        if shell.n:
            energy += float(np.sum(shell.rho_c * shell.thickness * shell.area * state.T_shell))
        if self._lid.present:
            energy += self._lid.heat_capacity * float(state.T_lid)
        return energy

    # ------------------------------------------------------------------ internals
    def _build_layout(self) -> None:
        """Split the liquid cells into interior runs (fast path) and the rest (general path)."""
        net = self.net
        nb_int = kn.N_INTERIOR_BLOCKS
        empty = np.zeros(0, dtype=np.int64)
        ny, nz = self._shape[1], self._shape[2]
        self._sx, self._sy = ny * nz, nz
        self._general = _Layout(
            np.arange(net.n_liq, dtype=np.int64), empty, empty, empty,
            np.zeros(nb_int + 1, dtype=np.int64), (0.0, 0.0, 0.0), 0.0,
        )  # fmt: skip
        self._interior: _Layout | None = None
        self._interior_area: tuple[float, float, float] = (0.0, 0.0, 0.0)
        if net.n_liq == 0:
            return
        # face area of a full interior face, per axis: the most common positive value (the
        # values are bitwise equal for all interior cells of one axis, but differ in the last
        # bit between axes); cut faces next to the wall differ and go through the general path
        areas = []
        for ax in range(3):
            vals = net.nbr_area[:, 2 * ax : 2 * ax + 2].ravel()
            u, cnt = np.unique(vals[vals > 0.0], return_counts=True)
            areas.append(float(u[np.argmax(cnt)]) if u.size else 0.0)
        mask = (net.nbr >= 0).all(axis=1)
        for d in range(6):
            mask &= net.nbr_area[:, d] == areas[d // 2]
        mask[net.special] = False
        if not mask.any():
            return
        idx = np.flatnonzero(mask).astype(np.int64)
        pflat = net.liq_flat[idx]
        brk = np.flatnonzero((np.diff(pflat) != 1) | (np.diff(idx) != 1)) + 1
        starts = np.concatenate([[0], brk]).astype(np.int64)
        ends = np.concatenate([brk, [idx.size]]).astype(np.int64)
        run_len = ends - starts
        cum = np.concatenate([[0], np.cumsum(run_len)])
        blk = np.searchsorted(cum, np.linspace(0, cum[-1], nb_int + 1), side="left")
        blk[0], blk[-1] = 0, run_len.size
        self._interior_area = (areas[0], areas[1], areas[2])
        self._interior = _Layout(
            np.flatnonzero(~mask).astype(np.int64),
            np.ascontiguousarray(pflat[starts]),
            np.ascontiguousarray(idx[starts]),
            np.ascontiguousarray(run_len),
            np.ascontiguousarray(blk, dtype=np.int64),
            (0.0, 0.0, 0.0),
            0.0,
        )

    def _choose_layout(self, k_flat: F64) -> _Layout:
        """The fast path needs a uniform, positive conductivity; otherwise use the general one."""
        if self._interior is None:
            return self._general
        k0 = float(k_flat[self.net.liq_flat[0]])
        # cheap test on the whole array first (the model fills it uniformly); otherwise gather
        if not (float(k_flat.min()) == k0 == float(k_flat.max())):
            kl = k_flat[self.net.liq_flat]
            if not (float(kl.min()) == k0 == float(kl.max())):
                return self._general
        if not k0 > 0.0:
            return self._general
        kh = 2.0 * k0 * k0 / (k0 + k0)
        gx, gy, gz = (a * kh / self._dx for a in self._interior_area)
        g = (gx, gy, gz)
        sg6 = 0.0
        for gi in g:
            sg6 += gi
            sg6 += gi
        return self._interior._replace(g=g, sg6=sg6)

    @staticmethod
    def _ids(coeffs: StepCoefficients) -> tuple[object, ...]:
        """Identity of the arrays (and the lid scalars) that the conductances were built from."""
        return (
            id(coeffs.k_liquid),
            id(coeffs.rho_c_liquid),
            id(coeffs.h_film),
            id(coeffs.h_out_shell),
            id(coeffs.h_dry),
            id(coeffs.h_surface_open),
            id(coeffs.h_surface_covered),
            float(coeffs.h_lid_inner),
            float(coeffs.h_lid_outer),
        )

    def _prepare(self, coeffs: StepCoefficients) -> tuple[float, float, float, float]:
        """Fill the conductance buffers. Returns (shell dt limit, G_hl, G_lo, G_opening)."""
        net, shell, lid = self.net, self.domain.shell, self._lid
        kn.reset_extra(self._extra, net.special)
        area, thick, rho_c, k, outer_area, ins_r, dry_area, couple_area = self._shell_static
        dt_shell = kn.shell_conductances(
            area,
            thick,
            rho_c,
            k,
            outer_area,
            ins_r,
            dry_area,
            net.edge_sum,
            self._couple_shell,
            net.couple_n,
            couple_area,
            _f64(coeffs.h_film),
            _f64(coeffs.h_out_shell),
            _f64(coeffs.h_dry),
            self._cs,
            self._g_out,
            self._g_dry,
            self._g_c,
            self._g_sum,
            self._extra,
        )
        kn.surface_conductances(
            net.surf_n,
            net.surf_flat,
            self._surf_area,
            _f64(coeffs.k_liquid).reshape(-1),
            net.volume,
            _f64(coeffs.h_surface_open),
            _f64(coeffs.h_surface_covered),
            self._f_cover,
            self._g_open,
            self._g_cov,
            self._extra,
        )
        g_hl = g_lo = g_opn = 0.0
        if lid.present:
            h_in = float(coeffs.h_lid_inner)
            h_out = float(coeffs.h_lid_outer)
            if h_in > 0.0:
                g_hl = h_in * lid.area
                g_opn = h_in * (1.0 - self._f_cover) * self._opening_area
            if h_out > 0.0 and lid.area > 0.0:
                g_lo = 1.0 / (lid.insulation_r / lid.area + 1.0 / (h_out * lid.area))
        del shell
        k_flat = _f64(coeffs.k_liquid).reshape(-1)
        kn.make_capacity(_f64(coeffs.rho_c_liquid).reshape(-1), net.volume, net.liq_flat, self._cap)
        self._layout = self._choose_layout(k_flat)
        out = (float(dt_shell), g_hl, g_lo, g_opn)
        self._prep = (self._ids(coeffs), out)
        return out

    @staticmethod
    def _dt_from_rate(max_rate: float) -> float:
        if not max_rate > 0.0:
            return math.inf
        return _STABILITY_MARGIN / max_rate

    def _spare(self, like: F64) -> F64:
        """The spare full-size liquid buffer (allocated on first use, cells outside copied)."""
        if self._out is None or self._out.shape != like.shape or self._out is like:
            self._out = like.copy()
        return self._out

    def _raise_non_finite(
        self, ts: F64, t_lid: float, t_hs: float, out_flat: F64 | None = None
    ) -> None:
        if out_flat is not None:
            bad = np.flatnonzero(~np.isfinite(out_flat[self.net.liq_flat]))
        else:
            bad = np.zeros(0, dtype=np.int64)
        if bad.size:
            flat = int(self.net.liq_flat[bad[0]])
            ijk = tuple(int(v) for v in np.unravel_index(flat, self._shape))
            raise SolverError(f"non-finite liquid temperature at cell {ijk} (flat index {flat})")
        bad = np.flatnonzero(~np.isfinite(ts))
        if bad.size:
            raise SolverError(f"non-finite shell temperature at node {int(bad[0])}")
        if not math.isfinite(t_lid + t_hs):
            raise SolverError("non-finite lid or headspace temperature")
        raise SolverError("non-finite energy flow (check the coefficients and sources)")


@cache
def _warmup() -> None:
    """Compile (or load from the on-disk cache) every kernel on a two-cell toy network."""
    f = np.float64
    n_liq, n_sh, n_c, n_s = 2, 1, 1, 1
    extra = np.zeros((n_liq, 2), dtype=f)
    flows = np.zeros(kn.N_FLOWS, dtype=f)
    liq_flat = np.arange(n_liq, dtype=np.int64)
    vol = np.ones(n_liq, dtype=f)
    temp = np.full(n_liq, 300.0, dtype=f)
    k = np.ones(n_liq, dtype=f)
    one_sh = np.ones(n_sh, dtype=f)
    one_c = np.ones(n_c, dtype=f)
    one_s = np.ones(n_s, dtype=f)
    zero_i = np.zeros(n_c, dtype=np.int64)
    special = np.zeros(1, dtype=np.int64)
    ts = np.full(n_sh, 310.0, dtype=f)
    edge_i = np.zeros(0, dtype=np.int64)
    edge_f = np.zeros(0, dtype=f)
    cs, g_out, g_dry, g_c, g_sum, net = (
        np.zeros(m, dtype=f) for m in (n_sh, n_sh, n_sh, n_c, n_sh, n_sh)
    )
    g_open, g_cov = np.zeros(n_s, dtype=f), np.zeros(n_s, dtype=f)

    kn.reset_extra(extra, special)
    kn.shell_conductances(
        one_sh, one_sh, one_sh, one_sh, one_sh, one_sh, one_sh, np.zeros(n_sh, dtype=f),
        zero_i, zero_i, one_c, one_c, one_sh, one_sh, cs, g_out, g_dry, g_c, g_sum, extra,
    )  # fmt: skip
    kn.surface_conductances(zero_i, zero_i, one_s, k, vol, one_s, one_s, 0.5, g_open, g_cov, extra)
    kn.surface_open_energy(
        zero_i, zero_i, one_s, temp, g_open, one_s, 0.5, 293.0, 1.0, extra, flows
    )
    kn.shell_substeps(
        2, 0.5, ts, 300.0, 300.0, True, 293.0, cs, g_out, g_dry, g_c, one_sh, edge_i, edge_i,
        edge_f, zero_i, zero_i, temp, zero_i, zero_i, g_cov, zero_i, 1.0, 1.0, 1.0, 1.0, net,
        extra, flows,
    )  # fmt: skip
    kn.shell_energy_change(cs, ts, ts)
    kn.make_capacity(k, vol, liq_flat, np.zeros(n_liq, dtype=f))
    kn.reset_extra_energy(extra, special)
    step_args, rate_args = _toy_liquid_args()
    kn.liquid_step_ser(*step_args)
    kn.liquid_max_rate_ser(*rate_args)


@cache
def _warmup_parallel() -> None:
    """Compile (or load) the parallel liquid kernels; only needed when threads are used."""
    step_args, rate_args = _toy_liquid_args()
    kn.liquid_step_par(*step_args)
    kn.liquid_max_rate_par(*rate_args)


def _toy_liquid_args() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Arguments of the liquid kernels on a two-cell toy network (general path only)."""
    f = np.float64
    i64 = np.int64
    liq_flat = np.arange(2, dtype=i64)
    nbr = np.array([[-1, 1, -1, -1, -1, -1], [0, -1, -1, -1, -1, -1]], dtype=np.int32)
    nbr_area = np.where(nbr >= 0, 1.0, 0.0).astype(f)
    ones = np.ones(2, dtype=f)
    extra = np.zeros((2, 2), dtype=f)
    empty = np.zeros(0, dtype=i64)
    blk = np.zeros(kn.N_INTERIOR_BLOCKS + 1, dtype=i64)
    gen = np.arange(2, dtype=i64)
    part = np.zeros((kn.N_BLOCKS, 3), dtype=f)
    step_args = (
        np.full(2, 300.0, dtype=f), ones, ones, ones, liq_flat, nbr, nbr_area, extra, gen,
        empty, empty, empty, blk, 1.0, 1.0, 1.0, 6.0, 1, 1, 1.0, 1.0, 1.0, np.zeros(2, dtype=f),
        part,
    )  # fmt: skip
    rate_args = (ones, ones, liq_flat, nbr, nbr_area, extra, gen, empty, empty, blk, 6.0, 1.0, part)
    return step_args, rate_args
