"""Incompressible Boussinesq flow on the MAC grid: projection method (WP21).

Equations (kinematic pressure ``p`` = pressure / rho0, in m^2/s^2; ``T_ref`` the bulk temperature):

    div u = 0
    du/dt + div(u u) = -grad p + div((nu + nu_t) grad u) + g beta (T - T_ref) z_hat

Time stepping per (sub)step: explicit Euler predictor for advection, viscosity and buoyancy with the
old pressure gradient, then an *incremental* projection: solve ``div(grad phi) = div(u*) / dt``,
``u = u* - dt grad phi``, ``p += phi``. Advection of momentum is in flux form (van Leer TVD
or central); advection of heat is computed from the projected velocity in flux form with a van
Leer limiter and returned as the per-cell source ``q_adv`` (W), to be added to the heat sources
of the thermal solver (which then uses the molecular conductivity). Every face flux is stored once
and added to one cell and subtracted from its neighbour, so ``q_adv`` sums to zero to round-off.

See ``docs/physics/flow.md`` for the discretisation, boundary conditions and validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numba
import numpy as np
from numpy.typing import DTypeLike, NDArray

from thermosim.core.types import F32, F64, Domain, LiquidGrid
from thermosim.flow import kernels as K
from thermosim.flow.grid import FlowGrid
from thermosim.flow.poisson import PoissonSolver

G_STANDARD = 9.80665
_SCHEMES = {"central": K.SCHEME_CENTRAL, "upwind": K.SCHEME_UPWIND, "vanleer": K.SCHEME_VANLEER}


@dataclass(frozen=True)
class FlowProps:
    """Fluid properties for one step, evaluated at the bulk temperature.

    Attributes:
        rho: density rho0, kg/m^3.
        beta: volumetric thermal expansion coefficient, 1/K.
        nu: kinematic viscosity, m^2/s.
        cp: specific heat, J/(kg K).
        T_ref: reference temperature of the buoyancy term, K. ``None`` uses the volume-weighted
            mean of the liquid temperature.
        g: gravitational acceleration, m/s^2.
        prandtl_t: turbulent Prandtl number for ``k_turb``.
    """

    rho: float
    beta: float
    nu: float
    cp: float
    T_ref: float | None = None
    g: float = G_STANDARD
    prandtl_t: float = 0.85


def props_from_physics(T_bulk: float, sg: float, *, g: float = G_STANDARD) -> FlowProps:
    """Wort properties at ``T_bulk`` (K) and specific gravity ``sg`` from ``physics.properties``."""
    from thermosim.physics import properties as P

    rho = float(P.wort_density(T_bulk, sg))
    return FlowProps(
        rho=rho,
        beta=float(P.wort_expansion(T_bulk, sg)),
        nu=float(P.wort_viscosity(T_bulk, sg)) / rho,
        cp=float(P.wort_specific_heat(T_bulk, sg)),
        T_ref=T_bulk,
        g=g,
    )


@dataclass(eq=False)
class FlowState:
    """Velocity and pressure on the MAC grid (dtype of the solver).

    Attributes:
        u, v, w: face velocities, m/s; shapes ``(nx+1, ny, nz)``, ``(nx, ny+1, nz)``,
            ``(nx, ny, nz+1)``. Exactly 0 on closed faces.
        p: cell pressure divided by rho0 (kinematic), m^2/s^2. Zero mean over the liquid.
        nut_max: largest eddy viscosity of the last step, m^2/s (0 without a subgrid model).
    """

    u: NDArray[np.floating]
    v: NDArray[np.floating]
    w: NDArray[np.floating]
    p: NDArray[np.floating]
    nut_max: float = 0.0


@dataclass(eq=False)
class FlowStepResult:
    """Result of :meth:`FlowSolver.step`.

    Attributes:
        q_adv: ``(nx, ny, nz)`` advective heat source, W per cell (sum = 0 to round-off), float64.
        k_turb: ``(nx, ny, nz)`` subgrid conductivity ``rho cp nu_t / Pr_t``, W/(m K), or None.
        max_divergence: largest ``|sum_faces w u|`` over the cells divided by the largest face
            velocity (dimensionless, = dx * max|div u| / max|u|), after the projection.
        poisson_iterations: CG iterations summed over the substeps.
        substeps: number of internal steps taken (the step is split to respect ``max_dt``).
        max_speed: largest face velocity magnitude after the step, m/s.
    """

    q_adv: F64
    k_turb: F64 | None
    max_divergence: float
    poisson_iterations: int
    substeps: int = 1
    max_speed: float = 0.0


@dataclass(eq=False)
class FlowSolver:
    """Projection-method flow solver on the liquid grid of a :class:`Domain`.

    Args:
        domain: a ``Domain`` (only ``domain.grid`` is used) or a ``LiquidGrid``.
        smagorinsky_cs: Smagorinsky constant (0.1-0.17); 0 switches the subgrid model off.
        dtype: dtype of the flow fields (float64 default; float32 for GPUs). Heat fluxes and
            their sums are always float64.
        momentum_scheme: ``"vanleer"`` (TVD, default), ``"central"`` or ``"upwind"``.
        scalar_scheme: advection scheme of the heat flux, same choices.
        poisson_tol: relative CG residual (default 1e-6 float64, 1e-5 float32); the
            divergence left after the projection is below ``poisson_tol`` times the velocity.
        poisson_max_iter: CG iteration cap.
        top_wall: ``None`` (default) for the physical flat free surface (w = 0, free slip); a
            pair ``(Ux, Uy)`` m/s makes the top a no-slip wall moving at that velocity (test
            hook for the lid-driven cavity and Rayleigh-Benard with a rigid lid).
        side_slip: ``(slip_x, slip_y)``; free slip on the *grid-boundary* walls normal to x /
            y (test hook: a thin box with ``slip_y`` is a 2-D problem, ``slip_x`` makes
            symmetry planes). Walls next to solid cells are always no slip.
        threads: Numba threads used by ``step`` (set for the calling thread). ``None`` picks one
            thread per 100 000 liquid cells (at most the available threads), because the launch
            overhead of the many small kernels outweighs the gain on small grids.
    """

    domain: Domain | LiquidGrid
    smagorinsky_cs: float = 0.0
    dtype: DTypeLike = np.float64
    momentum_scheme: str = "vanleer"
    scalar_scheme: str = "vanleer"
    poisson_tol: float | None = None
    poisson_max_iter: int = 200
    top_wall: tuple[float, float] | None = None
    side_slip: tuple[bool, bool] = (False, False)
    threads: int | None = None
    grid: FlowGrid = field(init=False)
    poisson: PoissonSolver = field(init=False)

    def __post_init__(self) -> None:
        lg = self.domain.grid if isinstance(self.domain, Domain) else self.domain
        self.dtype = np.dtype(self.dtype)
        if self.dtype not in (np.float32, np.float64):
            raise ValueError("dtype must be float32 or float64")
        if self.momentum_scheme not in _SCHEMES or self.scalar_scheme not in _SCHEMES:
            raise ValueError(f"schemes must be one of {sorted(_SCHEMES)}")
        self.grid = FlowGrid.from_liquid_grid(lg, self.dtype)
        g = self.grid
        shape = g.shape
        d = self.dtype
        self.poisson = PoissonSolver(
            g.wx, g.wy, g.wz, g.liq, tol=self.poisson_tol, max_iter=self.poisson_max_iter
        )
        nx, ny, nz = shape
        self._us = np.zeros((nx + 1, ny, nz), d)
        self._vs = np.zeros((nx, ny + 1, nz), d)
        self._ws = np.zeros((nx, ny, nz + 1), d)
        self._b = np.zeros(shape, d)
        self._phi = np.zeros(shape, d)
        self._T = np.zeros(shape, np.float64)  # T - T_ref of the current step
        self._nut = np.zeros(shape, d)
        self._Fx = np.zeros((nx + 1, ny, nz))
        self._Fy = np.zeros((nx, ny + 1, nz))
        self._Fz = np.zeros((nx, ny, nz + 1))
        self._part = np.zeros(max(nx, 1), np.float64)
        self._vol = g.volume
        # boundary ghost parameters
        self._fx = 1.0 if self.side_slip[0] else -1.0
        self._fy = 1.0 if self.side_slip[1] else -1.0
        if self.top_wall is None:
            self._top = (1.0, 0.0, 0.0)  # free slip: (f, vo_u, vo_v)
        else:
            self._top = (-1.0, 2.0 * self.top_wall[0], 2.0 * self.top_wall[1])
        if self.threads is None:
            pool = int(numba.config.NUMBA_NUM_THREADS)  # type: ignore[attr-defined]
            self.threads = max(1, min(pool, g.n_liquid // 100_000))
        self._mscheme = _SCHEMES[self.momentum_scheme]
        self._sscheme = _SCHEMES[self.scalar_scheme]

    # ------------------------------------------------------------------ state
    def initial_state(self) -> FlowState:
        """Fluid at rest (zero velocity and pressure)."""
        nx, ny, nz = self.grid.shape
        d = self.dtype
        return FlowState(
            u=np.zeros((nx + 1, ny, nz), d),
            v=np.zeros((nx, ny + 1, nz), d),
            w=np.zeros((nx, ny, nz + 1), d),
            p=np.zeros((nx, ny, nz), d),
        )

    def _max_abs(self, a: np.ndarray) -> float:
        part = self._part
        if part.shape[0] < a.shape[0]:
            part = self._part = np.zeros(a.shape[0], np.float64)
        K.max_abs(part, a)
        return float(part[: a.shape[0]].max())

    def max_speed(self, state: FlowState) -> float:
        """Largest face velocity magnitude, m/s."""
        return max(self._max_abs(state.u), self._max_abs(state.v), self._max_abs(state.w))

    def max_dt(self, state: FlowState, nu: float) -> float:
        """Largest stable time step, s: CFL <= 0.5 (``max_cell(|u|+|v|+|w|) dt / dx`` with the
        face velocities averaged to the cell centre) and half the explicit-viscosity limit
        ``dx^2 / (6 (nu + nu_t,max))``. ``inf`` if there is neither flow nor viscosity."""
        dx = self.grid.dx
        part = self._part
        if part.shape[0] < self.grid.shape[0]:
            part = self._part = np.zeros(self.grid.shape[0], np.float64)
        K.cfl_max(part, state.u, state.v, state.w)
        s = float(part[: self.grid.shape[0]].max())
        dt = math.inf
        if s > 0.0:
            dt = 0.5 * dx / s
        nu_eff = nu + state.nut_max
        if nu_eff > 0.0:
            dt = min(dt, dx * dx / (12.0 * nu_eff))
        return dt

    # ------------------------------------------------------------------ projection
    def divergence(self, state: FlowState) -> F64:
        """Cell divergence ``sum_faces(outward sign * w * u) / dx`` (1/s), float64, 0 outside."""
        g = self.grid
        u, v, w = (a.astype(np.float64) for a in (state.u, state.v, state.w))
        wx, wy, wz = (a.astype(np.float64) for a in (g.wx, g.wy, g.wz))
        d = (
            wx[1:] * u[1:]
            - wx[:-1] * u[:-1]
            + wy[:, 1:] * v[:, 1:]
            - wy[:, :-1] * v[:, :-1]
            + wz[:, :, 1:] * w[:, :, 1:]
            - wz[:, :, :-1] * w[:, :, :-1]
        )
        return np.where(g.liq, d / g.dx, 0.0)

    def _project(self, state: FlowState, us: np.ndarray, vs: np.ndarray, ws: np.ndarray, dt: float):
        """Write the divergence-free ``u = u* - dt grad phi`` into the state; ``p += phi``."""
        g = self.grid
        dx = g.dx
        K.div_rhs(self._b, us, vs, ws, g.wx, g.wy, g.wz, -dx / dt)
        # Absolute floor: |r|_2 <= tol dx/dt u*_max bounds every cell's divergence by tol u*_max.
        ustar = max(self._max_abs(us), self._max_abs(vs), self._max_abs(ws))
        tol = self.poisson.tol or 0.0
        atol = tol * (dx / dt) * ustar
        iters = self.poisson.solve(self._b, self._phi, atol)
        phi = self._phi
        dtdx = dt / dx
        K.project_u(state.u, us, phi, g.mu, dtdx)
        K.project_v(state.v, vs, phi, g.mv, dtdx)
        K.project_w(state.w, ws, phi, g.mw, dtdx)
        K.plane_sum(self._part, phi)
        mean = float(self._part[: phi.shape[0]].sum()) / max(g.n_liquid, 1)
        K.add_mean_free(state.p, phi, g.liq, mean)
        return iters

    def project(self, state: FlowState, dt: float = 1.0) -> int:
        """Make the velocity of ``state`` divergence free (pressure projection, no momentum
        update). Returns the CG iterations. ``dt`` only scales the pressure increment."""
        self._us[...] = state.u
        self._vs[...] = state.v
        self._ws[...] = state.w
        return self._project(state, self._us, self._vs, self._ws, dt)

    def max_divergence(self, state: FlowState) -> float:
        """``dx * max|div u| / max|u|`` (dimensionless); 0 for a field at rest."""
        g = self.grid
        part = self._part
        if part.shape[0] < g.shape[0]:
            part = self._part = np.zeros(g.shape[0], np.float64)
        K.div_max(part, state.u, state.v, state.w, g.wx, g.wy, g.wz)
        m = float(part[: g.shape[0]].max())
        s = self.max_speed(state)
        return m / s if s > 0.0 else 0.0

    # ------------------------------------------------------------------ one step
    def _substep(self, state: FlowState, props: FlowProps, dt: float) -> int:
        g = self.grid
        dx = g.dx
        use_nut = self.smagorinsky_cs > 0.0
        ftop, vou, vov = self._top
        if use_nut:
            K.smag_nut(
                self._nut,
                state.u,
                state.v,
                state.w,
                g.mu,
                g.mv,
                g.mw,
                g.liq,
                (self.smagorinsky_cs * dx) ** 2,
                1.0 / dx,
                self._fx,
                self._fy,
                ftop,
                vou,
                vov,
            )
            state.nut_max = self._max_abs(self._nut)
        K.rhs_u(
            self._us, state.u, state.v, state.w, state.p, g.mu, self._nut, props.nu, dt, dx,
            self._mscheme, self._fy, ftop, vou, use_nut,
        )  # fmt: skip
        K.rhs_v(
            self._vs, state.u, state.v, state.w, state.p, g.mv, self._nut, props.nu, dt, dx,
            self._mscheme, self._fx, ftop, vov, use_nut,
        )  # fmt: skip
        K.rhs_w(
            self._ws, state.u, state.v, state.w, state.p, g.mw, self._T, self._nut, props.nu, dt,
            dx, self._mscheme, self._fx, self._fy, use_nut, props.g * props.beta, 0.0,
        )  # fmt: skip
        return self._project(state, self._us, self._vs, self._ws, dt)

    def step(
        self, state: FlowState, T_liquid: F64 | F32, props: FlowProps, dt: float
    ) -> FlowStepResult:
        """Advance the velocity by ``dt`` seconds with the liquid temperature frozen at
        ``T_liquid``, and compute the advective heat source of that temperature field.

        The step is split into substeps of at most :meth:`max_dt` (viscosity from ``props``).
        ``q_adv`` is the average over the substeps of the flux-form advection of ``T_liquid``
        by the projected velocity of each substep. The caller's thermal solver must itself keep
        ``(|u|+|v|+|w|) dt_thermal / dx <= 0.5`` since it applies ``q_adv`` explicitly.

        Args:
            state: velocity and pressure, updated in place.
            T_liquid: ``(nx, ny, nz)`` temperature, K (value outside the liquid is ignored).
            props: fluid properties at the bulk temperature.
            dt: time step, s (> 0).
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.threads is not None and numba.get_num_threads() != self.threads:
            numba.set_num_threads(self.threads)
        g = self.grid
        if props.T_ref is not None:
            tref = float(props.T_ref)
        else:
            tref = float((np.asarray(T_liquid, np.float64) * g.volume).sum() / g.volume.sum())
        # Work with T - T_ref (float64): the buoyancy needs it, and advecting the offset field
        # keeps a divergence error of the projection from producing a heat source ~ T_ref div(u).
        np.subtract(T_liquid, tref, out=self._T, where=g.liq)
        self._T[~g.liq] = 0.0
        rcp_a = props.rho * props.cp * g.dx * g.dx
        q = np.zeros(g.shape, np.float64)
        remaining = dt
        iters = 0
        n_sub = 0
        while remaining > 1e-12 * dt:
            h = min(remaining, self.max_dt(state, props.nu))
            if remaining - h < 1e-9 * dt:
                h = remaining
            iters += self._substep(state, props, h)
            K.flux_x(self._Fx, self._T, state.u, g.mu, g.wx, g.liq, rcp_a, self._sscheme)
            K.flux_y(self._Fy, self._T, state.v, g.mv, g.wy, g.liq, rcp_a, self._sscheme)
            K.flux_z(self._Fz, self._T, state.w, g.mw, g.wz, g.liq, rcp_a, self._sscheme)
            K.flux_accumulate(q, self._Fx, self._Fy, self._Fz, h / dt)
            remaining -= h
            n_sub += 1
        k_turb = None
        if self.smagorinsky_cs > 0.0:
            k_turb = props.rho * props.cp * self._nut.astype(np.float64) / props.prandtl_t
        return FlowStepResult(
            q_adv=q,
            k_turb=k_turb,
            max_divergence=self.max_divergence(state),
            poisson_iterations=iters,
            substeps=n_sub,
            max_speed=self.max_speed(state),
        )

    # ------------------------------------------------------------------ output
    def slice_velocity(self, state: FlowState, j: int) -> tuple[F32, F32]:
        """Velocity at the cell centres of the vertical plane ``j``: ``(u, w)``, each ``(nx, nz)``
        float32 (m/s, index ``[i, k]``); 0 in non-liquid cells."""
        g = self.grid
        u = 0.5 * (state.u[:-1, j, :] + state.u[1:, j, :])
        w = 0.5 * (state.w[:, j, :-1] + state.w[:, j, 1:])
        liq = g.liq[:, j, :]
        return (
            np.where(liq, u, 0.0).astype(np.float32),
            np.where(liq, w, 0.0).astype(np.float32),
        )
