# Natural-convection flow solver (Phase 2, WP21)

Code: `src/thermosim/flow/` (`grid.py`, `kernels.py`, `poisson.py`, `solver.py`). Tests:
`tests/flow/`. Timing script: `tests/flow/perf_flow.py`.

The flow solver computes the buoyancy-driven motion of the liquid and returns the **advective heat
source** of every cell. It does not replace the Phase 1 thermal solver; WP22 couples the two (see
"Coupling contract").

## 1. Equations

Incompressible Navier-Stokes with the Boussinesq approximation. `p` is the **kinematic**
pressure (`pressure / rho0`, m^2/s^2), `T_ref` the reference (bulk) temperature, `nu_t` the
Smagorinsky eddy viscosity (0 when the subgrid model is off):

    div u = 0
    du/dt + div(u u) = -grad p + div((nu + nu_t) grad u) + g beta (T - T_ref) z_hat
    rho0 cp (dT/dt + div(u T)) = div(k grad T) + sources        (only the advection is done here)

The viscous term is `div(nu_eff grad u)`; the transposed part `div(nu_eff grad u^T)` is dropped
(it vanishes for constant `nu` because `div u = 0`; with the subgrid model it is a small
correction that is usually neglected in LES of buoyant flows).

Accuracy of the Boussinesq form: density, viscosity and `cp` are taken constant at the bulk
temperature (`FlowProps`, refreshed by the model with the coefficients). The neglected relative
density variation is `beta dT`: about 0.5 % for a 10 K difference in water and 1 % at 20 K, which is
the size of the error in the inertia terms. It is largest in the thermal boundary layer above the
burner or element (tens of kelvin across a few millimetres), where the buoyancy itself is exact
(`g beta dT` uses the local `T`) but the viscosity there differs from the bulk value by up to a
factor of two between 20 and 80 degC. This is the accepted error of requirement F4.

## 2. Grid and boundary conditions

MAC staggered grid on `LiquidGrid` (index order `[i, j, k] = [x, y, z]`):

| array | shape | location |
|---|---|---|
| `u` | `(nx+1, ny, nz)` | x-faces: face `i` is between cells `i-1` and `i` |
| `v` | `(nx, ny+1, nz)` | y-faces |
| `w` | `(nx, ny, nz+1)` | z-faces |
| `p`, `T`, `nu_t`, `q_adv` | `(nx, ny, nz)` | cell centres |

A face is **open** when both neighbouring cells are liquid; every other face (walls, base, free
surface, grid boundary) is closed and its normal velocity is exactly 0. The open fraction of a
face is `area / dx^2` from `LiquidGrid.area_x/y/z` (1 for a full face; the partial top layer and the
footprint correction of `geometry/mesh.py` make some differ from 1). Continuity and the pressure
operator use these fractions, so the flow is divergence free in exactly the sense in which the
thermal solver sees the cell volumes and face areas, and the advective heat flux through a face is
`rho cp u (w dx^2) T_face`. Momentum itself is discretised on the full-size staggered cells; the
fraction only enters continuity and the heat flux (the partial top layer is therefore treated as a
thin full cell for the momentum balance, a small error confined to the top row).

* **Walls, base** (staircase): no slip. The tangential velocity has a ghost value `-c` beyond a
  closed neighbour (the wall sits half a cell from the velocity sample), so the wall shear is
  `2 nu c / dx`.
* **Free surface**: flat, `w = 0`, free slip (`ghost = +c`, no shear) for `u` and `v`.
* **Test hooks** (not used by the model): `top_wall=(Ux, Uy)` makes the top a no-slip wall moving at
  that velocity (lid-driven cavity, rigid-lid Benard); `side_slip=(sx, sy)` makes the *grid
  boundary* walls normal to x / y free slip (a one-cell-thick box with `side_slip=(.., True)` is an
  exactly 2-D problem; `sx` gives symmetry planes). Solid cells inside the grid are always no slip.
* **Thermal walls**: handled by the Phase 1 shell coupling (the solver sees only the fluid).
  In flow mode `h_film` should be replaced by wall conduction to the first cell (WP22 decides
  on a wall function; none is applied here).

## 3. Time stepping

Per (sub)step, with `dt <= max_dt` (the public `step` splits longer steps automatically):

1. (Smagorinsky) `nu_t = (Cs dx)^2 |S|`, `|S| = sqrt(2 S_ij S_ij)`. The diagonal strain rates are at
   the cell centre; the three shear rates are evaluated at the cell edges with the wall ghost
   values and averaged as squares. No wall-damping function.
2. Explicit Euler predictor for advection, viscosity, buoyancy and the old pressure gradient:
   `u* = u + dt(-div(u u) + div(nu_eff grad u) + g beta (T - T_ref) z_hat - grad p)`.
3. **Incremental pressure projection**: solve `A phi = -(dx/dt) sum_faces(sigma w u*)` (A is the
   positive weighted 7-point Laplacian with Neumann walls), set `u = u* - dt grad phi` on the open
   faces and `p += phi - mean(phi)`. The increment form removes the first-order splitting error at
   the walls that the plain projection has in steady states.
4. Heat advection flux from the projected velocity (see 5).

Momentum advection is in flux form on the staggered control volumes (the advecting velocity at each
control-volume face is the average of the two nearest face velocities). The advected value uses the
van Leer limiter (`momentum_scheme="vanleer"`, default, second order TVD), central differences
(`"central"`) or first-order upwind (`"upwind"`).

**Time-step limit** (`FlowSolver.max_dt(state, nu)`):
`dt <= min(0.5 dx / max_cell(|u|+|v|+|w|), dx^2 / (12 (nu + nu_t,max)))`
where the cell velocity is the average of its face velocities. The first term is a CFL of 0.5 for
the TVD scheme with forward Euler; the second is half the explicit-diffusion stability limit
`dx^2 / (6 nu)`. The buoyancy (internal-wave) limit `N dt < 1` is far weaker for water in
a kettle and is not checked. The first-order time integration is adequate for a flow driven to a
quasi-steady plume structure; steady states of the discrete problem do not depend on `dt`.

## 4. Pressure solver

Matrix-free preconditioned conjugate gradients on the masked cell Laplacian. Preconditioner: one
symmetric geometric-multigrid V-cycle (red-black Gauss-Seidel, sum restriction, piecewise-constant
prolongation, 2x2x2 aggregation, coarse operator by re-discretisation so that walls stay sealed, direct
pseudo-inverse below 600 cells). The null space (constants) is removed by subtracting the
mean of the right-hand side. CG stops at `|r|_2 <= max(tol |b|_2, tol (dx/dt) u*_max)`; the second term
bounds the divergence left in every cell by `tol u*_max` and avoids chasing a vanishing residual
in a steady flow. Default `tol` = 1e-6 (float64), 1e-5 (float32); 6-8 CG iterations per step on
the kettle grids. Test 1 below uses `tol = 1e-10`.

## 5. Heat advection and the coupling contract

For each open face the solver stores the flux `F = rho cp u (w dx^2) T_face` (W, positive towards
+x/+y/+z) with `T_face` from the van Leer limiter (`scalar_scheme`); first order where the
upwind-upwind cell is missing. Then `q_adv[c] = sum(inflows) - sum(outflows)`: every flux is
stored once and enters one cell with + and its neighbour with -, so `sum(q_adv) = 0` to
round-off (tested to 1e-12 relative to `sum|q_adv|` for random fields, also in float32). The flux
is computed from `T - T_ref` (not the absolute temperature): for a divergence-free field this is
the same, and it keeps the small divergence tolerance of the projection from creating a source
`T_ref div u`. A uniform temperature gives `q_adv = 0` and, with no initial motion, no flow.

How WP22 should use it (`FlowSolver.step(state, T_liquid, props, dt) -> FlowStepResult`):

* `props = FlowProps(rho, beta, nu, cp, T_ref=None, g, prandtl_t)`; `props_from_physics(T_bulk, sg)`
  builds it from `physics/properties.py`. `T_ref = None` uses the volume-weighted mean.
* Add `result.q_adv` (W per cell, float64) to `StepCoefficients.q_cells` and use the molecular
  conductivity (plus `result.k_turb`, W/(m K), if the subgrid model is on) as `k_liquid`.
* `step` advances the velocity by `dt` with `T_liquid` frozen and splits `dt` into sub-steps
  of at most `max_dt` (`result.substeps`); `q_adv` is the time average over the sub-steps. The thermal
  solver applies `q_adv` explicitly, so its own time step must satisfy the same CFL:
  `0.5 dx / max_cell(|u|+|v|+|w|)` (`max_dt(state, nu)` provides it).
* `slice_velocity(state, j) -> (u, w)` gives `(nx, nz)` float32 cell-centre velocities (0 in
  non-liquid cells) for `Snapshot.velocity_u/w`.
* `FlowState` holds `u, v, w, p` (and `nut_max`); it is updated in place. `dtype` selects float32 for the
  GPU (fields and kernels run in that precision; fluxes, `q_adv` and all sums stay float64).

## 6. GPU port notes

All kernels are point-wise (`for i in prange(nx): for j: for k:` is one CUDA thread per cell or
face), read from and write to different arrays, and use no Python objects, so they map to
`@numba.cuda.jit` one to one. The reductions write one partial value per x-plane (`part`), summed by
the caller (deterministic, tested to give identical results with 1 and 2 threads). The CG loop and
the V-cycle are driven from Python (about 60 kernel launches per iteration); on a GPU the 41 000-cell
kettle grid would be launch-bound, so a persistent device-side driver or CUDA graphs are worth
considering in WP22/WP20.

## 7. Validation

All comparisons are against published values; the tests are in `tests/flow/`.

| case (test) | published | solver | error |
|---|---|---|---|
| Divergence after projection, random field in a cylinder mask, 20x20x21 (`test_projection`) | <= 1e-8 | 4e-10 (max over cells, relative to max abs velocity; 16 CG iterations) | passes |
| Lid-driven cavity Re = 100, 32^2, Ghia et al. (1982) Table I, u on x = 0.5, 15 points | table | max 2.2 % for |u| >= 0.05, max abs 0.0039 | < 5 % |
| same, Ghia Table II, v on y = 0.5 | table | max 2.8 % (|v| >= 0.05), max abs 0.0056 | < 5 % |
| Heated cavity Pr 0.71, Ra 1e3, 24^2, de Vahl Davis (1983) mean Nu | 1.118 | 1.1225 (1.1204 at 32^2) | +0.4 % |
| same, Ra 1e4, 24^2 | 2.243 | 2.3003 (2.2735 at 32^2, 2.2566 at 48^2) | +2.6 % (+1.4 %, +0.6 %) |
| same, Ra 1e5, 48^2 (slow) | 4.519 | 4.6156 (4.5700 at 64^2) | +2.1 % (+1.1 %) |
| peak u on x = 0.5 / peak w on z = 0.5, Ra 1e3 (units alpha/L) | 3.649 / 3.697 | 3.667 / 3.708 | +0.5 % / +0.3 % |
| same, Ra 1e4 | 16.178 / 19.617 | 16.198 / 19.531 | +0.1 % / -0.4 % |
| same, Ra 1e5 (slow, 48^2) | 34.73 / 68.59 | 34.96 / 68.08 | +0.7 % / -0.7 % |
| Energy: `sum(q_adv)` over `sum|q_adv|`, cylinder mask, random T | 0 | < 1e-12 (round-off) | passes |
| Uniform T: flow and `q_adv` | none | exactly 0 | passes |
| Rayleigh-Benard critical Ra, rigid-rigid, Pr 1, 24 cells over the layer, Chandrasekhar (1961) (slow) | 1707.76 | 1698.8 (growth rate zero crossing between Ra 1600 and 1800) | -0.5 % |
| Rayleigh-Benard Ra 1000 / 3000 (fast) | decays / grows | sigma < 0, w to 1e-14 / sigma > 0, convecting | passes |

Notes. The lid-driven cavity needs the lid-velocity hook; Ghia's v is the solver's `w` (the
thin dimension is y). The Rayleigh-Benard box has free-slip adiabatic side walls so a half-wave
`cos(pi x/W)` is the critical mode of the infinite layer (aspect `25/24` is within 3 % of the
critical half-wavelength 1.008). Figures in parentheses are finer grids; the Nusselt numbers
converge towards the published values at second order. The de Vahl Davis velocity tables are the
maxima along the mid-planes, which is what the tests extract.

## 8. Performance (`tests/flow/perf_flow.py`)

Intel Xeon 2.1 GHz, 4 cores shared with other jobs (load average 1-2 during the runs, so the
multi-thread numbers are pessimistic). The base is heated over a disk of radius 0.10 m (kettle:
6.2 kW, boxes: 2 kW) from rest; time step `0.9 max_dt` (cell-based CFL 0.5), measured over 30 s of
simulated time after 20 s of start-up. One `FlowSolver.step` call is one sub-step here. The solver
only; the Phase 1 thermal step is not included.

| case | cells | threads | ms / step | CG iterations / step | mean dt (s) | max speed (m/s) | wall per simulated second |
|---|---|---|---|---|---|---|---|
| box 32^3 (1 cm) | 32 768 | 1 / 4 | 14.6 / 16.3 | 7.2 | 0.038 | 0.11 | 0.39 s / 0.43 s |
| box 64^3 (1 cm) | 262 144 | 1 / 4 | 118 / 65 | 7.6 | 0.027 | 0.17 | 4.5 s / 2.5 s |
| 40 L kettle, 20 mm | 5 120 | 1 / 4 | 3.0 / 10.3 | 6.4 | 0.073 | 0.12 | 0.041 s / 0.14 s |
| 40 L kettle, 10 mm | 40 800 | 1 / 4 | 21.8 / 22.2 | 6.8 | 0.023 | 0.21 | 0.95 s / 0.97 s |

Below about 100 000 cells the parallel kernels do not pay (about 60 launches per CG iteration at 5-20 us
each); `FlowSolver(threads=None)` therefore uses one thread per 100 000 cells.

Projection for a 45-minute heat-up of the 40 L kettle at 10 mm on one CPU core: 2700 s x 0.95 s of
wall time per simulated second = about **43 minutes** if the time step stays at 0.023 s. The plume
speeds up as the water gets warmer (`beta` rises from 2e-4 to 7e-4 between 20 and 90 degC), which
lowers the CFL time step, and the Phase 1 thermal step and recording add to this, so plan for 1 to 1.5 h
on one core. At 20 mm the same heat-up takes about 2 minutes (0.041 s of wall time per simulated
second).

## 9. Limitations

* Staircase walls (first-order geometry error at the cylinder wall), no wall function.
* Euler time integration (first order) with a CFL <= 0.5 limit: about 0.02 s at 10 mm cells.
* Free surface is flat and rigid; evaporation cooling and surface tension are not modelled.
* Smagorinsky without wall damping slightly over-predicts `nu_t` near walls; it is optional.
* The partial top layer is a full cell in the momentum balance (see 2).
