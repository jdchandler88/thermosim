# Phase 2: Natural convection (flow) on the 3D grid

**Starts after Phase 1 sign-off.** These work packages are specified less
tightly than Phase 1 on purpose. The coordinator refines them, and updates
CONTRACTS.md, once WP20 reports and Phase 1 results are known.

**Target hardware:** an NVIDIA **GTX 1060** (Pascal architecture, CUDA compute
capability 6.1, 3 or 6 GB of memory). Consequences that every Phase 2 package
must respect:
- **Use CUDA 12.x toolchains and wheels.** CUDA 13 dropped support for Pascal
  cards, and some recent PyTorch/CuPy builds have dropped compute capability
  6.1. WP20 must confirm a pinned, working combination on Windows.
- **Use float32 for the flow fields.** Consumer Pascal cards run float64 at
  1/32 speed. Energy accounting may keep float64 accumulators.
- **Memory budget:** at most about 2.5 GB, so the 3 GB variant works. A 128³
  grid with about 10 float32 fields is roughly 85 MB, which is comfortable.
- **Always keep a CPU fallback** (Numba), so the program still runs on
  machines without an NVIDIA GPU.

## Shared design

- **Equations.** Incompressible Navier–Stokes with the Boussinesq
  approximation:
  - ∇·u = 0
  - ∂u/∂t + (u·∇)u = −∇p/ρ₀ + ν∇²u + g β (T − T₀) ẑ
  - ∂T/∂t + u·∇T = ∇·(α∇T) + sources

  Properties are evaluated at the bulk temperature and refreshed with the
  coefficients. The pure Boussinesq form is fine for ΔT below about 20 K in
  the bulk; document the error near the heated base.
- **Grid.** The same `LiquidGrid`. Velocities are stored on a MAC staggered
  grid: u on x-faces, v on y-faces, w on z-faces. Walls are staircase faces
  (u = 0 on any face not between two liquid cells).
- **Boundary conditions.** No-slip at the walls and base. Free slip
  (w = 0, ∂u/∂z = ∂v/∂z = 0) at the free surface, treated as a flat rigid
  surface. Thermal boundary conditions as in Phase 1 (shell coupling, surface
  losses). In flow mode, `h_film` is replaced by wall conduction to the first
  cell, optionally with a thermal wall function (WP21 decides).
- **Time stepping.** A projection (fractional step) method:
  1. advect (semi-Lagrangian, or 2nd-order upwind/TVD — WP20 recommends one);
  2. diffuse (explicit, or implicit for viscosity);
  3. add buoyancy;
  4. solve the pressure Poisson equation (CG with a multigrid or incomplete
     Cholesky preconditioner; geometric multigrid on the masked grid
     preferred);
  5. project.

  dt is limited by CFL ≤ 0.5 and by diffusion.
- **Turbulence.** A Smagorinsky subgrid model (C_s ≈ 0.1–0.17), switchable.
  Without it, the run is an implicit LES (no explicit subgrid model).
  Requirement F4 accepts that the small eddies are unresolved.
- **Coupling with Phase 1.** `ModelSettings.liquid_transport = "flow"` swaps
  the liquid update. The solver's liquid stage becomes advection plus
  molecular diffusion, while the shell, lid and headspace stages are reused
  unchanged. Energy accounting must still close; advection is conservative on
  the MAC grid with flux-form temperature advection.

## Decision (coordinator, Phase 2 kickoff)

- **The development container has no GPU.** GPU code can be written and checked
  here only with Numba's CUDA simulator (`NUMBA_ENABLE_CUDASIM=1`, slow but
  correct). Real GPU timing must run on the user's Windows PC.
- **Backend: Numba**, with **Numba CPU** first and **Numba CUDA** (`numba-cuda`,
  CUDA 12.x wheels) for the GTX 1060 later. Same language and toolchain as
  Phase 1; the kernels are written point-wise so the CPU `prange` kernels port
  to `@cuda.jit` almost line for line. The other candidates (CuPy, Taichi,
  PyTorch) are dropped: none is needed, and each adds a heavy dependency with
  uncertain Pascal support.
- **Coupling design:** the flow solver does **not** replace the Phase 1 thermal
  solver. Each step it (1) advances the velocity field using the current
  temperatures (buoyancy), then (2) computes the **advective heat flux** across
  every liquid face from that velocity, and returns it as an extra per-cell
  source array (W, summing to zero over the domain). The model adds it to
  `q_cells` and runs the existing `ThermalSolver` with the **molecular**
  conductivity (plus an optional subgrid turbulent conductivity). Shell, lid,
  losses and energy accounting are all reused unchanged, and energy still
  balances to round-off because the advective fluxes cancel in pairs.

## WP20: GPU check script (S)

- **Owns:** `benchmarks/gpu/**`, the optional dependency group `gpu` in
  `pyproject.toml` (`numba-cuda[cu12]`), `docs/phase2-backend.md`.
- **Task:** a script the user runs on Windows, `uv run --extra gpu python
  benchmarks/gpu/gpu_check.py`, that:
  1. reports whether CUDA is usable (device name, compute capability, memory,
     driver and runtime versions) with clear advice if not (driver too old,
     wrong wheel, no NVIDIA GPU);
  2. runs a correctness test of a small float32 3-D masked stencil kernel
     against a NumPy reference;
  3. times, on the GPU and on the CPU (Numba `prange`), a masked 7-point
     Jacobi sweep and a masked CG iteration at 64³, 96³ and 128³;
  4. prints a short, copy-pasteable summary.
- It must also run here under `NUMBA_ENABLE_CUDASIM=1` (tests do that at 16³).
- `docs/phase2-backend.md` records the decision above and how to read the
  script's output.

## WP21: Flow solver core, CPU (XL)

- **Owns:** `src/thermosim/flow/**`, `tests/flow/**`, `docs/physics/flow.md`.
- **Files:** `flow/grid.py` (MAC face masks and index tables built from
  `LiquidGrid`), `flow/kernels.py` (Numba CPU kernels, point-wise style, no
  Python objects inside, float64 for now with a dtype switch), `flow/poisson.py`
  (matrix-free pressure solve), `flow/solver.py`.
- **API:**
  - `FlowState`: `u`, `v`, `w` face velocities (m/s, MAC layout, shapes
    `(nx+1, ny, nz)` etc.), `p` cell pressure (Pa / ρ₀ kinematic is fine;
    document it).
  - `FlowSolver(domain, *, smagorinsky_cs=0.0, dtype=np.float64)`
  - `.initial_state() -> FlowState` (fluid at rest)
  - `.max_dt(state, nu) -> float` (CFL ≤ 0.5 and any explicit-viscosity limit)
  - `.step(state, T_liquid, props, dt) -> FlowStepResult`, where `props` gives
    ρ₀, β, ν, cp at the bulk temperature, and the result holds
    `q_adv` (nx, ny, nz) W per cell (sum ≈ 0 to round-off), optional
    `k_turb` (nx, ny, nz) W/(m K), `max_divergence`, `poisson_iterations`.
  - `slice_velocity(state, j) -> (u_slice, w_slice)` at cell centres, for
    `Snapshot.velocity_u/w`.
- **Numerics:**
  - Boussinesq buoyancy `g β (T − T_bulk)` on w-faces.
  - Momentum advection: semi-Lagrangian (robust, CFL-tolerant) or 2nd-order
    upwind; explicit viscosity (ν + ν_t).
  - Pressure: matrix-free preconditioned CG on the masked cell Laplacian with
    Neumann walls (fix the null space: zero-mean pressure). Jacobi
    preconditioner at minimum; a geometric multigrid preconditioner is a
    stretch goal. Converge to a relative residual of 1e-6 (configurable).
  - Projection makes the discrete divergence of every liquid cell ≤ 1e-8
    relative.
  - Walls: no-slip (face velocity 0 on any face touching non-liquid, plus
    wall-tangential no-slip via ghost values). Free surface: flat, w = 0,
    free-slip for u and v.
  - Heat advection flux: flux form with a TVD limiter (van Leer or minmod),
    ρ₀ cp T u across each open face, so every face flux leaves one cell and
    enters its neighbour.
  - Smagorinsky ν_t = (C_s Δ)² |S| when C_s > 0; k_turb = ρ cp ν_t / Pr_t
    (Pr_t = 0.85).
- **Validation (required tests):**
  1. Divergence after projection ≤ 1e-8 relative on a random field in a
     cylinder mask.
  2. Lid-driven cavity at Re = 100 (thin 3-D box or periodic-y; a test-only
     moving-lid boundary hook is fine) against Ghia et al. (1982) centreline
     u-velocity within 5%.
  3. Differentially heated square cavity at Ra = 10³ and 10⁴ (2-D-like thin
     box, Pr = 0.71) against de Vahl Davis (1983) average Nusselt number
     within 5%. Ra = 10⁵ as a slow test if feasible.
  4. Heat advection conserves energy: Σ q_adv = 0 to 1e-12 relative to its
     absolute sum; a uniform temperature produces no flow.
  5. Rayleigh–Bénard: a box heated from below starts convecting above
     Ra ≈ 1708 and stays still well below it (slow test acceptable).
- **Performance (report only):** ms per step and Poisson iterations at 32³,
  64³ and the 40 L kettle at 20 mm and 10 mm, single and multi-threaded.

## WP22: Integration and visualization (M)

- Model and runner support `liquid_transport = "flow"`. Snapshots carry
  `velocity_u` and `velocity_w` on the slice.
- GUI: an arrow overlay (decimated quiver) and/or streamlines on the slice,
  with a toggle; and a speed colormap option.
- Performance target (to be confirmed after WP20) on the GTX 1060: a 45 sim-min heat-up at
  10 mm cells in under 30 min on the user's GPU.

## WP23: Calibrating the Phase 1 effective conductivity (S)

- Run flow mode over a matrix of scenarios: burner vs element, three fill
  heights, two power levels.
- Fit `K_EFF_C` and the Ra exponent in `k_effective` so that Phase 1's
  time-to-target matches flow mode. Update WP03's defaults, with the
  calibration data stored in `docs/physics/k_eff_calibration.md`.
