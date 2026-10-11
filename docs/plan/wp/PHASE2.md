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

## WP20: Backend spike (S–M)

- **Goal:** choose the compute backend for the flow solver on the user's
  hardware.
- **Candidates** (target: the GTX 1060 on Windows):
  - Numba CPU (the baseline and required fallback);
  - Numba CUDA (custom kernels in the same language as Phase 1);
  - CuPy (raw kernels plus array operations);
  - Taichi (check that the project is still maintained and supports Pascal);
  - PyTorch (check the newest build that still supports compute capability
    6.1).
- **Task:**
  - Implement the same minimal 3D masked-grid pressure Poisson solve (a
    cylinder mask, 64³ and 128³) and one semi-Lagrangian advection step in
    each viable candidate.
  - Measure time per step, the effort to write custom kernels, install
    friction on Windows, and determinism.
- **Deliverable:** `docs/phase2-backend.md` with a recommendation, and the
  spike code in `spikes/phase2/` (not shipped).

## WP21: Flow solver core (XL)

- **Owns:** `src/thermosim/flow/**`, `tests/flow/**`.
- **Interface:** `FlowSolver(domain).step(state: FlowState, coeffs, dt) -> EnergyFlows`.
  It uses the same `ThermalState` plus velocity and pressure arrays.
- **Validation (required):**
  1. Lid-driven cavity at Re = 100 and 1000 against Ghia et al. (1982)
     centerline profiles (2D or a thin-3D test).
  2. Differentially heated square cavity at Ra = 10³–10⁶ against
     de Vahl Davis (1983) Nusselt numbers, within 2–5%.
  3. Rayleigh–Bénard onset near Ra_c ≈ 1708 between rigid plates.
  4. Divergence after projection below a tolerance (1e-8 relative).
  5. Energy conservation as in Phase 1.

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
