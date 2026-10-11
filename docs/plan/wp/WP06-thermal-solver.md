# WP06: Thermal network solver (Numba)

**Wave:** 1 · **Size:** L · **Depends on:** WP00 only. Develop and test
against `testing.synthetic.box_domain` and `uniform_coefficients`.

## Goal

A fast, energy-conserving, explicit finite-volume solver that advances the
liquid, shell, lid and headspace temperatures by one time step, given
coefficients from the physics layer. It contains no physics correlations
(decision D6).

## Owns

- `src/thermosim/solver/**` (`thermal.py`, `kernels.py`, `network.py`)
- `tests/solver/**`

## Interfaces

CONTRACTS §2 (`ThermalState`, `StepCoefficients`, `EnergyFlows`) and §10.

## Discretization (implement exactly; document any deviation in a PR contract-change request)

Notation: C is heat capacity (J/K), G is conductance (W/K), all temperatures
in K.

### Liquid cell p

- **Capacity:** C_p = `rho_c_liquid[p]` · `volume[p]`.
- **Between neighbors p and q:** G = A_face · k_h / dx, where k_h is the
  harmonic mean 2 k_p k_q / (k_p + k_q).
- **To shell node s** (one per coupling row c):
  G_c = A_c / (1/h_film[c] + t_s/(2 k_s)).
  The film coefficient replaces the unresolved boundary layer, so no extra
  half-cell liquid resistance is added. Document this.
- **Free surface** (cell p is in `surface_cells` with area A_s). With lid
  covered fraction f (0 if no lid) and top-layer half-depth δ = V_p/(2 A_s):
  - Open part: G_open = A_s(1−f) / (δ/k_p + 1/h_surface_open), going to T_amb.
    The explicit loss `q_evap_open` · A_s(1−f) is also always removed.
  - Covered part: G_cov = A_s f / (δ/k_p + 1/h_surface_covered), going to
    T_headspace.
- **Source:** `q_cells[p]`.

### Shell node s

- **Capacity:** C_s = `rho_c` · thickness · area.
- **In-plane edges:** `edge_conductance`.
- **Outer side:** G_out = 1 / ( t/(2 k A) + R_ins/A_out + 1/(h_out_shell · A_out) ),
  going to T_amb.
- **Dry inner side:** G_dry = A_dry / (1/h_dry + t/(2k)), going to T_headspace
  if there is a lid, else T_amb. In the no-lid case this loss is counted as
  `loss_dry_wall_open`.
- **Source:** `q_shell[s]`.

### Lid (only if `lid.present`)

- Capacity C_lid.
- **Headspace to lid:** G = h_lid_inner · A_lid.
- **Lid outer:** G = 1/(R_ins/A_lid + 1/(h_lid_outer · A_lid)), going to
  T_amb.

### Headspace (only if `lid.present`)

The headspace is **quasi-steady**, with zero heat capacity, to avoid stiffness.
At each (sub)step, T_hs is the conductance-weighted average of everything
connected to it:
- the covered surface cells;
- the dry walls;
- the lid;
- for a partial lid, an opening conductance G_open,hs = h_lid_inner · (1−f) ·
  πR², going to T_amb.

Energy through the opening is counted as `loss_lid_outer`. Its stored energy
change is 0; `stored_lid_headspace` holds the lid's change only.

### Time integration

Forward Euler for all nodes.

- **`stable_dt`** = 0.9 · min over liquid cells of C_p / ΣG_p. The sum ΣG_p
  includes every conductance touching p (neighbors, couplings, surface).
- **Shell, lid and headspace substeps:** if the caller's `dt` exceeds the
  shell or lid limit (0.9 · min C/ΣG over those nodes), take
  n = ceil(dt / dt_shell) equal substeps.
  - The liquid temperatures are held at their start-of-step values during the
    substeps.
  - Coupling energies are accumulated per liquid cell, then applied to the
    liquid once, at the end of the step.
  - This keeps the scheme conservative to round-off.
- **Liquid update:** T_p ← T_p + dt/C_p · (Σ fluxes + accumulated coupling
  energy/dt + q_cells − evaporation).
- **Energy accounting:**
  - Every flux is computed from the same temperatures used in the update and
    accumulated into `EnergyFlows` by category.
  - Stored energy per node is C·ΔT, using that step's C.
  - Result: `balance_error()` ≤ 1e-9 · total_in per step (round-off only).
- **NaN guard:** check `np.isfinite` on the results every step (it's cheap).
  Raise `SolverError` with the first bad index.
- **`step()` raises `SolverError`** if dt > `stable_dt(coeffs)` · 1.0001.

### Implementation guidance

- **Compact liquid indexing:** work on the N_liq liquid cells only. In
  `__init__`, precompute:
  - `liq_flat` (N_liq,), mapping compact index to flat index, and its inverse;
  - neighbor index `nbr` (N_liq, 6), with −1 for none, and face areas `nbr_A`
    (N_liq, 6);
  - coupling rows grouped by liquid cell (CSR format) and by shell node (CSR).
- **Kernels:** `@njit(cache=True, nogil=True, fastmath=False)` (`nogil` so the GUI thread stays responsive), using `parallel=True` with
  `prange` over liquid cells, and double buffering (read old, write new).
  - Accumulate energy with per-thread partial sums, or a separate serial
    reduction kernel. Parallel reductions must be deterministic within a
    tolerance.
- Shell edges are processed as an edge list in a serial loop. The shell is
  small (under about 20k nodes).
- Recomputing conductances each step is fine. Alternatively, cache the
  geometric parts in `__init__` and multiply by the coefficient arrays.
- **Warm up:** `__init__` calls the kernels once on a tiny array, so the Numba
  compile happens up front (and is cached on disk afterwards).

## Acceptance criteria (tests)

1. **Insulated uniform heating.** In a box with every loss coefficient 0 and
   `q_cells` uniform totaling P: the mean temperature rises at exactly
   P/ΣC per second, to 1e-10 relative.
2. **1D conduction.** A tall single-column box (1×1×200) with k uniform, its
   top held hot through a huge `h_surface_open` and its sides insulated,
   matches the analytic erfc solution for a semi-infinite slab within 1%
   before the heat front reaches the bottom.
3. **Newton cooling.** With very large k (a well-mixed box) and only
   `h_out_shell` on the shell: the mean temperature decays exponentially with
   the time constant computed from the lumped C and G, within 0.5%.
4. **Energy balance.** In a random-coefficient box with all features on (a
   lid, partial cover, sources in both the liquid and the shell, and
   evaporation): `balance_error` ≤ 1e-9 · total_in at every step, over 1000
   steps.
5. **Symmetry.** A symmetric setup gives a symmetric field, to 1e-12.
6. **Stability.** Stepping at `stable_dt` for 10^4 steps stays bounded (no
   new extrema beyond the source and boundary temperatures).
   `step(dt = 1.1 × stable_dt)` raises.
7. **Substeps.** A shell with an aluminum-like high in-plane conductance
   triggers substepping, and the results converge as dt is halved (the error
   shrinks in proportion to dt).
8. **Performance.** A 512k-liquid-cell box with a 20k-node shell takes under
   10 ms per step on a 4-core CI runner (record the timing; on CI it's a
   warning rather than a failure). It is a hard pass under 50 ms.
