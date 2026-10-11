# WP10: Validation suite, lumped reference model, benchmarks

**Wave:** Part A runs in wave 1 (in parallel); Part B runs in wave 2 (after
WP09) · **Size:** M

## Goal

Independent evidence that the simulator is right and fast enough: analytic
references, a simple single-temperature ("lumped") model for cross-checking,
grid-convergence studies, performance benchmarks, and a procedure for checking
against a real heat-up.

## Owns

- `src/thermosim/validation/**`
- `tests/validation/**`
- `benchmarks/**` (including `benchmarks/RESULTS.md`)
- `docs/validation.md`

## Part A (wave 1, needs only WP00)

1. **`validation/analytic.py`:** reference solutions as pure functions with
   their sources cited:
   - semi-infinite slab with a step surface temperature (erfc);
   - slab with convection at its surface (Carslaw & Jaeger);
   - a lumped body under Newton cooling with a constant source:
     T(t) = T∞ + P/G + (T0 − T∞ − P/G)·e^(−Gt/C);
   - 3D diffusion of a Gaussian pulse in an infinite medium.
2. **`validation/lumped.py`:** `LumpedModel(scenario, props, correlations)`.
   - One liquid temperature and one shell temperature.
   - It uses the **same** WP02 and WP03 functions, with the surface areas
     computed analytically from the cylinder: base πR², wetted side 2πRh, dry
     side 2πR(H_v − h), free surface πR², and the lid.
   - It integrates with `scipy.integrate.solve_ivp` (RK45, rtol 1e-8) to the
     target.
   - It returns the time-to-target and the time series.

   It's useful in its own right: it gives a quick estimate within seconds, and
   WP11 may show it as a "quick estimate". Until WP02 and WP03 merge, test it
   with `ConstantProps` and fixed h values.
3. **Tests for Part A:** the analytic functions satisfy their PDEs (checked by
   finite differences) and their limiting cases. The lumped model matches the
   analytic Newton cooling result to 1e-6.

## Part B (wave 2, after WP09)

4. **Solver against analytic solutions** (`tests/validation/test_solver_vs_analytic.py`),
   using `testing.synthetic` and the real solver. It overlaps with WP06's own
   tests on purpose: an independent author checks them.
5. **Full model against the lumped model:** for each example scenario with
   k_eff_scale = 100 (forcing a well-mixed liquid), the 3D time-to-target is
   within 3% of the lumped model's. Report the error for k_eff_scale = 1 as
   information.
6. **Grid convergence:** the 40 L propane example at dx = 0.02, 0.01 and
   0.005. Time-to-target must converge monotonically. Report the observed
   order and the Richardson-extrapolated value. The result at 0.005 must be
   within 2% of the extrapolated value. Mark this test `@pytest.mark.slow`
   and exclude it from default CI (run it weekly via
   `workflow_dispatch`/schedule; ask the coordinator to add the job).
7. **Energy conservation:** every example ends with
   `energy_balance_rel_error` < 1e-6.
8. **Sanity bounds:** a check against public rules of thumb, documented in the
   test. For example, 40 L from 15 to 99 °C on a 60k BTU/hr burner at 35%
   efficiency should take about 30–60 minutes, given ideal energy
   m·cp·ΔT ≈ 14 MJ at ≈ 6.2 kW delivered ≈ 38 minutes plus losses. Flag the
   result if it falls outside the band.
9. **Benchmarks** (`benchmarks/bench_*.py`, plain scripts, using
   `time.perf_counter`):
   - solver step time vs. cell count (64³, 96³, 128³);
   - coefficient refresh cost;
   - full runs of the examples at preview (10 mm) and standard (5 mm) sizes.

   Write `benchmarks/RESULTS.md` with the machine description. It must show
   requirement R2 being met, or say clearly by how much it misses.
10. **`docs/validation.md`:** how to check against a real heat-up.
    - **Equipment:** a thermometer or logger, a scale or measuring jug, notes
      on the weather. Where to put the probe, matching a scenario probe. Log
      at least every 30 s.
    - **Data:** CSV format examples that `load_measured_csv` accepts.
    - **Procedure:** run `thermosim compare RUN_DIR data.csv`. Interpret the
      bias and the time-to-target error. Tune burner `efficiency` first (it's
      the most uncertain input), then `k_eff_scale`.
    - **Results log:** a table to record each real check.

## Acceptance criteria

- Part A tests run in default CI and pass.
- Part B tests pass, except `slow`. `RESULTS.md` is populated.
- `docs/validation.md` exists and is clear enough for the user to follow
  without help.

## Known issue to investigate (found during wave 1)

The solver's parallel Numba kernels slow down catastrophically when the CPU is
oversubscribed. On a 4-core machine at load average about 18,
`test_newton_cooling_time_constant` ran for more than 30 minutes. The same test
takes 2.8 s with `NUMBA_NUM_THREADS=1`.

Investigate and fix, coordinating any `solver/` change through the coordinator:
- try the threading layers (`omp`, `tbb`, `workqueue`);
- use serial kernels below a cell-count threshold;
- cap the thread count.

Add a benchmark that runs while a CPU-burner process runs alongside. A user
running the GUI on Windows while other programs are busy must not hit this.
