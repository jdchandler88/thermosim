# Benchmark results (WP10)

Scripts in this directory (plain scripts, `uv run python benchmarks/<name>.py --help`):

| Script | What it measures |
|---|---|
| `bench_solver_step.py` | solver step time against the number of liquid cells (64^3, 96^3, 128^3), per thread count |
| `bench_coefficients.py` | cost of one coefficient refresh (`build_coefficients` + `stable_dt`) at 10 and 5 mm |
| `bench_examples.py` | full runs of the three examples at 10 and 5 mm; `--r2` is the requirement R2 case |
| `bench_oversubscription.py` | step time (or any command, such as a pytest run) next to busy-loop processes |
| `validation_report.py` | the tables of `docs/validation.md` section 6 |

## Machine

```
Linux 6.18.44-fc-v114 (Firecracker microVM), x86_64
Intel Xeon (Sapphire Rapids) @ 2.10 GHz, 4 vCPUs (1 thread per core, 4 cores), L2 8 MiB
(2 MiB per core), L3 260 MiB (shared with the host), 16 GB
Python 3.12.3, numpy 2.5.4, numba 0.68.0 (threading layer: omp / libgomp; tbb is not installed)
load average while measuring: 0.5-4 (the benchmarks themselves put 4 on it); other work on the
host is not visible from inside the VM
```

**The timings of this machine are noisy.** It is a shared VM: the same binary ran the R2 case in
329 s early in the session and in 455-535 s later in the day, and single solver steps of one
run vary between 0.85 and 3.4 ms. Numbers below were taken back to back where they are compared
(before/after on the same line of the same table); absolute values should be read as +-30%.
"Before" is the tree at the start of this work package (commit `24464d5`), run from an exported
copy with the same scripts.

## Requirement R2: 50 L kettle, 5 mm cells, 45 simulated minutes, under 5 minutes on 4 cores

`bench_examples.py --r2`: the propane kettle with 0.05 m^3 of water (402k liquid cells, 29k shell
nodes), `max_sim_time_s = 2700`, no GUI.

| | Wall time | Steps | Verdict |
|---|---|---|---|
| Before (WP09, as reported by the coordinator) | 370 s | | not met |
| Before (this machine, same session as "after") | 535 s | 110k | not met |
| After, run at the end of the work package | **455 s** | 116k | **not met** |
| After, earlier in the session when the machine was fast (intermediate version of the code) | 329 s | 110k | not met (10% over) |

**R2 is not met on this machine.** After the work, a typical run takes 455 s (50% over the
limit); the best I saw was 329 s (10% over). The speed-up over the start of the work package is
15-30%, measured back to back. What is left is dominated by the explicit time step, not by
overhead: 116,000 steps of about 23 ms (the limit is `C dx^2 / (6 k_eff)` with `k_eff` about 500
W/(m K) from the mixing closure), each costing about 3.9 ms on this machine (including its
share of the refresh), of which the liquid
kernel is about 2.7 ms (memory-bound; the best steps take 0.85 ms), the shell and surface
kernels 0.6 ms, and the coefficient refresh and bookkeeping about 0.6 ms (14%). Closing the gap
needs a different idea, not more tuning (see the end).

## Solver step time (`bench_solver_step.py`)

Box of `n^3` cubic cells of 5 mm, uniform conductivity, losses on, median of 30 steps (ms).

| Liquid cells | Before, 4 threads | After, 4 threads | After, 2 threads | After, 1 thread | Before, 1 thread | After, auto |
|---|---|---|---|---|---|---|
| 64^3 = 262k | 2.44 | 1.62 | 1.99 | 2.96 | 6.77 | 2.08 |
| 96^3 = 885k | 7.02 | 4.67 | 5.73 | 9.09 | 22.7 | 4.18 |
| 128^3 = 2.1M | 17.1 | 10.6 | 15.8 | 24.2 | 55.1 | 13.4 |

Single-thread speed is 2.3x better (interior fast path, no copy pass, conductances rebuilt only
on refresh); with four threads 1.5-1.7x. `auto` is the default: the thread tuner sometimes picks
fewer threads than the best fixed choice when the measurements are noisy, which costs up to 30%
here.

## Coefficient refresh (`bench_coefficients.py`)

Propane kettle, 50 L, state at t = 120 s (the base is boiling).

| dx | Liquid cells | Shell nodes | `build_coefficients`, before | after | `stable_dt`, before | after |
|---|---|---|---|---|---|---|
| 10 mm | 51k | 7.3k | 5.5 ms | 4.4 ms | 0.6 ms | 0.6 ms |
| 5 mm | 402k | 29k | 19.3 ms | 16.0 ms | 3.1 ms | 3.3 ms |

The refresh takes 14% of the R2 run (before: 13%, but with a cheaper film model); WP09 asked for
under 10%. At 10 mm it takes about half of the run time, because there are few, cheap steps and
2,700 refreshes. The refresh is a few hundred NumPy calls on arrays of the size of the shell;
the next saving would come from refreshing the film coefficient less often than the rest.

## The examples before and after fix A (`bench_examples.py`)

"Before" is the tree at `24464d5`, "after" the tree at the end of the work package. Base shell:
hottest base shell node during the run (the shell node is the mid-plane of the wall, so it
includes the conduction drop across the wall). Near wall: highest reading of the `near_wall`
probe (liquid, 15 mm inside the wall, 10 cm up).

| Example | dx | Base shell max (C) before / after | Near-wall max (C) before / after | Time to target before / after | Wall time before / after |
|---|---|---|---|---|---|
| `kettle_propane_40L` | 10 mm | 2458 / 159 | 106 / 106 | 48.3 / 45.2 min | 33.7 / 30.4 s |
| `kettle_electric_insulated` | 10 mm | 96 / 96 | no probe | 23.3 / 23.3 min | 16.2 / 13.3 s |
| `small_pot_stove` | 10 mm | 130 / 108 | 91 / 90 | 21.1 / 21.0 min | 4.1 / 4.0 s |
| `kettle_propane_40L` | 5 mm | 2486 / 160 | 106 / 106 | 48.5 / 45.2 min | 339 / 298 s |
| `kettle_electric_insulated` | 5 mm | 95 / 95 | no probe | 23.3 / 23.3 min | 120 / 105 s |
| `small_pot_stove` | 5 mm | 131 / 108 | 91 / 90 | 21.1 / 21.0 min | 18.0 / 17.3 s |

All energy balances are below 5e-14.

What the boiling model does and does not do:

* The propane kettle's base no longer runs at 2000+ C. The wetted surface sits about 10 K above
  the boiling point (Rohsenow, about 200 kW/m^2 needs 11 K) and the shell node 25-50 K above that
  because of the conduction drop through the 4 mm base. The k_eff hot side is capped at
  T_sat + 30 K, so `k_liquid` can no longer be driven by a runaway wall. (Its maximum is about
  424 W/(m K) in both versions: it is the Rayleigh scaling at a 100 K difference.)
* The time to target of the propane kettle is 7% shorter than before. I did not investigate it
  further; the old base shell sat 2000 C above the liquid, so the heat took a different path
  (more of it through the side wall, whose film coefficient was not limited the same way).
* **Liquid cells still exceed the boiling point.** The `near_wall` probe reads up to 106 C (the
  same as before), at the end of the run when the mean is 99 C, and the hottest liquid cells
  (next to the burner footprint) reach 139 C at 10 mm (before: 128 C; probably because the
  2000 C base spread the heat sideways before it entered the liquid). This is the effective-conductivity closure, not the wall: with
  `k_eff` about 400 W/(m K) a flux of 200 kW/m^2 needs a gradient of 500 K/m, i.e. 5 K per
  10 mm cell, and the liquid is not allowed to boil (no phase-change model, requirement L4), so
  the bottom layer superheats instead of mixing violently as real boiling would. The closure is
  to be calibrated with `k_eff_scale` (WP23).
* The immersed element of `kettle_electric_insulated` has no wall: its cells reach 125 C for
  the same reason; no boiling coefficient applies to them.

## Oversubscription (the "known issue")

`bench_oversubscription.py`: milliseconds per solver step of the propane kettle (50 L) while 16
busy-loop processes run on the 4 cores (load average about 20), each configuration in a fresh
interpreter. "Before" runs the tree at `24464d5`.

| dx | Config | Idle | 16 burners | Slowdown |
|---|---|---|---|---|
| 5 mm (402k cells) | before, default (4 threads, spinning OpenMP) | 6.51 | 67.2 | 10x |
| | before, `NUMBA_NUM_THREADS=1` | 12.7 | 60.0 | 4.7x |
| | **after, default (tuned, passive waits)** | 4.59 | 19.3 | 4.2x |
| | after, `THERMOSIM_THREADS=1` | 5.24 | 29.5 | 5.6x |
| | after, `THERMOSIM_THREADS=4` | 5.12 | 15.2 | 3.0x |
| 20 mm (6.4k cells) | before, default | 1.40 | 73.6 | **53x** |
| | before, `NUMBA_NUM_THREADS=1` | 1.51 | 6.65 | 4.4x |
| | **after, default (serial below 40k cells)** | 1.43 | 7.42 | 5.2x |
| | after, `THERMOSIM_THREADS=4` | 1.69 | 3.36 | 2.0x |

The WP06 test that took more than 30 minutes at load 18
(`tests/solver/test_solver_thermal.py::test_newton_cooling_time_constant`):

| | Idle | 16 burners |
|---|---|---|
| Before | 4.5 s | **timeout after 240 s** (more than 30 min in the original report) |
| After | 5.7 s | 25.6 s |

Findings:

1. **The cause is the spinning wait of the OpenMP threads, together with the barrier at the end of
   every parallel region.** The solver does two to four parallel regions per step of a few
   hundred microseconds each. With the default (active) wait policy a thread that arrives early
   spins, so with more runnable threads than cores the threads that have work are descheduled
   while the others burn their time slices; each region then costs scheduler time slices
   (milliseconds) instead of microseconds. The effect is worst for small grids, where the work
   per region is smallest (53x at 6.4k cells, 10x at 400k cells).
2. **Fixes in the code (all in `solver/`):**
   * `OMP_WAIT_POLICY=PASSIVE` is set (unless the user has set it) before the first parallel
     launch. On the idle machine it costs 0-10%; under load it turns 10-50x slowdowns into 3-5x.
   * grids below 40,000 liquid cells never use the threading layer (a serial variant of every
     kernel);
   * for larger grids the solver times real steps with 1, half and all threads for four steps
     each every 8 s and keeps the fastest, so a loaded machine falls back to fewer threads
     (and an idle one goes back to all). `THERMOSIM_THREADS=n` or `ThermalSolver(domain,
     threads=n)` fixes the count.
   * the thread limit is the size of the pool (`NUMBA_NUM_THREADS`), not the calling thread's
     current setting, which `numba.set_num_threads` leaves behind (this would otherwise let one
     solver lower the limit of the next).
3. **Threading layers.** Only `omp` (libgomp) and `workqueue` are available here (`tbb` needs
   the `tbb` package). `workqueue` was about 2x better than active-wait `omp` under load (28 vs
   71 ms/step at 5 mm, measured before the other fixes) but 60% slower when idle (9.5 vs 5.9 ms), and it is not safe when several Python threads
   launch parallel kernels (the GUI runs the model in a worker thread), so it is not used.
   Passive `omp` was as good as `workqueue` under load (31 ms/step) and within 15% of active
   `omp` when idle.
   On this VM active spinning also slows the main thread's own Python and NumPy work (the
   vCPUs appear to share resources), so passive waits are the better default here.
4. Running other work next to the solver gives it a smaller share of the CPU whatever the thread
   count; more threads get a larger share under CFS (4 threads were faster than 1 even at 20 mm
   with 16 competitors: 3.4 vs 5.6 ms). The tuner takes this into account because it measures.

## Test suite

`QT_QPA_PLATFORM=offscreen uv run python -m pytest -q` (the `slow` tests are skipped by default,
run them with `-m slow`):

| | Time |
|---|---|
| default threading | 534 passed, 2 skipped in **160 s** |
| `NUMBA_NUM_THREADS=1` | 534 passed, 2 skipped in 161 s |
| default threading, 4 busy-loop processes next to it | passed in 196 s |
| `-m slow` (grid convergence 20/10/5 mm, 512k-cell step time) | about 7 minutes |

The suite no longer needs `NUMBA_NUM_THREADS=1`. One test, which compared the time to target of
the headless run and the threaded runner to a relative 1e-6, failed under load: the two runs split
the simulation into chunks of different length (the chunk length follows the wall-clock speed),
and a chunk end shortens the step that crosses it, so the explicit-Euler trajectories differ at
the 2e-6 level. The tolerance is now 1e-4 (`tests/sim/test_sim_runner.py`).

## What would be needed to meet R2

The remaining cost is the number of explicit steps (116k) times a memory-bound step. Options, in
the order I would try them (all but the first change numerics or the scenario, so they need the
coordinator's decision):

1. Super-time-stepping (Runge-Kutta-Legendre) or an implicit/ADI diffusion solve for the liquid
   when `k_eff` is large: the step is limited by the mixing closure's diffusivity, not by
   accuracy. A 5-10x larger step keeps the scheme conservative.
2. A coarser grid in the bulk (the temperature field is smooth away from the walls and the
   base), with 5 mm cells only near the shell.
3. A larger `coeff_refresh_s` for the film coefficients (saves about 10% of the run).
4. Temporal blocking of the interior kernel (two steps per sweep); a complicated change for
   perhaps 30% of the kernel.
