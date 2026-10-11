# WP09: Model assembly, runner, CLI

**Wave:** 2 · **Size:** L · **Depends on:** WP01–WP07 merged.

## Goal

Wire everything into a working simulator: build the domain, solver and heat
source from a scenario; run the time loop with coefficient refreshes, energy
accounting, recording and stop criteria; provide the threaded `Runner` used by
the GUI (fast and real-time modes); and provide the `thermosim` command-line
tool.

## Owns

- `src/thermosim/sim/**` (`model.py`, `coefficients.py`, `runner.py`,
  `pacing.py`)
- `src/thermosim/cli.py`
- `tests/sim/**`, `tests/cli/**`

## Interfaces

CONTRACTS §4 (`Runner`, the events) and §12.

## Requirements

### `coefficients.py`: `build_coefficients(domain, state, scenario, source_output, cache) -> StepCoefficients`

All physics comes from WP02 and WP03. p = `pressure_at_altitude(altitude)`.
SG comes from the scenario.

| Field | Formula |
|---|---|
| `rho_c_liquid` | `wort_density(T_liquid, sg) * wort_specific_heat(T_liquid, sg)` per cell |
| `k_liquid` | Uniform: `k_effective(T_bulk=T_mean, T_hot=T_src, depth=fill_height, sg, scale=k_eff_scale)`. T_src is the area-weighted mean temperature of the shell nodes with burner weight > 0 (burner), or the weight-averaged temperature of the element cells (element). If the source is off, use the hottest wetted shell mean or T_mean, whichever is larger. |
| `h_film` | `h_film_liquid(T_shell[couple_shell], T_liquid[couple_cell], orient, L, sg)`, where orient = UP for BASE nodes and SIDE for SIDE nodes; L = D/4 (base) or fill height (side) |
| `h_out_shell` | `h_outer(T_s, T_amb, outer_face, char_length, wind, p, emissivity)`. T_s is the outer surface temperature: T_shell if uninsulated, otherwise one fixed-point estimate T_amb + (T_shell − T_amb)·R_out/(R_ins + R_out) using the previous h. For BASE nodes on a **burner** setup, set it to 0 while burner output > 0, because burner efficiency already accounts for the losses around the flame. Document this. |
| `h_dry` | With a lid: an enclosed-cavity natural convection estimate (the same helper as `h_headspace`). Without: `h_natural_vertical(T_shell, T_amb, H_v − h, p) + 0.5 * h_radiation(...)` (the inner wall sees the sky only partly). |
| `h_surface_open` | `h_outer(T_surf, T_amb, UP, D/4, wind, p, 0.95)` |
| `q_evap_open` | `q_evaporation(T_surf, T_amb, rh, p, h_conv)`, where h_conv is the convective part only (no radiation) |
| `h_surface_covered`, `h_lid_inner` | `h_headspace(T_surf, T_lid, sg, p)` |
| `h_lid_outer` | `h_outer(T_lid, T_amb, UP, D/4, wind, p, lid ε)` |
| `q_cells`, `q_shell` | `delivered_W * deposition.cell_weights` / `.shell_weights`, **recomputed every step** |

The coefficients other than the sources are refreshed every
`simulation.coeff_refresh_s` of simulated time. Profile this; the refresh must
take under 10% of the run time at default settings.

### `model.py`: `Model`

- **Construction:**
  - `domain = build_domain(scenario)`; `solver = ThermalSolver(domain)`;
    `source = build_heat_source(...)`.
  - Initial state: the liquid starts at `initial_temp_K`. The shell, lid and
    headspace start at `ambient_temp_K`.

    **[Proposed default — document it in the scenario docs]** the vessel
    starts at air temperature. Add an optional
    `scenario.vessel.initial_temp_K` only if the coordinator approves a
    contract change.
- **Time step:**
  - dt = min(`solver.stable_dt(coeffs)`, the time to the next sample,
    snapshot or coefficient refresh, the time remaining to max time).
  - Clamp dt to at most 1.0 s, so actuators and recording stay smooth.
- **Each step:**
  1. Get the command from the `CommandSource` (the schedule, manual input, or
     later PID; pass the target probe's temperature).
  2. `sp = source.power(cmd, t, dt)`.
  3. Fill the sources into the coefficients.
  4. Call `solver.step`.
  5. Accumulate `EnergyFlows`, gross energy, fuel and kWh.
- **Samples** are taken every `sample_interval_s`. **Snapshots** are taken
  every `snapshot_interval_s`, plus at t = 0 and at the end, via
  `geometry.extract_slice` plus the wall, base and lid temperatures.
- **Stop criteria:**
  - **TARGET_REACHED:** the target series is ≥ the target. Take one final
    sample and snapshot.
  - **PLATEAU:** after at least `plateau_window_s`, the target series rose by
    less than `plateau_rate_K_per_min` × window/60 over the last window
    **and** the mean command over that window is > 0.
  - **MAX_TIME.**
  - **USER_STOP** (from the runner).
  - **ERROR:** a `SolverError` is caught; the runner emits an `ErrorEvent` and
    the summary has `stop_reason` ERROR.
- `advance(until_t)` returns the `Sample`s and `Snapshot`s produced, in time
  order.

### `runner.py`: `SimulationRunner`

- A worker `threading.Thread` (daemon). Numba kernels release the GIL where
  possible (WP06 uses `nogil=True`; if not, request it via the coordinator).
- Events travel through a `queue.SimpleQueue`. `poll_events` drains up to
  `max_events` without blocking.
- **FAST mode:** calls `advance()` in chunks of about 50 ms of wall time,
  checking the pause and stop flags between chunks. It emits a
  `ProgressEvent` about every 100 ms of wall time.
- **REALTIME mode** (`pacing.py`):
  - The target sim time = t_sim0 + ∫ speed d(wall), integrated piecewise so
    that speed changes and pauses are exact.
  - The loop advances to the target and sleeps the remainder of a 20 ms tick.
  - If the simulation can't keep up, run flat out and report
    `realtime_lag_s` > 0.
  - Power commands go through `ManualCommand`.
- `set_power_command` and `set_speed` raise `ModeError` in FAST mode.
- **State transitions:** IDLE → RUNNING ⇄ PAUSED → FINISHED | STOPPED |
  FAILED. Calls that aren't valid for the current state are no-ops with a
  logged warning.
- **`result`** is available after a `FinishedEvent`. It has a `Summary` (via
  `recording.summarize`), the samples, a `SnapshotStore` and the scenario TOML.
- `run_headless(scenario)` runs in FAST mode, synchronously, without a thread.

### `cli.py`: `main(argv=None) -> int`

| Command | Behavior |
|---|---|
| `thermosim run SCENARIO.toml [--out DIR] [--cell-size M] [--max-time S] [--quiet]` | Headless run. Prints a progress bar (simple `\r` text, no extra dependency) and then the summary table. Writes `save_run` output to `--out`, which defaults to `runs/<scenario-name>-<timestamp>/`. |
| `thermosim validate SCENARIO.toml` | Loads and validates the scenario, and builds the domain (catching geometry errors). Prints the cell count, estimated memory, and estimated run time from a quick 50-step benchmark. |
| `thermosim examples [--write DIR]` | Lists the example scenarios, or writes them out. |
| `thermosim compare RUN_DIR MEASURED.csv [--probe NAME] [--temp-unit C\|F] [--time-unit s\|min]` | Prints the comparison metrics. |
| `thermosim gui [SCENARIO.toml]` | Imports `thermosim.gui.app` lazily and launches it. |

Exit codes: 0 success; 1 target not reached; 2 invalid scenario; 3 solver
error.

## Acceptance criteria

1. **End-to-end smoke test:** every example scenario at dx = 0.02 runs to
   TARGET_REACHED or PLATEAU in under 30 s on CI, with
   `energy_balance_rel_error` < 1e-6.
2. **Lumped agreement:** for the 40 L propane example with k_eff_scale = 50
   (well mixed) and wind at 0, the time-to-target is within 3% of WP10's
   lumped model given the same losses. (If WP10 isn't merged yet, compare
   against a local hand calculation: m·cp·ΔT / P_delivered, with losses
   disabled by setting a flag through a test-only hook.)
3. **Pacing:** in REALTIME mode at speed 10×, simulating 5 s takes 0.5 s
   ±20%. `set_speed(20)` mid-run takes effect. Pause stops the simulated time
   from advancing.
4. **Thread safety:** start, pause, set_power, resume and stop in quick
   succession must not deadlock (test under a 10 s timeout).
5. **CLI:** `run` on an example writes all 5 files. `validate` on a broken
   scenario exits 2 with a readable message. `--help` works for every
   subcommand.
6. **Performance (benchmark, not CI-gating):** the 50 L kettle at dx = 0.005,
   45 sim-min, finishes in under 5 min on a 4-core machine. Record the result
   in `benchmarks/RESULTS.md` (owned by WP10; send the numbers to WP10, or
   append via the coordinator).
