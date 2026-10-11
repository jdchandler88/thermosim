# WP07: Recording, snapshots, summary, CSV, measured-data comparison

**Wave:** 1 · **Size:** M · **Depends on:** WP00. It uses
`config.units.to_display` for US CSV export; until WP01 lands, guard that path
or use a local copy of the formulas in tests only.

## Goal

Capture time series and slice snapshots efficiently; compute the summary
(including time-to-target); export, save and load runs; and compare a run
against measured data from a real heat-up.

## Owns

- `src/thermosim/recording/**`
- `tests/recording/**`

## Interfaces

CONTRACTS §2 (`Sample`, `Snapshot`, `Summary`, `RunResult`) and §11.

## Requirements

1. **`Recorder`:** append-only.
   - `to_arrays()` returns columns: `t`, `T_mean`, `T_min`, `T_max`,
     `T_shell_mean`, `command`, `power_delivered`, `power_gross`, `fuel_kg`,
     `electric_kwh`, `probe_<name>` for each probe, and `energy_<field>` for
     each `EnergyFlows` field (cumulative J).
   - Appending must be O(1) amortized. Grow internal NumPy buffers
     geometrically, and build the `Sample` objects lazily if that's faster;
     the API stays the same.
2. **`SnapshotStore`:**
   - Holds snapshots in memory. If `max_in_memory` is exceeded and `spill_dir`
     is set, it writes the oldest to `.npz` chunk files and loads them back on
     access (an LRU cache of the last 64).
   - `nearest(t)` uses binary search.
   - `save` and `load` use a single compressed `.npz` file. `load(save(x))`
     equals x.
   - Memory budget: a 45-minute run at 5 s intervals on an 80×80 slice is
     about 540 snapshots × about 26 KB, roughly 15 MB. Fine in memory by
     default.
3. **`summarize`:**
   - `time_to_target_s` is the first time the target series (a probe or the
     mean) is ≥ the target, **linearly interpolated** between samples. It is
     None if never reached.
   - Energy totals come from the last sample's `cumulative`.
     `heating_efficiency` = stored_liquid / gross energy input. The gross
     energy is the integral of `power_gross`, computed by the trapezoid rule
     over the samples.
   - `energy_balance_rel_error` is as defined in CONTRACTS.
4. **`export_csv`:**
   - A header row with units, e.g. `t [s]`, `T_mean [°C]` (metric) or
     `[°F]` (US). Time is in seconds for both. Energy is in kJ (metric) or BTU
     (US).
   - UTF-8 with BOM, so Excel on Windows shows ° correctly.
   - Lines end in `\n`, via `newline=""` in the csv module.
5. **`save_run` / `load_run`:**
   - Writes a directory containing `summary.json` (with enums as strings),
     `timeseries.csv` (metric), `samples.npz` (exact, for reloading),
     `snapshots.npz` and `scenario.toml` (`RunResult.scenario_toml`).
   - Round-trips exactly.
6. **`load_measured_csv`:**
   - Auto-detects the delimiter (`,` `;` or tab) and decimal commas.
   - When the columns aren't given, auto-detects them: a time column named
     like `t`, `time` or `seconds`/`minutes`, and a temperature column named
     like `temp` or `T`, or else the first two numeric columns.
   - Accepts time as seconds, minutes, or `HH:MM:SS` clock times (converted to
     elapsed seconds from the first row).
   - Drops NaN rows, sorts by time, and converts to K.
   - Errors name the file and the line.
7. **`compare`:**
   - Interpolates the simulation onto the measured times, inside the
     overlapping range only.
   - Computes RMSE, maximum absolute error and bias (mean of sim − measured).
   - If a target is given, computes time-to-target for both, by the same
     interpolation, and the relative error.

## Acceptance criteria

- `time_to_target` interpolation is exact for a linear ramp.
- CSV export opens with the right columns and units, and is re-read in the
  test.
- `load_measured_csv` handles fixtures for: a comma CSV in seconds, a
  semicolon CSV with decimal commas in minutes, a clock-time CSV in °F, and an
  Inkbird/Tilt-style export (create a representative fixture and document its
  assumed format).
- `save_run`/`load_run` round-trip a synthetic `RunResult`.
- Appending 1e5 samples takes under 1 s.
