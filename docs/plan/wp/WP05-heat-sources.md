# WP05: Heat sources, actuators, power commands

**Wave:** 1 · **Size:** M · **Depends on:** WP00. Test with
`testing.synthetic.box_domain`, plus a hand-built cylinder-like `Domain` if
needed.

## Goal

Model how a requested power fraction becomes heat delivered into the domain:
command sources (a schedule, or manual input), actuators (SSR on/off, variable,
gas valve), and the spatial heat deposition of burners and elements.

## Owns

- `src/thermosim/sources/**` (`heat_sources.py`, `actuators.py`, `commands.py`,
  `build.py`)
- `tests/sources/**`

## Interfaces

CONTRACTS §4 (`Actuator`, `HeatSource`, `SourcePower`, `CommandSource`) and §9.

## Requirements

### Actuators

- **`VariableActuator`:** output = clip(command, 0, 1).
- **`OnOffActuator(cycle_period_s)`:** SSR time-proportioning.
  - Within each cycle [nT, (n+1)T), the output is ON for the first
    `command·T` seconds. The command is latched at the start of each cycle.
  - `output(command, t, dt)` returns the **exact time-average** of the on/off
    signal over [t, t+dt], including when dt spans several cycles or a partial
    one.
  - Test: integrated energy over 100 cycles = command × time, to 1e-12.
- **`GasValveActuator(min_fraction, response_time_s)`:**
  - A command below `min_fraction/2` is OFF (0).
  - Between `min_fraction/2` and `min_fraction`, the output is clamped to
    `min_fraction`. This is a burner's turndown limit.
  - The actual output follows the target with a first-order lag τ =
    `response_time_s`.
  - Return the exact average over dt of the analytic exponential solution.
  - `reset()` sets the output to 0.

### Command sources

- **`ScheduleCommand(steps)`:** piecewise-constant; the value at t is the last
  step with `t_s ≤ t`.
- **`ManualCommand(initial)`:** `set()` and `command()` are thread-safe (use a
  `threading.Lock`). Values are clipped to [0, 1].

### Heat sources

- **`BurnerSource`:**
  - gross_W = rated_power_W × actuator output.
  - delivered_W = efficiency × gross_W.
  - Deposition: over base shell nodes (`kind == BASE`). Each node's weight is
    the overlap of its column footprint with the burner's disk or ring
    (computed by sub-sampling the cell area, 8×8 points), normalized to sum
    to 1.
  - Raise `GeometryError` if the footprint overlaps no base node.
  - `fuel_kg(gross_J)` = gross_J / `PROPANE_LHV_J_PER_KG`.
- **`ElementSource`:**
  - gross_W = delivered_W = rated × output (all of the element's heat goes
    into the liquid).
  - Deposition: over liquid cells. Sample the polyline path at a spacing of
    ≤ dx/4. Each sample carries an equal share of length. Add each sample's
    length to the cell containing it, then normalize.
  - Raise `GeometryError(f"element segment at z={z:.3f} m is above the liquid
    (fill {h:.3f} m)")` if any sample is in a non-liquid cell or above the
    fill height. That covers dry-firing.
  - `electric_kwh(gross_J)` = gross_J / 3.6e6.
- **`power(command, t, dt)`** returns `SourcePower`. Delivered energy over a
  step is `delivered_W × dt`.
- **`build_heat_source(scenario, domain)`** dispatches on
  `heat_source.type`. `build_actuator(cfg)` dispatches on `actuator.type`.

## Acceptance criteria

- Deposition weights sum to 1 (to 1e-12) and are non-negative.
- A burner disk of diameter D on a large base deposits only within radius
  D/2 + dx·√2. A ring leaves its center empty.
- An element along the x-axis at z = 0.05 deposits only in cells with
  k = floor(0.05/dx). Per-cell weights are proportional to path length (test
  a 45° path).
- The SSR exact-average test, the gas valve step response (63.2% at τ, to
  1e-6) and the turndown behavior all pass.
- A `ManualCommand` stress test with 2 threads shows no exceptions and values
  always in [0, 1].
