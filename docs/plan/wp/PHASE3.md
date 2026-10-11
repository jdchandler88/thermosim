# Phase 3: PID controller in the loop

**Starts after Phase 2.** Scope is **decided**: the controller heats to a
single setpoint and holds it. There are no mash step schedules or setpoint
schedules.

## WP30: Controller model (M)

- **Owns:** `src/thermosim/control/**`, `tests/control/**`, and the
  config-schema additions made through the coordinator (`Scenario.controller`).
- **`PIDController(CommandSource)`:**
  - Gains Kp, Ki, Kd; sample period; derivative on measurement; a first-order
    filter on the derivative; output clamping to [0, 1]; anti-windup
    (conditional integration and back-calculation, selectable); setpoint
    weighting.
  - Its `command(t, probe_K)` updates only at multiples of its sample period
    (zero-order hold), like real hardware.
- **Sensor model:** a first-order lag (time constant τ_sensor, for a probe in a
  thermowell), optional Gaussian noise, and quantization (for example
  0.0625 °C for a DS18B20), applied to the probe value before the controller
  sees it.
- **Actuator:** reuses WP05's actuators. The PID output is the command, so SSR
  time-proportioning and the gas valve turndown are modeled automatically.
- **Metrics:** rise time, overshoot, settling time (±0.5 K), steady-state
  error, integral of absolute error, and actuator cycling count. Added to
  `Summary` through a contract change.
- **Tests:**
  - A closed loop on the lumped model (WP10) reaches the setpoint without
    windup overshoot when anti-windup is on.
  - Sensor lag increases overshoot, as expected.
  - Discrete behavior matches a hand-computed sequence.

## WP31: PID in the GUI and tuning aids (M)

- A controller tab in the editor: gains, a single setpoint, and
  sensor settings.
- **Real-time mode:** live gain and setpoint adjustment, and a chart showing
  the setpoint and output traces.
- **Tuning helpers:**
  - A step-test wizard: apply a step in open loop, fit a first-order plus dead
    time (FOPDT) model, and suggest gains by SIMC, Ziegler–Nichols or
    Cohen–Coon.
  - An option to export the fitted FOPDT parameters for use with real
    hardware.
- **Exit criteria:** the user can tune gains against the simulation and copy
  them to their controller.
