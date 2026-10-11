# ThermoSim Requirements

Status: **v3, agreed**. Every item is decided. The implementation plan in [`docs/plan/`](docs/plan/README.md)
is built on these requirements.

Every item below was agreed in discussion. The answers to the earlier open
questions are listed at the end.

---

## 1. Purpose

ThermoSim simulates heating a liquid (water or wort) in a vessel with a
configurable heat source, surrounded by the atmosphere. It models the heat, and
later the fluid flow, on a 3D grid, and lets you watch a 2D slice of it.

Goals, in priority order:

1. **Predict heating time.** How long does it take to bring a given batch to a
   target temperature in a given setup and given conditions?
2. **Learning and visualization.** See how heat (and later flow) moves through
   the vessel and liquid.
3. **PID tuning.** Tune a brewing controller against the simulation instead of
   real water.

## 2. Scope by phase

| Phase | Contents |
|---|---|
| **1** | 3D heat-only model, with the liquid's mixing approximated by an effective conductivity. Configurable vessel, lid, insulation, liquid, heat source and atmosphere. Fixed vertical slice viewer. Fast and real-time run modes. Time-to-temperature outputs. |
| **2** | Real buoyancy-driven flow (natural convection) on the same 3D grid, as a high-fidelity mode. Flow is shown in the slice. |
| **3** | Simulated PID controller in the loop, heating to a single setpoint and holding it (no mash step schedules). |
| **Later** | Boiling and boil-off (evaporation, volume loss, boil-off rate). A slice you can pan and rotate. |

The phases and their order are: real flow comes before the PID
controller. Boil-off comes later. Panning the slice comes later
**[Decided]**.

## 3. Platform and technology

- **T1** Written in Python.
- **T2** Runs on Windows, since all your personal machines are
  Windows. It also runs on macOS and Linux. The planned libraries
  support all three, and the automated tests run on all three.
- **T3** The solver uses NumPy, with Numba to compile the hot loops
  (it runs on the CPU). The GUI uses PySide6 (Qt) with pyqtgraph for charts and
  images. Setups are saved as TOML files. `uv` manages the project and its
  dependencies.
- **T4** The GPU backend for Phase 2 is chosen by a short trial phase (spike)
  before the flow solver is built. The target GPU is an **NVIDIA GTX 1060**
  (Pascal architecture, CUDA compute capability 6.1, 3 or 6 GB). That means
  CUDA 12.x tooling and single-precision (float32) math for the flow solver.
- **T5** The old Java/JavaFX code is deleted once the Python app replaces it.
  Git history keeps it.
- **T6** The simulation core is a library with no GUI dependency.
  The GUI and the command-line tool both sit on top of it.

## 4. Physical model

### 4.1 Domain and grid

- **P1** The model is fully 3D. It is not reduced to an axisymmetric
  (rotationally symmetric) 2D model.
- **P2** The liquid is a uniform grid of cubic cells with a
  configurable cell size. The top layer of cells can be partly filled, so any
  fill height is represented exactly.
- **P3** The vessel wall, base and lid are thin "shells" with their
  own nodes, not grid cells. A 1–3 mm wall is much thinner than a 5–10 mm grid
  cell; separate shell nodes let the wall's thickness, heat capacity and
  sideways heat spreading be modeled correctly. Sideways spreading matters
  because a base spreads burner heat outward from the flame.
- **P4** The outside atmosphere is a boundary condition, not a
  gridded volume of air.

### 4.2 Vessel

- **V1** Configurable.
- **V2** Cylindrical, with these settings: inner diameter, inner
  height, wall thickness, base thickness, and material. Material is a preset
  (stainless 304, aluminum, copper, carbon steel) or custom density, specific
  heat, conductivity and emissivity.
- **V3** Lid choices: none, full or partial. A partial
  lid is set by the fraction of the top it covers.
- **V4** Insulation is optional. It is set by
  thickness and conductivity, and can cover any of the side, base and lid.

### 4.3 Liquid

- **L1** Water or wort. Water is wort with a specific gravity of
  1.000.
- **L2** Inputs: volume (or fill height), starting temperature, and
  specific gravity.
- **L3** Density, specific heat, conductivity and boiling point are
  calculated from temperature and specific gravity using published
  correlations. Viscosity and thermal expansion are also calculated, for
  Phase 2.
- **L4** The simulation stops when the target temperature is reached.
  It does not model boiling.
- **L5** The run also stops, and says the target was not reached,
  if the temperature levels off below the target because losses equal the heat
  input, or if a maximum simulated time runs out.

### 4.4 Heat sources

- **H1** Configurable. Supported types:
  - **Propane burner under the base:** rated power and efficiency (the fraction
    of the burner's heat that reaches the vessel). The heat is
    applied over a disk or ring footprint on the base.
  - **Electric element inside the vessel:** wattage and shape.
    The shape is a path of points with a diameter, which covers straight rods,
    U-shapes and rings. All of the element's heat goes into the liquid it
    passes through. Setup is rejected if the element sits above the liquid
    line.
- **H2** The heat source can be driven several ways: on/off through
  an SSR with time-proportioning (switched on for a share of each fixed
  cycle), variable power, or a modulated gas valve.
- **H3** You can change the power while the simulation is running in
  real-time mode (see B3). In fast mode, power follows a
  schedule set before the run, which defaults to a constant value.

### 4.5 Atmosphere and heat losses

- **A1** Atmosphere inputs: ambient temperature, wind speed, relative
  humidity, and altitude. Altitude sets the air pressure, which in turn sets
  the boiling point and how fast water evaporates.
- **A2** Heat losses modeled:
  - from the outer walls, base and lid by natural convection, and by forced
    convection when there is wind;
  - by radiation;
  - from any uncovered liquid surface by convection, radiation and evaporation;
  - through insulation where it is fitted;
  - through the lid and headspace (the air between the liquid and the lid)
    where there is a lid.

### 4.6 Heat transport inside the liquid

- **F1** Phase 1 uses heat conduction with an effective conductivity
  that stands in for convection.
- **F2** The effective conductivity is recalculated during the run
  from standard natural-convection correlations. An advanced setting multiplies
  it by a calibration factor.
- **F3** Phase 2 solves the actual buoyancy-driven flow on the same
  grid.
- **F4** Phase 2 resolves the large convection cells and plumes; the
  grid is too coarse for the smallest turbulent eddies, so their mixing is
  approximated.
- **F5** Known limitation of Phase 1: the effective conductivity
  spreads heat equally in every direction. Directional effects, such as cold
  liquid staying below an element mounted high, are captured only in Phase 2.

## 5. Behavior

- **B1** The vertical slice is fixed at first. Panning through the
  volume comes later.
- **B2** The slice is the vertical plane through the center axis
  (y = 0).
- **B3** Two run modes:
  - **Fast mode** runs as fast as the computer allows, with the power following
    the schedule.
  - **Real-time mode** runs in step with the wall clock, optionally sped up by a
    factor you choose, and lets you change the power while it runs.
  - Both modes show the slice live. Snapshots are saved as the run goes, so you
    can scrub back and forth through it afterwards.
- **B4** You can pause, resume and stop a run.
- **B5** Setups are saved to and loaded from TOML files.
- **B6** A command-line mode runs a setup file without the GUI and
  writes the results to a folder.
- **B7** Units can be switched between metric and US customary in the GUI.
  **The default is US** (°F, gallons, BTU/hr, inches, mph). Internally
  everything is SI (temperatures in kelvin).

## 6. Outputs

- **O1** Temperature at configurable probe positions in the liquid.
- **O2** All of the following:
  - Time to reach the target temperature.
  - Average, minimum and maximum liquid temperature.
  - A chart of temperature over time.
  - A slice heat map. Phase 2 adds flow arrows.
  - Energy totals: heat put in, heat stored, and heat lost through each loss
    path.
  - Fuel or electricity used.
  - CSV export of the time series.
- **O3** You can import a measured heat-up from a CSV file, overlay
  it on the chart, and see error figures. This supports checking the
  simulation against a real batch (X2).

## 7. Performance

- **R1** A run should finish in minutes.
- **R2** At the standard cell size (5 mm), a 45-minute simulated
  heat-up of a 50 L kettle finishes in under 5 minutes on a typical 4-core
  Windows laptop. A "preview" cell size (10 mm) finishes in under 30 seconds.
- **R3** Real-time mode at 1× speed keeps up at any supported cell
  size.

## 8. Accuracy and validation

- **X1** Automated tests check the solver against cases with known
  answers:
  - Energy is conserved.
  - Heat conduction matches simple cases that have an exact formula.
  - A simple single-temperature ("lumped") energy-balance model agrees with the
    full model.
  - Results converge as the grid gets finer.
- **X2** You will probably check the simulation against a real heat-up. There
  is no fixed accuracy target ("close enough"). The goal is that, after tuning
  burner efficiency and `k_eff_scale` against one real heat-up, predictions
  for similar setups are useful in practice.

---

## Resolved questions

| Question | Answer |
|---|---|
| Run modes (B3) | Accepted as described: fast mode plus real-time mode with live power control |
| Graphics card | NVIDIA GTX 1060 (affects Phase 2 only; see T4) |
| Delete the Java code | Yes, once the Python app replaces it (T5) |
| Accuracy | "Close enough"; no fixed tolerance (X2) |
| Default units | US (B7) |
| Phase 3 behavior | A single setpoint: heat to it and hold. No mash step schedules |
| Earlier recommendations | All accepted, including the vessel starting at air temperature |
