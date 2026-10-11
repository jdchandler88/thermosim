# ThermoSim Requirements

Status: **draft v2**. The implementation plan in [`docs/plan/`](docs/plan/README.md)
is built on these requirements.

Each item is tagged:

- **[Decided]** — agreed in discussion.
- **[Proposed]** — a recommendation. The plan treats it as the baseline until
  you object.
- **[Open]** — a question that still needs an answer (listed at the end).

---

## 1. Purpose

ThermoSim simulates heating a liquid (water or wort) in a vessel with a
configurable heat source, surrounded by the atmosphere. It models the heat, and
later the fluid flow, on a 3D grid, and lets you watch a 2D slice of it.

Goals, in priority order **[Decided]**:

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
| **3** | Simulated PID controller in the loop. |
| **Later** | Boiling and boil-off (evaporation, volume loss, boil-off rate). A slice you can pan and rotate. |

The phases and their order are **[Decided]**: real flow comes before the PID
controller. Boil-off comes later **[Decided]**. Panning the slice comes later
**[Decided]**.

## 3. Platform and technology

- **T1 [Decided]** Written in Python.
- **T2 [Decided]** Runs on Windows, since all your personal machines are
  Windows. **[Proposed]** It also runs on macOS and Linux. The planned libraries
  support all three, and the automated tests run on all three.
- **T3 [Proposed]** The solver uses NumPy, with Numba to compile the hot loops
  (it runs on the CPU). The GUI uses PySide6 (Qt) with pyqtgraph for charts and
  images. Setups are saved as TOML files. `uv` manages the project and its
  dependencies.
- **T4 [Proposed]** The GPU backend for Phase 2 is chosen by a short trial
  phase (spike) before the flow solver is built. It depends on your graphics
  card ([Open] Q2).
- **T5 [Open]** Remove the old Java/JavaFX code once the Python app replaces it
  (Q3)? Until you answer, it is left untouched.
- **T6 [Proposed]** The simulation core is a library with no GUI dependency.
  The GUI and the command-line tool both sit on top of it.

## 4. Physical model

### 4.1 Domain and grid

- **P1 [Decided]** The model is fully 3D. It is not reduced to an axisymmetric
  (rotationally symmetric) 2D model.
- **P2 [Proposed]** The liquid is a uniform grid of cubic cells with a
  configurable cell size. The top layer of cells can be partly filled, so any
  fill height is represented exactly.
- **P3 [Proposed]** The vessel wall, base and lid are thin "shells" with their
  own nodes, not grid cells. A 1–3 mm wall is much thinner than a 5–10 mm grid
  cell; separate shell nodes let the wall's thickness, heat capacity and
  sideways heat spreading be modeled correctly. Sideways spreading matters
  because a base spreads burner heat outward from the flame.
- **P4 [Proposed]** The outside atmosphere is a boundary condition, not a
  gridded volume of air.

### 4.2 Vessel

- **V1 [Decided]** Configurable.
- **V2 [Proposed]** Cylindrical, with these settings: inner diameter, inner
  height, wall thickness, base thickness, and material. Material is a preset
  (stainless 304, aluminum, copper, carbon steel) or custom density, specific
  heat, conductivity and emissivity.
- **V3 [Decided]** Lid choices: none, full or partial. **[Proposed]** A partial
  lid is set by the fraction of the top it covers.
- **V4 [Decided]** Insulation is optional. **[Proposed]** It is set by
  thickness and conductivity, and can cover any of the side, base and lid.

### 4.3 Liquid

- **L1 [Decided]** Water or wort. Water is wort with a specific gravity of
  1.000.
- **L2 [Proposed]** Inputs: volume (or fill height), starting temperature, and
  specific gravity.
- **L3 [Proposed]** Density, specific heat, conductivity and boiling point are
  calculated from temperature and specific gravity using published
  correlations. Viscosity and thermal expansion are also calculated, for
  Phase 2.
- **L4 [Decided]** The simulation stops when the target temperature is reached.
  It does not model boiling.
- **L5 [Proposed]** The run also stops, and says the target was not reached,
  if the temperature levels off below the target because losses equal the heat
  input, or if a maximum simulated time runs out.

### 4.4 Heat sources

- **H1 [Decided]** Configurable. Supported types:
  - **Propane burner under the base:** rated power and efficiency (the fraction
    of the burner's heat that reaches the vessel). **[Proposed]** The heat is
    applied over a disk or ring footprint on the base.
  - **Electric element inside the vessel:** wattage and shape. **[Proposed]**
    The shape is a path of points with a diameter, which covers straight rods,
    U-shapes and rings. All of the element's heat goes into the liquid it
    passes through. Setup is rejected if the element sits above the liquid
    line.
- **H2 [Decided]** The heat source can be driven several ways: on/off through
  an SSR with time-proportioning (switched on for a share of each fixed
  cycle), variable power, or a modulated gas valve.
- **H3 [Decided]** You can change the power while the simulation is running in
  real-time mode (see B3). **[Proposed]** In fast mode, power follows a
  schedule set before the run, which defaults to a constant value.

### 4.5 Atmosphere and heat losses

- **A1 [Decided]** Atmosphere inputs: ambient temperature, wind speed, relative
  humidity, and altitude. Altitude sets the air pressure, which in turn sets
  the boiling point and how fast water evaporates.
- **A2 [Proposed]** Heat losses modeled:
  - from the outer walls, base and lid by natural convection, and by forced
    convection when there is wind;
  - by radiation;
  - from any uncovered liquid surface by convection, radiation and evaporation;
  - through insulation where it is fitted;
  - through the lid and headspace (the air between the liquid and the lid)
    where there is a lid.

### 4.6 Heat transport inside the liquid

- **F1 [Decided]** Phase 1 uses heat conduction with an effective conductivity
  that stands in for convection.
- **F2 [Proposed]** The effective conductivity is recalculated during the run
  from standard natural-convection correlations. An advanced setting multiplies
  it by a calibration factor.
- **F3 [Decided]** Phase 2 solves the actual buoyancy-driven flow on the same
  grid.
- **F4 [Proposed]** Phase 2 resolves the large convection cells and plumes; the
  grid is too coarse for the smallest turbulent eddies, so their mixing is
  approximated.
- **F5 [Proposed]** Known limitation of Phase 1: the effective conductivity
  spreads heat equally in every direction. Directional effects, such as cold
  liquid staying below an element mounted high, are captured only in Phase 2.

## 5. Behavior

- **B1 [Decided]** The vertical slice is fixed at first. Panning through the
  volume comes later.
- **B2 [Proposed]** The slice is the vertical plane through the center axis
  (y = 0).
- **B3 [Proposed]** Two run modes:
  - **Fast mode** runs as fast as the computer allows, with the power following
    the schedule.
  - **Real-time mode** runs in step with the wall clock, optionally sped up by a
    factor you choose, and lets you change the power while it runs.
  - Both modes show the slice live. Snapshots are saved as the run goes, so you
    can scrub back and forth through it afterwards.
- **B4 [Proposed]** You can pause, resume and stop a run.
- **B5 [Proposed]** Setups are saved to and loaded from TOML files.
- **B6 [Proposed]** A command-line mode runs a setup file without the GUI and
  writes the results to a folder.
- **B7 [Proposed]** Units can be switched between metric and US customary in
  the GUI. Internally everything is SI (temperatures in kelvin).

## 6. Outputs

- **O1 [Decided]** Temperature at configurable probe positions in the liquid.
- **O2 [Decided]** All of the following:
  - Time to reach the target temperature.
  - Average, minimum and maximum liquid temperature.
  - A chart of temperature over time.
  - A slice heat map. Phase 2 adds flow arrows.
  - Energy totals: heat put in, heat stored, and heat lost through each loss
    path.
  - Fuel or electricity used.
  - CSV export of the time series.
- **O3 [Proposed]** You can import a measured heat-up from a CSV file, overlay
  it on the chart, and see error figures. This supports checking the
  simulation against a real batch (X2).

## 7. Performance

- **R1 [Decided]** A run should finish in minutes.
- **R2 [Proposed]** At the standard cell size (5 mm), a 45-minute simulated
  heat-up of a 50 L kettle finishes in under 5 minutes on a typical 4-core
  Windows laptop. A "preview" cell size (10 mm) finishes in under 30 seconds.
- **R3 [Proposed]** Real-time mode at 1× speed keeps up at any supported cell
  size.

## 8. Accuracy and validation

- **X1 [Proposed]** Automated tests check the solver against cases with known
  answers:
  - Energy is conserved.
  - Heat conduction matches simple cases that have an exact formula.
  - A simple single-temperature ("lumped") energy-balance model agrees with the
    full model.
  - Results converge as the grid gets finer.
- **X2 [Decided]** You will probably check the simulation against a real
  heat-up. **[Open]** What accuracy do you expect (Q4)?

---

## Open questions

1. **Run modes.** Do fast and real-time modes, as described in B3, match what
   you meant by "real time mode"?
2. **Graphics card.** What graphics card is in the Windows machine you'd run
   this on (for example NVIDIA RTX 3060, AMD, or Intel integrated)? This only
   affects Phase 2.
3. **Java code.** Can the old Java code be deleted once the Python app replaces
   it? Git history keeps it either way.
4. **Accuracy.** How close does the predicted heating time need to be to a real
   heat-up (for example within 10%)?
5. **Units.** Which units should the GUI show by default: metric or US (°F,
   gallons, BTU/hr)?
6. **Phase 3 behavior.** Should the PID controller only heat up to a setpoint,
   or also hold it and step through mash rest schedules?
7. **[Proposed] items.** The plan builds on them; flag any you disagree with.
