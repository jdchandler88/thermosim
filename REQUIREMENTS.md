# ThermoSim Requirements

Status: **draft** — under review. Items are tagged:

- **[Decided]** — agreed in discussion.
- **[Proposed]** — a recommendation that needs your confirmation.
- **[Open]** — a question that still needs an answer (see the end of this document).

---

## 1. Purpose

ThermoSim simulates heating a liquid (water or wort) in a vessel with a
configurable heat source, surrounded by the atmosphere. It models the heat flow,
and later the fluid flow, on a 3D grid, and lets you watch a 2D slice of it.

Goals, in priority order **[Decided]**:

1. **Predict heating time.** How long does it take to bring a given batch to a
   target temperature in a given setup and under given conditions?
2. **Learning and visualization.** See how heat (and later flow) moves through
   the vessel and liquid.
3. **PID tuning.** Tune a brewing controller against the simulation instead of
   real water.

## 2. Scope by phase

| Phase | Contents |
|---|---|
| **1** | 3D heat-only model; the liquid uses an effective conductivity to stand in for convection. Configurable vessel, liquid, heat source and atmosphere. Fixed vertical slice viewer. Time-to-temperature outputs. |
| **2** | Real buoyancy-driven flow (natural convection) on the same 3D grid, as a high-fidelity mode. Flow is shown in the slice. |
| **3** | Simulated PID controller in the loop. |
| **Later** | Boiling and boil-off (evaporation, volume loss, boil-off rate). A slice you can pan or rotate. |

Phases 1–3 come from the staged plan **[Decided]**. Boil-off comes later
**[Decided]**. The 2D slice is fixed at first, with pan/rotate later
**[Decided]**. Putting the PID controller in its own phase after flow is
**[Proposed]**; it is the lowest priority, but it could come before Phase 2 if
you'd rather.

## 3. Platform and technology

- **T1 [Decided]** Written in Python.
- **T2 [Decided]** Runs on macOS.
- **T3 [Proposed]** The solver uses NumPy, with Numba to compile the hot loops.
- **T4 [Proposed]** If Phase 2 needs a GPU, use the Mac's Metal GPU (for example
  through Taichi or PyTorch MPS). CUDA is not available on Macs.
- **T5 [Proposed]** The existing Java/JavaFX code is removed once the Python app
  replaces it. The git history keeps it.
- **T6 [Proposed]** The simulation core is a library with no GUI dependency, so
  it can be tested and scripted on its own. The GUI and command-line tools sit
  on top of it.

## 4. Physical model

### 4.1 Domain and grid

- **P1 [Decided]** The model is fully 3D. It is not reduced to an axisymmetric
  (rotationally symmetric) 2D model.
- **P2 [Proposed]** A uniform grid of cubic cells with configurable spacing.
  Each cell is vessel material, liquid, air (headspace and surroundings) or heat
  source.
- **P3 [Proposed]** The modeled region is the vessel and its contents. The
  outside atmosphere is a boundary condition (an ambient temperature plus a
  heat transfer coefficient), not a gridded volume of air.

### 4.2 Vessel

- **V1 [Decided]** Configurable.
- **V2 [Proposed]** Parameters: shape (cylindrical first), inner diameter,
  height, wall thickness, base thickness, and material (a preset library such
  as stainless steel 304 and aluminum, or custom density, specific heat and
  conductivity).
- **V3 [Open]** Lid: none, full, or partial? See Q1.
- **V4 [Open]** Insulation jacket? See Q2.

### 4.3 Liquid

- **L1 [Decided]** Water or wort. Water is treated as wort with a specific
  gravity of 1.000.
- **L2 [Proposed]** Inputs: volume (or fill height), starting temperature, and
  specific gravity.
- **L3 [Proposed]** Density, specific heat and boiling point are calculated
  from specific gravity using published correlations. Viscosity and thermal
  expansion are added in Phase 2, because the flow model needs them.
- **L4 [Decided]** The simulation stops when the target temperature is reached.
  It does not model boiling.

### 4.4 Heat sources

- **H1 [Decided]** Configurable. Supported types:
  - **Propane burner under the base:** rated power (BTU/hr or kW) and an
    efficiency, meaning the fraction of the heat that actually reaches the
    vessel. [Proposed] The heat is applied to an area of the base with a chosen
    diameter.
  - **Electric element inside the vessel:** wattage, position, and shape (a
    straight rod or a loop, at a chosen height and orientation). [Proposed] All
    of its heat goes into the liquid next to it.
- **H2 [Decided]** The output can be driven several ways: on/off (an SSR with
  time-proportioning), variable power, or a modulated gas valve. Phase 1 uses a
  fixed power setting. Phase 3 lets the PID controller drive the output.
- **H3 [Open]** Can you change the power while the simulation is running, or
  only before it starts? See Q3.

### 4.5 Atmosphere and heat losses

- **A1 [Proposed]** Losses modeled:
  - from the outer walls to the air (natural convection, or forced convection
    if there is wind);
  - from the outer walls by radiation;
  - from the open liquid surface by convection and evaporation. Evaporation
    becomes a large loss as the liquid nears boiling, even before it actually
    boils.
- **A2 [Open]** Which atmosphere inputs? See Q4.

### 4.6 Heat transport inside the liquid

- **F1 [Decided]** Phase 1 uses heat conduction with an effective conductivity
  to stand in for convection.
- **F2 [Proposed]** The effective conductivity is not a fixed number. It is
  recalculated during the run from standard natural-convection correlations,
  so it follows how hard the liquid is being heated.
- **F3 [Decided]** Phase 2 solves the actual buoyancy-driven flow on the same
  grid.
- **F4 [Proposed]** Phase 2 accepts that the grid is too coarse to resolve
  every turbulent eddy. It reproduces the large convection cells and plumes,
  and the fine-scale mixing is approximated.

## 5. Behavior

- **B1 [Decided]** The vertical slice is fixed at first (Phase 1). Panning
  through the volume comes later.
- **B2 [Decided]** The slice is a vertical plane through the center axis. Which
  direction it faces is fixed at first.
- **B3 [Proposed]** Viewing: you watch a live slice while the simulation runs,
  and snapshots are saved as it goes, so afterwards you can scrub back and forth
  through the whole run. (You said "whatever is best"; this gives both.)
- **B4 [Proposed]** You can pause, resume and stop the simulation.
- **B5 [Proposed]** Setups (vessel, liquid, heat source, atmosphere) are saved
  to and loaded from a file, so you can keep a library of them.
- **B6 [Proposed]** Command-line mode: run a setup file without the GUI and
  write the results to a file, for running many scenarios.

## 6. Outputs

- **O1 [Decided]** A temperature reading at a configurable probe position
  (wall-mounted, through the top, and so on).
- **O2 [Proposed]**
  - Time to reach the target temperature, measured at the probe.
  - Average liquid temperature.
  - A chart of probe and average temperature over time.
  - A 2D slice heat map with a color scale. Phase 2 adds flow arrows or
    streamlines.
  - Energy totals: heat put in, heat absorbed by the liquid, and heat lost
    through each loss path.
  - Fuel or electricity used.
  - CSV export of the time series.
- **O3 [Open]** Confirm or trim this list. See Q5.

## 7. Performance

- **R1 [Decided]** A run should finish in minutes.
- **R2 [Proposed]** Target: at the default grid spacing, a full heat-up of
  about 45 minutes simulated time finishes in under 5 minutes on your Mac.
  Finer grids may take longer.
- **R3 [Open]** Is your Mac Apple Silicon or Intel? See Q6. This sets the
  performance target and the GPU options.

## 8. Accuracy and validation

- **X1 [Proposed]** Automated tests check the solver against cases with known
  answers:
  - Energy is conserved.
  - A perfectly insulated vessel heats at the rate a hand energy balance
    predicts.
  - Heat conduction matches simple cases that have an exact formula.
  - Results converge as the grid gets finer.
- **X2 [Open]** Will you check the simulation against a real heat-up, and what
  accuracy do you expect? See Q7.

---

## Open questions

1. **Lid:** none, full, partial, or all three as options?
2. **Insulation:** should the vessel be able to have an insulation jacket?
3. **Changing power mid-run:** should you be able to adjust the heat source
   while the simulation runs, or only set it beforehand?
4. **Atmosphere inputs:** which of these: ambient temperature, wind speed,
   humidity, altitude (which sets the boiling point and how fast water
   evaporates)?
5. **Outputs:** confirm or trim the list in O2.
6. **Mac:** Apple Silicon or Intel?
7. **Accuracy:** will you check against a real heat-up, and how close does the
   heating time need to be (for example within 10%)?
8. **Phase order:** should the PID controller (Phase 3) come before real flow
   (Phase 2)?
9. **Confirm every [Proposed] item**, or mark the ones you disagree with.
