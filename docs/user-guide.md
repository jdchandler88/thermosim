# ThermoSim user guide

ThermoSim predicts how a vessel of liquid heats up on a propane burner or with an
electric element: how long it takes to reach a target temperature, where the heat
goes, and how much fuel or electricity it uses. You describe the setup, run the
simulation, and look at the temperatures inside the liquid.

- Install and launch: see the [Windows instructions](../packaging/windows/README.md)
  (macOS and Linux: `uv run thermosim gui` in a clone of the repository).
- Requirements and design background: [REQUIREMENTS.md](../REQUIREMENTS.md).

The screenshots below were produced by the real simulator at a 10 mm cell size
(`python packaging/make_screenshots.py`).

## 1. The window

![The setup before a run](images/setup.png)

| Area | What it is |
|---|---|
| Left | The **setup editor**: tabs for vessel, lid and insulation, liquid, heat source, atmosphere, probes and target, and simulation settings. |
| Centre top | The **slice view**: temperature on a vertical cut through the middle of the vessel, with the probes marked. |
| Scrubber | Drag to look at any moment of the run; *Play* replays it at 10x to 1000x. |
| Centre bottom | The **chart**: probe and mean temperatures, the min-max band, the target (dashed red line), the heat source power (green, right axis) and any measured data. |
| Right | **Quick estimate**, then the **run summary** and the **energy breakdown**, then the **comparison** with measured data. |
| Toolbar | New, Open, Save, Run, Pause, Stop, real-time speed and power, the **Units** switch, Export CSV, Import measured. |

**Units.** US customary units are the default; switch to metric with the *Units*
box in the toolbar. Only the display changes: a setup is always stored in SI units, so
you can switch at any time without changing the setup. Numbers in the editor are shown
rounded (for example 8.661 in); the exact value is kept until you type a new one.

## 2. Creating a setup

Start from *File > Examples* (small pot on a stove, 40 L propane kettle, insulated
electric kettle) or *New*, and change what you need. Every field has a tooltip.
Invalid entries are outlined in red, and the message appears above the tabs; *Run* is
disabled until the setup is valid. Save your setup with *Save* (a `.toml` file you can
also edit in a text editor).

### What each setting means physically

**Vessel.** A cylinder with a flat base. *Inner diameter* and *inner height* set how
much liquid it holds and how much surface area loses heat. *Wall* and *base thickness*
and the *material* set the metal's heat capacity (it must be heated too) and how well
heat spreads sideways. Aluminum and copper conduct well; stainless steel conducts
poorly, so the base heats unevenly. A custom material needs density, specific heat,
conductivity and emissivity.

**Lid and insulation.** *Lid*: none, full, or partial (a covered fraction). A lid
mainly stops evaporation and open-surface radiation, which are the big losses near
boiling. *Insulation* is an optional layer on the side, base and/or lid: thicker or
lower conductivity means less loss through the wall.

**Liquid.** Give either the *volume* or the *fill height* (setting one clears the
other), the *initial temperature*, and the *specific gravity* (water is 1.000, wort is
about 1.040 to 1.080). Specific gravity changes the density and heat capacity of the
liquid.

**Heat source.**
- *Propane burner*: *rated power* (the gross fuel energy rate), *efficiency* (the
  share of that which reaches the vessel), and the *footprint* on the base, a disk or
  a ring, with its diameters. A small footprint concentrates heat in the middle of the
  base. The actuator is either an on/off cycle (the command sets the on-share of each
  cycle of the given *cycle period*) or a modulating valve (a *minimum fraction* it can
  hold, and a *response time*).
- *Electric element*: *rated power*, the *path* (x, y, z points of the element's
  centre line, origin at the centre of the inner base, z up) and its *diameter*. The
  whole path must be inside the liquid, or a run is refused and the offending point
  is highlighted (see section 7).

**Atmosphere.** *Ambient temperature*, *wind speed* (forced convection on the outside:
a breeze can double the side-wall loss), *relative humidity* and *altitude* (air
pressure sets the boiling point and the evaporation rate).

**Probes and target.** Probes are named points in the liquid that you can read like a
thermometer; they appear on the slice and the chart. The *target* is a temperature at
a chosen probe (or the volume-weighted *mean*); the run stops when it is reached.
*Power schedule*: the requested power as a fraction of the rated power, as steps over
time (the first step is at t = 0). It is used by fast runs; in real-time runs you set
the power with the slider instead.

**Simulation.**
- *Cell size*: the liquid is split into cubes of this size. Smaller is more accurate
  and slower; 5 mm is the standard, 10 mm is a quick preview, and the run time rises
  steeply (about 8 times more cells when halving the size, and smaller time steps too).
- *Maximum simulated time*: the run stops here if the target is never reached.
- *Run mode* and *real-time speed*: see sections 3 and 4.
- *Sample interval* (chart points) and *snapshot interval* (slice frames): how often
  results are recorded.
- *Plateau window / rate*: the run also stops, with "levelled off below target", when
  the temperature rises more slowly than this rate over the window (for example a boil
  that never reaches your target).
- *Coefficient refresh interval*: how often material properties and heat-transfer
  coefficients are updated as temperatures change.

**Model.** *Liquid transport* chooses how heat moves inside the liquid:
- *Effective conductivity* (the default, fast): convection is stood in for by a larger
  conductivity. The *effective conductivity factor* is a calibration knob that multiplies
  it; use 1 unless you are tuning against measurements.
- *Flow* (buoyancy-driven, much slower): the motion of the liquid is calculated, see
  "Flow mode" below. The *subgrid model constant* (0.1 by default; 0 switches the subgrid
  model off) and the *pressure solver tolerance* (leave empty for the default) only apply
  to this mode.

## 3. Quick estimate

Press **Estimate time to target** (right-hand column) for an answer in a fraction of a
second, using a simple one-temperature model of the whole vessel. It is good for a
first check of a setup (the example above: about 21:42 against 21:06 from the full
simulation) but it knows nothing about temperature differences inside the liquid.

## 4. Running in fast mode

Choose **Run (Fast)**. Before the run starts, ThermoSim checks the setup in the
background behind a small dialog (you can cancel it):

- it builds the vessel and heat source, and reports a problem next to the field that
  causes it (section 7);
- the very first run after installation first compiles the solver: **"Preparing solver
  (first run only)..."**; later runs skip this;
- it measures the solver speed and estimates the run time. If the estimate is over 10
  minutes you are offered the 10 mm preview cell size, to run anyway, or to cancel.

The run then goes as fast as it can, following the power schedule, and stops when the
target is reached, the temperature levels off, the maximum time is reached, or you
press **Stop**.

![A finished fast run](images/main-window-fast-run.png)

### Flow mode

Set **Model > Liquid transport** to *Flow* to calculate the buoyancy-driven motion of the
liquid instead of using an effective conductivity: warm liquid rises in a plume above the
burner or element, spreads under the surface and sinks along the cooler walls. It is the
more physical model, and it shows *where* the liquid circulates, but **it is much slower**:
every time step also solves for the velocity and pressure, and the time step is limited
by the speed of the flow (about 0.02 to 0.07 s instead of up to 1 s). The pre-run check
measures the cost of a few flow steps and includes it in the estimated run time, and the
preview cell size offered for a long flow run is 20 mm rather than 10 mm.

![Flow mode: the insulated electric kettle at 20 mm](images/flow-mode.png)

What you see:
- **Flow arrows** on the slice (toolbar above the slice, on by default): the in-plane
  velocity at the cell centres, thinned out to at most about 24 arrows across and scaled
  so that the longest arrow is about one and a half arrow spacings long (so only
  directions and *relative* speeds are shown; the peak speed is in the summary). Arrows
  slower than 2 % of the peak are left out.
- **Speed colours** shows the in-plane speed instead of the temperature; the colour bar
  switches to cm/s.
- Hovering over the slice also reads out the local flow speed.
- The **Run** box in the summary shows the **peak flow speed** seen in the slice plane
  (typically 5 to 20 cm/s in a heated kettle).

How it works, and what to keep in mind:
- The liquid is solved as an incompressible Boussinesq flow on the same grid (a
  projection method, with a Smagorinsky subgrid model for the unresolved eddies); the
  heat carried by the flow is added to the heat balance of every cell, so the energy
  balance still closes to round-off (below 1e-9 in every example). Vessel wall, base, lid,
  surface and loss models are the same as in the other mode.
- **Wall heat transfer is unchanged.** A 5 to 20 mm grid cannot resolve the thin thermal
  boundary layer at a wall, so the same wall film coefficients (including the nucleate
  boiling correction) are used in both modes as a wall function; flow mode changes how
  heat moves *inside* the liquid, not how it enters from the wall.
- The flow is calculated for the buoyancy of the liquid temperature only. Properties are
  taken at the bulk temperature, which is good to a few per cent for temperature differences
  below 20 K but is rougher in the layer right above a hot base. The free surface is treated as
  flat and free of friction; evaporation cooling is a heat loss only and does not drive flow.
- Use **20 mm** cells for a first look (a 40 L kettle runs in a few minutes), and 10 mm
  when you want the finer plume structure (about an hour for the 40 L propane example).
  Effective-conductivity mode remains the right tool for quick what-if runs.
- The time to target differs a little from the effective-conductivity result: at 20 mm the
  three examples take 0.4 % to 10 % longer in flow mode.

## 5. Running in real time

Choose **Run (Real-time)** to watch the run unfold at a speed you pick (*Speed*, 10x
means 10 simulated seconds per second) and to control the heat source live:

- the **Power** slider changes the requested power at once: a dotted marker with the new
  setting appears on the chart immediately, and the delivered power follows within a
  sample or two;
- the **Speed** box changes the speed; if the computer cannot keep up, the status bar
  shows the real-time lag;
- keyboard shortcuts (when you are not typing in a field): **Space** pause/resume,
  **Up/Down** power plus or minus 5 %, **+** and **-** speed.

![A real-time run at 10x with the power turned down to 40 %](images/realtime-run.png)

While a run is going, the setup cannot be edited. *Pause* freezes it; *Stop* ends it
and keeps what has been simulated so far.

## 6. Reading the outputs

- **Slice view.** Colours are temperature; *Color range* is fixed (initial to a little
  above the target) or automatic. Drag the scrubber, or press *Play*, to move through
  time. While a run is live the view follows it; moving the scrubber stops following
  (untick/tick *Follow live*).
- **Chart.** The white line is the mean liquid temperature, the grey band the coldest
  to warmest cell, coloured lines the probes. The gap between a base probe and the
  mean shows how much stratification you have. The yellow dashed line is the time
  shown in the slice.
- **Run summary.** Status, result (target reached / levelled off / maximum time /
  stopped), the **time to target**, simulated and wall-clock time, the final
  temperatures and the fuel (US: pounds of propane) or electricity used.
- **Energy.** Where the heat went: how much stayed in the liquid, in the vessel
  and lid, and the losses through the vessel surfaces, from the open surface (convection
  and radiation), by evaporation, through the lid, and from dry walls above the liquid.
  The *share* bars are relative to the heat delivered. A big *stored in vessel* share
  means a heavy vessel for the liquid it holds; big *loss* shares show where
  insulation or a lid would help.
- **Saving.** *File > Save run...* writes a folder with the summary, time series,
  slice snapshots and the setup used. *File > Open run...* loads one for playback (no
  simulation needed). If you close, start a new run or open another setup while a
  finished run is unsaved, ThermoSim asks first. *Export CSV* writes the time series as a
  spreadsheet file (temperatures, powers and energies in the displayed units; the fuel
  column is always in kg and electricity in kWh, and the CSV inside a saved run folder
  is always metric).

## 7. When something is wrong

Entries that are not valid on their own (a letter in a number field, a negative size)
are caught as you type. Problems that only show up when the vessel is built are caught
when you press *Run*, and shown in the red banner and on the field:

![An element that sticks out of the liquid](images/geometry-error.png)

Fix the highlighted field (here, a fill height lower than the element's legs, or move the
element) and run again. Relationships between fields (a probe above the liquid, a lid
insulation without a lid) are reported one at a time: fix one and the next may appear.

### Known limitations

- **Readings above boiling next to the heat source.** Probes or cells right next to a burner
  footprint or an element can show temperatures above the boiling point (up to about
  222 °F). The model does not boil the liquid; it uses an "effective conductivity" to stand in
  for convection, and very close to a strong heat source that shortcut lets the water there
  overheat. The average temperature and the time to target are not affected. Put probes
  away from the heat source for realistic readings. (In flow mode the liquid moves for real,
  but there is still no boiling.)
- **Run time at fine cell sizes.** A large kettle at the 5 mm cell size can take several
  minutes. The 10 mm size gives nearly identical heating times (within about 0.04%) in
  well under a minute.

## 8. Importing measured data and comparing

Use **Import measured** (toolbar or *File*) to overlay a thermometer log from a CSV
file. A dialog shows the first rows and lets you pick the time and temperature columns
and their units (time in seconds or minutes, temperature in C, F or K); clock times
such as `14:03:20` are understood. The data appears as orange points on the chart.

After a run, the **Measured vs simulated** box reports, for the target probe (or the
mean), the root-mean-square error, the maximum error, the bias (positive when the
simulation runs hot), and the time to target in both. If they differ systematically,
try the *effective conductivity factor*, the burner *efficiency*, the *wind speed*
and the *lid* settings first.

![Simulation against (illustrative) measured data](images/measured-comparison.png)

The same comparison is available without the GUI:
`thermosim compare RUN_DIR measured.csv`.

## 9. Command line

```
thermosim gui [setup.toml]            open the window
thermosim run setup.toml --out DIR    run without the window; prints the summary
thermosim validate setup.toml         check a setup and estimate memory and run time
thermosim examples [--write DIR]      list or copy the example setups
thermosim compare RUN_DIR data.csv    compare a saved run with measured data
```

`thermosim gui --fake` uses synthetic data instead of the simulator (for developers).
