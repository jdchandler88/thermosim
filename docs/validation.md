# Validating ThermoSim

This page explains how we know the simulator is right, and how you check it
against a real heat-up of your own vessel. Section 6 has the results of the
automated validation; `benchmarks/RESULTS.md` has the speed measurements.

## 1. Layers of evidence

| Layer | What it shows | Where | Status |
|---|---|---|---|
| Analytic references | Closed-form solutions of conduction problems (semi-infinite slab with a step or a convective surface, lumped Newton cooling, 3-D Gaussian pulse) that other code is checked against. Each satisfies its PDE (finite-difference test) and its limits. | `src/thermosim/validation/analytic.py`, `tests/validation/test_analytic.py` | Done |
| Lumped reference model | A 0-D model (one liquid, one shell, one lid temperature) that uses the same property and correlation functions and the same physics choices as the 3-D coefficient table. It matches exact Newton cooling to 1e-6 K. | `src/thermosim/validation/lumped.py`, `tests/validation/test_lumped.py` | Done |
| Solver against analytic solutions | The real solver on synthetic domains versus the references above: a convective surface (Carslaw & Jaeger 2.7, with a refinement check), a 3-D Gaussian pulse, lumped Newton heating, and a steady state through film and wall. | `tests/validation/test_solver_vs_analytic.py` | Done |
| Full model against lumped | With `k_eff_scale = 100` (well mixed) the 3-D time-to-target is within 3% of the lumped model's. Result: within 1% for all three examples (section 6). | `tests/validation/test_full_model.py` | Done |
| Grid convergence | 40 L propane example at dx = 20, 10, 5 mm; observed order and Richardson-extrapolated values. | `tests/validation/test_full_model.py` (`slow`) | Done |
| Energy conservation | `energy_balance_rel_error` < 1e-6 for every example (it is about 1e-14). | `tests/validation/test_full_model.py` | Done |
| Rules of thumb | 40 L from 15 to 99 C on a 60k BTU/hr burner at 35% efficiency takes 30-60 minutes (ideal 14 MJ at 6.2 kW is about 38 minutes, plus losses). The 3-D model gives 45 minutes. | `tests/validation/test_full_model.py` | Done |
| A real heat-up | Your own measurement (section 3). | your logbook, section 5 | Yours to do |

### The lumped model and the quick estimate

`LumpedModel(scenario).run()` returns the time to the target, the stop reason
(`target_reached`, `plateau` or `max_time`) and the time series of the liquid,
shell and lid temperatures and the delivered power. `quick_estimate(scenario)`
returns just the time to target (or `None`); it uses coefficients refreshed
every 30 s and a loose tolerance and takes about 0.1 s.

Its simplifications, which you should keep in mind when comparing it to the 3-D
model: a perfectly mixed liquid; a single shell temperature (no conduction around
the wall, except that the dry wall above the liquid is a fin rooted at the waterline);
the target is the liquid mean, whatever probe the scenario names; the heat source is
applied at its steady actuator value (no gas-valve lag, no SSR cycling); the base
outer loss is zero while a burner fires (as in the 3-D model), so a burner's heat is
delivered to the liquid directly. The liquid's heat capacity is V rho(T) cp(T), as in
the 3-D model (see the finding below).

Reference values from the lumped model for the example scenarios (full power, no
schedule), next to the ideal hand estimate m cp dT / delivered power:

| Example | Lumped time to target | Hand estimate | Ratio |
|---|---|---|---|
| `kettle_propane_40L` | 49.5 min | 38.0 min | 1.30 |
| `kettle_electric_insulated` | 26.0 min | 24.3 min | 1.07 |
| `small_pot_stove` | 20.9 min | 19.5 min | 1.07 |

**What it took for the two models to agree.** The first version of the lumped model
disagreed with the 3-D model by up to 9% for the propane kettle. Three differences were
found by comparing the loss flows of both models at the same state; none of them is
a bug in either model, but each is a real feature of the 3-D model that a one-temperature
model misses, and the lumped model now includes them:

1. *Where the burner's heat goes.* A burner heats the base, which is hot (about
   30-50 K above the liquid), while the side wall is only warmed by the liquid. Putting the
   burner's heat into the single shell node heated the whole wall to the base temperature
   and overstated the wall losses by about 3%. The base outer loss is zero while the burner
   fires, so its heat reaches the liquid whatever the base temperature; the lumped model
   now delivers it to the liquid directly.
2. *The dry wall is a fin.* The wall above the liquid is only 1.5 mm of steel; it cools to
   near ambient within about 3 cm of the waterline (3-D wall temperatures: 64 C just below the
   waterline, 33 C about 3 cm above it, 17 C at 7 cm, 11 C at the rim). Treating all of it as a wall at the
   liquid temperature overstated its losses (by about 3%). The lumped model uses an infinite-
   fin conductance, `pi D sqrt(g k t)`, with `g` the sum of the outer and inner coefficients.
3. *Heat capacity.* The 3-D coefficient table uses `rho(T) cp(T)` on a fixed volume, so the
   total heat capacity of the liquid falls by 3-4% between 20 and 100 C (no thermal
   expansion), whereas a fixed mass would keep it constant. The lumped model uses the same
   convention (`constant_mass=True` gives the physical value, which makes it 1.5-2% slower:
   the 3-D model is then 0.6%, 2.4% and 2.3% faster for the pot, the electric kettle and the
   propane kettle). **This is a small physical inconsistency of the 3-D model**: it makes the
   3-D heat-up about 1.5-2% faster than a fixed mass would. The coefficient table is
   specified by the contract (CONTRACTS section 12), so it is left as it is; it is reported
   to the coordinator.

## 2. Equipment for a real check

- **A thermometer or logger** that records at least every 30 s. A probe with a
  stated accuracy of 0.5 C or better. Check it in ice water (0 C) and just
  below boiling before the run; note the offset.
- **A scale or a measuring jug** to know the liquid volume to 2%. Weigh the
  water if you can: 1 kg is 1 L at room temperature.
- **A way to know the power**: the burner rating (BTU/hr or W) from its label,
  or the element's wattage. For propane, weigh the cylinder before and after if
  you can; it gives the energy used directly.
- **Weather notes**: air temperature at the vessel, wind (calm, light, about 2 m/s),
  altitude, and whether the vessel is covered or insulated. Write them down
  before you start: they are the scenario's atmosphere inputs.
- **Vessel dimensions**: inner diameter and height, wall and base thickness
  (a ruler and calipers; the wall of a kettle is about 1.5 mm).

### Probe placement

Put the probe where a scenario probe is, and use that probe in the comparison.
The examples use a probe on the axis (`center`, about half way up) and one
`near_wall`. Hold it in place with a clip rather than letting it rest on the
base or touch the wall: a probe on the metal reads the wall, not the liquid.
For the simplest and most robust check, stir well and use the `mean` series
against a probe at mid-depth on the axis. Record its position
(radius, height above the inside of the base) and put it in the scenario.

### Procedure

1. Fill the vessel to a known volume. Start from a known temperature; stir so
   it is uniform. Note that the simulator assumes **the vessel starts at air
   temperature**, so start with a vessel that has been standing in the air.
2. Start logging, then light the burner or switch on the element at the setting
   you will enter in the scenario (full power is the easiest to model).
3. Stir gently and steadily, or not at all, the same way each time; the
   `k_eff_scale` setting (section 4) describes how well mixed it is.
4. Stop at the target, or at about 90 C. Boiling is not modelled.
5. Note the end time, the gas used if you weighed it, and the final temperature.

## 3. Data format

`thermosim compare` reads a CSV through `load_measured_csv`. It accepts a comma,
semicolon or tab delimiter, decimal commas with a non-comma delimiter, a header
row, blank or `NaN` cells (those rows are dropped), a numeric time in seconds or
minutes, or a clock time. Any of these works:

```csv
time_s,temp_C
0,15.2
30,16.0
60,17.1
90,18.3
```

```csv
Time (min);Temperature (F)
0;59.4
0,5;60.1
1;61.0
```

```csv
time,temp
12:03:00,15.2
12:03:30,16.0
12:04:00,17.1
```

Clock times are converted to seconds from the first row. The time column is
found by name (`t`, `time`, `seconds`, `minutes`, ...) and the temperature
column by a name starting with `temp` (or `T`); otherwise the first two numeric
columns are used. Tell the tool the units, since they are not guessed:
`--temp-unit C|F` and `--time-unit s|min`.

## 4. Comparing and tuning

```
thermosim run my_vessel.toml --out runs/my_vessel
thermosim compare runs/my_vessel data.csv --probe center --temp-unit C --time-unit s
```

`--probe` is a probe name from the scenario or `mean`. The output has:

- **RMSE** and **max |error|** (K): how far the curves are apart overall and at worst.
- **Bias** (K): simulated minus measured, averaged over the run. Positive means the
  simulation is too hot.
- **Time to target**, simulated and measured, and the relative error.

How to read it:

| Symptom | Likely cause | What to change |
|---|---|---|
| Simulation reaches target early, bias positive, curves nearly parallel | Burner efficiency set too high | Lower `heat_source.efficiency` first |
| Simulation late, bias negative, parallel | Efficiency too low, or the real vessel loses less than modelled (lid, insulation) | Raise `efficiency`; check the lid and insulation entries |
| Gap grows near the end (high temperatures) | Losses wrong: wind, evaporation, humidity | Check `atmosphere` inputs and the lid |
| Probe near the wall leads the centre probe by more than measured | Stratification is too strong in the model | Raise `model.k_eff_scale` |
| Well-mixed liquid but the model shows strong gradients | Same | Raise `model.k_eff_scale` |
| Early part of the curve wrong, rest parallel | Initial temperature, the vessel's starting temperature, or probe offset | Re-check the starting temperature and the thermometer calibration |

**Tuning order.** Burner `efficiency` is the most uncertain input (real
burners range from about 0.25 to 0.6 depending on the pot and the wind), so
tune it first, using the time to target. Only then adjust `k_eff_scale`, which
changes how the heat spreads in the liquid (probe differences) rather than
how fast the mean rises. Change one thing at a time, and write each change in
the results log below. If after tuning the shape of the curve is still wrong,
say what you see: that is a model limit, and we want to know.

Rule of thumb for interpreting the match: a time-to-target error under 5% and a
bias under 1 K on a well-measured run is a good agreement; 5-10% is acceptable
given the uncertainty of the burner efficiency; more than that, check the
inputs before blaming the model.

## 5. Results log

Record every real check here (copy the table row).

| Date | Vessel and liquid | Source and setting | Weather | Probe | Tuned `efficiency` / `k_eff_scale` | Bias (K) | RMSE (K) | Time to target: measured / simulated (min) | Error | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| _example_ | 15 gal kettle, 40 L water, 15 C | 60k BTU/hr, full | 10 C, 2 m/s | center | 0.35 / 1 | +0.4 | 0.9 | 56.0 / 54.9 | -2% | _illustrative only, not a measurement_ |
|  |  |  |  |  |  |  |  |  |  |  |

## 6. Results of the automated validation (Part B)

Numbers from `benchmarks/validation_report.py` and the tests in `tests/validation/`, run on
the machine described in `benchmarks/RESULTS.md`.

**Solver versus analytic** (`test_solver_vs_analytic.py`; error relative to the initial
temperature difference unless stated):

| Case | Result |
|---|---|
| Convective surface, semi-infinite solid (C&J 2.7), dx = 5 mm | max error below 0.5%; the error falls by more than 2 when dx is halved |
| 3-D Gaussian pulse, width doubling (C&J 10.3), 33^3 cells | max error below 1% of the peak; energy conserved to 1e-12 (relative); variance within 1% of `sigma0^2 + 2 alpha t` |
| Well-mixed body with a source and a convective top | mean temperature within 0.2% of lumped Newton heating at 0.5, 1 and 3 time constants |
| Steady state through film and wall | within 0.2% of the series conductance of every node |

**Full model versus lumped model** (time to the liquid-mean target; the 3-D target probe is
set to `mean`; criterion: 3% at `k_eff_scale = 100`):

| Example | dx (mm) | `k_eff_scale` | 3-D (s) | Lumped (s) | Error |
|---|---|---|---|---|---|
| `small_pot_stove` | 20 | 100 | 1265 | 1257 | +0.71% |
| `kettle_electric_insulated` | 20 | 100 | 1547 | 1562 | -0.91% |
| `kettle_propane_40L` | 40 | 100 | 2946 | 2971 | -0.84% |
| `small_pot_stove` | 10 | 1 | 1261 | 1257 | +0.32% |
| `kettle_electric_insulated` | 10 | 1 | 1533 | 1562 | -1.83% |
| `kettle_propane_40L` | 10 | 1 | 2712 | 2971 | -8.72% |

The last three rows are for information: at the default `k_eff_scale = 1` the liquid
stratifies (hot at the bottom, cooler at the top), so the surface, which loses most of the heat
by evaporation, is colder than the mean and the mean rises faster than in a perfectly mixed
liquid. The propane kettle, heated from a small footprint under a deep liquid, shows it most.
The well-mixed errors change by less than 0.4 percentage points between cell sizes of 40, 30
and 20 mm.

**Grid convergence** of the 40 L propane example, dx = 20, 10, 5 mm (`slow` test, about 6
minutes):

| Quantity | 20 mm | 10 mm | 5 mm | Order | Extrapolated | 5 mm error |
|---|---|---|---|---|---|---|
| Time to the mean-temperature target (s) | 2713.29 | 2712.13 | 2712.27 | not monotone | - | spread 1.2 s (0.04%) |
| Rise at `center`, t = 1500 s (K) | 48.7264 | 48.7001 | 48.6948 | 2.3 | 48.6934 | 0.003% |
| Rise at `near_wall`, t = 1500 s (K) | 56.1949 | 56.2302 | 56.2342 | 3.1 | 56.2347 | 0.001% |

The mean-temperature time to target is almost independent of the cell size, because the whole
run is dominated by the energy balance; it agrees to 0.04% over a factor of 4 in dx but is
not strictly monotone at that level (the differences are 1.2 s and then -0.14 s), so no observed order
can be given for it (the criterion "converges monotonically" is met by the local temperatures,
not by this integral quantity). The probe temperatures converge monotonically at second to third
order, and the 5 mm result is within 0.01% of the Richardson-extrapolated value (criterion: 2%).

**Energy balance.** `energy_balance_rel_error` of every example at 20 mm: 2.4e-16
(electric), 1.0e-14 (propane), 4.4e-16 (pot). Limit: 1e-6.

**Rule of thumb.** The 40 L propane example reaches 99 C in 45.2 minutes (band: 30-60;
ideal energy only: 38.0).

**Wall boiling.** A burner delivers about 200 kW/m^2 through a 0.2 m footprint. With natural
convection alone the base shell reached more than 2000 C; with nucleate boiling at the wall
(`h_film_liquid`, Rohsenow) the base shell node stays at about 160 C: the wetted surface is
about 10 K above the boiling point, and the rest is the conduction drop across the 4 mm steel
base. Liquid cells next to the base still exceed the boiling point by up to about 40 K at the end of
the heat-up (bulk boiling and boil-off are outside the model, requirement L4), and the
`near_wall` probe reads up to 106 C as the mean approaches 99 C. See `docs/physics/correlations.md`.

**Performance.** See `benchmarks/RESULTS.md`.

**To run the weekly job** (the `slow` tests), the CI needs a scheduled or
`workflow_dispatch` job with `pytest -m slow tests/validation`; the default job should use
`-m "not slow"`.
