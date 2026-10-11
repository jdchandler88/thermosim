# Validating ThermoSim

This page explains how we know the simulator is right, and how you check it
against a real heat-up of your own vessel. Parts marked **PLACEHOLDER** are
filled in when the end-to-end validation (WP10 Part B) lands.

## 1. Layers of evidence

| Layer | What it shows | Where | Status |
|---|---|---|---|
| Analytic references | Closed-form solutions of conduction problems (semi-infinite slab with a step or a convective surface, lumped Newton cooling, 3-D Gaussian pulse) that other code is checked against. Each satisfies its PDE (finite-difference test) and its limits. | `src/thermosim/validation/analytic.py`, `tests/validation/test_analytic.py` | Done |
| Lumped reference model | A 0-D model (one liquid, one shell, one lid temperature) that uses the same property and correlation functions and the same physics choices as the 3-D coefficient table. It matches exact Newton cooling to 1e-6 K. | `src/thermosim/validation/lumped.py`, `tests/validation/test_lumped.py` | Done |
| Solver against analytic solutions | The real solver on synthetic domains versus the references above. | `tests/validation/test_solver_vs_analytic.py` | **PLACEHOLDER** (Part B) |
| Full model against lumped | With `k_eff_scale = 100` (well mixed) the 3-D time-to-target is within 3% of the lumped model's. | `tests/validation/` | **PLACEHOLDER** (Part B) |
| Grid convergence | 40 L propane example at dx = 20, 10, 5 mm; observed order and Richardson-extrapolated time. | `tests/validation/` (`slow`) | **PLACEHOLDER** (Part B) |
| Energy conservation | `energy_balance_rel_error` < 1e-6 for every example. | `tests/validation/` | **PLACEHOLDER** (Part B) |
| Rules of thumb | 40 L from 15 to 99 C on a 60k BTU/hr burner at 35% efficiency takes 30-60 minutes (ideal 14 MJ at 6.2 kW is about 38 minutes, plus losses). | `tests/validation/` | Lumped model checked; 3-D **PLACEHOLDER** |
| A real heat-up | Your own measurement (section 3). | your logbook, section 5 | Yours to do |

### The lumped model and the quick estimate

`LumpedModel(scenario).run()` returns the time to the target, the stop reason
(`target_reached`, `plateau` or `max_time`) and the time series of the liquid,
shell and lid temperatures and the delivered power. `quick_estimate(scenario)`
returns just the time to target (or `None`); it uses coefficients refreshed
every 30 s and a loose tolerance and takes about 0.1 s.

Its simplifications, which you should keep in mind when comparing it to the 3-D
model: a perfectly mixed liquid; a single shell temperature (no conduction around
the wall); the target is the liquid mean, whatever probe the scenario names; the
heat source is applied at its steady actuator value (no gas-valve lag, no SSR
cycling); the base outer loss is zero while a burner fires (as in the 3-D model).

Reference values from the lumped model for the example scenarios (full power, no
schedule), next to the ideal hand estimate m cp dT / delivered power:

| Example | Lumped time to target | Hand estimate | Ratio |
|---|---|---|---|
| `kettle_propane_40L` | 54.3 min | 38.0 min | 1.43 |
| `kettle_electric_insulated` | 26.6 min | 24.3 min | 1.10 |
| `small_pot_stove` | 21.7 min | 19.5 min | 1.11 |

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

**PLACEHOLDER.** To be filled in when WP09 is merged:

- Solver versus analytic: maximum error for each case.
- Full model versus lumped (`k_eff_scale = 100`): error per example (target: under 3%);
  and for `k_eff_scale = 1`, as information.
- Grid convergence of the 40 L propane example at dx = 20, 10 and 5 mm: time to
  target, observed order, Richardson-extrapolated value, and the error at 5 mm
  (target: under 2%).
- Energy balance error per example (target: under 1e-6).
- The rule-of-thumb band result.
- Performance: see `benchmarks/RESULTS.md`.
