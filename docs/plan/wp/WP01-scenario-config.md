# WP01: Scenario schema, units, TOML I/O, example scenarios

**Wave:** 1 · **Size:** M · **Depends on:** WP00

## Goal

Define the user-facing setup ("scenario") model: its validation, saving and
loading as TOML, unit conversion for display, and a set of example scenarios.
The GUI (WP08) generates its forms from this schema, so the metadata must be
complete.

## Owns

- `src/thermosim/config/**`
- `src/thermosim/scenarios/*.toml`
- `tests/config/**`

## Interfaces

Implement CONTRACTS §5 exactly.

## Requirements

1. **Every numeric field** has `title`, `description`, sensible `ge`/`le`
   bounds and `json_schema_extra={"quantity": ...}`. Every model has
   `model_config = ConfigDict(extra="forbid")`.
2. **Cross-field validation** (`model_validator`). Raise `ScenarioError`
   wrapping pydantic's error with a readable message:
   - Exactly one of `liquid.volume_m3` and `liquid.fill_height_m` is set.
   - The fill height (given, or computed as V/(πR²)) is ≤ `vessel.inner_height_m`.
   - `lid.mode == "partial"` requires 0 < `covered_fraction` < 1. `"full"`
     forces 1.0 and `"none"` ignores the value.
   - Burner: `footprint_outer_diameter_m` ≤ the outer diameter of the base,
     and the inner diameter is less than the outer.
   - Element: at least 2 path points, and every point lies inside the inner
     cylinder (r ≤ R − diameter/2, 0 ≤ z ≤ inner height). Whether the element
     is above the liquid is checked by WP04, not here.
   - Probe names are unique and none is called `"mean"`. `target.probe` is
     `"mean"` or an existing probe name.
   - Each probe position is inside the inner cylinder and below the fill
     height.
   - `power_schedule` times are sorted and the first is at `t_s == 0`.
     Commands are in [0, 1].
   - `insulation.covers` containing `"lid"` requires a lid.
   - Material keys exist in `physics.properties.SOLIDS`. Check against the
     hard-coded key list `stainless_304, aluminum, copper, carbon_steel`, so
     there is no import dependency on WP02.
   - `simulation.cell_size_m` is between 0.002 and 0.05, and the inner
     diameter is at least 6 cells.
   - `model.liquid_transport == "flow"` raises `ScenarioError("flow mode is
     not available until Phase 2")`.
3. **Units** (`units.py`). US is the default unit system in the GUI and CLI
   (REQUIREMENTS B7). The metric display uses °C, L, kW, mm, m/s, min,
   kPa. US uses °F, gal (US), BTU/hr, in, mph, min. Temperature differences
   convert by the scale factor only. Round-tripping `to_display` and
   `from_display` must agree to within 1e-12 relative. Provide
   `unit_label()` for every quantity.
4. **TOML I/O.**
   - `save` then `load` round-trips exactly.
   - Tuples are written as arrays, and `None` fields are omitted.
   - The file starts with a comment header saying units are SI and
     temperatures are in kelvin.
   - Errors name the file, the field path and the problem.
5. **Example scenarios** (package data). Give realistic values and comments:
   - `kettle_propane_40L.toml`: 15 gal (57 L) stainless kettle, 40 L water at
     15 °C, a 60,000 BTU/hr burner at efficiency 0.35, no lid, 10 °C and
     2 m/s wind, target 100 °C minus 1 K at the mean.
   - `kettle_electric_insulated.toml`: 40 L kettle with a 5500 W U-shaped
     element 5 cm above the base, 25 L wort at SG 1.050, partial lid at 0.5,
     side insulation, target 99 °C at a probe on the side wall 10 cm up.
   - `small_pot_stove.toml`: 8 L aluminum pot, 5 L water, 2500 W burner at
     efficiency 0.5, full lid, indoors at 21 °C, target 90 °C.
6. Provide `defaults.py: default_scenario() -> Scenario`, which is the first
   example.

## Acceptance criteria

- Every validation rule has a passing test and a failing test.
- Every example loads, round-trips, and passes validation.
- Unit round trips are property-tested with `hypothesis`.
- `quantity_of(Vessel, "inner_diameter_m") == "length"`, and every numeric
  field across all models has a quantity (a test walks the schema).

## Notes

- The GUI reads `Scenario.model_json_schema()` together with `quantity_of`.
  Keep the field order meaningful, because it becomes the form order.
