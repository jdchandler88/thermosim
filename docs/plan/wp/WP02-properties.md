# WP02: Material and fluid properties

**Wave:** 1 · **Size:** M · **Depends on:** WP00

## Goal

Provide accurate, vectorized property functions for the solid materials, wort
and water (as a function of temperature and specific gravity), humid air, and
atmospheric pressure.

## Owns

- `src/thermosim/physics/properties.py` (it may split into private modules
  under `physics/_props_*.py`)
- `tests/physics/test_properties.py`

## Interfaces

Implement CONTRACTS §6 exactly.

## Requirements

1. **Valid range** is 273.15 K to 393.15 K (0–120 °C), and SG from 1.000 to
   1.130. Outside that range, clip the inputs and emit a single
   `warnings.warn` per call site (use `stacklevel`). Don't raise.
2. **Water (SG = 1.000)** must match IAPWS-95 reference values at 1 atm (the
   NIST Webbook) within these tolerances, from 0 to 100 °C:
   - density ±0.05%
   - specific heat ±0.3%
   - conductivity ±1%
   - viscosity ±2%
   - expansion coefficient ±3% for T ≥ 10 °C; ±0.2e-4 1/K absolute below
     that (it passes through 0 near 4 °C)

   Use explicit correlations (for example Kell 1975 for density) or cubic
   interpolation over an embedded table. **Embed the reference table in the
   test file**, with its source cited.
3. **Wort.** Treat it as a sucrose solution at the same °Plato.
   - Plato conversion: the standard cubic, °P = −616.868 + 1111.14·SG −
     630.272·SG² + 135.997·SG³, and its numerical inverse.
   - Density: `sg × water_density(T)` corrected for temperature using sucrose
     solution data, or simply `sg × ρ_water(T)`. Document the choice.
   - Specific heat, conductivity and viscosity: published sucrose-solution
     correlations (for example from the sugar-technology literature). Cite the
     source in the docstring.
   - Boiling-point elevation: from sucrose-solution data or Raoult's law with
     an activity correction. Expected size: about 0.3–1 K across 10–20 °P.
     Combine with pressure through Antoine or Clausius–Clapeyron for water.
4. **Boiling point.** `wort_boiling_point(1.000, 101325) = 373.12 K ± 0.05`.
   At 1600 m (≈ 83.5 kPa) water boils at about 95 °C ± 0.3.
5. **Saturation pressure.** Use Buck (1996) or the IAPWS-IF97 saturation line,
   within ±0.2% from 0 to 120 °C.
6. **Latent heat** within ±0.5% of the steam tables.
7. **Air.** Dry-air properties from 250 to 400 K are enough (the humidity
   effect on the convection properties is negligible). The ideal-gas density
   uses `p`. Viscosity uses Sutherland's law. Conductivity is a polynomial
   fit. Expansion is 1/T. Within ±2% of standard tables.
8. **`pressure_at_altitude`.** The ISA troposphere formula, giving 101325 Pa
   at 0 m.
9. **`SOLIDS`.** Room-temperature values with citations, for example:
   - stainless 304: ρ 8000, cp 500, k 16.2, ε 0.35 (dull)
   - aluminum: ρ 2700, cp 900, k 205, ε 0.10
   - copper: ρ 8960, cp 385, k 390, ε 0.07
   - carbon steel: ρ 7850, cp 490, k 45, ε 0.80

   `solid_props(obj)` also accepts any object with `density`,
   `specific_heat`, `conductivity` and `emissivity` attributes.
10. **Performance.** Each function must handle a 1e6-element array in under
    50 ms. No Python loops over elements.

## Acceptance criteria

- The reference-table tests pass at the tolerances above.
- Wort trends are monotonic and in the right direction: as SG rises, cp
  falls, viscosity rises and the boiling point rises.
- Scalar inputs return scalars (a 0-d float); array inputs return arrays of
  the same shape.
