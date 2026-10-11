# WP03: Heat-transfer correlations and effective conductivity

**Wave:** 1 · **Size:** M · **Depends on:** WP00. It uses WP02's *function
API*; until WP02 merges, test with `thermosim.testing.synthetic.ConstantProps`.

## Goal

Provide every surface heat-transfer coefficient, evaporation flux, and the
effective conductivity of the liquid. Together these decide how accurate the
Phase 1 model is.

## Owns

- `src/thermosim/physics/correlations.py`
- `tests/physics/test_correlations.py`
- `docs/physics/correlations.md`: a derivation and reference table for every
  correlation

## Interfaces

Implement CONTRACTS §7 exactly. Get properties by calling
`thermosim.physics.properties`. Evaluate film properties at the film
temperature (T_s + T_∞)/2. Signs follow D9.

## Requirements

1. **External natural convection (air):**
   - Vertical wall: Churchill & Chu (1975), the full-range equation.
   - Horizontal plates (Lloyd & Moran; Incropera Table 9.x):
     - hot surface facing up: Nu = 0.54 Ra^¼ for 1e4 < Ra ≤ 1e7, and
       0.15 Ra^⅓ above that;
     - hot surface facing down: Nu = 0.27 Ra^¼ (or Raithby & Hollands).
     - The characteristic length L = A/P. Callers pass L.
   - The "cold" cases (surface colder than the air) swap the up and down
     correlations.
   - For |ΔT| < 1e-3 K, return the ΔT → 0 limit or a small floor (for example
     1 W/m² K). Never NaN.
2. **Forced convection:**
   - Cylinder in crossflow: Churchill & Bernstein (1977).
   - Flat plate in parallel flow: laminar 0.664 Re^½ Pr^⅓, or mixed laminar
     and turbulent above Re 5e5.
   - **Mixing natural and forced:** `h_mixed`, (h_n³ + h_f³)^⅓.
3. **Radiation.** The exact linearization ε σ (T_s² + T_∞²)(T_s + T_∞), with
   surroundings at T_amb.
4. **`h_outer`** dispatches on `OuterFace`:
   - SIDE: vertical natural convection mixed with a cylinder in crossflow;
   - DOWN: horizontal plate facing down mixed with a plate in parallel flow;
   - UP: horizontal plate facing up mixed with a plate in parallel flow;
   - then adds radiation.

   Vectorized over node arrays.
5. **Evaporation from an open surface.** Use the Chilton–Colburn heat and mass
   transfer analogy:
   - h_m = h_conv / (ρ_air cp_air Le^(2/3)), with Le = α/D_AB.
   - Vapor densities from saturation pressure at T_surf, and RH × saturation
     pressure at T_amb (ideal gas).
   - ṁ = h_m (ρ_v,s − ρ_v,∞), floored at 0; q = ṁ · h_fg(T_surf).
   - `h_conv` for the free surface is the UP correlation, with L = D/4.

   Sanity check: water at 95 °C, still air at 20 °C, RH 0.5 gives
   roughly 3–8 kW/m². Cite the comparison source in the test.
6. **Liquid-side film coefficient (`h_film_liquid`).** Natural convection in
   the liquid, using wort properties:
   - base (wall faces UP into the liquid, heating from below): Nu =
     0.15 Ra^⅓, with L = the base's characteristic length (passed in);
   - side: Churchill & Chu, with L = the fill height.

   A wall *colder* than the liquid uses the reversed correlation. The result
   is clipped to [50, 5000] W/m² K. This is subcooled natural convection; no
   nucleate boiling, by requirement L4. Document that above about 95 °C, real
   coefficients rise because of subcooled boiling.
7. **Covered surface to headspace (`h_headspace`).** A full lid gives a
   saturated headspace: evaporation from the surface and condensation on the
   lid carry heat. Model it as an enclosed-cavity natural convection
   coefficient plus an evaporation–condensation term from the
   saturation-density difference between T_surface and T_lid, using the same
   analogy. Document the assumptions.
8. **Effective conductivity (`k_effective`).**
   - k_eff = k_mol · max(1, Nu), with Nu = C · Ra^⅓.
   - Ra = g β |T_hot − T_bulk| H³ / (ν α), using wort properties at T_bulk,
     where H is the depth.
   - Default C = 0.1, a Rayleigh–Bénard-type scaling.
   - Multiply by `scale`. Return a scalar or an array.

   **Document clearly** that this is a modeling closure to be calibrated in
   WP23 (Phase 2). Expose `K_EFF_C` as a module constant.
9. No function may return NaN or a negative coefficient for any input in the
   valid ranges. Add a hypothesis test over the ranges.

## Acceptance criteria

- Each correlation matches a worked textbook example (Incropera or Çengel; cite
  the example number) within 5%.
- The evaporation sanity check passes.
- Every function is vectorized; a 1e5-node `h_outer` evaluation takes under
  20 ms.
- `docs/physics/correlations.md` lists each equation, its validity range, and
  its reference.
