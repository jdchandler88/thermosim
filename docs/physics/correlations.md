# Heat-transfer correlations (WP03)

Module: `thermosim.physics.correlations`. SI units, temperatures in K, pressures in Pa.
Heat-flux sign (D9): positive = heat leaving the surface, so `q = h (T_surface - T_ref)`.
All functions broadcast over NumPy arrays and never return NaN or negative values. Air
properties come from `properties.air_props` at the film temperature `(T_s + T_inf)/2`;
wort properties from the `wort_*` functions.

Ra = g beta |dT| L^3 / (nu alpha) for air (beta = 1/T_film). For |dT| < 1 mK the correlations
are evaluated at 1 mK and the natural-convection result is floored at 1 W/(m^2 K).

## Air side (outer surfaces and free surface)

| Function | Equation | Validity | Reference |
|---|---|---|---|
| `h_natural_vertical` | Nu = {0.825 + 0.387 Ra^(1/6) / [1 + (0.492/Pr)^(9/16)]^(8/27)}^2 | all Ra, all Pr | Churchill & Chu (1975); Incropera Eq. 9.26 |
| `h_natural_horizontal_up` (hot face up) | Nu = 0.54 Ra^(1/4) (Ra <= 1e7); 0.15 Ra^(1/3) (Ra > 1e7) | 1e4 < Ra < 1e11, Pr >= 0.7 | Lloyd & Moran (1974); Incropera Eqs. 9.30, 9.31 |
| `h_natural_horizontal_down` (hot face down) | Nu = 0.27 Ra^(1/4) | 1e5 < Ra < 1e10 | Lloyd & Moran (1974); Incropera Eq. 9.32 |
| cold faces | up and down correlations swap | | same |
| `h_forced_cylinder` | Nu = 0.3 + 0.62 Re^(1/2) Pr^(1/3) / [1 + (0.4/Pr)^(2/3)]^(1/4) * [1 + (Re/282000)^(5/8)]^(4/5) | Re Pr > 0.2 | Churchill & Bernstein (1977); Incropera Eq. 7.57 |
| `h_forced_plate` | Re <= 5e5: Nu = 0.664 Re^(1/2) Pr^(1/3); Re > 5e5: Nu = (0.037 Re^(4/5) - 871) Pr^(1/3) | 0.6 < Pr < 60 | Incropera Eqs. 7.30, 7.38 |
| `h_mixed` | h = (h_n^3 + h_f^3)^(1/3) | | Churchill (1977) blending, n = 3 |
| `h_radiation` | h_r = eps sigma (T_s^2 + T_inf^2)(T_s + T_inf) | grey, small body in large enclosure at T_inf | exact factor of eps sigma (Ts^4 - Ti^4)/(Ts - Ti) |

`h_outer(T_s, T_inf, face, char_length, wind, p, emissivity, diameter=None)`:

* `SIDE`: `h_mixed(h_natural_vertical(height = char_length), h_forced_cylinder(D = diameter))` + radiation;
* `DOWN`: `h_mixed(h_natural_horizontal_down, h_forced_plate)` + radiation;
* `UP`: `h_mixed(h_natural_horizontal_up, h_forced_plate)` + radiation;
* with `wind = 0` the forced term is dropped.

Large inputs (> 2000 elements, scalar `p`) take air properties from a 0.1 K film-temperature
grid (nearest node, < 0.02 % in k); 1e5 nodes take about 16 ms.

## Evaporation from an open surface

Chilton-Colburn analogy (Incropera Eq. 6.60): `h_m = h_conv / (rho_air cp_air Le^(2/3))`,
`Le = alpha / D_AB`, with air properties and the vapour diffusivity at the film temperature.
Vapour densities: ideal gas, `rho_v,s = p_sat(T_surf) / (R_v T_surf)` and
`rho_v,inf = RH p_sat(T_amb) / (R_v T_amb)`. `m = max(0, h_m (rho_v,s - rho_v,inf))`,
`q = m h_fg(T_surf)`.

`h_conv` for the free surface is the UP correlation with L = D/4. Water activity of the wort
(about 1 % vapour-pressure lowering) and Stefan-flow effects are neglected, which slightly
over-predicts.

Sanity check (see tests): water at 95 C, 20 C still air, RH 0.5, D = 0.3 m gives about
8.9 kW/m^2, within 25 % of the ASHRAE pool-evaporation equation
(`w = A (p_w - p_a)(0.089 + 0.0782 V) / Y`, ASHRAE HVAC Applications ch. 6, about 7.4 kW/m^2 at
V = 0). This is slightly above the "roughly 3-8 kW/m^2" range of the work-package text; the
test accepts 3-10 kW/m^2.

## Liquid-side film coefficient `h_film_liquid(T_wall, T_bulk, orientation, L, sg, p=101325)`

Wort properties at the film temperature; `Ra = g beta |dT| L^3 rho^2 cp / (mu k)`
(beta floored at 1e-6 /K because water's beta is negative below 4 C). Each of the two
temperatures is clipped to 0-120 C before averaging (the property fits stop there); the
temperature difference itself is the true one.

* Wall facing UP into the liquid (the base), hot wall: `Nu = 0.15 Ra^(1/3)`, L passed in
  (Incropera Eq. 9.31 form, applied for all Ra).
* Hot wall facing DOWN, or cold wall facing UP: `Nu = 0.27 Ra^(1/4)` (Eq. 9.32).
* Cold wall facing DOWN: as hot facing UP.
* SIDE: Churchill-Chu with Pr of the wort, L = fill height.
* The natural-convection value is clipped to [50, 5000] W/(m^2 K).

**Nucleate boiling at the wall (added in WP10).** A kettle base at 50-200 kW/m^2 does not
superheat by hundreds of kelvin: it runs (sub-cooled) nucleate boiling, with a wall superheat of
5-25 K above the local boiling point. For `T_wall > T_sat(p, sg)` (`wort_boiling_point`, so
altitude and gravity are included through the optional pressure `p`) the film coefficient is
the smooth maximum of the natural-convection value and a boiling coefficient:

* flux, Rohsenow (1952), Incropera Eq. 10.5:
  `q = mu_l h_fg [g (rho_l - rho_v)/sigma]^(1/2) [cp_l (T_wall - T_sat) / (C_sf h_fg Pr_l)]^3`,
  with `C_sf = 0.013` (water on polished stainless steel or copper, Incropera Table 10.1) and
  `n = 1`; saturated-liquid wort properties at `T_sat`, `sigma` from the IAPWS 1994 release,
  vapour density from the ideal-gas law. About 140 kW/m^2 at a 10 K superheat;
* capped at the critical heat flux (Zuber 1958, Incropera Eq. 10.7, about 1.1 MW/m^2);
* linearised to `h_b = q / (T_wall - T_bulk)` (zero where the wall is not hotter than the
  bulk), and combined as `h = (h_nc^4 + h_b^4)^(1/4)`. `q` grows with the cube of the superheat,
  so the combination is continuous with a continuous slope at `T_sat`.

`h_film_through_wall(T_node, T_bulk, orientation, L, sg, r_wall, p)` is the form used by the
model. The shell node temperature is the wall mid-plane value, so the wetted surface is colder
by the conduction drop `q * t/(2k)` (25 K for 4 mm stainless at 200 kW/m^2); the boiling
surface temperature is found from the series balance `h(T_w)(T_w - T_bulk) = (T_node - T_w)/r_wall`
by bisection (film properties frozen at `(T_sat + T_bulk)/2`), and `h` is evaluated there. Nodes
below the boiling point simply get `h_film_liquid` at the node temperature.

Scope (requirement L4 unchanged): this only keeps a heated wall from superheating unphysically.
Bulk boiling, vapour holdup, boil-off and film boiling are not modelled, and the liquid cells
next to a strongly heated wall can still exceed `T_sat` because the effective-conductivity
mixing closure carries the heat away more slowly than real boiling-driven mixing would.
`k_effective` is called with a hot-side temperature of at most `T_sat + 30 K` for the same
reason.

## Covered surface to headspace `h_headspace`

`h = h_cavity + h_evap`, with q = h (T_surface - T_lid).

* `h_cavity`: horizontal layer heated from below, gap `HEADSPACE_GAP_M = 0.10 m`,
  Nu = max(1, 0.069 Ra^(1/3) Pr^0.074) (Globe & Dropkin 1959; Incropera Eq. 9.49); Nu = 1 when
  the lid is warmer than the surface.
* `h_evap`: net vapour flux `h_m (a_w rho_v,sat(T_surface) - rho_v,sat(T_lid)) h_fg / (T_surface - T_lid)`,
  with h_m from the analogy applied to `h_cavity`. The lid is assumed to condense the saturated
  headspace at its own temperature; `a_w` is the wort's water activity from its boiling-point
  elevation (p / p_sat(T_boil)). Floored at 0; the dT -> 0 limit is the 1 mK secant.

Assumptions: well-mixed saturated gas, uniform lid temperature, dry-air properties, fixed gap.

## Effective conductivity `k_effective`

`k_eff = k_mol max(1, scale C Ra^(1/3))`, `Ra = g beta |T_hot - T_bulk| H^3 / (nu alpha)`,
wort properties at `T_bulk`, `H` = liquid depth, `C = K_EFF_C = 0.1`.

**This is a modelling closure, not a measured property.** It represents the mixing by
natural convection in the Phase 1 conduction-only model through a Rayleigh-Benard-type
scaling (Nu ~ Ra^(1/3)); the constant and `scale` are to be calibrated against measurements in
WP23. Applying `scale` to the Nusselt number (rather than to the whole product) keeps
`k_eff >= k_mol`.

## Contract change requests

1. `h_outer` has no cylinder diameter for the crossflow correlation. An optional trailing
   argument `diameter=None` was added (falls back to `char_length`); the caller should pass
   `Shell.outer_diameter`.
2. `h_headspace` has no headspace height. A fixed gap `HEADSPACE_GAP_M` is assumed; the
   signature should gain a `gap` (m) argument.
3. `k_effective`'s `scale` is applied to the Nusselt enhancement (see above).
