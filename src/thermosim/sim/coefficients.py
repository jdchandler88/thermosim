"""Physics coefficients for the solver, assembled from properties and correlations (WP09).

`build_coefficients` fills a `StepCoefficients` from the current state; it is called every
`simulation.coeff_refresh_s` of simulated time. The heat-source terms (`q_cells`, `q_shell`)
change every step and are filled by `apply_source`.

Modelling choices (all SI, temperatures in K):

* Liquid volumetric heat capacity comes from a 1-D table of ``rho(T, sg) * cp(T, sg)``
  interpolated linearly (0.1 K spacing), because evaluating the property correlations on every
  cell of a fine grid would dominate the run time.
* ``k_liquid`` is uniform: `k_effective` at the mean liquid temperature with the temperature of
  the heat source as the hot side (burner: weighted mean of the shell nodes under the
  footprint; element: weighted mean of the element cells). When the source is off, the larger
  of the wetted-area-weighted mean temperature of the wetted shell and the mean liquid
  temperature is used.
* Heated wall. ``h_film_liquid`` includes nucleate boiling above the local boiling point
  (Rohsenow), so a heated base stays within a few kelvin of saturation instead of
  superheating by hundreds of kelvin; the hot-side temperature handed to ``k_effective`` is
  capped at the boiling point + 30 K. Bulk boiling and boil-off are not modelled (L4).
* Outer shell coefficient. For insulated nodes the outer surface temperature is estimated by one
  fixed-point step with the previous coefficient. **Burner setups: the base nodes get
  ``h_out_shell = 0`` while the burner output is above zero**, because the burner efficiency
  already accounts for the losses around the flame and the hot gas under the base.
* The dry inner wall uses `h_headspace` against the headspace temperature when there is a lid;
  without a lid it uses vertical-wall natural convection to ambient plus half of the
  radiation coefficient (the inner wall sees the sky only partly).
* The gap passed to `h_headspace` is ``inner_height - fill_height`` (at least 1 cm).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from thermosim.config.schema import Burner, Scenario
from thermosim.core.protocols import SourcePower
from thermosim.core.types import (
    F64,
    I64,
    Domain,
    HeatDeposition,
    OuterFace,
    ShellKind,
    StepCoefficients,
    ThermalState,
)
from thermosim.physics import correlations as cr
from thermosim.physics import properties as pr

_TABLE_T_MIN = 273.2
_TABLE_T_MAX = 393.0
_TABLE_DT = 0.1
_SURFACE_EMISSIVITY = 0.95  # free liquid surface
_H_OUT_GUESS = 10.0  # W/(m^2 K), first estimate of the outer coefficient
_MIN_GAP_M = 0.01
_SUPERHEAT_CAP_K = 30.0  # K above the boiling point, cap of the k_effective hot side
_AIR_T_MAX = 449.0  # air property fits are valid to 450 K


def _water_T(T: F64 | float) -> F64 | float:
    """Clip a temperature to the validity range of the water/wort property fits (K).

    Shell nodes under a burner can exceed 393 K; their film properties are evaluated at the
    clipped value (the film temperature is only ever half-way to the liquid anyway).
    """
    return np.clip(T, 273.2, 392.9)  # type: ignore[no-any-return]


@dataclass
class CoefficientCache:
    """Static data and previous values reused by every refresh."""

    p: float
    sg: float
    T_sat: float
    T_amb: float
    wind: float
    rh: float
    k_eff_scale: float
    burner: bool
    deposition: HeatDeposition
    table_T: F64
    table_rho_c: F64
    liq_flat: I64
    liq_volume: F64
    liq_volume_sum: float
    gap: float
    wall_emissivity_inner: float
    face: I64
    base_mask: np.ndarray
    row_orient: I64
    row_length: F64
    row_rw: F64
    has_cell_weights: bool
    has_shell_weights: bool
    src_shell_idx: I64
    src_shell_w: F64
    src_cell_idx: I64
    src_cell_w: F64
    wetted_shell: I64
    wetted_w: F64
    h_out_prev: F64
    h_lid_prev: float = _H_OUT_GUESS
    coeffs: StepCoefficients | None = None
    h_out_full: F64 = field(default_factory=lambda: np.zeros(0))
    h_out_burner: F64 = field(default_factory=lambda: np.zeros(0))
    refresh_count: int = 0


def make_cache(
    domain: Domain, scenario: Scenario, deposition: HeatDeposition, vessel_emissivity: float
) -> CoefficientCache:
    """Precompute everything that does not change during a run."""
    grid, shell = domain.grid, domain.shell
    sg = float(scenario.liquid.specific_gravity)
    atm = scenario.atmosphere
    liq_flat = np.flatnonzero(grid.is_liquid.reshape(-1)).astype(np.int64)
    table_T = np.arange(_TABLE_T_MIN, _TABLE_T_MAX + _TABLE_DT, _TABLE_DT)
    # Liquid mass per cell is fixed at the fill temperature (the grid does not expand), so the
    # volumetric heat capacity is rho(T0) cp(T): the batch's heat capacity is m cp(T).
    rho0 = float(pr.wort_density(float(scenario.liquid.initial_temp_K), sg))
    table = rho0 * np.asarray(pr.wort_specific_heat(table_T, sg))
    D = 2.0 * domain.inner_radius
    base = shell.kind == int(ShellKind.BASE)
    row_base = base[shell.couple_shell]
    row_orient = np.where(row_base, int(OuterFace.UP), int(OuterFace.SIDE)).astype(np.int64)
    row_length = np.where(row_base, D / 4.0, domain.fill_height).astype(np.float64)
    cw = np.asarray(deposition.cell_weights).reshape(-1)
    sw = np.asarray(deposition.shell_weights)
    src_cell_idx = np.flatnonzero(cw > 0.0).astype(np.int64)
    src_shell_idx = np.flatnonzero(sw > 0.0).astype(np.int64)
    wetted = np.flatnonzero(shell.wetted_area > 0.0).astype(np.int64)
    p = float(pr.pressure_at_altitude(atm.altitude_m))
    return CoefficientCache(
        p=p,
        sg=sg,
        T_sat=float(pr.wort_boiling_point(sg, p)),
        T_amb=float(atm.ambient_temp_K),
        wind=float(atm.wind_speed_m_s),
        rh=float(atm.relative_humidity),
        k_eff_scale=float(scenario.model.k_eff_scale),
        burner=isinstance(scenario.heat_source, Burner),
        deposition=deposition,
        table_T=table_T,
        table_rho_c=table,
        liq_flat=liq_flat,
        liq_volume=np.ascontiguousarray(grid.volume.reshape(-1)[liq_flat], dtype=np.float64),
        liq_volume_sum=float(grid.volume.reshape(-1)[liq_flat].sum()),
        gap=max(domain.inner_height - domain.fill_height, _MIN_GAP_M),
        wall_emissivity_inner=vessel_emissivity,
        face=np.asarray(shell.outer_face, dtype=np.int64),
        base_mask=base,
        row_orient=row_orient,
        row_length=row_length,
        row_rw=np.asarray(
            shell.thickness[shell.couple_shell] / (2.0 * shell.k[shell.couple_shell]),
            dtype=np.float64,
        ),
        has_cell_weights=bool(src_cell_idx.size),
        has_shell_weights=bool(src_shell_idx.size),
        src_shell_idx=src_shell_idx,
        src_shell_w=sw[src_shell_idx].astype(np.float64),
        src_cell_idx=src_cell_idx,
        src_cell_w=cw[src_cell_idx].astype(np.float64),
        wetted_shell=wetted,
        wetted_w=np.asarray(shell.wetted_area)[wetted].astype(np.float64),
        h_out_prev=np.full(shell.n, _H_OUT_GUESS),
    )


def mean_liquid_T(domain: Domain, T_liquid: F64) -> float:
    """Volume-weighted mean liquid temperature, K."""
    vol = domain.grid.volume
    return float(np.sum(T_liquid * vol) / np.sum(vol))


def _source_temperature(
    cache: CoefficientCache, state: ThermalState, on: bool, t_mean: float
) -> float:
    """Hot-side temperature for k_effective, K."""
    if on:
        if cache.has_shell_weights:
            w = cache.src_shell_w
            return float(np.sum(w * state.T_shell[cache.src_shell_idx]) / np.sum(w))
        if cache.has_cell_weights:
            w = cache.src_cell_w
            return float(np.sum(w * state.T_liquid.reshape(-1)[cache.src_cell_idx]) / np.sum(w))
    if cache.wetted_shell.size:
        w = cache.wetted_w
        wet = float(np.sum(w * state.T_shell[cache.wetted_shell]) / np.sum(w))
        return max(wet, t_mean)
    return t_mean


def build_coefficients(
    domain: Domain,
    state: ThermalState,
    scenario: Scenario,
    source_output: SourcePower | None,
    cache: CoefficientCache,
) -> StepCoefficients:
    """Refresh every physics coefficient from the current state (see the module docstring).

    The same `StepCoefficients` object is updated in place on every call after the first.
    """
    grid, shell, lid = domain.grid, domain.shell, domain.lid
    p, sg, T_amb = cache.p, cache.sg, cache.T_amb
    D = 2.0 * domain.inner_radius
    T_flat = state.T_liquid.reshape(-1)
    T_liq = T_flat[cache.liq_flat]
    T_mean = float(np.dot(T_liq, cache.liq_volume)) / cache.liq_volume_sum
    on = source_output is not None and source_output.delivered_W > 0.0

    coeffs = cache.coeffs
    if coeffs is None:
        shape = grid.shape
        n_s = len(grid.surface_cells)
        coeffs = StepCoefficients(
            T_amb=T_amb,
            k_liquid=np.zeros(shape),
            rho_c_liquid=np.zeros(shape),
            h_film=np.zeros(len(shell.couple_shell)),
            h_out_shell=np.zeros(shell.n),
            h_dry=np.zeros(shell.n),
            h_surface_open=np.zeros(n_s),
            q_evap_open=np.zeros(n_s),
            h_surface_covered=np.zeros(n_s),
            h_lid_inner=0.0,
            h_lid_outer=0.0,
            q_cells=np.zeros(shape),
            q_shell=np.zeros(shell.n),
        )
        cache.coeffs = coeffs
        cache.h_out_full = np.zeros(shell.n)
        cache.h_out_burner = np.zeros(shell.n)

    # --- liquid
    coeffs.rho_c_liquid.reshape(-1)[cache.liq_flat] = np.interp(
        T_liq, cache.table_T, cache.table_rho_c
    )
    # The hot-side temperature of the mixing closure cannot exceed the boiling point by more
    # than the nucleate-boiling superheat: the wall is clamped by boiling (h_film_liquid).
    T_src = min(_source_temperature(cache, state, on, T_mean), cache.T_sat + _SUPERHEAT_CAP_K)
    k_eff = float(
        cr.k_effective(_water_T(T_mean), T_src, domain.fill_height, sg, cache.k_eff_scale)
    )
    coeffs.k_liquid.fill(k_eff)

    # --- liquid-side film (nucleate boiling acts on the wetted surface, which is colder than
    # the shell node by the half-wall conduction drop, hence the series form)
    if len(cache.row_orient):
        coeffs.h_film[:] = cr.h_film_through_wall(
            state.T_shell[shell.couple_shell],
            T_flat[shell.couple_cell],
            cache.row_orient,
            cache.row_length,
            sg,
            cache.row_rw,
            p,
        )

    # --- outer shell
    T_s = state.T_shell
    r_ins = shell.insulation_r
    r_out = 1.0 / np.maximum(cache.h_out_prev, 1e-6)
    T_outer = np.where(r_ins > 0.0, T_amb + (T_s - T_amb) * r_out / (r_ins + r_out), T_s)
    h_out = np.asarray(
        cr.h_outer(
            np.minimum(T_outer, _AIR_T_MAX),
            T_amb,
            cache.face,
            shell.char_length,
            cache.wind,
            p,
            shell.emissivity,
            diameter=shell.outer_diameter,
        )
    )
    cache.h_out_prev[:] = h_out
    cache.h_out_full[:] = h_out
    cache.h_out_burner[:] = h_out
    if cache.burner:
        cache.h_out_burner[cache.base_mask] = 0.0
    coeffs.h_out_shell = cache.h_out_burner if (cache.burner and on) else cache.h_out_full

    # --- dry inner wall
    if lid.present:
        coeffs.h_dry[:] = cr.h_headspace(
            _water_T(T_s), _water_T(state.T_headspace), sg, p, cache.gap
        )
    else:
        T_dry = np.minimum(T_s, _AIR_T_MAX)
        h_nat = cr.h_natural_vertical(T_dry, T_amb, cache.gap, p)
        h_rad = cr.h_radiation(T_dry, T_amb, cache.wall_emissivity_inner)
        coeffs.h_dry[:] = np.asarray(h_nat) + 0.5 * np.asarray(h_rad)

    # --- free surface
    surf = grid.surface_cells
    if len(surf):
        T_surf = T_flat[surf]
        if (not lid.present) or lid.covered_fraction < 1.0:
            h_open = np.asarray(
                cr.h_outer(T_surf, T_amb, OuterFace.UP, D / 4.0, cache.wind, p, _SURFACE_EMISSIVITY)
            )
            coeffs.h_surface_open[:] = h_open
            h_conv = h_open - np.asarray(cr.h_radiation(T_surf, T_amb, _SURFACE_EMISSIVITY))
            coeffs.q_evap_open[:] = cr.q_evaporation(T_surf, T_amb, cache.rh, p, h_conv)
        if lid.present:
            coeffs.h_surface_covered[:] = cr.h_headspace(
                _water_T(T_surf), _water_T(state.T_lid), sg, p, cache.gap
            )
    if lid.present:
        if len(surf):
            area = grid.surface_area
            T_surf_mean = float(np.sum(T_flat[surf] * area) / np.sum(area))
        else:
            T_surf_mean = T_mean
        coeffs.h_lid_inner = float(
            cr.h_headspace(_water_T(T_surf_mean), _water_T(state.T_lid), sg, p, cache.gap)
        )
        r_l = lid.insulation_r
        r_o = 1.0 / max(cache.h_lid_prev, 1e-6)
        T_lid_outer = (
            T_amb + (state.T_lid - T_amb) * r_o / (r_l + r_o) if r_l > 0.0 else state.T_lid
        )
        h_lo = float(
            cr.h_outer(T_lid_outer, T_amb, OuterFace.UP, D / 4.0, cache.wind, p, lid.emissivity)
        )
        cache.h_lid_prev = h_lo
        coeffs.h_lid_outer = h_lo

    if source_output is not None:
        apply_source(coeffs, cache, source_output)
    cache.refresh_count += 1
    return coeffs


def apply_source(coeffs: StepCoefficients, cache: CoefficientCache, sp: SourcePower) -> None:
    """Set the per-step source terms (and the burner base-loss switch) from `sp`."""
    P = float(sp.delivered_W)
    if cache.has_cell_weights:
        np.multiply(cache.deposition.cell_weights, P, out=coeffs.q_cells)
    if cache.has_shell_weights:
        np.multiply(cache.deposition.shell_weights, P, out=coeffs.q_shell)
    if cache.burner:
        coeffs.h_out_shell = cache.h_out_burner if P > 0.0 else cache.h_out_full
