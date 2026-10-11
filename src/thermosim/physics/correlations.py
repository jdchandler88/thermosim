"""Heat-transfer correlations and effective conductivity (CONTRACTS.md section 7).

WP00 stub, implemented by WP03. Every function returns W/m^2 K (h_*) or W/m^2 (q_*,
positive = leaving the surface), and is vectorized.
"""

from __future__ import annotations

import numpy as np

from thermosim.core.types import F64, OuterFace

Num = float | F64


def h_natural_vertical(T_s: Num, T_inf: Num, height: Num, p: Num) -> Num:
    """Churchill-Chu natural convection on a vertical wall. T in K, height in m, p in Pa.
    Returns W/(m^2 K)."""
    raise NotImplementedError


def h_natural_horizontal_up(T_s: Num, T_inf: Num, L: Num, p: Num) -> Num:
    """Natural convection, hot surface facing up (hot-up/cold-down plate). L in m."""
    raise NotImplementedError


def h_natural_horizontal_down(T_s: Num, T_inf: Num, L: Num, p: Num) -> Num:
    """Natural convection, hot surface facing down. L in m."""
    raise NotImplementedError


def h_forced_cylinder(T_s: Num, T_inf: Num, D: Num, U: Num, p: Num) -> Num:
    """Churchill-Bernstein crossflow over a cylinder. D in m, U in m/s."""
    raise NotImplementedError


def h_forced_plate(T_s: Num, T_inf: Num, L: Num, U: Num, p: Num) -> Num:
    """Laminar/turbulent flat-plate forced convection. L in m, U in m/s."""
    raise NotImplementedError


def h_mixed(h_nat: Num, h_forced: Num) -> Num:
    """Mixed convection, (h_n^3 + h_f^3)^(1/3)."""
    raise NotImplementedError


def h_radiation(T_s: Num, T_inf: Num, emissivity: Num) -> Num:
    """Exact linearization eps*sigma*(Ts^2+Ti^2)(Ts+Ti). Returns W/(m^2 K)."""
    raise NotImplementedError


def h_outer(
    T_s: Num,
    T_inf: Num,
    face: OuterFace | np.ndarray,
    char_length: Num,
    wind: Num,
    p: Num,
    emissivity: Num,
) -> Num:
    """Convenience: picks the correlation for `face`, mixes with forced wind, adds radiation.
    Returns W/(m^2 K)."""
    raise NotImplementedError


def q_evaporation(T_surf: Num, T_amb: Num, rh: Num, p: Num, h_conv: Num) -> Num:
    """Evaporative heat flux via the Chilton-Colburn analogy, W/m^2; 0 if negative."""
    raise NotImplementedError


def evaporation_mass_flux(T_surf: Num, T_amb: Num, rh: Num, p: Num, h_conv: Num) -> Num:
    """Evaporative mass flux, kg/(m^2 s)."""
    raise NotImplementedError


def h_film_liquid(
    T_wall: Num, T_bulk: Num, orientation: OuterFace | np.ndarray, L: Num, sg: Num
) -> Num:
    """Liquid-side natural-convection film coefficient for wall -> liquid (Phase 1).
    orientation is the direction the WALL faces into the liquid: base -> UP, side -> SIDE.
    Returns W/(m^2 K)."""
    raise NotImplementedError


def h_headspace(T_liquid_surface: Num, T_lid: Num, sg: Num, p: Num) -> Num:
    """Covered surface <-> headspace coefficient, including condensation. W/(m^2 K)."""
    raise NotImplementedError


def k_effective(T_bulk: Num, T_hot: Num, depth: Num, sg: Num, scale: float = 1.0) -> Num:
    """Effective liquid conductivity, W/(m K), >= molecular k; Nu(Ra) based."""
    raise NotImplementedError
