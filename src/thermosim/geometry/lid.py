"""Lumped lid and headspace (WP04)."""

from __future__ import annotations

import math
from typing import Any

from thermosim.core.errors import ScenarioError
from thermosim.core.types import LidSpec
from thermosim.physics import properties


def build_lid(
    scenario: Any, inner_radius: float, inner_height: float, fill_height: float
) -> LidSpec:
    """Lid specification from `scenario.lid` (and `scenario.insulation` if it covers the lid).

    Args:
        scenario: object with `lid` (mode, covered_fraction, material, thickness_m) and
            `insulation` (None or thickness_m, conductivity_W_mK, emissivity, covers).
        inner_radius: R, m.
        inner_height: H_v, m.
        fill_height: h, m.

    Returns:
        LidSpec. Without a lid every field is zero (the headspace is ambient air).
        With a lid: area = f pi R^2 (m^2), heat_capacity = rho cp t area (J/K),
        headspace_volume = pi R^2 (H_v - h) f (m^3).
    """
    lid = scenario.lid
    if lid.mode == "none":
        return LidSpec(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if lid.mode == "full":
        fraction = 1.0
    elif lid.mode == "partial":
        fraction = float(lid.covered_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ScenarioError("lid covered_fraction must be in (0, 1]")
    else:
        raise ScenarioError(f"unknown lid mode {lid.mode!r}")
    mat = properties.solid_props(lid.material)
    area = fraction * math.pi * inner_radius**2
    ins = scenario.insulation
    insulated = ins is not None and "lid" in ins.covers
    return LidSpec(
        present=True,
        covered_fraction=fraction,
        area=area,
        heat_capacity=mat.density * mat.specific_heat * float(lid.thickness_m) * area,
        emissivity=float(ins.emissivity) if insulated else mat.emissivity,
        insulation_r=float(ins.thickness_m) / float(ins.conductivity_W_mK) if insulated else 0.0,
        headspace_volume=math.pi
        * inner_radius**2
        * max(inner_height - fill_height, 0.0)
        * fraction,
    )
