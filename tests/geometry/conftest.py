"""Fixtures for the geometry tests: scenario-shaped objects and fixed solid properties."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from thermosim.physics import properties
from thermosim.physics.properties import SolidProps

STAINLESS = SolidProps("stainless_304", 8000.0, 500.0, 16.2, 0.35)
ALUMINUM = SolidProps("aluminum", 2700.0, 900.0, 167.0, 0.1)


def _fake_solid_props(material: Any) -> SolidProps:
    if isinstance(material, str):
        return {"stainless_304": STAINLESS, "aluminum": ALUMINUM}[material]
    return SolidProps(
        "custom",
        material.density,
        material.specific_heat,
        material.conductivity,
        material.emissivity,
    )


@pytest.fixture(autouse=True)
def _patch_solid_props(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(properties, "solid_props", _fake_solid_props)


ScenarioFactory = Callable[..., Any]


@pytest.fixture
def make_scenario() -> ScenarioFactory:
    """Factory of Scenario-shaped SimpleNamespace objects (SI units)."""

    def factory(
        *,
        diameter: float = 0.30,
        height: float = 0.40,
        fill_height: float | None = None,
        volume: float | None = None,
        wall: float = 0.002,
        base: float = 0.003,
        dx: float = 0.01,
        material: Any = "stainless_304",
        lid_mode: str = "none",
        lid_fraction: float = 1.0,
        insulation: Any = None,
        probes: list[tuple[str, tuple[float, float, float]]] | None = None,
    ) -> Any:
        if fill_height is None and volume is None:
            fill_height = 0.8 * height
        if probes is None:
            probes = [("center", (0.0, 0.0, 0.5 * (fill_height or 0.1)))]
        return SimpleNamespace(
            vessel=SimpleNamespace(
                shape="cylinder",
                inner_diameter_m=diameter,
                inner_height_m=height,
                wall_thickness_m=wall,
                base_thickness_m=base,
                material=material,
            ),
            lid=SimpleNamespace(
                mode=lid_mode,
                covered_fraction=lid_fraction,
                material="stainless_304",
                thickness_m=0.001,
            ),
            insulation=insulation,
            liquid=SimpleNamespace(volume_m3=volume, fill_height_m=fill_height),
            probes=[SimpleNamespace(name=n, position_m=p) for n, p in probes],
            simulation=SimpleNamespace(cell_size_m=dx),
        )

    return factory
