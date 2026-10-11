"""Heat-source deposition and power conversion (WP05).

Uses the synthetic box domain for exact geometry and real domains built from the example
scenarios (with a coarse cell size for speed).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from thermosim.config.io import example_scenarios
from thermosim.config.schema import Element as ElementCfg
from thermosim.config.schema import Scenario
from thermosim.core.errors import GeometryError
from thermosim.core.types import Domain, ShellKind
from thermosim.geometry.build import build_domain
from thermosim.sources.actuators import VariableActuator
from thermosim.sources.heat_sources import (
    BurnerSource,
    ElementSource,
    deposition_for_burner,
    deposition_for_element,
)
from thermosim.testing.synthetic import box_domain

DX = 0.01


def _burner_cfg(outer: float, inner: float = 0.0, footprint: str = "disk") -> SimpleNamespace:
    return SimpleNamespace(
        footprint=footprint,
        footprint_outer_diameter_m=outer,
        footprint_inner_diameter_m=inner,
    )


def _element_cfg(path: list[tuple[float, float, float]]) -> ElementCfg:
    return ElementCfg(rated_power_W=5000.0, path_m=path)


def _cell_scenario_with_liquid(name: str, cell_size: float, **liquid_update: object) -> Scenario:
    scenario = example_scenarios()[name]
    return scenario.model_copy(
        update={
            "simulation": scenario.simulation.model_copy(update={"cell_size_m": cell_size}),
            "liquid": scenario.liquid.model_copy(update=liquid_update),
        }
    )


def _real_domain(name: str, cell_size: float, **liquid_update: object) -> Domain:
    return build_domain(_cell_scenario_with_liquid(name, cell_size, **liquid_update))


def _base_radius(domain: Domain) -> tuple[np.ndarray, np.ndarray]:
    base = np.flatnonzero(domain.shell.kind == ShellKind.BASE)
    r = np.hypot(domain.shell.position[base, 0], domain.shell.position[base, 1])
    return base, r


# ----------------------------------------------------------------- burner
def test_burner_disk_weights_are_normalized_and_radius_bounded() -> None:
    domain = box_domain(40, 40, 4, DX)
    D = 0.12
    dep = deposition_for_burner(_burner_cfg(D), domain)

    assert dep.shell_weights.min() >= 0.0
    assert dep.shell_weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert not dep.cell_weights.any()

    base, r = _base_radius(domain)
    weights = dep.shell_weights
    side = np.flatnonzero(domain.shell.kind == ShellKind.SIDE)
    assert not weights[side].any()

    bound = D / 2 + DX * math.sqrt(2)
    assert not weights[base[r > bound]].any()
    # Nodes whose whole square lies inside the disk all get weight.
    interior = r < D / 2 - DX / math.sqrt(2)
    assert np.all(weights[base[interior]] > 0.0)


def test_burner_ring_leaves_center_empty() -> None:
    domain = box_domain(40, 40, 4, DX)
    dep = deposition_for_burner(_burner_cfg(0.12, 0.06, "ring"), domain)
    base, r = _base_radius(domain)
    weights = dep.shell_weights

    assert dep.shell_weights.sum() == pytest.approx(1.0, abs=1e-12)
    hole = r < 0.03 - DX / math.sqrt(2)
    assert not weights[base[hole]].any()
    ring = (r > 0.03 + DX / math.sqrt(2)) & (r < 0.06 - DX / math.sqrt(2))
    assert np.all(weights[base[ring]] > 0.0)
    assert not weights[base[r > 0.06 + DX * math.sqrt(2)]].any()


def test_burner_footprint_over_no_base_node_raises() -> None:
    domain = box_domain(10, 10, 4, DX)
    with pytest.raises(GeometryError):
        deposition_for_burner(_burner_cfg(1.0e-4), domain)


def test_burner_on_real_domain_deposits_under_footprint_only() -> None:
    domain = _real_domain("kettle_propane_40L", 0.02)
    D = float(example_scenarios()["kettle_propane_40L"].heat_source.footprint_outer_diameter_m)
    dep = deposition_for_burner(_burner_cfg(D), domain)

    assert dep.shell_weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert dep.shell_weights.min() >= 0.0
    base, r = _base_radius(domain)
    assert not dep.shell_weights[base[r > D / 2 + 0.02 * math.sqrt(2)]].any()
    assert dep.shell_weights[base].sum() == pytest.approx(1.0, abs=1e-12)


# ----------------------------------------------------------------- element
def test_element_along_x_axis_deposits_only_in_its_layer() -> None:
    domain = box_domain(12, 10, 8, DX)
    k = math.floor(0.05 / DX)
    dep = deposition_for_element(_element_cfg([(-0.05, 0.0, 0.05), (0.05, 0.0, 0.05)]), domain)
    w = dep.cell_weights

    assert w.min() >= 0.0
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert not dep.shell_weights.any()
    off_layer = np.ones(w.shape[2], dtype=bool)
    off_layer[k] = False
    assert not w[:, :, off_layer].any()
    # The path covers cells i = 1..10 (x from -0.05 to 0.05); each gets 1/10 of the heat.
    for i in range(1, 11):
        assert w[i].sum() == pytest.approx(0.1, abs=1e-9)
    assert w[0].sum() == 0.0
    assert w[11].sum() == 0.0


def test_element_45_degree_path_weights_proportional_to_length() -> None:
    domain = box_domain(20, 20, 12, DX)
    a = np.array([-0.03, 0.0, 0.02])
    b = np.array([0.03, 0.0, 0.08])
    dep = deposition_for_element(_element_cfg([tuple(a), tuple(b)]), domain)
    w = dep.cell_weights

    # Reference: exact-ish per-cell length from a very fine sampling of the segment.
    length = float(np.linalg.norm(b - a))
    n_fine = 400_000
    s = (np.arange(n_fine) + 0.5) / n_fine
    pts = a + s[:, None] * (b - a)
    grid = domain.grid
    ijk = np.floor((pts - grid.origin) / DX).astype(int)
    ref = np.zeros(grid.shape)
    np.add.at(ref, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1.0 / n_fine)

    n_samples = math.ceil(length / (DX / 16) - 1e-9)
    assert np.max(np.abs(w - ref)) <= 1.5 / n_samples
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_element_above_liquid_raises() -> None:
    domain = box_domain(10, 10, 4, DX)  # fill height 0.04 m
    with pytest.raises(GeometryError, match="above the liquid"):
        deposition_for_element(_element_cfg([(-0.02, 0.0, 0.06), (0.02, 0.0, 0.06)]), domain)


def test_element_partly_above_liquid_raises() -> None:
    domain = box_domain(10, 10, 4, DX)
    with pytest.raises(GeometryError, match=r"element segment at z=0\.0\d\d m"):
        deposition_for_element(_element_cfg([(-0.03, 0.0, 0.02), (0.03, 0.0, 0.06)]), domain)


def test_element_on_real_domain_bottom_leg_in_expected_layer() -> None:
    domain = _real_domain("kettle_electric_insulated", 0.02)
    k = math.floor(0.05 / 0.02)
    dep = deposition_for_element(_element_cfg([(-0.10, 0.0, 0.05), (0.10, 0.0, 0.05)]), domain)
    w = dep.cell_weights
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    off_layer = np.ones(w.shape[2], dtype=bool)
    off_layer[k] = False
    assert not w[:, :, off_layer].any()
    assert np.all(domain.grid.is_liquid[w > 0])


def test_element_full_example_path_fits_the_liquid() -> None:
    domain = _real_domain("kettle_electric_insulated", 0.02)
    cfg = example_scenarios()["kettle_electric_insulated"].heat_source
    assert isinstance(cfg, ElementCfg)
    dep = deposition_for_element(cfg, domain)
    assert dep.cell_weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_element_dry_firing_when_liquid_is_too_shallow() -> None:
    # 5 L of wort in the 0.32 m vessel: the fill is about 0.1 m, below the 0.25 m legs.
    # Probes are dropped here: the example probe sits above this shallow fill, and build_domain
    # would reject it before the element check is reached.
    scenario = _cell_scenario_with_liquid("kettle_electric_insulated", 0.02, volume_m3=0.005)
    scenario = scenario.model_copy(update={"probes": []})
    domain = build_domain(scenario)
    with pytest.raises(GeometryError, match="above the liquid"):
        deposition_for_element(scenario.heat_source, domain)  # type: ignore[arg-type]


# ----------------------------------------------------------------- source power
def test_burner_power_is_efficiency_times_gross() -> None:
    dep = deposition_for_burner(_burner_cfg(0.1), box_domain(10, 10, 4, DX))
    src = BurnerSource(17585.0, 0.35, VariableActuator(), dep)
    power = src.power(1.0, 0.0, 10.0)
    assert power.gross_W == pytest.approx(17585.0)
    assert power.delivered_W == pytest.approx(0.35 * 17585.0)
    assert src.fuel_kg(46.35e6) == pytest.approx(1.0)
    assert src.electric_kwh(3.6e6) == 0.0


def test_element_power_is_all_delivered() -> None:
    dep = deposition_for_element(
        _element_cfg([(-0.02, 0.0, 0.01), (0.02, 0.0, 0.01)]), box_domain(10, 10, 4, DX)
    )
    src = ElementSource(5500.0, VariableActuator(), dep)
    power = src.power(0.5, 0.0, 10.0)
    assert power.gross_W == pytest.approx(2750.0)
    assert power.delivered_W == pytest.approx(2750.0)
    assert src.electric_kwh(3.6e6) == pytest.approx(1.0)
    assert src.fuel_kg(46.35e6) == 0.0


# ----------------------------------------------------------------- near-wall element samples
@pytest.mark.parametrize("wall_offset_cells", [0.6, 0.3])
def test_ring_element_near_wall_deposits_fully(wall_offset_cells: float) -> None:
    domain = _real_domain("kettle_propane_40L", 0.02)
    grid = domain.grid
    radius = domain.inner_radius - wall_offset_cells * grid.dx
    theta = np.linspace(0.0, 2.0 * math.pi, 129)  # closed loop
    path = [(radius * math.cos(t), radius * math.sin(t), 0.05) for t in theta]

    # Dense sweep of the same ring: at 0.6 dx every sample is in a liquid cell, at 0.3 dx some
    # are not. The 0.3 dx case therefore exercises the nearest-cell fallback.
    dense = np.linspace(0.0, 2.0 * math.pi, 20000, endpoint=False)
    xyz = np.stack(
        [radius * np.cos(dense), radius * np.sin(dense), np.full(len(dense), 0.05)], axis=1
    )
    ijk = np.floor((xyz - grid.origin) / grid.dx).astype(int)
    outside_cells = ~grid.is_liquid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    if wall_offset_cells < 0.5:
        assert outside_cells.any()

    dep = deposition_for_element(_element_cfg(path), domain)
    w = dep.cell_weights
    assert w.min() >= 0.0
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert not w[~grid.is_liquid].any()
    assert not dep.shell_weights.any()


def test_segment_above_fill_height_still_raises() -> None:
    domain = _real_domain("kettle_propane_40L", 0.02)
    z = domain.fill_height + 0.02
    with pytest.raises(GeometryError, match="above the liquid"):
        deposition_for_element(_element_cfg([(-0.05, 0.0, z), (0.05, 0.0, z)]), domain)


def test_segment_outside_vessel_raises() -> None:
    domain = box_domain(10, 10, 4, DX)  # inner radius 0.05 m
    with pytest.raises(GeometryError, match="outside the vessel"):
        deposition_for_element(_element_cfg([(0.06, 0.0, 0.01), (0.06, 0.0, 0.02)]), domain)
