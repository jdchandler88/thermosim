"""Scenario TOML I/O: round trips, file format, and error messages."""

from __future__ import annotations

import tomllib
from importlib.resources import files
from pathlib import Path

import pytest

from thermosim.config.defaults import default_scenario
from thermosim.config.io import (
    example_scenarios,
    load_scenario,
    save_scenario,
    scenario_from_toml,
    scenario_to_toml,
)
from thermosim.config.schema import Scenario
from thermosim.core.errors import ScenarioError


def test_save_then_load_round_trips_exactly(tmp_path: Path) -> None:
    for name, scenario in example_scenarios().items():
        path = tmp_path / f"{name}.toml"
        save_scenario(scenario, path)
        assert load_scenario(path) == scenario


def test_text_round_trip_is_exact_for_every_example() -> None:
    for scenario in example_scenarios().values():
        assert scenario_from_toml(scenario_to_toml(scenario)) == scenario


def test_header_states_units_and_kelvin() -> None:
    text = scenario_to_toml(default_scenario())
    header = text.splitlines()[:3]
    assert all(line.startswith("#") for line in header)
    joined = " ".join(header).lower()
    assert "si" in joined
    assert "kelvin" in joined


def test_tuples_are_written_as_arrays() -> None:
    electric = example_scenarios()["kettle_electric_insulated"]
    data = tomllib.loads(scenario_to_toml(electric))
    path = data["heat_source"]["path_m"]
    assert isinstance(path, list)
    assert isinstance(path[0], list)
    assert path[0] == [-0.10, 0.0, 0.25]
    assert data["probes"][0]["position_m"] == [0.0, 0.0, 0.15]


def test_none_fields_are_omitted() -> None:
    propane = default_scenario()  # sets volume_m3 only and has no insulation
    text = scenario_to_toml(propane)
    data = tomllib.loads(text)
    assert "fill_height_m" not in data["liquid"]
    assert "insulation" not in data
    assert "[insulation]" not in text


def test_insulation_covers_is_written_as_sorted_array() -> None:
    electric = example_scenarios()["kettle_electric_insulated"]
    data = electric.model_copy(deep=True)
    assert data.insulation is not None
    data.insulation.covers = {"side", "base"}
    parsed = tomllib.loads(scenario_to_toml(data))
    assert parsed["insulation"]["covers"] == ["base", "side"]
    assert scenario_from_toml(scenario_to_toml(data)) == data


def test_enum_written_as_plain_string() -> None:
    parsed = tomllib.loads(scenario_to_toml(default_scenario()))
    assert parsed["simulation"]["mode"] == "fast"


def test_load_error_names_file_and_field_path(tmp_path: Path) -> None:
    text = scenario_to_toml(default_scenario()).replace(
        "inner_diameter_m = 0.36", "inner_diameter_m = -0.36"
    )
    path = tmp_path / "broken.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ScenarioError) as info:
        load_scenario(path)
    message = str(info.value)
    assert str(path) in message
    assert "vessel.inner_diameter_m" in message


def test_cross_field_error_names_file_and_field(tmp_path: Path) -> None:
    text = scenario_to_toml(default_scenario()).replace('mode = "none"', 'mode = "partial"')
    path = tmp_path / "partial.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ScenarioError) as info:
        load_scenario(path)
    assert str(path) in str(info.value)
    assert "lid.covered_fraction" in str(info.value)


def test_invalid_toml_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "junk.toml"
    path.write_text("name = \n[vessel", encoding="utf-8")
    with pytest.raises(ScenarioError) as info:
        load_scenario(path)
    assert str(path) in str(info.value)
    assert "TOML" in str(info.value)


def test_missing_file_raises_scenario_error(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError) as info:
        load_scenario(tmp_path / "absent.toml")
    assert "absent.toml" in str(info.value)


def test_scenario_from_toml_rejects_missing_required_field() -> None:
    text = scenario_to_toml(default_scenario()).replace('name = "', 'label = "')
    with pytest.raises(ScenarioError, match="name"):
        scenario_from_toml(text)


def test_examples_are_package_data() -> None:
    folder = files("thermosim") / "scenarios"
    names = sorted(entry.name for entry in folder.iterdir() if entry.name.endswith(".toml"))
    assert names == [
        "kettle_electric_insulated.toml",
        "kettle_propane_40L.toml",
        "small_pot_stove.toml",
    ]


def test_example_values_match_the_spec() -> None:
    examples = example_scenarios()

    propane = examples["kettle_propane_40L"]
    assert propane.vessel.inner_height_m == pytest.approx(0.56)
    assert propane.liquid.volume_m3 == pytest.approx(0.040)
    assert propane.liquid.initial_temp_K == pytest.approx(288.15)
    assert propane.lid.mode == "none"
    assert propane.insulation is None
    assert propane.atmosphere.wind_speed_m_s == pytest.approx(2.0)
    assert propane.atmosphere.ambient_temp_K == pytest.approx(283.15)
    assert propane.target.temp_K == pytest.approx(372.15)
    assert propane.target.probe == "mean"

    electric = examples["kettle_electric_insulated"]
    assert electric.heat_source.type == "element"  # type: ignore[union-attr]
    assert electric.heat_source.rated_power_W == pytest.approx(5500.0)  # type: ignore[union-attr]
    assert electric.liquid.specific_gravity == pytest.approx(1.050)
    assert electric.liquid.volume_m3 == pytest.approx(0.025)
    assert electric.lid.mode == "partial"
    assert electric.lid.covered_fraction == pytest.approx(0.5)
    assert electric.insulation is not None
    assert electric.insulation.covers == {"side"}
    assert electric.target.probe == "side_wall_10cm"
    assert electric.target.temp_K == pytest.approx(372.15)
    probe = next(p for p in electric.probes if p.name == "side_wall_10cm")
    assert probe.position_m[2] == pytest.approx(0.10)

    pot = examples["small_pot_stove"]
    assert pot.vessel.material == "aluminum"
    assert pot.lid.mode == "full"
    assert pot.heat_source.efficiency == pytest.approx(0.5)  # type: ignore[union-attr]
    assert pot.heat_source.rated_power_W == pytest.approx(2500.0)  # type: ignore[union-attr]
    assert pot.atmosphere.ambient_temp_K == pytest.approx(294.15)
    assert pot.target.temp_K == pytest.approx(363.15)


def test_examples_are_scenarios() -> None:
    for scenario in example_scenarios().values():
        assert isinstance(scenario, Scenario)
