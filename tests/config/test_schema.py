"""Scenario schema: every validation rule has a passing and a failing test."""

from __future__ import annotations

import copy
from typing import Any, Literal, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from thermosim.config import schema
from thermosim.config.defaults import default_scenario
from thermosim.config.io import example_scenarios
from thermosim.config.schema import Scenario
from thermosim.config.units import quantity_of
from thermosim.core.errors import ScenarioError


def _dump(scenario: Scenario) -> dict[str, Any]:
    return scenario.model_dump(mode="json", exclude_none=True)


def _propane() -> dict[str, Any]:
    return _dump(default_scenario())


def _electric() -> dict[str, Any]:
    return _dump(example_scenarios()["kettle_electric_insulated"])


def _build(data: dict[str, Any]) -> Scenario:
    return Scenario.model_validate(copy.deepcopy(data))


# ------------------------------------------------------------------ examples and defaults
def test_default_is_first_example() -> None:
    assert default_scenario().name.startswith("Propane kettle")
    assert default_scenario() == example_scenarios()["kettle_propane_40L"]


def test_examples_pass_validation() -> None:
    examples = example_scenarios()
    assert set(examples) == {"kettle_propane_40L", "kettle_electric_insulated", "small_pot_stove"}
    for scenario in examples.values():
        assert Scenario.model_validate(_dump(scenario)) == scenario


def test_extra_keys_are_rejected() -> None:
    data = _propane()
    data["vessel"]["diameter_typo_m"] = 0.3
    with pytest.raises(ValidationError):
        _build(data)


def test_schema_version_must_be_one() -> None:
    data = _propane()
    data["schema_version"] = 2
    with pytest.raises(ValidationError):
        _build(data)


# ------------------------------------------------------------------ liquid
@pytest.mark.parametrize("fill_key", ["volume_m3", "fill_height_m"])
def test_liquid_one_fill_value_passes(fill_key: str) -> None:
    data = _propane()
    data["liquid"] = {"initial_temp_K": 290.0, fill_key: 0.03 if fill_key == "volume_m3" else 0.3}
    _build(data)


def test_liquid_both_fill_values_rejected() -> None:
    data = _propane()
    data["liquid"]["fill_height_m"] = 0.2
    with pytest.raises(ScenarioError, match="exactly one"):
        _build(data)


def test_liquid_no_fill_value_rejected() -> None:
    data = _propane()
    del data["liquid"]["volume_m3"]
    with pytest.raises(ScenarioError, match="exactly one"):
        _build(data)


def test_fill_height_above_vessel_rejected() -> None:
    data = _propane()
    data["liquid"] = {"initial_temp_K": 290.0, "fill_height_m": 0.9}
    with pytest.raises(ScenarioError, match=r"exceeds vessel\.inner_height_m"):
        _build(data)


def test_computed_fill_height_above_vessel_rejected() -> None:
    data = _propane()
    data["liquid"]["volume_m3"] = 0.2  # 2.0 m of fill in a 0.36 m vessel
    with pytest.raises(ScenarioError, match=r"exceeds vessel\.inner_height_m"):
        _build(data)


def test_specific_gravity_bounds() -> None:
    data = _propane()
    data["liquid"]["specific_gravity"] = 2.0
    with pytest.raises(ValidationError):
        _build(data)


# ------------------------------------------------------------------ lid
def test_partial_lid_fraction_passes() -> None:
    data = _propane()
    data["lid"] = {"mode": "partial", "covered_fraction": 0.5}
    assert _build(data).lid.covered_fraction == 0.5


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_partial_lid_fraction_must_be_strictly_between_zero_and_one(fraction: float) -> None:
    data = _propane()
    data["lid"] = {"mode": "partial", "covered_fraction": fraction}
    with pytest.raises(ScenarioError, match="covered_fraction"):
        _build(data)


def test_full_lid_forces_fraction_to_one() -> None:
    data = _propane()
    data["lid"] = {"mode": "full", "covered_fraction": 0.3}
    assert _build(data).lid.covered_fraction == 1.0


def test_no_lid_ignores_fraction() -> None:
    data = _propane()
    data["lid"] = {"mode": "none", "covered_fraction": 0.3}
    assert _build(data).lid.covered_fraction == 0.3


# ------------------------------------------------------------------ insulation
def test_insulation_lid_cover_requires_lid() -> None:
    data = _electric()
    data["insulation"]["covers"] = ["side", "lid"]
    data["lid"] = {"mode": "none"}
    with pytest.raises(ScenarioError, match="requires a lid"):
        _build(data)


def test_insulation_lid_cover_with_lid_passes() -> None:
    data = _electric()
    data["insulation"]["covers"] = ["side", "lid"]
    data["lid"] = {"mode": "full"}
    assert _build(data).insulation is not None


def test_insulation_covers_is_a_set_of_known_surfaces() -> None:
    data = _electric()
    data["insulation"]["covers"] = ["roof"]
    with pytest.raises(ValidationError):
        _build(data)


# ------------------------------------------------------------------ materials
@pytest.mark.parametrize("key", ["stainless_304", "aluminum", "copper", "carbon_steel"])
def test_preset_materials_pass(key: str) -> None:
    data = _propane()
    data["vessel"]["material"] = key
    assert _build(data).vessel.material == key


def test_unknown_material_rejected() -> None:
    data = _propane()
    data["vessel"]["material"] = "unobtainium"
    with pytest.raises(ValidationError, match="unknown material"):
        _build(data)


def test_unknown_lid_material_rejected() -> None:
    data = _propane()
    data["lid"] = {"mode": "full", "material": "unobtainium"}
    with pytest.raises(ValidationError, match="unknown material"):
        _build(data)


def test_custom_material_passes() -> None:
    data = _propane()
    data["vessel"]["material"] = {
        "density": 2700.0,
        "specific_heat": 900.0,
        "conductivity": 200.0,
        "emissivity": 0.1,
    }
    assert not isinstance(_build(data).vessel.material, str)


# ------------------------------------------------------------------ simulation
def test_cell_size_bounds() -> None:
    data = _propane()
    data["simulation"]["cell_size_m"] = 0.001
    with pytest.raises(ValidationError):
        _build(data)
    data["simulation"]["cell_size_m"] = 0.06
    with pytest.raises(ValidationError):
        _build(data)


def test_cell_size_lower_bound_passes() -> None:
    data = _propane()
    data["simulation"]["cell_size_m"] = 0.002
    _build(data)


def _narrow_vessel(diameter_m: float) -> dict[str, Any]:
    data = _propane()
    data["vessel"]["inner_diameter_m"] = diameter_m
    data["liquid"] = {"initial_temp_K": 290.0, "fill_height_m": 0.1}
    data["probes"] = []
    data["simulation"]["cell_size_m"] = 0.05
    return data


def test_vessel_must_be_six_cells_across() -> None:
    with pytest.raises(ScenarioError, match="6 cells"):
        _build(_narrow_vessel(0.25))  # five 5 cm cells


def test_six_cells_across_passes() -> None:
    _build(_narrow_vessel(0.30))  # exactly six 5 cm cells


def test_flow_mode_is_accepted_with_defaults() -> None:
    data = _propane()
    data["model"]["liquid_transport"] = "flow"
    sc = _build(data)
    assert sc.model.liquid_transport == "flow"
    assert sc.model.smagorinsky_cs == 0.1
    assert sc.model.poisson_tol is None


def test_flow_settings_are_validated() -> None:
    data = _propane()
    data["model"]["smagorinsky_cs"] = -0.1
    with pytest.raises(ValidationError):
        _build(data)


# ------------------------------------------------------------------ burner
def test_burner_footprint_fits_base() -> None:
    data = _propane()
    outer = data["vessel"]["inner_diameter_m"] + 2 * data["vessel"]["wall_thickness_m"]
    data["heat_source"]["footprint_outer_diameter_m"] = outer + 0.01
    with pytest.raises(ScenarioError, match="footprint_outer_diameter_m"):
        _build(data)


def test_burner_footprint_equal_to_base_passes() -> None:
    data = _propane()
    outer = data["vessel"]["inner_diameter_m"] + 2 * data["vessel"]["wall_thickness_m"]
    data["heat_source"]["footprint_outer_diameter_m"] = outer
    _build(data)


def test_burner_ring_inner_must_be_less_than_outer() -> None:
    data = _propane()
    data["heat_source"]["footprint"] = "ring"
    data["heat_source"]["footprint_inner_diameter_m"] = 0.2
    with pytest.raises(ScenarioError, match="footprint_inner_diameter_m"):
        _build(data)


def test_burner_ring_passes() -> None:
    data = _propane()
    data["heat_source"]["footprint"] = "ring"
    data["heat_source"]["footprint_inner_diameter_m"] = 0.1
    assert _build(data).heat_source.footprint == "ring"  # type: ignore[union-attr]


# ------------------------------------------------------------------ element
def test_element_needs_two_points() -> None:
    data = _electric()
    data["heat_source"]["path_m"] = [[0.0, 0.0, 0.1]]
    with pytest.raises(ValidationError):
        _build(data)


def test_element_point_outside_vessel_rejected() -> None:
    data = _electric()
    data["heat_source"]["path_m"][0] = [0.2, 0.0, 0.25]  # radius 0.2 > 0.16 - 0.005
    with pytest.raises(ScenarioError, match="path_m"):
        _build(data)


def test_element_point_above_vessel_rejected() -> None:
    data = _electric()
    data["heat_source"]["path_m"][0] = [-0.10, 0.0, 0.6]  # above the 0.5 m inner height
    with pytest.raises(ScenarioError, match="path_m"):
        _build(data)


def test_element_point_below_base_rejected() -> None:
    data = _electric()
    data["heat_source"]["path_m"][1] = [-0.10, 0.0, -0.01]
    with pytest.raises(ScenarioError, match="path_m"):
        _build(data)


def test_element_diameter_counts_toward_wall_clearance() -> None:
    data = _electric()
    data["heat_source"]["path_m"][0] = [0.156, 0.0, 0.25]  # inside 0.16 but not 0.16 - 0.01/2
    data["heat_source"]["diameter_m"] = 0.01
    with pytest.raises(ScenarioError, match="path_m"):
        _build(data)


def test_element_inside_vessel_passes() -> None:
    assert _build(_electric()).heat_source.type == "element"  # type: ignore[union-attr]


# ------------------------------------------------------------------ probes and target
def test_duplicate_probe_names_rejected() -> None:
    data = _electric()
    data["probes"].append({"name": "center", "position_m": [0.0, 0.0, 0.1]})
    with pytest.raises(ScenarioError, match="unique"):
        _build(data)


def test_probe_named_mean_rejected() -> None:
    data = _electric()
    data["probes"][0]["name"] = "mean"
    with pytest.raises(ScenarioError, match="reserved"):
        _build(data)


def test_probe_outside_cylinder_rejected() -> None:
    data = _electric()
    data["probes"][0]["position_m"] = [0.2, 0.0, 0.1]
    with pytest.raises(ScenarioError, match="inside the vessel"):
        _build(data)


def test_probe_above_fill_height_rejected() -> None:
    data = _electric()
    data["probes"][0]["position_m"] = [0.0, 0.0, 0.35]  # fill is 0.311 m
    with pytest.raises(ScenarioError, match="below the fill height"):
        _build(data)


def test_probe_below_base_rejected() -> None:
    data = _electric()
    data["probes"][0]["position_m"] = [0.0, 0.0, -0.01]
    with pytest.raises(ScenarioError, match="inside the vessel"):
        _build(data)


def test_probe_inside_liquid_passes() -> None:
    assert len(_build(_electric()).probes) == 2


def test_target_probe_must_exist() -> None:
    data = _electric()
    data["target"]["probe"] = "nowhere"
    with pytest.raises(ScenarioError, match=r"target\.probe"):
        _build(data)


def test_target_mean_passes() -> None:
    data = _electric()
    data["target"]["probe"] = "mean"
    _build(data)


def test_target_temperature_bounds() -> None:
    data = _electric()
    data["target"]["temp_K"] = 500.0
    with pytest.raises(ValidationError):
        _build(data)


# ------------------------------------------------------------------ power schedule
def test_schedule_must_start_at_zero() -> None:
    data = _propane()
    data["power_schedule"] = [{"t_s": 10.0, "command": 1.0}]
    with pytest.raises(ScenarioError, match="t_s = 0"):
        _build(data)


def test_schedule_must_be_sorted() -> None:
    data = _propane()
    data["power_schedule"] = [
        {"t_s": 0.0, "command": 1.0},
        {"t_s": 600.0, "command": 0.5},
        {"t_s": 300.0, "command": 0.2},
    ]
    with pytest.raises(ScenarioError, match="strictly increasing"):
        _build(data)


def test_schedule_times_must_be_distinct() -> None:
    data = _propane()
    data["power_schedule"] = [{"t_s": 0.0, "command": 1.0}, {"t_s": 0.0, "command": 0.5}]
    with pytest.raises(ScenarioError, match="strictly increasing"):
        _build(data)


def test_schedule_command_bounds() -> None:
    data = _propane()
    data["power_schedule"] = [{"t_s": 0.0, "command": 1.2}]
    with pytest.raises(ValidationError):
        _build(data)


def test_schedule_command_negative_rejected() -> None:
    data = _propane()
    data["power_schedule"] = [{"t_s": 0.0, "command": -0.1}]
    with pytest.raises(ValidationError):
        _build(data)


def test_schedule_must_not_be_empty() -> None:
    data = _propane()
    data["power_schedule"] = []
    with pytest.raises(ValidationError):
        _build(data)


def test_valid_schedule_passes() -> None:
    data = _propane()
    data["power_schedule"] = [{"t_s": 0.0, "command": 1.0}, {"t_s": 600.0, "command": 0.5}]
    assert len(_build(data).power_schedule) == 2


# ------------------------------------------------------------------ schema walk
def _numeric(annotation: Any) -> bool:
    """True if the annotation mentions float or int as a value type (not a Literal value)."""
    if annotation in (float, int):
        return True
    origin = get_origin(annotation)
    if origin is Literal:
        return False
    return any(_numeric(arg) for arg in get_args(annotation))


def _config_models() -> list[type[BaseModel]]:
    found: list[type[BaseModel]] = []
    for name in dir(schema):
        obj = getattr(schema, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseModel)
            and obj.__module__ == schema.__name__
        ):
            found.append(obj)
    return found


def test_every_numeric_field_has_a_quantity_and_bounds() -> None:
    models = _config_models()
    assert len(models) >= 15
    checked = 0
    for model in models:
        for name, info in model.model_fields.items():
            if not _numeric(info.annotation):
                continue
            checked += 1
            assert quantity_of(model, name) is not None, f"{model.__name__}.{name} has no quantity"
            assert info.title, f"{model.__name__}.{name} has no title"
            assert info.description, f"{model.__name__}.{name} has no description"
    assert checked >= 40  # 44 numeric fields at the time of writing


def test_every_field_has_a_title_and_description() -> None:
    for model in _config_models():
        for name, info in model.model_fields.items():
            assert info.title, f"{model.__name__}.{name} has no title"
            assert info.description, f"{model.__name__}.{name} has no description"


def test_model_config_forbids_extra_everywhere() -> None:
    for model in _config_models():
        assert model.model_config.get("extra") == "forbid", model.__name__


def test_quantity_spot_checks() -> None:
    assert quantity_of(schema.Vessel, "inner_diameter_m") == "length"
    assert quantity_of(schema.Atmosphere, "ambient_temp_K") == "temperature"
    assert quantity_of(schema.Liquid, "specific_gravity") == "specific_gravity"
    assert quantity_of(schema.Burner, "rated_power_W") == "power"
    assert quantity_of(schema.Liquid, "volume_m3") == "volume"
    assert quantity_of(schema.Vessel, "material") is None
    assert quantity_of(schema.Atmosphere, "altitude_m") == "altitude"
    assert quantity_of(schema.SimSettings, "plateau_rate_K_per_min") == "temperature_rate"


def _has_bound(info: Any) -> bool:
    for meta in info.metadata:
        if any(getattr(meta, attr, None) is not None for attr in ("ge", "gt", "le", "lt")):
            return True
    return False


def test_scalar_numeric_fields_have_bounds() -> None:
    """Scalar numeric fields carry ge/gt/le/lt. Coordinates are bounded on their items."""
    for model in _config_models():
        for name, info in model.model_fields.items():
            if info.annotation not in (float, int, float | None):
                continue
            assert _has_bound(info), f"{model.__name__}.{name} has no ge/gt/le/lt bound"
