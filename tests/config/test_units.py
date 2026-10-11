"""Unit conversion: known values, labels, and property-based round trips."""

from __future__ import annotations

import math
from typing import get_args

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import BaseModel

from thermosim.config.schema import Quantity, Vessel
from thermosim.config.units import from_display, quantity_of, to_display, unit_label

QUANTITIES: tuple[str, ...] = get_args(Quantity)
SYSTEMS: tuple[str, ...] = ("metric", "us")
REL = 1e-12


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL, abs_tol=0.0)


# ------------------------------------------------------------------ known values
def test_temperature_fixed_points() -> None:
    assert to_display(273.15, "temperature", "metric") == pytest.approx(0.0, abs=1e-12)
    assert to_display(373.15, "temperature", "metric") == pytest.approx(100.0)
    assert to_display(373.15, "temperature", "us") == pytest.approx(212.0)
    assert to_display(273.15, "temperature", "us") == pytest.approx(32.0)


def test_temperature_delta_uses_scale_only() -> None:
    assert to_display(10.0, "temperature_delta", "metric") == pytest.approx(10.0)
    assert to_display(10.0, "temperature_delta", "us") == pytest.approx(18.0)
    assert from_display(18.0, "temperature_delta", "us") == pytest.approx(10.0)


def test_volume_gallon_and_litre() -> None:
    # 15 US gallons is the 57 L kettle in the propane example.
    assert from_display(15.0, "volume", "us") == pytest.approx(0.0567812, rel=1e-6)
    assert to_display(0.040, "volume", "metric") == pytest.approx(40.0)


def test_length_inch_and_millimetre() -> None:
    assert to_display(0.0254, "length", "us") == pytest.approx(1.0)
    assert to_display(0.005, "length", "metric") == pytest.approx(5.0)


def test_power_btu_per_hour() -> None:
    assert to_display(1000.0, "power", "us") == pytest.approx(3412.14, rel=1e-5)
    assert to_display(2500.0, "power", "metric") == pytest.approx(2.5)


def test_speed_mph() -> None:
    assert to_display(1.0, "speed", "us") == pytest.approx(2.23694, rel=1e-5)


def test_time_is_seconds_in_both_systems() -> None:
    assert to_display(600.0, "time", "metric") == pytest.approx(600.0)
    assert to_display(600.0, "time", "us") == pytest.approx(600.0)


def test_temperature_rate_and_altitude() -> None:
    # K/min stays the same number in degC/min; US multiplies by 1.8 (no offset).
    assert to_display(0.05, "temperature_rate", "metric") == pytest.approx(0.05)
    assert to_display(0.05, "temperature_rate", "us") == pytest.approx(0.09)
    assert unit_label("temperature_rate", "metric") == "\u00b0C/min"
    assert unit_label("temperature_rate", "us") == "\u00b0F/min"
    assert to_display(1000.0, "altitude", "metric") == pytest.approx(1000.0)
    assert to_display(1000.0, "altitude", "us") == pytest.approx(3280.8399, rel=1e-6)
    assert unit_label("altitude", "us") == "ft"
    assert unit_label("altitude", "metric") == "m"


def test_engineering_units() -> None:
    assert to_display(1.0, "conductivity", "us") == pytest.approx(0.5778, rel=1e-4)
    assert to_display(1000.0, "density", "us") == pytest.approx(62.428, rel=1e-4)
    assert to_display(4186.8, "specific_heat", "us") == pytest.approx(1.0)
    assert to_display(1.0, "thermal_resistance", "us") == pytest.approx(5.678, rel=1e-4)
    assert to_display(1.0e-4, "area", "metric") == pytest.approx(1.0)


def test_fraction_and_specific_gravity_are_unchanged() -> None:
    for system in SYSTEMS:
        assert to_display(0.35, "fraction", system) == pytest.approx(0.35)
        assert to_display(1.05, "specific_gravity", system) == pytest.approx(1.05)


def test_arrays_broadcast() -> None:
    kelvin = np.array([273.15, 373.15])
    out = to_display(kelvin, "temperature", "us")
    assert isinstance(out, np.ndarray)
    assert out == pytest.approx([32.0, 212.0])
    back = from_display(out, "temperature", "us")
    assert isinstance(back, np.ndarray)
    assert back == pytest.approx(kelvin, rel=REL)


def test_unknown_system_is_rejected() -> None:
    with pytest.raises(ValueError, match="unit system"):
        to_display(1.0, "length", "imperial")  # type: ignore[arg-type]


# ------------------------------------------------------------------ labels
@pytest.mark.parametrize("quantity", QUANTITIES)
@pytest.mark.parametrize("system", SYSTEMS)
def test_every_quantity_has_a_label(quantity: str, system: str) -> None:
    label = unit_label(quantity, system)  # type: ignore[arg-type]
    assert isinstance(label, str)


def test_labels_match_the_spec_units() -> None:
    assert unit_label("temperature", "us") == "°F"
    assert unit_label("temperature", "metric") == "°C"
    assert unit_label("volume", "us") == "gal"
    assert unit_label("volume", "metric") == "L"
    assert unit_label("power", "us") == "BTU/hr"
    assert unit_label("power", "metric") == "kW"
    assert unit_label("length", "us") == "in"
    assert unit_label("length", "metric") == "mm"
    assert unit_label("speed", "us") == "mph"
    assert unit_label("speed", "metric") == "m/s"
    assert unit_label("time", "us") == "s"
    assert unit_label("time", "metric") == "s"


# ------------------------------------------------------------------ round trips (property)
@given(
    kelvin=st.floats(min_value=200.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    system=st.sampled_from(SYSTEMS),
)
def test_temperature_si_round_trip(kelvin: float, system: str) -> None:
    display = to_display(kelvin, "temperature", system)  # type: ignore[arg-type]
    assert _close(from_display(display, "temperature", system), kelvin)  # type: ignore[arg-type]


@given(
    display=st.floats(min_value=-100.0, max_value=300.0, allow_nan=False, allow_infinity=False),
    system=st.sampled_from(SYSTEMS),
)
def test_temperature_display_round_trip(display: float, system: str) -> None:
    kelvin = from_display(display, "temperature", system)  # type: ignore[arg-type]
    assume(kelvin > 0.0)
    back = to_display(kelvin, "temperature", system)  # type: ignore[arg-type]
    assert math.isclose(back, display, rel_tol=REL, abs_tol=1e-9)


@given(
    value=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    quantity=st.sampled_from([q for q in QUANTITIES if q != "temperature"]),
    system=st.sampled_from(SYSTEMS),
)
def test_non_temperature_round_trip(value: float, quantity: str, system: str) -> None:
    display = to_display(value, quantity, system)  # type: ignore[arg-type]
    assert _close(from_display(display, quantity, system), value)  # type: ignore[arg-type]
    assert _close(to_display(from_display(value, quantity, system), quantity, system), value)  # type: ignore[arg-type]


@given(
    value=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    quantity=st.sampled_from(["fraction", "dimensionless", "specific_gravity"]),
    system=st.sampled_from(SYSTEMS),
)
def test_identity_units_round_trip_exactly(value: float, quantity: str, system: str) -> None:
    display = to_display(value, quantity, system)  # type: ignore[arg-type]
    assert from_display(display, quantity, system) == value  # type: ignore[arg-type]


# ------------------------------------------------------------------ quantity_of
def test_quantity_of_reads_field_tags() -> None:
    assert quantity_of(Vessel, "inner_diameter_m") == "length"
    assert quantity_of(Vessel, "material") is None


def test_quantity_of_unknown_field_raises() -> None:
    class Empty(BaseModel):
        pass

    with pytest.raises(KeyError):
        quantity_of(Empty, "nope")
