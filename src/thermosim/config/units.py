"""Unit conversion for display and input (CONTRACTS.md section 5).

Internally everything is SI (D1); conversions happen only here. Each display unit is an
affine map of the SI value:

    display = (si + offset_si) * scale + offset_disp

Temperatures use the affine part (kelvin to Celsius or Fahrenheit). Temperature
differences use the scale only (1 K = 1.8 F). Metric display uses degC, L, kW, mm, m/s,
min. US display uses degF, gal (US), BTU/hr, in, mph, min (REQUIREMENTS B7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, get_args

import numpy as np
from pydantic import BaseModel

from thermosim.config.schema import Quantity

UnitSystem = Literal["metric", "us"]

_QUANTITIES: frozenset[str] = frozenset(get_args(Quantity))


@dataclass(frozen=True)
class _Conv:
    label: str
    scale: float
    offset_si: float = 0.0  # added to the SI value before scaling
    offset_disp: float = 0.0  # added after scaling


_KELVIN_OFFSET = -273.15

_CONVERSIONS: dict[UnitSystem, dict[Quantity, _Conv]] = {
    "metric": {
        "temperature": _Conv("°C", 1.0, _KELVIN_OFFSET),
        "temperature_delta": _Conv("°C", 1.0),
        "length": _Conv("mm", 1000.0),
        "area": _Conv("cm²", 1.0e4),
        "volume": _Conv("L", 1000.0),
        "power": _Conv("kW", 1.0e-3),
        "speed": _Conv("m/s", 1.0),
        "time": _Conv("min", 1.0 / 60.0),
        "fraction": _Conv("", 1.0),
        "specific_gravity": _Conv("SG", 1.0),
        "conductivity": _Conv("W/(m·K)", 1.0),
        "density": _Conv("kg/m³", 1.0),
        "specific_heat": _Conv("J/(kg·K)", 1.0),
        "thermal_resistance": _Conv("m²·K/W", 1.0),
        "dimensionless": _Conv("", 1.0),
    },
    "us": {
        # 1 K = 1.8 F; 0 degC = 32 degF
        "temperature": _Conv("°F", 1.8, _KELVIN_OFFSET, 32.0),
        "temperature_delta": _Conv("°F", 1.8),
        "length": _Conv("in", 1.0 / 0.0254),
        "area": _Conv("in²", 1.0 / 6.4516e-4),
        "volume": _Conv("gal", 1.0 / 0.003785411784),  # US gallon
        "power": _Conv("BTU/hr", 3.412141633),
        "speed": _Conv("mph", 2.2369362920544),
        "time": _Conv("min", 1.0 / 60.0),
        "fraction": _Conv("", 1.0),
        "specific_gravity": _Conv("SG", 1.0),
        "conductivity": _Conv("BTU/(hr·ft·°F)", 1.0 / 1.730734666),
        "density": _Conv("lb/ft³", 1.0 / 16.01846337),
        "specific_heat": _Conv("BTU/(lb·°F)", 1.0 / 4186.8),
        "thermal_resistance": _Conv("ft²·°F·hr/BTU", 5.678263),
        "dimensionless": _Conv("", 1.0),
    },
}


def _conv(quantity: Quantity, system: UnitSystem) -> _Conv:
    try:
        return _CONVERSIONS[system][quantity]
    except KeyError as exc:
        raise ValueError(f"unknown quantity {quantity!r} or unit system {system!r}") from exc


def to_display(
    value: float | np.ndarray, quantity: Quantity, system: UnitSystem
) -> float | np.ndarray:
    """Convert an SI value to the display unit of `system` for `quantity`.

    Works on scalars and arrays. Temperatures are kelvin in, degC or degF out.
    """
    conv = _conv(quantity, system)
    return (value + conv.offset_si) * conv.scale + conv.offset_disp


def from_display(
    value: float | np.ndarray, quantity: Quantity, system: UnitSystem
) -> float | np.ndarray:
    """Convert a value entered in the display unit of `system` back to SI."""
    conv = _conv(quantity, system)
    return (value - conv.offset_disp) / conv.scale - conv.offset_si


def unit_label(quantity: Quantity, system: UnitSystem) -> str:
    """Display unit label, e.g. "°F", "gal", "BTU/hr". Empty for dimensionless values."""
    return _conv(quantity, system).label


def quantity_of(model: type[BaseModel], field_name: str) -> Quantity | None:
    """The `quantity` tag of a pydantic model field, or None if it has none.

    Raises KeyError if the model has no field of that name.
    """
    info = model.model_fields.get(field_name)
    if info is None:
        raise KeyError(f"{model.__name__} has no field {field_name!r}")
    extra = info.json_schema_extra
    if isinstance(extra, dict):
        tag = extra.get("quantity")
        if isinstance(tag, str) and tag in _QUANTITIES:
            return cast(Quantity, tag)
    return None
