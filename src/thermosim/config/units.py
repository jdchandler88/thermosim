"""Unit conversion for display and input (CONTRACTS.md section 5). WP00 stub, implemented by WP01.

Internally everything is SI (D1); conversions happen only here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from thermosim.config.schema import Quantity

UnitSystem = Literal["metric", "us"]


def to_display(
    value: float | np.ndarray, quantity: Quantity, system: UnitSystem
) -> float | np.ndarray:
    """Convert an SI value to the display unit of `system` for `quantity`."""
    raise NotImplementedError


def from_display(
    value: float | np.ndarray, quantity: Quantity, system: UnitSystem
) -> float | np.ndarray:
    """Convert a value entered in the display unit of `system` back to SI."""
    raise NotImplementedError


def unit_label(quantity: Quantity, system: UnitSystem) -> str:
    """Display unit label, e.g. "°F", "gal", "BTU/hr"."""
    raise NotImplementedError


def quantity_of(model: type[BaseModel], field_name: str) -> Quantity | None:
    """The `quantity` tag of a pydantic model field, or None if it has none."""
    raise NotImplementedError
