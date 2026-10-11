"""Display formatting shared by the GUI widgets (SI in, display strings out)."""

from __future__ import annotations

import math

import numpy as np

from thermosim.config.units import UnitSystem, to_display, unit_label

DEFAULT_UNITS: UnitSystem = "us"  # REQUIREMENTS B7: US is the default

_J_PER_KWH = 3.6e6
_J_PER_BTU = 1055.05585262
_KG_PER_LB = 0.45359237
_M_PER_IN = 0.0254


def temp(value_K: float, system: UnitSystem) -> float:
    """Kelvin to display temperature."""
    return float(to_display(value_K, "temperature", system))


def temp_array(values_K: np.ndarray, system: UnitSystem) -> np.ndarray:
    return np.asarray(to_display(values_K, "temperature", system))


def temp_label(system: UnitSystem) -> str:
    return unit_label("temperature", system)


def fmt_temp(value_K: float | None, system: UnitSystem, digits: int = 1) -> str:
    if value_K is None or not math.isfinite(value_K):
        return "-"
    return f"{temp(value_K, system):.{digits}f} {temp_label(system)}"


def fmt_delta(value_K: float, system: UnitSystem, digits: int = 2) -> str:
    value = float(to_display(value_K, "temperature_delta", system))
    return f"{value:.{digits}f} {unit_label('temperature_delta', system)}"


def plot_length(value_m: float | np.ndarray, system: UnitSystem) -> float | np.ndarray:
    """Length for the slice view: metres (metric) or inches (US)."""
    return value_m / _M_PER_IN if system == "us" else value_m


def plot_length_label(system: UnitSystem) -> str:
    return "in" if system == "us" else "m"


def fmt_clock(t_s: float | None) -> str:
    """mm:ss, or h:mm:ss when an hour or longer."""
    if t_s is None or not math.isfinite(t_s):
        return "-"
    total = round(max(t_s, 0.0))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_energy(joules: float, system: UnitSystem) -> str:
    if system == "us":
        text, unit = f"{joules / _J_PER_BTU:,.0f}", "BTU"
    else:
        text, unit = f"{joules / _J_PER_KWH:,.3f}", "kWh"
    if text.lstrip("-").replace(",", "").strip("0.") == "":
        text = text.lstrip("-")  # avoid "-0"
    return f"{text} {unit}"


def fmt_mass(kg: float, system: UnitSystem) -> str:
    if system == "us":
        return f"{kg / _KG_PER_LB:.3f} lb"
    return f"{kg:.3f} kg"


# Decimal places used when showing an editable quantity, per (quantity, unit system).
_DECIMALS: dict[tuple[str, str], int] = {
    ("temperature", "metric"): 1,
    ("temperature", "us"): 1,
    ("temperature_delta", "metric"): 1,
    ("temperature_delta", "us"): 1,
    ("temperature_rate", "metric"): 2,
    ("temperature_rate", "us"): 2,
    ("length", "metric"): 1,
    ("length", "us"): 3,
    ("altitude", "metric"): 0,
    ("altitude", "us"): 0,
    ("area", "metric"): 1,
    ("area", "us"): 2,
    ("volume", "metric"): 2,
    ("volume", "us"): 3,
    ("power", "metric"): 3,
    ("power", "us"): 0,
    ("speed", "metric"): 2,
    ("speed", "us"): 2,
    ("time", "metric"): 1,
    ("time", "us"): 1,
}


def fmt_quantity(value: float, quantity: str, system: UnitSystem) -> str:
    """Text for an editor field: a display-unit value with a sensible number of decimals.

    Trailing zeros are dropped. Values too small for the fixed decimals keep 3 significant
    digits instead, so a small thickness never shows as 0. Quantities without an entry here
    (conductivities, densities, fractions, ...) use 4 significant digits.
    """
    if not math.isfinite(value):
        return str(value)
    decimals = _DECIMALS.get((quantity, system))
    if decimals is None:
        return f"{value:.4g}"
    if value != 0.0 and abs(value) < 10.0 ** (2 - decimals):
        return f"{value:.3g}"
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text
