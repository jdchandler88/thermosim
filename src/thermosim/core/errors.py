"""Exception hierarchy (CONTRACTS.md section 3). Frozen after WP00."""

from __future__ import annotations


class ThermoSimError(Exception): ...


class ScenarioError(ThermoSimError): ...  # invalid / inconsistent scenario


class GeometryError(ThermoSimError): ...  # e.g. element above liquid line, probe outside liquid


class ModeError(ThermoSimError): ...  # e.g. set_power_command in FAST mode


class SolverError(ThermoSimError): ...  # NaN / instability detected
