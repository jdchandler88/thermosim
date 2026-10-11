"""Model assembly, runner and headless entry point (WP09). Public API re-exported here."""

from __future__ import annotations

from .model import Model
from .runner import SimulationRunner, run_headless

__all__ = ["Model", "SimulationRunner", "run_headless"]
