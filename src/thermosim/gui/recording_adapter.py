"""Adapter over `thermosim.recording` (WP07), degrading gracefully until it lands.

Until WP07 merges, the functions in `thermosim.recording` are stubs that raise
`NotImplementedError`. Each capability is probed once with harmless arguments: only a
`NotImplementedError` means "not available"; any other outcome means the real
implementation is present. The GUI disables the related action (with a tooltip) when a
capability is missing.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from thermosim.config.units import UnitSystem
from thermosim.core.types import Sample

UNAVAILABLE_TIP = "Needs the recording module (WP07), which is not available yet."


def _implemented(call: Callable[[], Any]) -> bool:
    try:
        call()
    except NotImplementedError:
        return False
    except Exception:
        return True
    return True


@lru_cache(maxsize=1)
def export_csv_available() -> bool:
    try:
        from thermosim.recording import export_csv
    except ImportError:
        return False

    def probe() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_csv([], Path(tmp) / "probe.csv", "metric")

    return _implemented(probe)


@lru_cache(maxsize=1)
def measured_available() -> bool:
    try:
        from thermosim.recording import load_measured_csv
    except ImportError:
        return False
    return _implemented(lambda: load_measured_csv(Path("__thermosim_probe__.csv")))


@lru_cache(maxsize=1)
def compare_available() -> bool:
    try:
        from thermosim.recording import compare
    except ImportError:
        return False
    return _implemented(lambda: compare([], "mean", None, None))  # type: ignore[arg-type]


def reset_probes() -> None:
    """Forget cached availability (for tests)."""
    export_csv_available.cache_clear()
    measured_available.cache_clear()
    compare_available.cache_clear()


def export_csv(samples: list[Sample], path: Path, system: UnitSystem) -> None:
    from thermosim.recording import export_csv as real

    real(samples, path, system)


def load_measured(
    path: Path,
    time_col: str | None,
    temp_col: str | None,
    temp_unit: str,
    time_unit: str,
) -> Any:
    from thermosim.recording import load_measured_csv

    return load_measured_csv(
        path,
        time_col=time_col,
        temp_col=temp_col,
        temp_unit=temp_unit,  # type: ignore[arg-type]
        time_unit=time_unit,  # type: ignore[arg-type]
    )


def compare(samples: list[Sample], probe: str, measured: Any, target_K: float | None) -> Any:
    from thermosim.recording import compare as real

    return real(samples, probe, measured, target_K)
