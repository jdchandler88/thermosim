"""Geometry: grid, shell, probes and slice (WP04). Public API re-exported from build."""

from __future__ import annotations

from .build import (
    build_domain,
    cell_index_at,
    extract_slice,
    fill_height_from_volume,
    probe_values,
)

__all__ = [
    "build_domain",
    "cell_index_at",
    "extract_slice",
    "fill_height_from_volume",
    "probe_values",
]
