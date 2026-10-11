"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import os

import pytest

from thermosim.core.types import Domain
from thermosim.testing.synthetic import box_domain

# GUI tests need a display-less Qt platform on CI and on developer machines without one.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: long-running test, skipped unless selected with -m slow (or -m 'slow or ...')",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``slow`` tests unless the ``-m`` expression mentions them (CI runs them weekly)."""
    if "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow test: run with `pytest -m slow`")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def box() -> Domain:
    """The 10 x 8 x 6 cell box with 1 cm cells, used across the tests."""
    return box_domain(10, 8, 6, 0.01)
