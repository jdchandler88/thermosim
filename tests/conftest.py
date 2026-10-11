"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import os

import pytest

from thermosim.core.types import Domain
from thermosim.testing.synthetic import box_domain

# GUI tests need a display-less Qt platform on CI and on developer machines without one.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running test (deselect with -m 'not slow')")


@pytest.fixture(scope="session")
def box() -> Domain:
    """The 10 x 8 x 6 cell box with 1 cm cells, used across the tests."""
    return box_domain(10, 8, 6, 0.01)
