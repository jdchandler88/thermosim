"""Solver test configuration: registers the ``slow`` marker (pyproject.toml is not ours)."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running test (deselect with -m 'not slow')")
