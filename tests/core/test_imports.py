"""Every module in the package imports cleanly (no circular imports, no missing names)."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import thermosim
from thermosim import cli


def test_all_modules_import() -> None:
    names = [info.name for info in pkgutil.walk_packages(thermosim.__path__, prefix="thermosim.")]
    assert names, "no thermosim modules were found"
    for name in names:
        importlib.import_module(name)


def test_cli_help_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower()
