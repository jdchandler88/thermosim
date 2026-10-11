"""Fixtures for the GUI-with-real-simulator tests (pytest-qt, offscreen)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox

from thermosim.gui import recording_adapter
from thermosim.gui.main_window import MainWindow


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    s = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    s.clear()
    return s


@pytest.fixture
def make_window(qtbot, settings: QSettings) -> Callable[..., MainWindow]:  # type: ignore[no-untyped-def]
    """A MainWindow using the REAL runner and pre-run checks."""

    def build() -> MainWindow:
        window = MainWindow(settings=settings)
        qtbot.addWidget(window)
        window.show()
        return window

    return build


@pytest.fixture(autouse=True)
def _reset_adapter() -> None:
    recording_adapter.reset_probes()


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never block on a modal prompt: discard unsaved changes by default."""
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
