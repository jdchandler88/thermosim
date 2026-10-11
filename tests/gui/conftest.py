"""Fixtures for GUI tests (pytest-qt, offscreen)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from gui_helpers import RecordingRunner
from PySide6.QtCore import QSettings

from thermosim.config.schema import Scenario
from thermosim.core.protocols import Runner
from thermosim.core.types import RunMode
from thermosim.gui import recording_adapter
from thermosim.gui.main_window import MainWindow


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    s = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    s.clear()
    return s


@pytest.fixture
def make_window(qtbot, settings: QSettings) -> Callable[..., MainWindow]:  # type: ignore[no-untyped-def]
    """Build a MainWindow whose runner factory makes quick RecordingRunners."""

    def build(fast_rate: float = 6000.0, realtime_tick: float = 0.01) -> MainWindow:
        def factory(scenario: Scenario, mode: RunMode) -> Runner:
            return RecordingRunner(scenario, mode, fast_rate=fast_rate, tick_s=realtime_tick)

        window = MainWindow(settings=settings, runner_factory=factory)
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
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Discard,
    )
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
