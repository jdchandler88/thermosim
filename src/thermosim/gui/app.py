"""GUI entry point: `thermosim.gui.app.main(argv) -> int` (wired to `thermosim gui` by WP09)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from thermosim.config.io import load_scenario
from thermosim.core.errors import ScenarioError
from thermosim.gui.fake_runner import FakeRunner
from thermosim.gui.main_window import MainWindow

APP_NAME = "ThermoSim"
ORG_NAME = "ThermoSim"


def make_icon() -> QIcon:
    """A simple generated icon: an orange disc with a white 'T'."""
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(230, 90, 20))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(2, 2, 60, 60)
    p.setPen(QColor(255, 255, 255))
    font = QFont()
    font.setBold(True)
    font.setPixelSize(40)
    p.setFont(font)
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "T")
    p.end()
    return QIcon(pix)


def create_application(argv: list[str]) -> QApplication:
    """The QApplication (created once), with name, organisation and icon set."""
    app = QApplication.instance()
    if app is None:
        # Fractional scaling (e.g. Windows at 150%) is passed through, not rounded.
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        app = QApplication(argv)
    assert isinstance(app, QApplication)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setWindowIcon(make_icon())
    return app


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(prog="thermosim gui", description="ThermoSim desktop GUI.")
    parser.add_argument("scenario", nargs="?", type=Path, help="setup (TOML) file to open")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="force the fake runner (synthetic data; the default until the real runner is wired)",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI. Returns the process exit code."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    args, qt_args = parse_args(args_list)
    app = create_application([sys.argv[0] if sys.argv else APP_NAME, *qt_args])
    factory = (lambda sc, mode: FakeRunner(sc, mode)) if args.fake else None
    window = MainWindow(runner_factory=factory)
    if args.scenario is not None:
        try:
            window.load_scenario(load_scenario(args.scenario), args.scenario)
        except ScenarioError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
