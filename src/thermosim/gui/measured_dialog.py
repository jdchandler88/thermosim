"""Dialog to pick the columns and units of a measured-temperature CSV file."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from thermosim.config.units import UnitSystem

_PREVIEW_ROWS = 12


@dataclass(frozen=True)
class MeasuredSelection:
    path: Path
    time_col: str | None
    temp_col: str | None
    temp_unit: str  # "C" | "F" | "K"
    time_unit: str  # "s" | "min"


def read_preview(
    path: Path, max_rows: int = _PREVIEW_ROWS
) -> tuple[list[str], list[list[str]], bool]:
    """(column names, first rows, file_has_header) of a CSV file."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        rows = []
        for row in reader:
            if any(cell.strip() for cell in row):
                rows.append([c.strip() for c in row])
            if len(rows) > max_rows:
                break
    if not rows:
        return [], [], False

    def numeric(text: str) -> bool:
        try:
            float(text)
        except ValueError:
            return False
        return True

    has_header = not all(numeric(c) for c in rows[0] if c)
    if has_header:
        return rows[0], rows[1:], True
    return [f"column {i + 1}" for i in range(len(rows[0]))], rows, False


class MeasuredImportDialog(QDialog):
    """Preview of the parsed columns with dropdowns for time and temperature."""

    def __init__(self, path: Path, system: UnitSystem, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import measured data")
        self._path = path
        self.columns, rows, self.has_header = read_preview(path)

        self.time_combo = QComboBox()
        self.temp_combo = QComboBox()
        for combo in (self.time_combo, self.temp_combo):
            combo.addItems(self.columns)
        if len(self.columns) > 1:
            self.temp_combo.setCurrentIndex(1)
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(["s", "min"])
        self.temp_unit_combo = QComboBox()
        self.temp_unit_combo.addItems(["C", "F", "K"])
        self.temp_unit_combo.setCurrentText("F" if system == "us" else "C")

        self.preview = QTableWidget(len(rows), len(self.columns))
        self.preview.setHorizontalHeaderLabels(self.columns)
        for r, row in enumerate(rows):
            for c, cell in enumerate(row[: len(self.columns)]):
                self.preview.setItem(r, c, QTableWidgetItem(cell))

        form = QFormLayout()
        form.addRow("Time column", self.time_combo)
        form.addRow("Time unit", self.time_unit_combo)
        form.addRow("Temperature column", self.temp_combo)
        form.addRow("Temperature unit", self.temp_unit_combo)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"File: {path.name}"))
        lay.addWidget(self.preview)
        lay.addLayout(form)
        lay.addWidget(buttons)
        self.resize(560, 460)

    def selection(self) -> MeasuredSelection:
        names_usable = self.has_header
        return MeasuredSelection(
            path=self._path,
            time_col=self.time_combo.currentText() if names_usable else None,
            temp_col=self.temp_combo.currentText() if names_usable else None,
            temp_unit=self.temp_unit_combo.currentText(),
            time_unit=self.time_unit_combo.currentText(),
        )
