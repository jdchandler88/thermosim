"""Right-hand summary panel: status, timings, energy breakdown, fuel and comparison."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from thermosim.config.units import UnitSystem
from thermosim.core.types import EnergyFlows, Sample, StopReason, Summary
from thermosim.gui.formatting import (
    DEFAULT_UNITS,
    fmt_clock,
    fmt_delta,
    fmt_energy,
    fmt_mass,
    fmt_temp,
)

_STOP_TEXT = {
    StopReason.TARGET_REACHED: "Target reached",
    StopReason.PLATEAU: "Levelled off below target",
    StopReason.MAX_TIME: "Maximum time reached",
    StopReason.USER_STOP: "Stopped by user",
    StopReason.ERROR: "Error",
}

_ROWS: list[tuple[str, str]] = [
    ("Heat in (liquid + vessel)", "in"),
    ("Stored in liquid", "stored_liquid"),
    ("Stored in vessel", "stored_shell"),
    ("Stored in lid/headspace", "stored_lid_headspace"),
    ("Loss: vessel surfaces", "loss_shell_outer"),
    ("Loss: open surface (conv/rad)", "loss_surface_convrad"),
    ("Loss: evaporation", "loss_surface_evap"),
    ("Loss: lid", "loss_lid_outer"),
    ("Loss: dry wall (no lid)", "loss_dry_wall_open"),
]


class SummaryPanel(QWidget):
    def __init__(self, system: UnitSystem = DEFAULT_UNITS) -> None:
        super().__init__()
        self._system: UnitSystem = system
        self._flows: EnergyFlows | None = None
        self._summary: Summary | None = None
        self._sample: Sample | None = None
        self._comparison: Any = None
        self._comparison_note = ""
        self._progress: tuple[float, float, float] | None = None
        self._status = "Idle"
        self._peak_speed: float | None = None

        self.status_label = QLabel()
        self.reason_label = QLabel()
        self.time_to_target_label = QLabel()
        self.times_label = QLabel()
        self.final_label = QLabel()
        self.fuel_label = QLabel()
        self.peak_speed_label = QLabel()
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b52b27;")
        self.error_label.hide()

        top = QGroupBox("Run")
        form = QFormLayout(top)
        form.addRow("Status", self.status_label)
        form.addRow("Result", self.reason_label)
        form.addRow("Time to target", self.time_to_target_label)
        form.addRow("Sim / wall time", self.times_label)
        form.addRow("Temperatures", self.final_label)
        form.addRow("Fuel / electricity", self.fuel_label)
        self._peak_speed_title = QLabel("Peak flow speed")
        form.addRow(self._peak_speed_title, self.peak_speed_label)

        self.energy_table = QTableWidget(len(_ROWS), 3)
        self.energy_table.setHorizontalHeaderLabels(["Energy", "Amount", "Share of heat in"])
        self.energy_table.verticalHeader().setVisible(False)
        self.energy_table.horizontalHeader().setStretchLastSection(True)
        self.energy_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._bars: list[QProgressBar] = []
        for r, (label, _) in enumerate(_ROWS):
            self.energy_table.setItem(r, 0, QTableWidgetItem(label))
            self.energy_table.setItem(r, 1, QTableWidgetItem("-"))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            self._bars.append(bar)
            self.energy_table.setCellWidget(r, 2, bar)
        self.energy_table.resizeColumnToContents(0)
        energy_box = QGroupBox("Energy")
        QVBoxLayout(energy_box).addWidget(self.energy_table)

        self.comparison_label = QLabel("No measured data imported.")
        self.comparison_label.setWordWrap(True)
        comp_box = QGroupBox("Measured vs simulated")
        QVBoxLayout(comp_box).addWidget(self.comparison_label)

        lay = QVBoxLayout(self)
        lay.addWidget(top)
        lay.addWidget(self.error_label)
        lay.addWidget(energy_box, 1)
        lay.addWidget(comp_box)
        self._render()

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        self._flows = None
        self._summary = None
        self._sample = None
        self._progress = None
        self._peak_speed = None
        self.error_label.hide()
        self._render()

    def set_status(self, text: str) -> None:
        self._status = text
        self._render()

    def set_progress(self, sim_s: float, wall_s: float, max_s: float) -> None:
        self._progress = (sim_s, wall_s, max_s)
        self._render()

    def set_sample(self, sample: Sample) -> None:
        self._sample = sample
        self._flows = sample.cumulative
        self._render()

    def set_summary(self, summary: Summary) -> None:
        self._summary = summary
        self._flows = summary.energy
        self._render()

    def set_peak_speed(self, speed_m_s: float | None) -> None:
        """Peak flow speed (m/s) seen so far in the slice; None hides the row (no flow data)."""
        self._peak_speed = speed_m_s
        self._render()

    def set_comparison(self, comparison: Any, note: str = "") -> None:
        self._comparison = comparison
        self._comparison_note = note
        self._render()

    def set_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))

    def set_system(self, system: UnitSystem) -> None:
        self._system = system
        self._render()

    # ------------------------------------------------------------------ rendering
    def _render(self) -> None:
        sysm = self._system
        s = self._summary
        self.status_label.setText(self._status)
        self.reason_label.setText(_STOP_TEXT.get(s.stop_reason, str(s.stop_reason)) if s else "-")
        self.time_to_target_label.setText(
            fmt_clock(s.time_to_target_s) if s and s.time_to_target_s is not None else "-"
        )
        if s is not None:
            self.times_label.setText(f"{fmt_clock(s.sim_time_s)} / {fmt_clock(s.wall_time_s)}")
        elif self._progress is not None:
            sim, wall, _ = self._progress
            self.times_label.setText(f"{fmt_clock(sim)} / {fmt_clock(wall)}")
        else:
            self.times_label.setText("-")
        if s is not None:
            self.final_label.setText(
                f"mean {fmt_temp(s.final_T_mean, sysm)}, "
                + ", ".join(f"{k} {fmt_temp(v, sysm)}" for k, v in s.final_probes.items())
            )
        elif self._sample is not None:
            self.final_label.setText(
                f"mean {fmt_temp(self._sample.T_mean, sysm)}, min "
                f"{fmt_temp(self._sample.T_min, sysm)}, max {fmt_temp(self._sample.T_max, sysm)}"
            )
        else:
            self.final_label.setText("-")
        fuel, kwh = (
            (s.fuel_kg, s.electric_kwh)
            if s
            else ((self._sample.fuel_kg, self._sample.electric_kwh) if self._sample else (0, 0))
        )
        if fuel > 0:
            self.fuel_label.setText(f"{fmt_mass(fuel, sysm)} propane")
        elif kwh > 0:
            self.fuel_label.setText(f"{kwh:.3f} kWh electricity")
        else:
            self.fuel_label.setText("-")
        peak = self._peak_speed
        self._peak_speed_title.setVisible(peak is not None)
        self.peak_speed_label.setVisible(peak is not None)
        if peak is not None:
            self.peak_speed_label.setText(f"{peak * 100:.1f} cm/s (in the slice plane)")
        self._render_energy()
        self._render_comparison()

    def _render_energy(self) -> None:
        flows = self._flows
        total_in = flows.total_in() if flows else 0.0
        for r, (_, attr) in enumerate(_ROWS):
            value = 0.0
            if flows is not None:
                value = total_in if attr == "in" else float(getattr(flows, attr))
            item = self.energy_table.item(r, 1)
            assert item is not None
            item.setText(fmt_energy(value, self._system) if flows is not None else "-")
            share = round(100.0 * value / total_in) if total_in > 0 else 0
            self._bars[r].setValue(max(0, min(share, 100)))

    def _render_comparison(self) -> None:
        c = self._comparison
        if c is None:
            self.comparison_label.setText(self._comparison_note or "No measured data imported.")
            return
        sysm = self._system
        lines = [
            f"RMSE {fmt_delta(c.rmse_K, sysm)}, max error {fmt_delta(c.max_abs_K, sysm)}, "
            f"bias {fmt_delta(c.bias_K, sysm)}"
        ]
        if c.time_to_target_sim_s is not None and c.time_to_target_meas_s is not None:
            rel = c.time_to_target_error_rel
            lines.append(
                f"Time to target: sim {fmt_clock(c.time_to_target_sim_s)}, measured "
                f"{fmt_clock(c.time_to_target_meas_s)}"
                + (f" ({rel * 100:+.1f}%)" if rel is not None else "")
            )
        self.comparison_label.setText("\n".join(lines))
