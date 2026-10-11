"""Main window: toolbar, setup editor, slice view, chart, scrubber and summary (WP08)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QElapsedTimer, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from thermosim.config.io import example_scenarios, load_scenario, save_scenario
from thermosim.config.schema import Scenario
from thermosim.config.units import UnitSystem
from thermosim.core.errors import ModeError, ScenarioError
from thermosim.core.protocols import (
    ErrorEvent,
    FinishedEvent,
    ProgressEvent,
    Runner,
    RunnerState,
    SampleEvent,
    SnapshotEvent,
)
from thermosim.core.types import RunMode, RunResult, Sample, Snapshot, SnapshotStoreLike
from thermosim.gui import recording_adapter as rec
from thermosim.gui.chart import Chart
from thermosim.gui.editor import ScenarioEditor
from thermosim.gui.formatting import DEFAULT_UNITS, fmt_clock
from thermosim.gui.measured_dialog import MeasuredImportDialog
from thermosim.gui.runner_factory import make_runner
from thermosim.gui.slice_view import SliceView
from thermosim.gui.snapshot_list import ListSnapshotStore
from thermosim.gui.summary_panel import SummaryPanel

RunnerFactory = Callable[[Scenario, RunMode], Runner]
_SLIDER_STEPS = 1000
_PLAYBACK_SPEEDS = [("10x", 10.0), ("60x", 60.0), ("300x", 300.0), ("1000x", 1000.0)]


class MainWindow(QMainWindow):
    """The ThermoSim desktop window."""

    def __init__(
        self,
        settings: QSettings | None = None,
        runner_factory: RunnerFactory | None = None,
        poll_interval_ms: int = 33,
    ) -> None:
        super().__init__()
        self._settings = settings if settings is not None else QSettings("ThermoSim", "ThermoSim")
        self._runner_factory: RunnerFactory = runner_factory or make_runner
        self._system: UnitSystem = self._load_units()
        self._path: Path | None = None
        self._runner: Runner | None = None
        self._active = False
        self._run_mode = RunMode.FAST
        self._samples: list[Sample] = []
        self._store: SnapshotStoreLike = ListSnapshotStore()
        self._result: RunResult | None = None
        self._measured: Any = None
        self._displayed: Snapshot | None = None
        self._updating_scrub = False
        self._playing = False
        self._play_t = 0.0
        self._last_progress: ProgressEvent | None = None
        self._last_error = ""

        self.editor = ScenarioEditor(self._system)
        self.slice_view = SliceView(self._system)
        self.chart = Chart(self._system)
        self.summary = SummaryPanel(self._system)

        self._build_actions()
        self._build_toolbar()
        self._build_layout()

        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self._tick)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(33)
        self._play_timer.timeout.connect(self._play_step)
        self._play_clock = QElapsedTimer()

        self.editor.changed.connect(self._on_editor_changed)
        self._status_label = QLabel()
        self.statusBar().addWidget(self._status_label, 1)

        self.resize(1500, 900)
        self.new_scenario(force=True)
        self._update_actions()
        self._update_status()

    # ------------------------------------------------------------------ construction
    def _load_units(self) -> UnitSystem:
        value = self._settings.value("units", DEFAULT_UNITS)
        return "metric" if value == "metric" else "us"

    def _action(self, text: str, slot: Callable[[], Any], tip: str = "") -> QAction:
        act = QAction(text, self)
        act.triggered.connect(lambda _checked=False: slot())
        if tip:
            act.setToolTip(tip)
        return act

    def _build_actions(self) -> None:
        self.act_new = self._action("New", lambda: self.new_scenario())
        self.act_open = self._action("Open...", lambda: self.open_file())
        self.act_save = self._action("Save", lambda: self.save())
        self.act_save_as = self._action("Save As...", lambda: self.save_as())
        self.act_run_fast = self._action("Run (Fast)", lambda: self.start_run(RunMode.FAST))
        self.act_run_rt = self._action("Run (Real-time)", lambda: self.start_run(RunMode.REALTIME))
        self.act_pause = self._action("Pause", self.pause_resume)
        self.act_stop = self._action("Stop", self.stop_run)
        self.act_export = self._action("Export CSV", lambda: self.export_csv())
        self.act_import = self._action("Import measured", lambda: self.import_measured())
        self.act_quit = self._action("Quit", self.close)
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        for act in (self.act_new, self.act_open):
            file_menu.addAction(act)
        self._examples_menu = file_menu.addMenu("Examples")
        for name in example_scenarios():
            act = QAction(name.replace("_", " "), self)
            act.triggered.connect(lambda _c=False, n=name: self.load_example(n))
            self._examples_menu.addAction(act)
        for act in (self.act_save, self.act_save_as, self.act_export, self.act_import):
            file_menu.addAction(act)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)
        run_menu = menu.addMenu("&Run")
        for act in (self.act_run_fast, self.act_run_rt, self.act_pause, self.act_stop):
            run_menu.addAction(act)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        for act in (self.act_new, self.act_open, self.act_save, self.act_save_as):
            tb.addAction(act)
        tb.addSeparator()
        self.run_button = QToolButton()
        self.run_button.setText("Run")
        self.run_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.run_button.setDefaultAction(self.act_run_fast)
        from PySide6.QtWidgets import QMenu

        run_menu = QMenu(self.run_button)
        run_menu.addAction(self.act_run_fast)
        run_menu.addAction(self.act_run_rt)
        self.run_button.setMenu(run_menu)
        self.run_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        tb.addWidget(self.run_button)
        tb.addAction(self.act_pause)
        tb.addAction(self.act_stop)
        tb.addSeparator()
        tb.addWidget(QLabel(" Speed "))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 100.0)
        self.speed_spin.setDecimals(1)
        self.speed_spin.setSingleStep(0.5)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSuffix("x")
        self.speed_spin.valueChanged.connect(self._on_speed)
        tb.addWidget(self.speed_spin)
        tb.addWidget(QLabel("  Power "))
        self.power_slider = QSlider(Qt.Orientation.Horizontal)
        self.power_slider.setRange(0, 100)
        self.power_slider.setValue(100)
        self.power_slider.setFixedWidth(140)
        self.power_slider.valueChanged.connect(self._on_power)
        tb.addWidget(self.power_slider)
        self.power_label = QLabel("100%")
        self.power_label.setMinimumWidth(40)
        tb.addWidget(self.power_label)
        tb.addSeparator()
        tb.addWidget(QLabel(" Units "))
        self.units_combo = QComboBox()
        self.units_combo.addItem("Metric", "metric")
        self.units_combo.addItem("US", "us")
        self.units_combo.setCurrentIndex(self.units_combo.findData(self._system))
        self.units_combo.currentIndexChanged.connect(
            lambda _i: self.set_units(self.units_combo.currentData())
        )
        tb.addWidget(self.units_combo)
        tb.addSeparator()
        tb.addAction(self.act_export)
        tb.addAction(self.act_import)

    def _build_layout(self) -> None:
        # scrubber row
        self.play_button = QToolButton()
        self.play_button.setText("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setRange(0, _SLIDER_STEPS)
        self.scrubber.valueChanged.connect(self._on_scrub)
        self.time_label = QLabel("t = --:--")
        self.time_label.setMinimumWidth(100)
        self.playback_speed = QComboBox()
        for label, factor in _PLAYBACK_SPEEDS:
            self.playback_speed.addItem(label, factor)
        self.playback_speed.setCurrentIndex(1)
        self.follow_live = QCheckBox("Follow live")
        self.follow_live.setChecked(True)
        scrub_row = QHBoxLayout()
        scrub_row.addWidget(self.play_button)
        scrub_row.addWidget(self.scrubber, 1)
        scrub_row.addWidget(self.time_label)
        scrub_row.addWidget(QLabel("Playback"))
        scrub_row.addWidget(self.playback_speed)
        scrub_row.addWidget(self.follow_live)

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        vsplit = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.slice_view, 1)
        tl.addLayout(scrub_row)
        vsplit.addWidget(top)
        vsplit.addWidget(self.chart)
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 2)
        cl.addWidget(vsplit)

        hsplit = QSplitter(Qt.Orientation.Horizontal)
        self.editor.setMinimumWidth(430)
        hsplit.addWidget(self.editor)
        hsplit.addWidget(center)
        hsplit.addWidget(self.summary)
        hsplit.setStretchFactor(0, 0)
        hsplit.setStretchFactor(1, 1)
        hsplit.setStretchFactor(2, 0)
        hsplit.setSizes([440, 760, 320])
        self.setCentralWidget(hsplit)

    # ------------------------------------------------------------------ state helpers
    @property
    def system(self) -> UnitSystem:
        return self._system

    @property
    def is_active(self) -> bool:
        """True from start until the run's FinishedEvent/ErrorEvent has been handled."""
        return self._active

    @property
    def runner(self) -> Runner | None:
        return self._runner

    @property
    def result(self) -> RunResult | None:
        return self._result

    @property
    def store(self) -> SnapshotStoreLike:
        return self._store

    @property
    def samples(self) -> list[Sample]:
        return self._samples

    @property
    def displayed_snapshot(self) -> Snapshot | None:
        return self._displayed

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def scenario_path(self) -> Path | None:
        return self._path

    def status_text(self) -> str:
        return str(self._status_label.text())

    def _update_title(self) -> None:
        name = self.editor.data().get("name", "") or "Untitled"
        star = "*" if self.editor.is_dirty() else ""
        self.setWindowTitle(f"{name}{star} - ThermoSim")

    def _on_editor_changed(self) -> None:
        self._update_title()
        self._update_actions()

    def _update_actions(self) -> None:
        valid = self.editor.is_valid()
        active = self._active
        run_ok = valid and not active
        tip = "" if valid else f"Fix the setup first:\n{self.editor.error_message()}"
        for act, base in (
            (self.act_run_fast, "Run in fast mode"),
            (self.act_run_rt, "Run in real time"),
        ):
            act.setEnabled(run_ok)
            act.setToolTip(base if valid else tip)
        self.act_pause.setEnabled(active)
        paused = self._runner is not None and self._runner.state == RunnerState.PAUSED
        self.act_pause.setText("Resume" if paused else "Pause")
        self.act_stop.setEnabled(active)
        rt = active and self._run_mode == RunMode.REALTIME
        self.speed_spin.setEnabled(rt)
        self.power_slider.setEnabled(rt)
        self.editor.setEnabled(not active)
        self.act_new.setEnabled(not active)
        self.act_open.setEnabled(not active)
        self._examples_menu.setEnabled(not active)
        have_samples = bool(self._samples)
        csv_ok = rec.export_csv_available()
        self.act_export.setEnabled(csv_ok and have_samples)
        self.act_export.setToolTip(
            "Export the time series as CSV" if csv_ok else rec.UNAVAILABLE_TIP
        )
        meas_ok = rec.measured_available()
        self.act_import.setEnabled(meas_ok)
        self.act_import.setToolTip(
            "Overlay measured temperatures from a CSV file" if meas_ok else rec.UNAVAILABLE_TIP
        )
        self.scrubber.setEnabled(len(self._store) > 0)
        self.play_button.setEnabled(len(self._store) > 1)

    def _update_status(self) -> None:
        state = self._runner.state.value if self._runner is not None else "idle"
        p = self._last_progress
        sim = p.sim_time_s if p else 0.0
        wall = p.wall_time_s if p else 0.0
        lag = p.realtime_lag_s if p else 0.0
        self._status_label.setText(
            f"State: {state}   |   sim {fmt_clock(sim)}   |   wall {fmt_clock(wall)}"
            f"   |   real-time lag {lag:.1f} s"
        )
        self.summary.set_status(state.capitalize())

    # ------------------------------------------------------------------ units
    def set_units(self, system: UnitSystem) -> None:
        """Switch the display units everywhere. Stored SI values are untouched."""
        if system not in ("metric", "us"):
            raise ValueError(system)
        self._system = system
        self._settings.setValue("units", system)
        idx = self.units_combo.findData(system)
        if self.units_combo.currentIndex() != idx:
            self.units_combo.blockSignals(True)
            self.units_combo.setCurrentIndex(idx)
            self.units_combo.blockSignals(False)
        self.editor.set_units(system)
        self.slice_view.set_system(system)
        self.chart.set_system(system)
        self.summary.set_system(system)

    # ------------------------------------------------------------------ files
    def _maybe_save(self) -> bool:
        if not self.editor.is_dirty():
            return True
        name = self.editor.data().get("name", "Untitled")
        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            f"Save changes to '{name}'?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self.save()
        return reply == QMessageBox.StandardButton.Discard

    def _reset_run_views(self, scenario: Scenario | None) -> None:
        self._samples = []
        self._store = ListSnapshotStore()
        self._result = None
        self._displayed = None
        self._last_progress = None
        self.slice_view.clear()
        self.chart.reset(scenario)
        self.summary.reset()
        self.summary.set_comparison(None)
        self._update_scrubber_range()
        self.time_label.setText("t = --:--")
        self.chart.set_cursor(None)

    def load_scenario(self, scenario: Scenario, path: Path | None = None) -> None:
        """Load `scenario` into the editor and reset the run views."""
        self._path = path
        self.editor.set_scenario(scenario)
        self.slice_view.set_scenario(scenario)
        self.speed_spin.blockSignals(True)
        self.speed_spin.setValue(scenario.simulation.realtime_speed)
        self.speed_spin.blockSignals(False)
        self._reset_run_views(scenario)
        if self._measured is not None:
            self.chart.set_measured(self._measured.t, self._measured.T, self._measured.label)
        self._update_title()
        self._update_actions()

    def new_scenario(self, force: bool = False) -> bool:
        if not force and not self._maybe_save():
            return False
        template = example_scenarios()["small_pot_stove"].model_copy(update={"name": "New setup"})
        self.load_scenario(template, None)
        return True

    def load_example(self, name: str) -> bool:
        if not self._maybe_save():
            return False
        self.load_scenario(example_scenarios()[name], None)
        return True

    def open_file(self, path: Path | str | None = None) -> bool:
        if not self._maybe_save():
            return False
        if path is None:
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Open setup", "", "Setup files (*.toml);;All files (*)"
            )
            if not chosen:
                return False
            path = chosen
        try:
            scenario = load_scenario(Path(path))
        except ScenarioError as exc:
            QMessageBox.warning(self, "Cannot open setup", str(exc))
            return False
        self.load_scenario(scenario, Path(path))
        return True

    def save(self, path: Path | str | None = None) -> bool:
        target = Path(path) if path is not None else self._path
        if target is None:
            return self.save_as()
        scenario = self.editor.scenario()
        if scenario is None:
            self.statusBar().showMessage("Fix the setup errors before saving.", 5000)
            return False
        save_scenario(scenario, target)
        self._path = target
        self.editor.mark_clean()
        self._update_title()
        return True

    def save_as(self, path: Path | str | None = None) -> bool:
        if path is None:
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Save setup as", "", "Setup files (*.toml)"
            )
            if not chosen:
                return False
            path = chosen if chosen.endswith(".toml") else chosen + ".toml"
        return self.save(path)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._active and self._runner is not None:
            self._runner.stop()
        if not self._maybe_save():
            event.ignore()
            return
        self._timer.stop()
        self._play_timer.stop()
        event.accept()

    # ------------------------------------------------------------------ running
    def start_run(self, mode: RunMode | None = None) -> bool:
        if self._active:
            return False
        scenario = self.editor.scenario()
        if scenario is None:
            self.statusBar().showMessage("The setup has errors; fix them before running.", 5000)
            return False
        mode = mode or self._run_mode
        self._run_mode = mode
        speed = self.speed_spin.value() if mode == RunMode.REALTIME else 1.0
        sim = scenario.simulation.model_copy(
            update={
                "mode": mode,
                "realtime_speed": speed
                if mode == RunMode.REALTIME
                else scenario.simulation.realtime_speed,
            }
        )
        run_scenario = scenario.model_copy(update={"simulation": sim})
        self._stop_playback()
        self._reset_run_views(run_scenario)
        self.slice_view.set_scenario(run_scenario)
        self._last_error = ""
        self.power_slider.blockSignals(True)
        self.power_slider.setValue(round(run_scenario.power_schedule[0].command * 100))
        self.power_slider.blockSignals(False)
        self.power_label.setText(f"{self.power_slider.value()}%")
        self._runner = self._runner_factory(run_scenario, mode)
        self._active = True
        self.follow_live.setChecked(True)
        self._runner.start()
        self._timer.start()
        self._update_actions()
        self._update_status()
        return True

    def pause_resume(self) -> None:
        r = self._runner
        if r is None or not self._active:
            return
        if r.state == RunnerState.PAUSED:
            r.resume()
        else:
            r.pause()
        self._update_actions()
        self._update_status()

    def stop_run(self) -> None:
        if self._runner is not None and self._active:
            self._runner.stop()

    def _on_speed(self, value: float) -> None:
        if self._runner is not None and self._active and self._run_mode == RunMode.REALTIME:
            try:
                self._runner.set_speed(value)
            except ModeError:
                pass

    def _on_power(self, value: int) -> None:
        self.power_label.setText(f"{value}%")
        if self._runner is not None and self._active and self._run_mode == RunMode.REALTIME:
            try:
                self._runner.set_power_command(value / 100.0)
            except ModeError:
                pass

    def _tick(self) -> None:
        """Poll the runner (30 Hz) and update the views. Never blocks."""
        runner = self._runner
        if runner is None:
            self._timer.stop()
            return
        events = runner.poll_events()
        new_samples: list[Sample] = []
        newest: Snapshot | None = None
        finished: RunResult | None = None
        error: ErrorEvent | None = None
        for ev in events:
            if isinstance(ev, SampleEvent):
                new_samples.append(ev.sample)
            elif isinstance(ev, SnapshotEvent):
                if isinstance(self._store, ListSnapshotStore):
                    self._store.add(ev.snapshot)
                newest = ev.snapshot
            elif isinstance(ev, ProgressEvent):
                self._last_progress = ev
            elif isinstance(ev, FinishedEvent):
                finished = ev.result
            elif isinstance(ev, ErrorEvent):
                error = ev
        if new_samples:
            self._samples.extend(new_samples)
            self.chart.add_samples(new_samples)
            self.summary.set_sample(new_samples[-1])
        if self._last_progress is not None:
            p = self._last_progress
            self.summary.set_progress(p.sim_time_s, p.wall_time_s, p.max_time_s)
        if newest is not None:
            self._update_scrubber_range()
            if self.follow_live.isChecked():
                self._show_snapshot(newest)
                self._set_scrubber_to(newest.t)
        if error is not None:
            self._last_error = error.message
            self.summary.set_error(error.message)
            self._end_run()
        elif finished is not None:
            self._on_finished(finished)
        self._update_status()

    def _end_run(self) -> None:
        self._active = False
        self._timer.stop()
        self._update_actions()

    def _on_finished(self, result: RunResult) -> None:
        self._result = result
        self._samples = list(result.samples)
        self._store = result.snapshots
        self.chart.set_samples(self._samples)
        self.summary.set_summary(result.summary)
        self._last_progress = ProgressEvent(
            result.summary.sim_time_s, result.summary.wall_time_s, result.summary.sim_time_s, 0.0
        )
        self.summary.set_progress(result.summary.sim_time_s, result.summary.wall_time_s, 0.0)
        self._end_run()
        self._update_scrubber_range()
        if len(self._store):
            last = self._store[len(self._store) - 1]
            self._show_snapshot(last)
            self._set_scrubber_to(last.t)
        self._update_comparison()

    # ------------------------------------------------------------------ playback / scrub
    def _t_max(self) -> float:
        times = self._store.times() if len(self._store) else []
        return times[-1] if times else 0.0

    def _update_scrubber_range(self) -> None:
        self.scrubber.setEnabled(len(self._store) > 0)
        self.play_button.setEnabled(len(self._store) > 1)

    def _set_scrubber_to(self, t: float) -> None:
        t_max = self._t_max()
        value = round(_SLIDER_STEPS * t / t_max) if t_max > 0 else 0
        self._updating_scrub = True
        self.scrubber.setValue(min(max(value, 0), _SLIDER_STEPS))
        self._updating_scrub = False

    def _show_snapshot(self, snap: Snapshot) -> None:
        self._displayed = snap
        self.slice_view.set_snapshot(snap)
        self.chart.set_cursor(snap.t)
        self.time_label.setText(f"t = {fmt_clock(snap.t)}")

    def scrub_to(self, t: float) -> None:
        """Show the snapshot nearest to simulated time `t` (what dragging the slider does)."""
        if not len(self._store):
            return
        if self._active:
            self.follow_live.setChecked(False)
        self._show_snapshot(self._store.nearest(t))
        self._set_scrubber_to(t)

    def _on_scrub(self, value: int) -> None:
        if self._updating_scrub or not len(self._store):
            return
        if self._active:
            self.follow_live.setChecked(False)
        t = self._t_max() * value / _SLIDER_STEPS
        self._show_snapshot(self._store.nearest(t))

    def toggle_playback(self) -> None:
        if self._playing:
            self._stop_playback()
            return
        if len(self._store) < 2:
            return
        if self._active:
            self.follow_live.setChecked(False)
        t_max = self._t_max()
        cur = self._displayed.t if self._displayed is not None else 0.0
        self._play_t = 0.0 if cur >= t_max - 1e-9 else cur
        self._playing = True
        self.play_button.setText("Pause")
        self._play_clock.start()
        self._play_timer.start()

    def _stop_playback(self) -> None:
        self._playing = False
        self._play_timer.stop()
        self.play_button.setText("Play")

    def _play_step(self) -> None:
        dt_wall = self._play_clock.restart() / 1000.0
        factor = float(self.playback_speed.currentData())
        self._play_t += dt_wall * factor
        t_max = self._t_max()
        if self._play_t >= t_max:
            self._play_t = t_max
            self._stop_playback()
        self._show_snapshot(self._store.nearest(self._play_t))
        self._set_scrubber_to(self._play_t)

    # ------------------------------------------------------------------ export / import
    def export_csv(self, path: Path | str | None = None) -> bool:
        if not (rec.export_csv_available() and self._samples):
            return False
        if path is None:
            chosen, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV files (*.csv)")
            if not chosen:
                return False
            path = chosen
        rec.export_csv(self._samples, Path(path), self._system)
        return True

    def import_measured(self, path: Path | str | None = None) -> bool:
        if not rec.measured_available():
            return False
        if path is None:
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Import measured data", "", "CSV files (*.csv);;All files (*)"
            )
            if not chosen:
                return False
            path = chosen
        dialog = MeasuredImportDialog(Path(path), self._system, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return False
        sel = dialog.selection()
        try:
            series = rec.load_measured(
                sel.path, sel.time_col, sel.temp_col, sel.temp_unit, sel.time_unit
            )
        except Exception as exc:
            QMessageBox.warning(self, "Cannot import", str(exc))
            return False
        self.apply_measured(series)
        return True

    def apply_measured(self, series: Any) -> None:
        """Overlay a measured series (has .t [s], .T [K], .label) and update the comparison."""
        self._measured = series
        self.chart.set_measured(series.t, series.T, series.label)
        self._update_comparison()

    def _update_comparison(self) -> None:
        if self._measured is None:
            self.summary.set_comparison(None)
            return
        if not rec.compare_available():
            self.summary.set_comparison(None, f"Measured data shown. {rec.UNAVAILABLE_TIP}")
            return
        if not self._samples:
            self.summary.set_comparison(None, "Run the simulation to compare with measured data.")
            return
        scenario = self.editor.scenario()
        target_K = scenario.target.temp_K if scenario else None
        probe = scenario.target.probe if scenario else "mean"
        names = list(self._samples[-1].probes)
        candidates = [probe] + [n for n in names if n != probe]
        last: Exception | None = None
        for cand in candidates:
            try:
                comparison = rec.compare(self._samples, cand, self._measured, target_K)
            except Exception as exc:
                last = exc
                continue
            self.summary.set_comparison(comparison)
            return
        self.summary.set_comparison(None, f"Comparison failed: {last}")
