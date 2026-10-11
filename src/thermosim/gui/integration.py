"""GUI integration of the real simulator (WP11).

* `run_preflight` builds the model off the GUI thread, so geometry errors can be shown next
  to the offending field, and measures how fast the solver is to estimate the run time.
  It also performs the first-run Numba warm-up.
* `PreflightController` runs it on a worker thread behind a cancellable modal progress
  dialog.
* `QuickEstimatePanel` is the button and read-out for the lumped-model estimate.
* `locate_error` maps a geometry/scenario error message to the editor field it is about.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from thermosim.config.schema import Scenario
from thermosim.config.units import UnitSystem
from thermosim.core.errors import ThermoSimError
from thermosim.gui.formatting import fmt_clock, fmt_temp

LONG_RUN_S = 600.0  # warn when the estimated wall time exceeds this
PREVIEW_CELL_SIZE_M = 0.010  # "10 mm is a quick preview" (schema)
TIMING_STEPS = 50

STATUS_CHECK = "Checking the setup..."
STATUS_WARMUP = "Preparing solver (first run only)..."
STATUS_TIMING = "Estimating the run time..."

_solver_warm = False  # the Numba kernels have been compiled/loaded in this process


@dataclass
class PreflightResult:
    """Outcome of the checks made before a run."""

    ok: bool
    cancelled: bool = False
    error: str = ""
    error_field: str | None = None
    liquid_cells: int = 0
    total_cells: int = 0
    shell_nodes: int = 0
    sim_per_wall: float = 0.0  # simulated seconds per wall second, early in the run
    est_sim_s: float | None = None  # simulated time the run is expected to need
    est_wall_s: float | None = None  # estimated wall time of the run
    quick_estimate_s: float | None = None  # lumped-model time to target

    @property
    def is_long(self) -> bool:
        return self.est_wall_s is not None and self.est_wall_s > LONG_RUN_S


def locate_error(scenario: Scenario, message: str) -> str | None:
    """The dotted editor path (e.g. 'liquid.fill_height_m') a build error message is about."""
    m = message.lower()
    if "fill height" in m or ("liquid" in m and "exceeds" in m):
        return (
            "liquid.fill_height_m"
            if scenario.liquid.fill_height_m is not None
            else "liquid.volume_m3"
        )
    if m.startswith("probe") or " probe " in m[:20]:
        return "probes"
    if "element" in m:
        return _element_row(scenario, m) or "heat_source.path_m"
    if "burner" in m or "footprint" in m:
        return "heat_source.footprint_outer_diameter_m"
    if "cell size" in m or "no liquid cell" in m:
        return "simulation.cell_size_m"
    if "thickness" in m:
        return "vessel.wall_thickness_m"
    return None


def _element_row(scenario: Scenario, message: str) -> str | None:
    """Path of the element vertex ('heat_source.path_m.ROW.2') whose z the message names."""
    found = re.search(r"z=(-?\d+\.?\d*)", message)
    path = getattr(scenario.heat_source, "path_m", None)
    if found is None or path is None:
        return None
    z = float(found.group(1))
    for row, point in enumerate(path):
        if abs(point[2] - z) < 6e-4:
            return f"heat_source.path_m.{row}.2"
    return None


def run_preflight(
    scenario: Scenario,
    cancel: threading.Event | None = None,
    status: Callable[[str], None] | None = None,
) -> PreflightResult:
    """Build the model, warm up the solver and time it. Meant for a worker thread.

    Never raises for scenario/geometry problems: they are returned in the result.
    """
    global _solver_warm
    from thermosim.sim.model import Model
    from thermosim.sources.commands import ScheduleCommand
    from thermosim.validation.lumped import quick_estimate

    cancel = cancel or threading.Event()
    say = status or (lambda _msg: None)

    def cancelled() -> PreflightResult | None:
        return PreflightResult(ok=False, cancelled=True) if cancel.is_set() else None

    say(STATUS_CHECK)
    try:
        model = Model(scenario, ScheduleCommand(scenario))
    except ThermoSimError as exc:
        return PreflightResult(
            ok=False, error=str(exc), error_field=locate_error(scenario, str(exc))
        )
    if (res := cancelled()) is not None:
        return res
    dom = model.domain
    result = PreflightResult(
        ok=True,
        liquid_cells=int(dom.grid.is_liquid.sum()),
        total_cells=int(dom.grid.is_liquid.size),
        shell_nodes=int(dom.shell.n),
    )

    try:
        if not _solver_warm:
            say(STATUS_WARMUP)
        model.advance(0.0)
        model.advance(scenario.simulation.max_sim_time_s, max_steps=2)  # compiles the kernels
        _solver_warm = True
        if (res := cancelled()) is not None:
            return res
        say(STATUS_TIMING)
        t0, sim0 = time.perf_counter(), model.state.t
        model.advance(scenario.simulation.max_sim_time_s, max_steps=TIMING_STEPS)
        wall = max(time.perf_counter() - t0, 1e-9)
        result.sim_per_wall = (model.state.t - sim0) / wall
    except ThermoSimError as exc:
        return PreflightResult(ok=False, error=str(exc), error_field=None)

    try:
        result.quick_estimate_s = quick_estimate(scenario)
    except Exception:  # the estimate is optional; never block a run on it
        result.quick_estimate_s = None
    result.est_sim_s = (
        result.quick_estimate_s
        if result.quick_estimate_s is not None
        else float(scenario.simulation.max_sim_time_s)
    )
    if result.sim_per_wall > 0:
        result.est_wall_s = result.est_sim_s / result.sim_per_wall
    return result


class PreflightController(QObject):
    """Runs `run_preflight` on a worker thread behind a cancellable modal progress dialog.

    `done` is emitted once, on the GUI thread, with the `PreflightResult` (`cancelled` set if
    the user cancelled).
    """

    done = Signal(object)

    def __init__(self, parent: QWidget | None, scenario: Scenario) -> None:
        super().__init__(parent)
        self._scenario = scenario
        self._cancel = threading.Event()
        self._status = STATUS_CHECK
        self._result: PreflightResult | None = None
        self._finished = False
        self._dialog = QProgressDialog(STATUS_CHECK, "Cancel", 0, 0, parent)
        self._dialog.setWindowTitle("ThermoSim")
        self._dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._dialog.setMinimumDuration(0)
        self._dialog.setAutoClose(False)
        self._dialog.setAutoReset(False)
        self._dialog.setMinimumWidth(360)
        self._dialog.canceled.connect(self.cancel)
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._poll)
        self._thread = threading.Thread(target=self._work, name="Preflight", daemon=True)

    @property
    def status_text(self) -> str:
        return str(self._dialog.labelText())

    def start(self) -> None:
        self._dialog.show()
        self._timer.start()
        self._thread.start()

    def cancel(self) -> None:
        """Ask the worker to stop; `done` follows (with `cancelled`) once it notices."""
        self._cancel.set()
        self._dialog.setLabelText("Cancelling...")

    def _set_status(self, text: str) -> None:
        self._status = text  # read on the GUI thread by _poll

    def _work(self) -> None:
        try:
            self._result = run_preflight(self._scenario, self._cancel, self._set_status)
        except Exception as exc:  # never leave the dialog waiting
            self._result = PreflightResult(ok=False, error=f"{type(exc).__name__}: {exc}")

    def _poll(self) -> None:
        if self._finished:
            return
        if self._result is None:
            if not self._cancel.is_set() and self._dialog.labelText() != self._status:
                self._dialog.setLabelText(self._status)
            return
        self._finished = True
        self._timer.stop()
        self._dialog.canceled.disconnect(self.cancel)
        self._dialog.close()
        self._dialog.deleteLater()
        result = self._result
        if self._cancel.is_set() and result.ok:
            result = PreflightResult(ok=False, cancelled=True)
        self.done.emit(result)


class QuickEstimatePanel(QGroupBox):
    """Button and read-out for the lumped-model time-to-target estimate (under 1 s)."""

    requested = Signal()

    def __init__(self, system: UnitSystem = "us") -> None:
        super().__init__("Quick estimate")
        self._system: UnitSystem = system
        self._last: tuple[float | None, float] | None = None  # (seconds, target K)
        self.button = QPushButton("Estimate time to target")
        self.button.setToolTip(
            "A fast single-temperature (lumped) estimate, before running the full simulation."
        )
        self.button.clicked.connect(lambda _c=False: self.requested.emit())
        self.label = QLabel("Not estimated yet.")
        self.label.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.button)
        lay.addWidget(self.label)

    def set_system(self, system: UnitSystem) -> None:
        self._system = system
        if self._last is not None:
            self.show_result(*self._last)

    def show_result(self, seconds: float | None, target_K: float) -> None:
        self._last = (seconds, target_K)
        target = fmt_temp(target_K, self._system)
        if seconds is None:
            self.label.setText(f"Target {target} is not reached within the maximum time.")
        else:
            self.label.setText(
                f"About {fmt_clock(seconds)} to reach {target} (lumped model; the full "
                "simulation will differ somewhat)."
            )

    def show_error(self, message: str) -> None:
        self._last = None
        self.label.setText(f"Cannot estimate: {message}")

    def reset(self) -> None:
        self._last = None
        self.label.setText("Not estimated yet.")


def describe_preflight(result: PreflightResult) -> str:
    """One-line status text for a successful preflight."""
    parts = [f"{result.liquid_cells:,} liquid cells"]
    if result.est_wall_s is not None:
        wall = result.est_wall_s
        parts.append(
            f"estimated run time {wall:.0f} s"
            if wall < 90
            else f"estimated run time {fmt_clock(wall)}"
        )
    return ", ".join(parts)
