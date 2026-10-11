"""Time-series chart: probes, mean, min-max band, target, power and measured data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from thermosim.config.schema import Scenario
from thermosim.config.units import UnitSystem, to_display, unit_label
from thermosim.core.types import Sample
from thermosim.gui.formatting import DEFAULT_UNITS, fmt_clock, temp, temp_array, temp_label

_PROBE_COLORS = [
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (204, 121, 167),
    (213, 94, 0),
    (153, 153, 255),
]
_NICE_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 21600, 43200]


class TimeAxis(pg.AxisItem):  # type: ignore[misc]
    """Bottom axis in seconds, labelled mm:ss, or h:mm when the span exceeds an hour."""

    def tickSpacing(self, minVal: float, maxVal: float, size: float) -> list[tuple[float, float]]:
        span = max(maxVal - minVal, 1e-9)
        want = max(size / 90.0, 2.0)
        step = next((s for s in _NICE_STEPS if span / s <= want), _NICE_STEPS[-1])
        return [(float(step), 0.0)]

    def tickStrings(self, values: Any, scale: float, spacing: float) -> list[str]:
        long = (self.range[1] - self.range[0]) > 3600 if hasattr(self, "range") else False
        out = []
        for v in values:
            total = round(max(v, 0.0))
            if long:
                out.append(f"{total // 3600}:{(total % 3600) // 60:02d}")
            else:
                out.append(f"{total // 60:02d}:{total % 60:02d}")
        return out


class Chart(QWidget):
    """Live and replayed temperature/power chart."""

    def __init__(self, system: UnitSystem = DEFAULT_UNITS) -> None:
        super().__init__()
        self._system: UnitSystem = system
        self._t: list[float] = []
        self._mean: list[float] = []
        self._min: list[float] = []
        self._max: list[float] = []
        self._power: list[float] = []
        self._probes: dict[str, list[float]] = {}
        self._probe_curves: dict[str, Any] = {}
        self._measured: tuple[np.ndarray, np.ndarray, str] | None = None
        self._target_K: float | None = None

        self._plot = pg.PlotWidget(axisItems={"bottom": TimeAxis(orientation="bottom")})
        pi = self._plot.getPlotItem()
        self._pi = pi
        pi.showGrid(x=True, y=True, alpha=0.25)
        self._legend = pi.addLegend(offset=(10, 10))
        for name in ("left", "bottom", "right"):
            pi.getAxis(name).enableAutoSIPrefix(False)
        pi.setLabel("bottom", "Time", units="mm:ss")

        # band, mean, target, cursor
        self._min_curve = pg.PlotDataItem(pen=pg.mkPen(None))
        self._max_curve = pg.PlotDataItem(pen=pg.mkPen(None))
        self._band = pg.FillBetweenItem(self._min_curve, self._max_curve, brush=(255, 255, 255, 40))
        pi.addItem(self._band)
        self._mean_curve = pi.plot(pen=pg.mkPen((255, 255, 255), width=2.5), name="Mean")
        self._target_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen((255, 90, 90), width=1.5, style=Qt.PenStyle.DashLine)
        )
        self._target_line.setVisible(False)
        pi.addItem(self._target_line)
        self._cursor = pg.InfiniteLine(
            angle=90, pen=pg.mkPen((255, 200, 0), width=1.5, style=Qt.PenStyle.DashLine)
        )
        self._cursor.setVisible(False)
        pi.addItem(self._cursor)
        self._measured_item = pg.ScatterPlotItem(
            size=6, pen=pg.mkPen(None), brush=pg.mkBrush(255, 160, 40)
        )
        self._measured_item.setVisible(False)
        pi.addItem(self._measured_item)

        # power on the right axis
        self._p2 = pg.ViewBox()
        pi.showAxis("right")
        pi.scene().addItem(self._p2)
        pi.getAxis("right").linkToView(self._p2)
        self._p2.setXLink(pi)
        self._power_curve = pg.PlotCurveItem(pen=pg.mkPen((120, 220, 120), width=1.5))
        self._p2.addItem(self._power_curve)
        self._legend.addItem(self._power_curve, "Power")
        pi.vb.sigResized.connect(self._sync_views)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._plot)
        self._refresh_labels()

    # ------------------------------------------------------------------ views
    def _sync_views(self) -> None:
        self._p2.setGeometry(self._pi.vb.sceneBoundingRect())
        self._p2.linkedViewChanged(self._pi.vb, self._p2.XAxis)

    def _refresh_labels(self) -> None:
        self._pi.setLabel("left", "Temperature", units=temp_label(self._system))
        self._pi.setLabel("right", "Power", units=unit_label("power", self._system))

    # ------------------------------------------------------------------ accessors
    @property
    def temperature_unit(self) -> str:
        return str(self._pi.getAxis("left").labelUnits)

    @property
    def power_unit(self) -> str:
        return str(self._pi.getAxis("right").labelUnits)

    @property
    def n_points(self) -> int:
        return len(self._t)

    @property
    def probe_names(self) -> list[str]:
        return list(self._probes)

    @property
    def has_measured(self) -> bool:
        return self._measured is not None

    @property
    def cursor_time(self) -> float | None:
        return float(self._cursor.value()) if self._cursor.isVisible() else None

    @property
    def target_display_value(self) -> float | None:
        return float(self._target_line.value()) if self._target_line.isVisible() else None

    def mean_display_data(self) -> np.ndarray:
        """The mean curve's y values as drawn (display units)."""
        y = self._mean_curve.yData
        return np.asarray([] if y is None else y)

    # ------------------------------------------------------------------ data
    def reset(self, scenario: Scenario | None = None) -> None:
        """Clear all series (keeping measured data) and set up for `scenario`."""
        self._t.clear()
        self._mean.clear()
        self._min.clear()
        self._max.clear()
        self._power.clear()
        for curve in self._probe_curves.values():
            self._pi.removeItem(curve)
            self._legend.removeItem(curve)
        self._probe_curves.clear()
        self._probes.clear()
        if scenario is not None:
            self._target_K = scenario.target.temp_K
            for i, probe in enumerate(scenario.probes):
                color = _PROBE_COLORS[i % len(_PROBE_COLORS)]
                curve = self._pi.plot(pen=pg.mkPen(color, width=1.5), name=probe.name)
                self._probe_curves[probe.name] = curve
                self._probes[probe.name] = []
        self._redraw()

    def add_samples(self, samples: list[Sample]) -> None:
        for s in samples:
            self._append(s)
        if samples:
            self._redraw()

    def set_samples(self, samples: list[Sample]) -> None:
        """Replace the series with `samples` (used when a run finishes)."""
        self._t.clear()
        self._mean.clear()
        self._min.clear()
        self._max.clear()
        self._power.clear()
        for series in self._probes.values():
            series.clear()
        for s in samples:
            self._append(s)
        self._redraw()

    def _append(self, s: Sample) -> None:
        self._t.append(s.t)
        self._mean.append(s.T_mean)
        self._min.append(s.T_min)
        self._max.append(s.T_max)
        self._power.append(s.power_delivered)
        for name, series in self._probes.items():
            series.append(s.probes.get(name, np.nan))

    def set_system(self, system: UnitSystem) -> None:
        self._system = system
        self._refresh_labels()
        self._redraw()

    def set_measured(self, t_s: np.ndarray, T_K: np.ndarray, label: str = "Measured") -> None:
        if self._measured is not None:
            self._legend.removeItem(self._measured_item)
        self._measured = (np.asarray(t_s, dtype=float), np.asarray(T_K, dtype=float), label)
        self._legend.addItem(self._measured_item, label)
        self._measured_item.setVisible(True)
        self._redraw()

    def clear_measured(self) -> None:
        if self._measured is not None:
            self._legend.removeItem(self._measured_item)
        self._measured = None
        self._measured_item.setVisible(False)
        self._redraw()

    def set_cursor(self, t_s: float | None) -> None:
        if t_s is None:
            self._cursor.setVisible(False)
        else:
            self._cursor.setValue(t_s)
            self._cursor.setVisible(True)

    def _redraw(self) -> None:
        sysm = self._system
        t = np.asarray(self._t, dtype=float)
        self._mean_curve.setData(t, temp_array(np.asarray(self._mean, dtype=float), sysm))
        self._min_curve.setData(t, temp_array(np.asarray(self._min, dtype=float), sysm))
        self._max_curve.setData(t, temp_array(np.asarray(self._max, dtype=float), sysm))
        for name, curve in self._probe_curves.items():
            curve.setData(t, temp_array(np.asarray(self._probes[name], dtype=float), sysm))
        power = np.asarray(to_display(np.asarray(self._power, dtype=float), "power", sysm))
        self._power_curve.setData(t, power)
        if power.size:
            self._p2.setYRange(0.0, max(float(power.max()) * 1.15, 1e-9), padding=0)
        if self._target_K is not None:
            self._target_line.setValue(temp(self._target_K, sysm))
            self._target_line.setVisible(True)
        if self._measured is not None:
            mt, mT, _ = self._measured
            self._measured_item.setData(mt, temp_array(mT, sysm))
        self._pi.setTitle(None)

    def clock_text(self, t_s: float) -> str:
        return fmt_clock(t_s)
