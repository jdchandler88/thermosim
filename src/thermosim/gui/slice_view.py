"""Slice heat map (y = 0 plane) with vessel walls, lid, heat source and probes (WP08 req. 2)."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from thermosim.config.schema import Burner, Element, Scenario
from thermosim.config.units import UnitSystem, from_display
from thermosim.core.types import Snapshot
from thermosim.gui.formatting import (
    DEFAULT_UNITS,
    fmt_temp,
    plot_length,
    plot_length_label,
    temp,
    temp_label,
)


def get_colormap() -> Any:
    """inferno, falling back to viridis."""
    for name in ("inferno", "viridis"):
        try:
            return pg.colormap.get(name)
        except Exception:
            continue
    return pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])


class _ColoredLines(pg.GraphicsObject):  # type: ignore[misc]
    """A set of thick line segments, each with its own colour."""

    def __init__(self, width: float = 6.0) -> None:
        super().__init__()
        self._lines: list[tuple[QPointF, QPointF, QColor]] = []
        self._rect = QRectF()
        self._width = width

    def set_lines(self, lines: list[tuple[QPointF, QPointF, QColor]]) -> None:
        self.prepareGeometryChange()
        self._lines = lines
        if lines:
            xs = [p.x() for a, b, _ in lines for p in (a, b)]
            ys = [p.y() for a, b, _ in lines for p in (a, b)]
            self._rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        else:
            self._rect = QRectF()
        self.update()

    @property
    def n_lines(self) -> int:
        return len(self._lines)

    def boundingRect(self) -> QRectF:
        pad = 0.02 * max(self._rect.width(), self._rect.height(), 1e-6)
        return self._rect.adjusted(-pad, -pad, pad, pad)

    def paint(self, painter: Any, *args: Any) -> None:
        for a, b, color in self._lines:
            pen = QPen(color, self._width)
            pen.setCosmetic(True)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(pen)
            painter.drawLine(a, b)


class SliceView(QWidget):
    """Heat map of the latest or scrubbed snapshot."""

    def __init__(self, system: UnitSystem = DEFAULT_UNITS) -> None:
        super().__init__()
        self._system: UnitSystem = system
        self._scenario: Scenario | None = None
        self._snapshot: Snapshot | None = None
        self._cmap = get_colormap()
        self._mode = "fixed"
        self._fixed_K: tuple[float, float] = (293.15, 373.15)
        self._levels_K: tuple[float, float] = self._fixed_K
        self._rect_disp = QRectF()
        self._decor: list[Any] = []

        # --- controls strip
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Fixed", "fixed")
        self._mode_combo.addItem("Auto", "auto")
        self._lo_edit = QLineEdit()
        self._hi_edit = QLineEdit()
        for edit in (self._lo_edit, self._hi_edit):
            edit.setMaximumWidth(70)
            edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._range_unit = QLabel()
        strip = QHBoxLayout()
        strip.addWidget(QLabel("Color range"))
        strip.addWidget(self._mode_combo)
        strip.addWidget(self._lo_edit)
        strip.addWidget(QLabel("to"))
        strip.addWidget(self._hi_edit)
        strip.addWidget(self._range_unit)
        strip.addStretch(1)
        self._mode_combo.currentIndexChanged.connect(self._on_mode)
        self._lo_edit.editingFinished.connect(self._on_range_edit)
        self._hi_edit.editingFinished.connect(self._on_range_edit)

        # --- plot
        self._glw = pg.GraphicsLayoutWidget()
        self._plot = self._glw.addPlot(row=0, col=0)
        self._plot.setAspectLocked(True)
        self._plot.showGrid(x=False, y=False)
        self._image = pg.ImageItem()
        self._plot.addItem(self._image)
        self._walls = _ColoredLines(width=7.0)
        self._walls.setZValue(10)
        self._plot.addItem(self._walls)
        self._colorbar = pg.ColorBarItem(
            values=self._fixed_K, colorMap=self._cmap, interactive=False, width=16
        )
        self._glw.addItem(self._colorbar, row=0, col=1)
        self._plot.scene().sigMouseMoved.connect(self._on_mouse)

        self._readout = QLabel(" ")
        self._readout.setMinimumHeight(18)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(strip)
        lay.addWidget(self._glw, 1)
        lay.addWidget(self._readout)
        self._refresh_axes()
        self._refresh_range_edits()

    # ------------------------------------------------------------------ public API
    @property
    def snapshot(self) -> Snapshot | None:
        return self._snapshot

    @property
    def levels_K(self) -> tuple[float, float]:
        return self._levels_K

    @property
    def color_mode(self) -> str:
        return self._mode

    @property
    def readout_text(self) -> str:
        return str(self._readout.text())

    def displayed_image(self) -> np.ndarray | None:
        """The RGBA array currently on screen (nx, nz, 4), or None."""
        img = self._image.image
        return None if img is None else np.asarray(img)

    def n_wall_segments(self) -> int:
        return int(self._walls.n_lines)

    def set_system(self, system: UnitSystem) -> None:
        self._system = system
        self._refresh_axes()
        self._rebuild_decor()
        self._refresh_range_edits()
        self._render()

    def set_scenario(self, scenario: Scenario | None) -> None:
        self._scenario = scenario
        if scenario is not None:
            self._fixed_K = (scenario.liquid.initial_temp_K, scenario.target.temp_K + 2.0)
        self._rebuild_decor()
        self._refresh_range_edits()
        self._render()

    def set_color_mode(self, mode: str) -> None:
        """'auto' (current min/max) or 'fixed'."""
        idx = self._mode_combo.findData(mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)

    def set_fixed_range_K(self, lo: float, hi: float) -> None:
        self._fixed_K = (lo, hi)
        self._refresh_range_edits()
        self._render()

    def set_snapshot(self, snap: Snapshot | None) -> None:
        self._snapshot = snap
        self._render()

    def clear(self) -> None:
        self._snapshot = None
        self._image.clear()
        self._walls.set_lines([])
        self._readout.setText(" ")

    def readout_at(self, x_disp: float, z_disp: float) -> str:
        """Hover text for a point in plot (display length) coordinates."""
        snap = self._snapshot
        if snap is None or self._rect_disp.isEmpty():
            return " "
        r = self._rect_disp
        nx, nz = snap.slice_T.shape
        if not (r.left() <= x_disp < r.right() and r.top() <= z_disp < r.bottom()):
            return " "
        i = min(int((x_disp - r.left()) / r.width() * nx), nx - 1)
        k = min(int((z_disp - r.top()) / r.height() * nz), nz - 1)
        value = float(snap.slice_T[i, k])
        unit = plot_length_label(self._system)
        coords = f"x, z: {x_disp:.1f}, {z_disp:.1f} {unit}"
        if not math.isfinite(value):
            return coords
        return f"{coords}   {fmt_temp(value, self._system)}"

    # ------------------------------------------------------------------ internals
    def _refresh_axes(self) -> None:
        unit = plot_length_label(self._system)
        self._plot.setLabel("bottom", "x", units=unit)
        self._plot.setLabel("left", "z", units=unit)
        self._colorbar.axis.setLabel(temp_label(self._system))
        self._range_unit.setText(temp_label(self._system))

    def _refresh_range_edits(self) -> None:
        lo, hi = self._fixed_K
        for edit, value in ((self._lo_edit, lo), (self._hi_edit, hi)):
            edit.blockSignals(True)
            edit.setText(f"{temp(value, self._system):.1f}")
            edit.blockSignals(False)
        enabled = self._mode == "fixed"
        self._lo_edit.setEnabled(enabled)
        self._hi_edit.setEnabled(enabled)

    def _on_mode(self) -> None:
        self._mode = str(self._mode_combo.currentData())
        self._refresh_range_edits()
        self._render()

    def _on_range_edit(self) -> None:
        try:
            lo = float(from_display(float(self._lo_edit.text()), "temperature", self._system))
            hi = float(from_display(float(self._hi_edit.text()), "temperature", self._system))
        except ValueError:
            self._refresh_range_edits()
            return
        if hi > lo:
            self._fixed_K = (lo, hi)
        self._refresh_range_edits()
        self._render()

    def _compute_levels(self, snap: Snapshot | None) -> tuple[float, float]:
        if self._mode == "auto" and snap is not None and np.isfinite(snap.slice_T).any():
            lo = float(np.nanmin(snap.slice_T))
            hi = float(np.nanmax(snap.slice_T))
            if hi - lo < 0.1:
                hi = lo + 0.1
            return lo, hi
        return self._fixed_K

    def _color(self, value_K: float) -> QColor:
        lo, hi = self._levels_K
        frac = min(max((value_K - lo) / max(hi - lo, 1e-9), 0.0), 1.0)
        r, g, b, a = self._cmap.map(np.array([frac]), mode="byte")[0]
        return QColor(int(r), int(g), int(b), int(a))

    def _render(self) -> None:
        snap = self._snapshot
        self._levels_K = self._compute_levels(snap)
        lo, hi = self._levels_K
        self._colorbar.setLevels((temp(lo, self._system), temp(hi, self._system)))
        if snap is None:
            self._image.clear()
            self._walls.set_lines([])
            return
        norm = (snap.slice_T.astype(np.float64) - lo) / max(hi - lo, 1e-9)
        valid = np.isfinite(norm)
        rgba = self._cmap.map(np.clip(np.nan_to_num(norm), 0.0, 1.0), mode="byte")
        rgba = np.asarray(rgba, dtype=np.uint8)
        rgba[..., 3] = np.where(valid, 255, 0)
        self._image.setImage(rgba, autoLevels=False, levels=(0, 255))
        nx, nz = snap.slice_T.shape
        dx = float(snap.x_centers[1] - snap.x_centers[0]) if nx > 1 else 0.005
        x0 = float(snap.x_centers[0]) - dx / 2
        z0 = float(snap.z_centers[0]) - dx / 2
        x0d = float(plot_length(x0, self._system))
        z0d = float(plot_length(z0, self._system))
        wd = float(plot_length(dx * nx, self._system))
        hd = float(plot_length(dx * nz, self._system))
        rect = QRectF(x0d, z0d, wd, hd)
        first = self._rect_disp.isEmpty()
        self._rect_disp = rect
        self._image.setRect(rect)
        if first:
            self._plot.autoRange()
        self._update_walls(snap)

    def _wall_x(self) -> float:
        sc = self._scenario
        if sc is None:
            return 0.0
        return sc.vessel.inner_diameter_m / 2 + sc.vessel.wall_thickness_m / 2

    def _update_walls(self, snap: Snapshot) -> None:
        sc = self._scenario
        wx = self._wall_x()
        if sc is None and snap.x_centers.size:
            wx = float(snap.x_centers[-1])
        L = lambda v: float(plot_length(v, self._system))  # noqa: E731
        lines: list[tuple[QPointF, QPointF, QColor]] = []

        def add_polyline(xs: np.ndarray, zs: np.ndarray, temps: np.ndarray) -> None:
            for n in range(len(xs) - 1):
                tm = float(0.5 * (temps[n] + temps[n + 1]))
                lines.append(
                    (
                        QPointF(L(xs[n]), L(zs[n])),
                        QPointF(L(xs[n + 1]), L(zs[n + 1])),
                        self._color(tm),
                    )
                )

        add_polyline(np.full(len(snap.wall_left_z), -wx), snap.wall_left_z, snap.wall_left_T)
        add_polyline(np.full(len(snap.wall_right_z), wx), snap.wall_right_z, snap.wall_right_T)
        base_z = -(sc.vessel.base_thickness_m / 2) if sc is not None else 0.0
        add_polyline(snap.base_x, np.full(len(snap.base_x), base_z), snap.base_T)
        if snap.lid_T is not None and sc is not None and sc.lid.mode != "none":
            r = sc.vessel.inner_diameter_m / 2
            top = sc.vessel.inner_height_m + sc.lid.thickness_m / 2
            frac = sc.lid.covered_fraction if sc.lid.mode == "partial" else 1.0
            lines.append(
                (
                    QPointF(L(-r), L(top)),
                    QPointF(L(-r + 2 * r * frac), L(top)),
                    self._color(snap.lid_T),
                )
            )
        self._walls.set_lines(lines)

    def _rebuild_decor(self) -> None:
        for item in self._decor:
            self._plot.removeItem(item)
        self._decor.clear()
        sc = self._scenario
        if sc is None:
            return
        L = lambda v: float(plot_length(v, self._system))  # noqa: E731
        r = sc.vessel.inner_diameter_m / 2
        wx = r + sc.vessel.wall_thickness_m / 2
        base_t = sc.vessel.base_thickness_m
        height = sc.vessel.inner_height_m
        dx = sc.simulation.cell_size_m

        # Faint vessel outline above the liquid, so the empty headspace is visible.
        outline = pg.PlotCurveItem(
            [L(-wx), L(-wx), L(wx), L(wx)],
            [L(height), L(-base_t / 2), L(-base_t / 2), L(height)],
            pen=pg.mkPen((140, 140, 140, 120), width=1),
        )
        self._add_decor(outline)

        src = sc.heat_source
        if isinstance(src, Burner):
            bar_h = max(0.006, 0.02 * height)
            outer = src.footprint_outer_diameter_m / 2
            inner = src.footprint_inner_diameter_m / 2 if src.footprint == "ring" else 0.0
            spans = [(-outer, outer)] if inner <= 0 else [(-outer, -inner), (inner, outer)]
            hatch_pen = pg.mkPen((255, 140, 0), width=1.5)
            for a, b in spans:
                self._add_decor(self._hatched_bar(a, b, -base_t - bar_h, -base_t, bar_h, hatch_pen))
        elif isinstance(src, Element):
            pts = src.path_m
            for (x1, y1, z1), (x2, y2, z2) in pairwise(pts):
                on_plane = 0.5 * (abs(y1) + abs(y2)) <= max(dx, src.diameter_m)
                alpha = 255 if on_plane else 70
                pen = pg.mkPen((80, 200, 255, alpha), width=3, style=Qt.PenStyle.DashLine)
                self._add_decor(pg.PlotCurveItem([L(x1), L(x2)], [L(z1), L(z2)], pen=pen))

        for probe in sc.probes:
            px, py, pz = probe.position_m
            alpha = 255 if abs(py) <= dx else 90
            dot = pg.ScatterPlotItem(
                [L(px)],
                [L(pz)],
                size=9,
                pen=pg.mkPen(255, 255, 255, alpha),
                brush=pg.mkBrush(0, 0, 0, min(alpha, 160)),
            )
            self._add_decor(dot)
            label = pg.TextItem(probe.name, color=(255, 255, 255, alpha), anchor=(0, 1))
            label.setPos(L(px), L(pz))
            self._add_decor(label)

    def _hatched_bar(
        self, x0: float, x1: float, z0: float, z1: float, step: float, pen: Any
    ) -> Any:
        """A rectangle outline filled with diagonal hatch lines (display units)."""
        L = lambda v: float(plot_length(v, self._system))  # noqa: E731
        xs: list[float] = []
        zs: list[float] = []
        h = z1 - z0
        x = x0 - h
        while x < x1:
            ax, bx = x, x + h
            az, bz = z0, z1
            if ax < x0:  # clip the left end
                az = z0 + (x0 - ax)
                ax = x0
            if bx > x1:  # clip the right end
                bz = z1 - (bx - x1)
                bx = x1
            if bz > az:
                xs += [L(ax), L(bx)]
                zs += [L(az), L(bz)]
            x += step
        outline_x = [L(x0), L(x1), L(x1), L(x0), L(x0)]
        outline_z = [L(z0), L(z0), L(z1), L(z1), L(z0)]
        group = pg.PlotCurveItem(outline_x, outline_z, pen=pen)
        hatch = pg.PlotCurveItem(xs, zs, pen=pen, connect="pairs", parent=group)
        hatch.setZValue(1)
        return group

    def _add_decor(self, item: Any) -> None:
        item.setZValue(20)
        self._plot.addItem(item)
        self._decor.append(item)

    def _on_mouse(self, pos: Any) -> None:
        vb = self._plot.vb
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        point = vb.mapSceneToView(pos)
        self._readout.setText(self.readout_at(point.x(), point.y()))
