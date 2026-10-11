"""Setup editor generated from the pydantic `Scenario` schema (WP08 requirement 1).

The editor keeps the scenario as a plain JSON-like dict in SI. Every widget reads and
writes one path in that dict. After each edit the dict is validated with
`Scenario.model_validate`; pydantic errors (and `ScenarioError` messages, which start with
a field path) are mapped back onto the widgets as a red border plus a tooltip, and the
cross-field messages appear in a banner. Display conversion (metric or US) happens only
at the widget boundary, with `thermosim.config.units`.
"""

from __future__ import annotations

import copy
import re
import types
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from inspect import isclass
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from thermosim.config.schema import (
    MATERIAL_KEYS,
    CustomMaterial,
    Scenario,
)
from thermosim.config.units import (
    UnitSystem,
    from_display,
    quantity_of,
    to_display,
    unit_label,
)
from thermosim.core.errors import ScenarioError
from thermosim.gui.formatting import DEFAULT_UNITS, fmt_quantity

Path = tuple[str | int, ...]

_ERR_STYLE = "border: 1px solid #d9534f; background: rgba(217, 83, 79, 0.18);"
_ERR_CELL = QColor(217, 83, 79, 90)

# Values for required fields that have no schema default, used when the editor has to
# create a new sub-object (switching a union, enabling insulation, adding a row).
_FALLBACKS: dict[str, Any] = {
    "Insulation.thickness_m": 0.025,
    "Insulation.conductivity_W_mK": 0.04,
    "Burner.rated_power_W": 5000.0,
    "Burner.footprint_outer_diameter_m": 0.18,
    "Element.rated_power_W": 2000.0,
    "Element.path_m": [[-0.05, 0.0, 0.03], [0.05, 0.0, 0.03]],
    "ProbeCfg.name": "probe",
    "ProbeCfg.position_m": [0.0, 0.0, 0.05],
    "PowerStep.t_s": 0.0,
}

# Editing one of a pair of alternatives clears the other (liquid volume or fill height).
_EXCLUSIVE: dict[str, str] = {
    "liquid.volume_m3": "liquid.fill_height_m",
    "liquid.fill_height_m": "liquid.volume_m3",
}

_TABS: list[tuple[str, list[str]]] = [
    ("Vessel", ["vessel"]),
    ("Lid && insulation", ["lid", "insulation"]),
    ("Liquid", ["liquid"]),
    ("Heat source", ["heat_source"]),
    ("Atmosphere", ["atmosphere"]),
    ("Probes/target", ["probes", "target", "power_schedule"]),
    ("Simulation", ["simulation", "model"]),
]

_PATH_RE = re.compile(r"^([A-Za-z_]\w*(?:\[[^\]]*\])?(?:\.[A-Za-z_]\w*(?:\[[^\]]*\])?)*)\s*:")


# ----------------------------------------------------------------------------- helpers
def pkey(path: Path | list[str | int]) -> str:
    return ".".join(str(p) for p in path)


def _plain(value: Any) -> Any:
    """Convert defaults to JSON-like data (dict/list/str/float)."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set | frozenset):
        return sorted(value)
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def default_dict(cls: type[BaseModel], carry: dict[str, Any] | None = None) -> dict[str, Any]:
    """A valid-looking starting dict for a new instance of `cls`."""
    out: dict[str, Any] = {}
    for name, info in cls.model_fields.items():
        key = f"{cls.__name__}.{name}"
        if carry and name in carry:
            out[name] = copy.deepcopy(carry[name])
        elif key in _FALLBACKS:
            out[name] = copy.deepcopy(_FALLBACKS[key])
        elif info.is_required():
            out[name] = None
        else:
            out[name] = _plain(info.get_default(call_default_factory=True))
    return out


def _strip_optional(ann: Any) -> tuple[Any, bool]:
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(ann) if a is not type(None)]
        if len(args) < len(get_args(ann)) and len(args) == 1:
            return args[0], True
    return ann, False


def _is_model(tp: Any) -> bool:
    return isclass(tp) and issubclass(tp, BaseModel)


def _title(info: FieldInfo, name: str) -> str:
    return info.title or name.replace("_", " ").capitalize()


class _Slot(QWidget):
    """A container that holds exactly one child widget, replaceable."""

    def __init__(self) -> None:
        super().__init__()
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._child: QWidget | None = None

    def set(self, widget: QWidget | None) -> None:
        if self._child is not None:
            self._lay.removeWidget(self._child)
            self._child.setParent(None)
            self._child.deleteLater()
        self._child = widget
        if widget is not None:
            self._lay.addWidget(widget)


# ----------------------------------------------------------------------------- targets
class _WidgetTarget:
    """Error marking for a leaf widget."""

    def __init__(self, widget: QWidget) -> None:
        self.widget = widget
        self._tip = widget.toolTip()
        self._style = widget.styleSheet()

    def mark(self, msg: str | None, rest: Path) -> None:
        if msg is None:
            self.widget.setStyleSheet(self._style)
            self.widget.setToolTip(self._tip)
            self.widget.setProperty("invalid", False)
        else:
            self.widget.setStyleSheet(_ERR_STYLE)
            self.widget.setToolTip(msg)
            self.widget.setProperty("invalid", True)


@dataclass
class _Col:
    field: str | None  # key in the row dict; None when the row is itself the tuple
    idx: int | None  # index into a tuple value
    header: str
    quantity: str | None  # None for text columns


class _TableTarget:
    """A table of rows. Errors colour the offending cell (or the first cell of a row)."""

    def __init__(self, table: QTableWidget, cols: list[_Col]) -> None:
        self.table = table
        self.cols = cols
        self._tip = table.toolTip()
        self._marked: list[QTableWidgetItem] = []

    def clear(self) -> None:
        for item in self._marked:
            item.setBackground(QColor(0, 0, 0, 0))
            item.setToolTip("")
        self._marked.clear()
        self.table.setProperty("invalid", False)
        self.table.setToolTip(self._tip)

    def mark(self, msg: str | None, rest: Path) -> None:
        if msg is None:
            self.clear()
            return
        self.table.setProperty("invalid", True)
        self.table.setToolTip(msg)
        if rest and isinstance(rest[0], int) and rest[0] < self.table.rowCount():
            row = rest[0]
            col = 0
            if len(rest) > 1:
                field = rest[1] if isinstance(rest[1], str) else None
                idx = rest[2] if len(rest) > 2 and isinstance(rest[2], int) else None
                for c, spec in enumerate(self.cols):
                    if spec.field == field and (idx is None or spec.idx == idx):
                        col = c
                        break
            item = self.table.item(row, col)
            if item is not None:
                item.setBackground(_ERR_CELL)
                item.setToolTip(msg)
                self._marked.append(item)


# ----------------------------------------------------------------------------- editor
class ScenarioEditor(QWidget):
    """Tabbed form editing a `Scenario`, built from the pydantic schema."""

    changed = Signal()  # any edit or load
    validated = Signal(bool, str)  # (is valid, message for the banner)

    def __init__(self, system: UnitSystem = DEFAULT_UNITS) -> None:
        super().__init__()
        self._system: UnitSystem = system
        self._data: dict[str, Any] = {}
        self._baseline: dict[str, Any] = {}
        self._scenario: Scenario | None = None
        self._valid = False
        self._message = ""
        self._targets: dict[str, _WidgetTarget | _TableTarget] = {}
        self._widgets: dict[str, QWidget] = {}
        self._refreshers: list[tuple[str, Callable[[], None]]] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._buttons: dict[str, tuple[QPushButton, QPushButton]] = {}
        self._name_slot = _Slot()
        outer.addWidget(self._name_slot)
        self._banner = QLabel()
        self._banner.setWordWrap(True)
        self._banner.setTextFormat(Qt.TextFormat.PlainText)
        self._banner.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._banner.setStyleSheet(
            "QLabel { color: #b52b27; background: rgba(217,83,79,0.15); padding: 4px; }"
        )
        self._banner.hide()
        outer.addWidget(self._banner)
        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, 1)

    # ------------------------------------------------------------------ public API
    @property
    def system(self) -> UnitSystem:
        return self._system

    def set_scenario(self, scenario: Scenario) -> None:
        """Load a scenario: rebuild every form from it and mark it clean."""
        self._data = scenario.model_dump(mode="json")
        if self._data.get("insulation") is not None:
            self._data["insulation"]["covers"] = sorted(self._data["insulation"]["covers"])
        self._baseline = copy.deepcopy(self._data)
        self._rebuild()
        self._validate()
        self.changed.emit()

    def scenario(self) -> Scenario | None:
        """The current scenario, or None if the inputs are invalid."""
        return self._scenario

    def data(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def is_valid(self) -> bool:
        return self._valid

    def error_message(self) -> str:
        return self._message

    def is_dirty(self) -> bool:
        return self._data != self._baseline

    def mark_clean(self) -> None:
        self._baseline = copy.deepcopy(self._data)
        self.changed.emit()

    def set_units(self, system: UnitSystem) -> None:
        """Switch the displayed unit system. Stored SI values do not change."""
        self._system = system
        for _, fn in list(self._refreshers):
            fn()

    def field_widget(self, path: str) -> QWidget:
        """The widget bound to a dotted path such as 'vessel.inner_diameter_m'."""
        return self._widgets[path]

    def paths(self) -> list[str]:
        return list(self._widgets)

    def show_external_error(self, path: str, message: str) -> None:
        """Mark the field at dotted `path` (e.g. 'liquid.fill_height_m') with an error found
        outside schema validation (geometry checks), and show `message` in the banner.

        The marking disappears at the next edit, which re-validates the form.
        """
        loc: list[str | int] = [int(p) if p.isdigit() else p for p in path.split(".") if p]
        found = self._find_target(tuple(loc))
        if found is not None:
            target, rest = found
            target.mark(message, rest)
        self._banner.setText(message)
        self._banner.setVisible(True)
        self._mark_tabs([(tuple(loc), message)])

    def set_value_si(self, path: str, value: float) -> None:
        """Set a numeric field to an exact SI value (as if typed, but without display rounding)."""
        key = tuple(int(p) if p.isdigit() else p for p in path.split("."))
        self._set(key, float(value))
        for ref_key, fn in self._refreshers:
            if ref_key == path:
                fn()
        self._edited()

    def set_field_text(self, path: str, text: str) -> None:
        """Type `text` into a text/number field as a user would."""
        widget = self._widgets[path]
        assert isinstance(widget, QLineEdit)
        widget.setText(text)
        widget.textEdited.emit(text)

    # ------------------------------------------------------------------ data access
    def _get(self, path: Path) -> Any:
        cur: Any = self._data
        for seg in path:
            cur = cur[seg]
        return cur

    def _set(self, path: Path, value: Any) -> None:
        cur: Any = self._data
        for seg in path[:-1]:
            cur = cur[seg]
        cur[path[-1]] = value

    def _safe_get(self, path: Path) -> Any:
        try:
            return self._get(path)
        except (KeyError, IndexError, TypeError):
            return None

    # ------------------------------------------------------------------ registry
    def _register(self, path: Path, target: _WidgetTarget | _TableTarget, widget: QWidget) -> None:
        key = pkey(path)
        self._targets[key] = target
        self._widgets[key] = widget

    def _on_refresh(self, path: Path, fn: Callable[[], None]) -> None:
        self._refreshers.append((pkey(path), fn))

    def _purge(self, prefix: str) -> None:
        for key in [k for k in self._targets if k.startswith(prefix)]:
            del self._targets[key]
            self._widgets.pop(key, None)
        self._refreshers = [(k, f) for k, f in self._refreshers if not k.startswith(prefix)]

    def _edited(self) -> None:
        self._validate()
        self.changed.emit()

    # ------------------------------------------------------------------ building
    def _rebuild(self) -> None:
        self._targets.clear()
        self._widgets.clear()
        self._refreshers.clear()
        self._buttons.clear()
        while self._tabs.count():
            old = self._tabs.widget(0)
            self._tabs.removeTab(0)
            if old is not None:
                old.deleteLater()
        # Name field
        name_box = QWidget()
        row = QHBoxLayout(name_box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Setup name"))
        name_edit = QLineEdit()
        name_edit.setToolTip(Scenario.model_fields["name"].description or "")
        row.addWidget(name_edit, 1)
        name_edit.setText(str(self._data.get("name", "")))
        name_edit.textEdited.connect(lambda text: self._text_edited(("name",), text))
        self._register(("name",), _WidgetTarget(name_edit), name_edit)
        self._name_slot.set(name_box)

        fields = Scenario.model_fields
        for title, names in _TABS:
            page = QWidget()
            lay = QVBoxLayout(page)
            for name in names:
                widget = self._container_for(Scenario, name, fields[name], (name,))
                lay.addWidget(widget)
            lay.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(page)
            self._tabs.addTab(scroll, title.replace("&&", "&"))

    def _text_edited(self, path: Path, text: str) -> None:
        self._set(path, text)
        self._edited()

    def _container_for(
        self, cls: type[BaseModel], name: str, info: FieldInfo, path: Path
    ) -> QWidget:
        """Widget for a top-level field: always a group box (or a bare leaf)."""
        kind, _ = self._classify(info)
        if kind in ("number", "text", "literal", "enum", "flags", "material"):
            box = QGroupBox(_title(info, name))
            form = QFormLayout(box)
            label, widget = self._leaf(cls, name, info, path)
            form.addRow(label, widget)
            return box
        return self._complex(cls, name, info, path)

    def _classify(self, info: FieldInfo) -> tuple[str, Any]:
        inner, optional = _strip_optional(info.annotation)
        origin = get_origin(inner)
        if origin in (Union, types.UnionType):
            members = get_args(inner)
            if all(_is_model(m) for m in members):
                return "union", members
            if set(members) == {str, CustomMaterial}:
                return "material", None
        if _is_model(inner):
            return ("optional_model" if optional else "model"), inner
        if origin is list:
            elem = get_args(inner)[0]
            if _is_model(elem):
                return "list_model", elem
            if get_origin(elem) is tuple:
                return "list_tuple", elem
        if origin is set:
            return "flags", get_args(get_args(inner)[0])
        if origin is Literal:
            return "literal", get_args(inner)
        if isclass(inner) and issubclass(inner, Enum):
            return "enum", inner
        if inner in (float, int):
            return "number", optional
        if inner is str:
            return "text", None
        return "unsupported", None

    def _leaf(
        self, cls: type[BaseModel], name: str, info: FieldInfo, path: Path
    ) -> tuple[QWidget, QWidget]:
        kind, extra = self._classify(info)
        tip = info.description or ""
        title = _title(info, name)
        if kind == "number":
            quantity = quantity_of(cls, name) or "dimensionless"
            edit = self._number_edit(path, quantity, bool(extra))
            edit.setToolTip(tip)
            label = QLabel()

            def refresh_label(label: QLabel = label, q: str = quantity, t: str = title) -> None:
                unit = unit_label(q, self._system)  # type: ignore[arg-type]
                label.setText(f"{t} ({unit})" if unit else t)

            refresh_label()
            self._on_refresh(path, refresh_label)
            label.setToolTip(tip)
            return label, edit
        label = QLabel(title)
        label.setToolTip(tip)
        if kind == "text":
            text_edit = QLineEdit()
            text_edit.setToolTip(tip)
            text_edit.setText(str(self._safe_get(path) or ""))
            text_edit.textEdited.connect(lambda text, p=path: self._text_edited(p, text))
            self._register(path, _WidgetTarget(text_edit), text_edit)
            return label, text_edit
        if kind in ("literal", "enum"):
            values = list(extra) if kind == "literal" else [m.value for m in extra]
            return label, self._combo(path, values, tip)
        if kind == "flags":
            return label, self._flags(path, list(extra), tip)
        if kind == "material":
            return label, self._material(path, tip)
        raise TypeError(f"unsupported field {cls.__name__}.{name}: {info.annotation}")

    def _build_model(self, cls: type[BaseModel], path: Path) -> QWidget:
        host = QWidget()
        form = QFormLayout(host)
        form.setContentsMargins(4, 4, 4, 4)
        for name, info in cls.model_fields.items():
            ann = info.annotation
            if get_origin(ann) is Literal and len(get_args(ann)) == 1:
                continue  # constant tag such as "type" or "shape"
            fpath = (*path, name)
            kind, _ = self._classify(info)
            if kind in ("union", "model", "optional_model", "list_model", "list_tuple"):
                form.addRow(self._complex(cls, name, info, fpath))
            else:
                label, widget = self._leaf(cls, name, info, fpath)
                form.addRow(label, widget)
        return host

    def _complex(self, cls: type[BaseModel], name: str, info: FieldInfo, path: Path) -> QGroupBox:
        kind, extra = self._classify(info)
        box = QGroupBox(_title(info, name))
        box.setToolTip(info.description or "")
        lay = QVBoxLayout(box)
        if kind == "model":
            lay.addWidget(self._build_model(extra, path))
        elif kind == "optional_model":
            box.setCheckable(True)
            slot = _Slot()
            lay.addWidget(slot)
            self._optional_model(box, slot, extra, path)
        elif kind == "union":
            self._union(lay, extra, path, str(info.discriminator))
        elif kind == "list_model":
            self._table_for_model(lay, extra, path, info)
        elif kind == "list_tuple":
            self._table_for_tuples(lay, cls, name, path, info)
        else:
            raise TypeError(f"unsupported field {cls.__name__}.{name}: {info.annotation}")
        return box

    # ------------------------------------------------------------------ leaf widgets
    def _number_edit(self, path: Path, quantity: str, allow_none: bool) -> QLineEdit:
        edit = QLineEdit()
        edit.setAlignment(Qt.AlignmentFlag.AlignRight)

        def refresh() -> None:
            value = self._safe_get(path)
            if value is None:
                text = ""
            elif isinstance(value, int | float):
                shown = to_display(float(value), quantity, self._system)  # type: ignore[arg-type]
                text = fmt_quantity(float(shown), quantity, self._system)
            else:
                text = str(value)
            edit.blockSignals(True)
            edit.setText(text)
            edit.blockSignals(False)

        def on_text(text: str) -> None:
            text = text.strip()
            value: Any
            if text == "" and allow_none:
                value = None
            else:
                try:
                    value = float(from_display(float(text), quantity, self._system))  # type: ignore[arg-type]
                except ValueError:
                    value = text  # pydantic reports it at this widget
            self._set(path, value)
            other = _EXCLUSIVE.get(pkey(path))
            if other is not None and value is not None:
                other_path = tuple(other.split("."))
                self._set(other_path, None)
                for key, fn in self._refreshers:
                    if key == other:
                        fn()
            self._edited()

        edit.textEdited.connect(on_text)
        self._on_refresh(path, refresh)
        refresh()
        self._register(path, _WidgetTarget(edit), edit)
        return edit

    def _combo(self, path: Path, values: list[Any], tip: str) -> QComboBox:
        combo = QComboBox()
        for v in values:
            combo.addItem(str(v), v)
        combo.setToolTip(tip)

        def refresh() -> None:
            combo.blockSignals(True)
            idx = combo.findData(self._safe_get(path))
            combo.setCurrentIndex(max(idx, 0))
            combo.blockSignals(False)

        def on_change() -> None:
            self._set(path, combo.currentData())
            self._edited()

        combo.currentIndexChanged.connect(lambda _i: on_change())
        self._on_refresh(path, refresh)
        refresh()
        self._register(path, _WidgetTarget(combo), combo)
        return combo

    def _flags(self, path: Path, values: list[Any], tip: str) -> QWidget:
        host = QWidget()
        lay = QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        boxes: dict[Any, QCheckBox] = {}
        for v in values:
            cb = QCheckBox(str(v))
            boxes[v] = cb
            lay.addWidget(cb)
        host.setToolTip(tip)

        def refresh() -> None:
            current = self._safe_get(path) or []
            for v, cb in boxes.items():
                cb.blockSignals(True)
                cb.setChecked(v in current)
                cb.blockSignals(False)

        def on_toggle() -> None:
            self._set(path, sorted(v for v, cb in boxes.items() if cb.isChecked()))
            self._edited()

        for cb in boxes.values():
            cb.toggled.connect(lambda _c: on_toggle())
        self._on_refresh(path, refresh)
        refresh()
        self._register(path, _WidgetTarget(host), host)
        return host

    def _material(self, path: Path, tip: str) -> QWidget:
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        combo = QComboBox()
        for key in MATERIAL_KEYS:
            combo.addItem(key, key)
        combo.addItem("custom...", "__custom__")
        combo.setToolTip(tip)
        slot = _Slot()
        lay.addWidget(combo)
        lay.addWidget(slot)
        key_prefix = pkey(path) + "."

        def show_custom() -> None:
            self._purge(key_prefix)
            if isinstance(self._safe_get(path), dict):
                slot.set(self._build_model(CustomMaterial, path))
            else:
                slot.set(None)

        def refresh() -> None:
            value = self._safe_get(path)
            combo.blockSignals(True)
            idx = combo.findData("__custom__" if isinstance(value, dict) else value)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

        def on_change() -> None:
            data = combo.currentData()
            if data == "__custom__":
                if not isinstance(self._safe_get(path), dict):
                    self._set(
                        path,
                        {
                            "density": 8000.0,
                            "specific_heat": 500.0,
                            "conductivity": 16.0,
                            "emissivity": 0.3,
                        },
                    )
            else:
                self._set(path, data)
            show_custom()
            self._edited()

        combo.currentIndexChanged.connect(lambda _i: on_change())
        self._register(path, _WidgetTarget(combo), combo)
        self._on_refresh(path, refresh)
        refresh()
        show_custom()
        return host

    # ------------------------------------------------------------------ containers
    def _optional_model(
        self, box: QGroupBox, slot: _Slot, model: type[BaseModel], path: Path
    ) -> None:
        prefix = pkey(path) + "."

        def populate() -> None:
            self._purge(prefix)
            if self._safe_get(path) is None:
                slot.set(None)
            else:
                slot.set(self._build_model(model, path))

        def refresh() -> None:
            box.blockSignals(True)
            box.setChecked(self._safe_get(path) is not None)
            box.blockSignals(False)

        def on_toggle(on: bool) -> None:
            self._set(path, default_dict(model) if on else None)
            populate()
            self._edited()

        box.toggled.connect(on_toggle)
        refresh()
        populate()
        self._register(path, _WidgetTarget(box), box)

    def _union(
        self,
        lay: QVBoxLayout,
        members: tuple[type[BaseModel], ...],
        path: Path,
        disc: str,
    ) -> None:
        tags = [m.model_fields[disc].default for m in members]
        combo = QComboBox()
        for tag in tags:
            combo.addItem(str(tag).replace("_", " ").capitalize(), tag)
        slot = _Slot()
        lay.addWidget(combo)
        lay.addWidget(slot)
        prefix = pkey(path) + "."

        def current_member() -> type[BaseModel]:
            tag = (self._safe_get(path) or {}).get(disc)
            return members[tags.index(tag)] if tag in tags else members[0]

        def populate() -> None:
            self._purge(prefix)
            slot.set(self._build_model(current_member(), path))

        def refresh() -> None:
            combo.blockSignals(True)
            idx = combo.findData((self._safe_get(path) or {}).get(disc))
            combo.setCurrentIndex(max(idx, 0))
            combo.blockSignals(False)

        def on_change() -> None:
            tag = combo.currentData()
            member = members[tags.index(tag)]
            old = self._safe_get(path) or {}
            carry = {k: v for k, v in old.items() if k == "rated_power_W"}
            self._set(path, default_dict(member, carry) | {disc: tag})
            populate()
            self._edited()

        combo.currentIndexChanged.connect(lambda _i: on_change())
        self._register(path, _WidgetTarget(combo), combo)
        self._on_refresh(path, refresh)
        refresh()
        populate()

    # ------------------------------------------------------------------ tables
    def _model_columns(self, model: type[BaseModel]) -> list[_Col]:
        cols: list[_Col] = []
        for name, info in model.model_fields.items():
            inner, _ = _strip_optional(info.annotation)
            quantity = quantity_of(model, name)
            title = _title(info, name)
            if get_origin(inner) is tuple:
                for i, axis in enumerate("xyz"):
                    cols.append(_Col(name, i, f"{title} {axis}", quantity or "length"))
            elif inner is str:
                cols.append(_Col(name, None, title, None))
            else:
                cols.append(_Col(name, None, title, quantity or "dimensionless"))
        return cols

    def _table_for_model(
        self, lay: QVBoxLayout, model: type[BaseModel], path: Path, info: FieldInfo
    ) -> None:
        cols = self._model_columns(model)

        def new_row() -> Any:
            rows = self._get(path)
            row = default_dict(model)
            if model.__name__ == "ProbeCfg":
                row["name"] = f"probe{len(rows) + 1}"
                if rows:
                    row["position_m"] = list(rows[-1]["position_m"])
            elif model.__name__ == "PowerStep" and rows:
                row["t_s"] = rows[-1]["t_s"] + 60.0
                row["command"] = rows[-1]["command"]
            return row

        self._table(lay, path, cols, new_row, info.description or "")

    def _table_for_tuples(
        self, lay: QVBoxLayout, cls: type[BaseModel], name: str, path: Path, info: FieldInfo
    ) -> None:
        quantity = quantity_of(cls, name) or "length"
        cols = [_Col(None, i, axis, quantity) for i, axis in enumerate("xyz")]

        def new_row() -> Any:
            rows = self._get(path)
            return list(rows[-1]) if rows else [0.0, 0.0, 0.05]

        self._table(lay, path, cols, new_row, info.description or "")

    def _table(
        self,
        lay: QVBoxLayout,
        path: Path,
        cols: list[_Col],
        new_row: Callable[[], Any],
        tip: str,
    ) -> None:
        table = QTableWidget(0, len(cols))
        table.setToolTip(tip)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setMinimumHeight(110)
        add_btn = QPushButton("Add")
        del_btn = QPushButton("Remove")
        buttons = QHBoxLayout()
        buttons.addWidget(add_btn)
        buttons.addWidget(del_btn)
        buttons.addStretch(1)
        lay.addWidget(table)
        lay.addLayout(buttons)
        target = _TableTarget(table, cols)

        def cell_value(row: Any, col: _Col) -> Any:
            v = row[col.field] if col.field is not None else row
            return v[col.idx] if col.idx is not None else v

        def set_cell(row: Any, col: _Col, value: Any) -> None:
            if col.idx is None:
                assert col.field is not None
                row[col.field] = value
            elif col.field is None:
                row[col.idx] = value
            else:
                row[col.field][col.idx] = value

        def header(col: _Col) -> str:
            if col.quantity is None:
                return col.header
            unit = unit_label(col.quantity, self._system)  # type: ignore[arg-type]
            return f"{col.header} ({unit})" if unit else col.header

        def refresh() -> None:
            rows = self._safe_get(path) or []
            table.blockSignals(True)
            table.setRowCount(len(rows))
            table.setHorizontalHeaderLabels([header(c) for c in cols])
            for r, row in enumerate(rows):
                for c, col in enumerate(cols):
                    v = cell_value(row, col)
                    if col.quantity is not None and isinstance(v, int | float):
                        text = fmt_quantity(
                            float(to_display(float(v), col.quantity, self._system)),  # type: ignore[arg-type]
                            col.quantity,
                            self._system,
                        )
                    else:
                        text = "" if v is None else str(v)
                    table.setItem(r, c, QTableWidgetItem(text))
            table.blockSignals(False)
            table.horizontalHeader().setStretchLastSection(True)

        def on_item(item: QTableWidgetItem) -> None:
            rows = self._get(path)
            col = cols[item.column()]
            row = rows[item.row()]
            text = item.text().strip()
            value: Any = text
            if col.quantity is not None:
                try:
                    value = float(from_display(float(text), col.quantity, self._system))  # type: ignore[arg-type]
                except ValueError:
                    value = text
            set_cell(row, col, value)
            self._edited()

        def add() -> None:
            self._get(path).append(new_row())
            refresh()
            self._edited()

        def remove() -> None:
            rows = self._get(path)
            sel = sorted({i.row() for i in table.selectedIndexes()}, reverse=True)
            if not sel and rows:
                sel = [len(rows) - 1]
            for r in sel:
                del rows[r]
            refresh()
            self._edited()

        table.itemChanged.connect(on_item)
        add_btn.clicked.connect(add)
        del_btn.clicked.connect(remove)
        self._register(path, target, table)
        self._on_refresh(path, refresh)
        refresh()
        self._buttons[pkey(path)] = (add_btn, del_btn)

    def table_buttons(self, path: str) -> tuple[QPushButton, QPushButton]:
        """(add, remove) buttons of the table at `path`."""
        return self._buttons[path]

    # ------------------------------------------------------------------ validation
    def _validate(self) -> None:
        for target in self._targets.values():
            target.mark(None, ())
        errors: list[tuple[Path, str]] = []
        self._scenario = None
        try:
            self._scenario = Scenario.model_validate(copy.deepcopy(self._data))
        except ValidationError as exc:
            for err in exc.errors():
                msg = str(err["msg"])
                if msg.startswith("Value error, "):
                    msg = msg[len("Value error, ") :]
                loc = tuple(err["loc"])
                if loc and loc[-1] == "str" and isinstance(self._safe_get(loc[:-1]), dict):
                    continue  # the 'str' branch of a str|CustomMaterial union
                errors.append((loc, msg))
        except ScenarioError as exc:
            errors.append((self._path_from_message(str(exc)), str(exc)))
        self._valid = self._scenario is not None
        lines = []
        for loc, msg in errors:
            found = self._find_target(loc)
            if found is not None:
                target, rest = found
                target.mark(msg, rest)
            lines.append(f"{pkey(loc)}: {msg}" if loc and not _PATH_RE.match(msg) else msg)
        self._message = "\n".join(lines)
        self._banner.setText(self._message)
        self._banner.setVisible(bool(self._message))
        self._mark_tabs(errors)
        self.validated.emit(self._valid, self._message)

    @staticmethod
    def _path_from_message(message: str) -> Path:
        m = _PATH_RE.match(message)
        if not m:
            return ()
        segs: list[str | int] = []
        for part in m.group(1).split("."):
            base, _, rest = part.partition("[")
            segs.append(base)
            if rest:
                idx = rest.rstrip("]")
                if idx.isdigit():
                    segs.append(int(idx))
        return tuple(segs)

    def _find_target(self, loc: Path) -> tuple[_WidgetTarget | _TableTarget, Path] | None:
        segs = list(loc)
        variants = [segs] + [segs[:i] + segs[i + 1 :] for i in range(1, len(segs))]
        best: tuple[int, _WidgetTarget | _TableTarget, Path] | None = None
        for cand in variants:
            for n in range(len(cand), 0, -1):
                key = pkey(cand[:n])
                target = self._targets.get(key)
                if target is not None:
                    if best is None or n > best[0]:
                        best = (n, target, tuple(cand[n:]))
                    break
        return None if best is None else (best[1], best[2])

    def _mark_tabs(self, errors: list[tuple[Path, str]]) -> None:
        bad = {str(loc[0]) for loc, _ in errors if loc}
        for i, (title, names) in enumerate(_TABS):
            text = title.replace("&&", "&")
            if any(n in bad for n in names):
                text += " !"
            self._tabs.setTabText(i, text)
