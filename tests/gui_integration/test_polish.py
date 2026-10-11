"""Display formatting, the editor's number fields and the CLI progress bar."""

from __future__ import annotations

import io

import pytest
from gi_helpers import quick_scenario

from thermosim.cli import _ProgressBar
from thermosim.config.units import from_display
from thermosim.core.protocols import ProgressEvent
from thermosim.gui.formatting import fmt_quantity


@pytest.mark.parametrize(
    ("value", "quantity", "system", "expected"),
    [
        (8.66141732283, "length", "us", "8.661"),
        (0.0197, "length", "us", "0.0197"),  # too small for 3 decimals: keep 3 sig. digits
        (1.5, "length", "us", "1.5"),
        (170.0, "temperature", "us", "170"),
        (170.04, "temperature", "us", "170"),
        (68.123, "temperature", "us", "68.1"),
        (-0.04, "temperature", "metric", "-0.04"),
        (9.0, "length", "metric", "9"),
        (17060.4, "power", "us", "17060"),
        (0.0, "length", "us", "0"),
        (0.0163, "conductivity", "us", "0.0163"),
    ],
)
def test_fmt_quantity(value: float, quantity: str, system: str, expected: str) -> None:
    assert fmt_quantity(value, quantity, system) == expected  # type: ignore[arg-type]


def test_editor_shows_short_values_and_keeps_si_untouched(qtbot, make_window) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(quick_scenario())
    editor = window.editor
    before = editor.data()
    diameter = editor.field_widget("vessel.inner_diameter_m").text()  # type: ignore[attr-defined]
    assert len(diameter.split(".")[-1]) <= 3
    # editing a different field leaves every other stored SI value exactly as it was
    editor.set_field_text("target.temp_K", "171")
    after = editor.data()
    expected_K = float(from_display(171.0, "temperature", "us"))
    assert after["target"]["temp_K"] == pytest.approx(expected_K)
    after["target"]["temp_K"] = before["target"]["temp_K"]
    assert after == before
    # switching units and back does not change them either
    window.set_units("metric")
    window.set_units("us")
    assert editor.data()["vessel"] == before["vessel"]


class _Stream(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _feed(bar: _ProgressBar, n: int = 400) -> None:
    for i in range(n + 1):
        bar(
            ProgressEvent(
                sim_time_s=i * 1.0, wall_time_s=i * 0.01, max_time_s=float(n), realtime_lag_s=0.0
            )
        )
    bar.done()


def test_progress_bar_on_a_terminal_uses_one_line() -> None:
    stream = _Stream(tty=True)
    _feed(_ProgressBar(stream))
    out = stream.getvalue()
    assert out.count("\r") == 401
    assert out.count("\n") == 1


def test_progress_bar_not_on_a_terminal_prints_about_every_ten_percent() -> None:
    stream = _Stream(tty=False)
    _feed(_ProgressBar(stream))
    out = stream.getvalue()
    assert "\r" not in out
    assert 9 <= out.count("\n") <= 12
    assert "100.0%" in out
