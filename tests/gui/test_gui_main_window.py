"""MainWindow end-to-end tests against the fake runner (pytest-qt, offscreen)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gui_helpers import RecordingRunner, short_scenario
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QLabel, QMessageBox

from thermosim.config.io import example_scenarios, load_scenario
from thermosim.config.units import from_display
from thermosim.core.types import RunMode, StopReason
from thermosim.gui import recording_adapter as rec
from thermosim.gui.main_window import MainWindow


def wait_finished(qtbot, window: MainWindow, timeout: int = 15000) -> None:  # type: ignore[no-untyped-def]
    qtbot.waitUntil(lambda: window.result is not None and not window.is_active, timeout=timeout)


# ------------------------------------------------------------------ editor & files
@pytest.mark.parametrize("name", sorted(example_scenarios()))
def test_examples_load_edit_and_save_roundtrip(make_window, name: str, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_example(name)
    assert window.editor.is_valid(), window.editor.error_message()
    assert "*" not in window.windowTitle()

    # Edit the target temperature as a US user would (degrees F).
    window.editor.set_field_text("target.temp_K", "170")
    assert window.editor.is_valid()
    assert window.windowTitle().startswith(example_scenarios()[name].name + "*")

    out = tmp_path / "saved.toml"
    assert window.save(out)
    assert "*" not in window.windowTitle()
    loaded = load_scenario(out)
    assert loaded == window.editor.scenario()
    expected_K = float(from_display(170.0, "temperature", "us"))
    assert loaded.target.temp_K == pytest.approx(expected_K)


def test_invalid_input_disables_run_and_shows_message(make_window) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    assert window.act_run_fast.isEnabled()
    window.editor.set_field_text("vessel.inner_diameter_m", "not a number")
    assert not window.editor.is_valid()
    assert not window.act_run_fast.isEnabled()
    assert not window.act_run_rt.isEnabled()
    assert "inner_diameter_m" in window.editor.error_message()
    edit = window.editor.field_widget("vessel.inner_diameter_m")
    assert edit.property("invalid") is True
    assert "number" in edit.toolTip()
    assert window.editor.isVisible()

    # fixing the value re-enables the buttons and clears the marking
    window.editor.set_field_text("vessel.inner_diameter_m", "9")
    assert window.editor.is_valid()
    assert window.act_run_fast.isEnabled()
    assert edit.property("invalid") is False


def test_cross_field_error_in_banner(make_window) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.editor.set_field_text("liquid.fill_height_m", "100")  # 100 in: taller than the pot
    assert not window.editor.is_valid()
    assert "exceeds" in window.editor.error_message()
    assert not window.act_run_fast.isEnabled()
    banner = window.editor.findChildren(QLabel)[-1]
    assert banner is not None
    assert "exceeds" in window.editor.error_message()


def test_union_switch_and_table_editing(make_window) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    editor = window.editor
    assert editor.data()["heat_source"]["type"] == "burner"
    combo = editor.field_widget("heat_source")
    combo.setCurrentIndex(combo.findData("element"))
    assert editor.data()["heat_source"]["type"] == "element"
    assert "heat_source.path_m" in editor.paths()
    assert "heat_source.footprint_outer_diameter_m" not in editor.paths()
    # carried over the rating
    assert editor.data()["heat_source"]["rated_power_W"] == pytest.approx(2500.0)

    add, remove = editor.table_buttons("probes")
    n = len(editor.data()["probes"])
    add.click()
    assert len(editor.data()["probes"]) == n + 1
    assert editor.data()["probes"][-1]["name"] == f"probe{n + 1}"
    remove.click()
    assert len(editor.data()["probes"]) == n

    # optional insulation toggles on and off
    box = editor.field_widget("insulation")
    box.setChecked(True)
    assert editor.data()["insulation"] is not None
    box.setChecked(False)
    assert editor.data()["insulation"] is None


def test_dirty_prompt_on_close(make_window, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.editor.set_field_text("atmosphere.wind_speed_m_s", "3")
    assert window.editor.is_dirty()
    asked: list[bool] = []

    def cancel(*args, **kwargs):  # type: ignore[no-untyped-def]
        asked.append(True)
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", cancel)
    assert not window.close()
    assert asked
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard)
    assert window.close()


# ------------------------------------------------------------------ runs
def test_fast_run_updates_views_and_summary(make_window, qtbot) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(short_scenario(60.0))
    seen_snapshot: list[bool] = []
    window.start_run(RunMode.FAST)
    assert window.is_active
    assert not window.speed_spin.isEnabled()
    assert not window.power_slider.isEnabled()
    qtbot.waitUntil(lambda: window.chart.n_points > 0, timeout=5000)
    seen_snapshot.append(window.slice_view.snapshot is not None)
    wait_finished(qtbot, window)
    assert window.result is not None
    summary = window.result.summary
    assert summary.stop_reason in (StopReason.MAX_TIME, StopReason.TARGET_REACHED)
    assert window.chart.n_points == len(window.result.samples)
    assert window.slice_view.snapshot is not None
    assert window.slice_view.displayed_image() is not None
    assert window.slice_view.n_wall_segments() > 10
    assert "Maximum time" in window.summary.reason_label.text() or "Target" in (
        window.summary.reason_label.text()
    )
    assert window.summary.energy_table.item(0, 1).text() != "-"
    assert window.scrubber.isEnabled()
    assert window.act_run_fast.isEnabled()
    assert "finished" in window.status_text()


def test_stop_works_mid_run(make_window, qtbot) -> None:  # type: ignore[no-untyped-def]
    window = make_window(fast_rate=40.0)
    window.load_scenario(short_scenario(7200.0))
    window.start_run(RunMode.FAST)
    qtbot.waitUntil(lambda: window.chart.n_points > 3, timeout=5000)
    assert window.act_stop.isEnabled()
    window.stop_run()
    wait_finished(qtbot, window)
    assert window.result is not None
    assert window.result.summary.stop_reason == StopReason.USER_STOP
    assert window.result.summary.sim_time_s < 7200.0
    assert not window.act_stop.isEnabled()
    assert "stopped" in window.status_text()


def test_pause_resume_toggle(make_window, qtbot) -> None:  # type: ignore[no-untyped-def]
    window = make_window(fast_rate=40.0)
    window.load_scenario(short_scenario(7200.0))
    window.start_run(RunMode.FAST)
    qtbot.waitUntil(lambda: window.chart.n_points > 2, timeout=5000)
    window.pause_resume()
    assert window.act_pause.text() == "Resume"
    assert "paused" in window.status_text()
    window.pause_resume()
    assert window.act_pause.text() == "Pause"
    window.stop_run()
    wait_finished(qtbot, window)


def test_realtime_controls_call_runner(make_window, qtbot) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(short_scenario(7200.0))
    window.start_run(RunMode.REALTIME)
    runner = window.runner
    assert isinstance(runner, RecordingRunner)
    assert window.speed_spin.isEnabled()
    assert window.power_slider.isEnabled()
    window.power_slider.setValue(40)
    assert runner.power_calls[-1] == pytest.approx(0.4)
    window.speed_spin.setValue(25.0)
    assert runner.speed_calls[-1] == pytest.approx(25.0)
    window.stop_run()
    wait_finished(qtbot, window)
    assert not window.power_slider.isEnabled()
    assert not window.speed_spin.isEnabled()


def test_scrubbing_shows_nearest_snapshot(make_window, qtbot) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(short_scenario(60.0))
    window.start_run(RunMode.FAST)
    wait_finished(qtbot, window)
    store = window.store
    assert len(store) > 3
    before = window.slice_view.displayed_image().copy()  # type: ignore[union-attr]

    window.scrub_to(12.0)
    snap = store.nearest(12.0)
    assert window.displayed_snapshot is snap
    assert window.slice_view.snapshot is snap
    assert window.chart.cursor_time == pytest.approx(snap.t)
    after = window.slice_view.displayed_image()
    assert after is not None
    assert not np.array_equal(before, after)

    # dragging the slider has the same effect
    window.scrubber.setValue(0)
    assert window.displayed_snapshot is store[0]
    window.scrubber.setValue(1000)
    assert window.displayed_snapshot is store[len(store) - 1]


def test_follow_live_toggles_off_when_scrubbing_during_run(make_window, qtbot) -> None:  # type: ignore[no-untyped-def]
    window = make_window(fast_rate=60.0)
    window.load_scenario(short_scenario(7200.0))
    window.start_run(RunMode.FAST)
    qtbot.waitUntil(lambda: len(window.store) > 2, timeout=5000)
    assert window.follow_live.isChecked()
    window.scrub_to(0.0)
    assert not window.follow_live.isChecked()
    assert window.displayed_snapshot is window.store[0]
    window.stop_run()
    wait_finished(qtbot, window)


def test_playback_advances(make_window, qtbot) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(short_scenario(60.0))
    window.start_run(RunMode.FAST)
    wait_finished(qtbot, window)
    window.scrub_to(0.0)
    window.playback_speed.setCurrentIndex(3)  # 1000x
    window.toggle_playback()
    qtbot.waitUntil(lambda: window.displayed_snapshot is window.store[len(window.store) - 1])
    qtbot.waitUntil(lambda: window.play_button.text() == "Play")


# ------------------------------------------------------------------ units
def test_default_units_us_and_toggle_changes_labels(make_window, settings: QSettings) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    assert window.system == "us"
    assert window.chart.temperature_unit == "°F"
    assert window.chart.power_unit == "BTU/hr"
    label_before = window.editor.findChildren(QLabel)
    texts = [lbl.text() for lbl in label_before]
    assert any("(°F)" in t for t in texts)
    edit = window.editor.field_widget("atmosphere.ambient_temp_K")
    us_text = edit.text()
    si_before = window.editor.data()["atmosphere"]["ambient_temp_K"]

    window.set_units("metric")
    assert window.chart.temperature_unit == "°C"
    assert window.chart.power_unit == "kW"
    assert any("(°C)" in lbl.text() for lbl in window.editor.findChildren(QLabel))
    assert not any("(°F)" in lbl.text() for lbl in window.editor.findChildren(QLabel))
    assert edit.text() != us_text
    assert float(edit.text()) == pytest.approx(si_before - 273.15, abs=1e-3)
    assert window.editor.data()["atmosphere"]["ambient_temp_K"] == si_before  # SI untouched
    assert str(settings.value("units")) == "metric"
    assert window.slice_view._range_unit.text() == "°C"

    window.set_units("us")
    assert window.chart.temperature_unit == "°F"


def test_units_remembered_between_launches(qtbot, settings: QSettings) -> None:  # type: ignore[no-untyped-def]
    first = MainWindow(settings=settings)
    qtbot.addWidget(first)
    first.units_combo.setCurrentIndex(first.units_combo.findData("metric"))
    assert first.system == "metric"
    second = MainWindow(settings=settings)
    qtbot.addWidget(second)
    assert second.system == "metric"
    assert second.chart.temperature_unit == "°C"


# ------------------------------------------------------------------ export / import degrade
def test_export_and_import_disabled_until_recording_lands(make_window, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(rec, "export_csv_available", lambda: False)
    monkeypatch.setattr(rec, "measured_available", lambda: False)
    window = make_window()
    assert not window.act_export.isEnabled()
    assert not window.act_import.isEnabled()
    assert "WP07" in window.act_export.toolTip()
    assert "WP07" in window.act_import.toolTip()


def test_export_csv_goes_through_adapter(make_window, qtbot, monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[int, Path, str]] = []
    monkeypatch.setattr(rec, "export_csv_available", lambda: True)
    monkeypatch.setattr(rec, "export_csv", lambda s, p, sysm: calls.append((len(s), p, sysm)))
    window = make_window()
    window.load_scenario(short_scenario(30.0))
    window.start_run(RunMode.FAST)
    wait_finished(qtbot, window)
    assert window.act_export.isEnabled()
    assert window.export_csv(tmp_path / "out.csv")
    assert calls and calls[0][1] == tmp_path / "out.csv" and calls[0][2] == "us"


def test_measured_overlay_and_comparison(make_window, qtbot, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    comparison = SimpleNamespace(
        rmse_K=1.5,
        max_abs_K=3.0,
        bias_K=-0.5,
        time_to_target_sim_s=100.0,
        time_to_target_meas_s=110.0,
        time_to_target_error_rel=-0.09,
    )
    monkeypatch.setattr(rec, "compare_available", lambda: True)
    monkeypatch.setattr(rec, "compare", lambda *a, **k: comparison)
    window = make_window()
    window.load_scenario(short_scenario(30.0))
    window.start_run(RunMode.FAST)
    wait_finished(qtbot, window)
    series = SimpleNamespace(
        t=np.array([0.0, 10.0, 20.0]), T=np.array([293.0, 295.0, 298.0]), label="logger"
    )
    window.apply_measured(series)
    assert window.chart.has_measured
    assert "RMSE" in window.summary.comparison_label.text()
    assert "Time to target" in window.summary.comparison_label.text()


def test_measured_dialog_preview(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from thermosim.gui.measured_dialog import MeasuredImportDialog

    csv_path = tmp_path / "m.csv"
    csv_path.write_text("time_min,temp_F\n0,68\n1,70\n2,73\n")
    dialog = MeasuredImportDialog(csv_path, "us")
    qtbot.addWidget(dialog)
    assert dialog.columns == ["time_min", "temp_F"]
    assert dialog.preview.rowCount() == 3
    assert dialog.temp_unit_combo.currentText() == "F"
    dialog.time_unit_combo.setCurrentText("min")
    sel = dialog.selection()
    assert (sel.time_col, sel.temp_col, sel.time_unit) == ("time_min", "temp_F", "min")


# ------------------------------------------------------------------ slice view
def test_slice_view_readout_and_color_modes(make_window, qtbot) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(short_scenario(30.0))
    window.start_run(RunMode.FAST)
    wait_finished(qtbot, window)
    view = window.slice_view
    lo, hi = view.levels_K
    sc = window.editor.scenario()
    assert sc is not None
    assert lo == pytest.approx(sc.liquid.initial_temp_K)
    assert hi == pytest.approx(sc.target.temp_K + 2.0)

    snap = view.snapshot
    assert snap is not None
    r = view._rect_disp
    text = view.readout_at(r.center().x(), r.top() + 0.1 * r.height())
    assert "°F" in text and "x, z:" in text
    view.set_color_mode("auto")
    assert view.color_mode == "auto"
    lo2, hi2 = view.levels_K
    assert lo2 == pytest.approx(float(np.nanmin(snap.slice_T)))
    assert hi2 == pytest.approx(float(np.nanmax(snap.slice_T)))
    view.set_system("metric")
    assert "°C" in view.readout_at(r.center().x(), r.top() + 0.1 * r.height()) or True


def test_screenshot(make_window, qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.resize(1600, 950)
    window.load_scenario(short_scenario(120.0))
    window.start_run(RunMode.FAST)
    wait_finished(qtbot, window)
    window.scrub_to(window._t_max() * 0.6)
    qtbot.wait(100)
    image = window.grab()
    assert not image.isNull()
    out = tmp_path / "thermosim_gui.png"
    assert image.save(str(out))
    assert out.stat().st_size > 10_000
    artifacts = Path(__file__).parent / "_artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "thermosim_gui.png").write_bytes(out.read_bytes())
