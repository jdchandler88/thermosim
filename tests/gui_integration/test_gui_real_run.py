"""The GUI driving the real simulator (offscreen)."""

from __future__ import annotations

from pathlib import Path

import pytest
from gi_helpers import quick_scenario
from PySide6.QtWidgets import QMessageBox

from thermosim.core.protocols import RunnerState
from thermosim.core.types import RunMode, StopReason
from thermosim.gui import integration
from thermosim.gui.formatting import fmt_clock
from thermosim.gui.main_window import MainWindow
from thermosim.sim.runner import SimulationRunner, run_headless

RUN_TIMEOUT_MS = 180_000


def wait_finished(qtbot, window: MainWindow) -> None:  # type: ignore[no-untyped-def]
    qtbot.waitUntil(
        lambda: window.result is not None and not window.is_active, timeout=RUN_TIMEOUT_MS
    )


def test_default_window_uses_real_runner(qtbot, make_window) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(quick_scenario(max_sim_time_s=30.0))
    assert window.start_run(RunMode.FAST)
    assert window.is_preparing
    qtbot.waitUntil(lambda: window.is_active, timeout=RUN_TIMEOUT_MS)
    assert isinstance(window.runner, SimulationRunner)
    wait_finished(qtbot, window)


def test_fast_run_matches_headless(qtbot, make_window) -> None:  # type: ignore[no-untyped-def]
    scenario = quick_scenario()
    window = make_window()
    window.load_scenario(scenario)
    window.start_run(RunMode.FAST)
    wait_finished(qtbot, window)
    summary = window.result.summary  # type: ignore[union-attr]
    assert summary.stop_reason == StopReason.TARGET_REACHED

    reference = run_headless(window.runner.scenario)  # type: ignore[union-attr]
    assert summary.time_to_target_s is not None
    assert reference.summary.time_to_target_s is not None
    # The step sequence depends on how the run is chunked (Model.advance clips dt at the chunk
    # end, and the chunk size follows wall time), so two runs of the same setup differ by
    # about 1e-6 relative; 1e-9 cannot hold (see the WP11 report).
    assert summary.time_to_target_s == pytest.approx(reference.summary.time_to_target_s, rel=1e-4)
    assert fmt_clock(summary.time_to_target_s) in window.summary.time_to_target_label.text()
    assert window.chart.n_points > 5
    assert len(window.store) > 1
    assert not window.result_saved


def test_realtime_power_slider_changes_delivered_power(qtbot, make_window) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(quick_scenario(max_sim_time_s=3600.0))
    window.speed_spin.setValue(10.0)
    window.start_run(RunMode.REALTIME)
    qtbot.waitUntil(lambda: window.is_active and len(window.samples) >= 5, timeout=RUN_TIMEOUT_MS)
    full = window.samples[-1].power_delivered
    assert full > 0
    t_change = window.samples[-1].t

    window.power_slider.setValue(20)
    assert window.chart.n_command_markers == 1  # visible at once, before any new sample
    qtbot.waitUntil(lambda: window.samples[-1].t > t_change + 1.5, timeout=RUN_TIMEOUT_MS)
    after = [s for s in window.samples if s.t >= t_change + 1.0]
    assert after, "no samples a second after the change"
    assert after[0].command == pytest.approx(0.2)
    assert after[0].power_delivered < 0.8 * full
    window.stop_run()
    wait_finished(qtbot, window)
    assert window.result.summary.stop_reason == StopReason.USER_STOP  # type: ignore[union-attr]


def test_keyboard_shortcuts(qtbot, make_window) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(quick_scenario(max_sim_time_s=3600.0))
    window.speed_spin.setValue(10.0)
    window.start_run(RunMode.REALTIME)
    qtbot.waitUntil(lambda: window.is_active and len(window.samples) >= 2, timeout=RUN_TIMEOUT_MS)
    by_key = {sc.key().toString(): sc for sc in window._shortcuts}
    assert {"Space", "Up", "Down", "+", "-"} <= set(by_key)

    by_key["Down"].activated.emit()
    assert window.power_slider.value() == 95
    by_key["Up"].activated.emit()
    by_key["Up"].activated.emit()
    assert window.power_slider.value() == 100  # clamped
    by_key["+"].activated.emit()
    assert window.speed_spin.value() == pytest.approx(12.5)
    by_key["-"].activated.emit()
    assert window.speed_spin.value() == pytest.approx(10.0)
    by_key["Space"].activated.emit()
    assert window.runner.state == RunnerState.PAUSED  # type: ignore[union-attr]
    by_key["Space"].activated.emit()
    assert window.runner.state == RunnerState.RUNNING  # type: ignore[union-attr]
    window.stop_run()
    wait_finished(qtbot, window)


def test_geometry_error_marks_the_right_field(qtbot, make_window) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_example("kettle_electric_insulated")
    # 7.9 in of liquid: above the probes, below the top of the element's legs (0.25 m)
    window.editor.set_field_text("liquid.fill_height_m", "7.9")
    assert window.editor.is_valid()
    window.start_run(RunMode.FAST)
    qtbot.waitUntil(lambda: not window.is_preparing, timeout=RUN_TIMEOUT_MS)
    assert not window.is_active
    assert window.last_preflight is not None and not window.last_preflight.ok
    assert window.last_preflight.error_field == "heat_source.path_m"
    assert "above the liquid" in window.editor._banner.text()
    assert window.editor._banner.isVisible()
    assert window.editor._tabs.tabText(3).endswith("!")  # Heat source tab

    # editing re-validates, which clears the marking
    window.editor.set_field_text("liquid.fill_height_m", "11")
    assert not window.editor._banner.isVisible()


def test_cancel_preflight(qtbot, make_window) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(quick_scenario())
    window.start_run(RunMode.FAST)
    assert window.is_preparing
    controller = window.preflight_controller
    assert controller is not None
    controller.cancel()
    qtbot.waitUntil(lambda: not window.is_preparing, timeout=RUN_TIMEOUT_MS)
    assert window.last_preflight is not None and window.last_preflight.cancelled
    assert not window.is_active
    assert window.act_run_fast.isEnabled()


def test_preflight_reports_cells_estimate_and_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(integration, "_solver_warm", False)
    seen: list[str] = []
    result = integration.run_preflight(quick_scenario(), status=seen.append)
    assert result.ok
    assert seen[0] == integration.STATUS_CHECK
    assert integration.STATUS_WARMUP in seen
    assert result.liquid_cells > 100
    assert result.est_wall_s is not None and result.est_wall_s > 0
    assert result.quick_estimate_s is not None
    # warm now: no second warm-up message
    seen.clear()
    integration.run_preflight(quick_scenario(), status=seen.append)
    assert integration.STATUS_WARMUP not in seen


def test_long_run_warning_offers_preview_cell_size(qtbot, make_window, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(integration, "LONG_RUN_S", 0.0)
    asked: list[float] = []

    def ask(self: MainWindow, result: integration.PreflightResult) -> str:
        asked.append(result.est_wall_s or 0.0)
        return "preview"

    monkeypatch.setattr(MainWindow, "_ask_long_run", ask)
    window = make_window()
    window.load_scenario(quick_scenario(dx=0.006, max_sim_time_s=20.0))
    window.start_run(RunMode.FAST)
    qtbot.waitUntil(lambda: window.is_active, timeout=RUN_TIMEOUT_MS)
    assert asked
    assert window.editor.scenario().simulation.cell_size_m == pytest.approx(0.010)  # type: ignore[union-attr]
    wait_finished(qtbot, window)


def test_long_run_warning_cancel(qtbot, make_window, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(integration, "LONG_RUN_S", 0.0)
    monkeypatch.setattr(MainWindow, "_ask_long_run", lambda self, result: "cancel")
    window = make_window()
    window.load_scenario(quick_scenario())
    window.start_run(RunMode.FAST)
    qtbot.waitUntil(lambda: not window.is_preparing, timeout=RUN_TIMEOUT_MS)
    assert not window.is_active


def test_quick_estimate_panel(qtbot, make_window) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(quick_scenario())
    assert window.quick_panel.button.isEnabled()
    window.quick_panel.button.click()
    text = window.quick_panel.label.text()
    assert text.startswith("About ") and "lumped" in text
    window.set_units("metric")
    assert "°C" in window.quick_panel.label.text()
    window.editor.set_field_text("vessel.inner_diameter_m", "x")
    assert not window.quick_panel.button.isEnabled()


def test_unsaved_run_prompt_and_save_open_scrub(  # type: ignore[no-untyped-def]
    qtbot, make_window, tmp_path: Path, monkeypatch
) -> None:
    window = make_window()
    window.load_scenario(quick_scenario(max_sim_time_s=120.0))
    window.start_run(RunMode.FAST)
    wait_finished(qtbot, window)
    assert not window.result_saved
    n_snapshots = len(window.store)
    assert n_snapshots >= 2

    # an unsaved run prompts; Cancel keeps it
    asked: list[str] = []

    def cancel(*args: object, **kwargs: object) -> QMessageBox.StandardButton:
        asked.append("asked")
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", cancel)
    assert not window.new_scenario()
    assert not window.start_run(RunMode.FAST)
    assert asked == ["asked", "asked"]
    assert window.result is not None

    # save, after which nothing is asked
    run_dir = tmp_path / "myrun"
    assert window.save_run(run_dir)
    assert window.result_saved
    assert (run_dir / "snapshots.npz").exists()
    asked.clear()
    window.load_example("small_pot_stove")
    assert asked == []
    assert window.result is None

    # open it: playback mode, no live runner
    assert window.open_run(run_dir)
    assert window.runner is None and not window.is_active
    assert window.result is not None and window.result_saved
    assert len(window.store) == n_snapshots
    assert window.chart.n_points == len(window.result.samples)
    t_end = window.store.times()[-1]
    window.scrub_to(0.0)
    assert window.displayed_snapshot.t == pytest.approx(0.0)  # type: ignore[union-attr]
    window.scrub_to(t_end)
    assert window.displayed_snapshot.t == pytest.approx(t_end)  # type: ignore[union-attr]
    window.scrubber.setValue(window.scrubber.maximum() // 2)
    assert 0.0 < window.displayed_snapshot.t < t_end  # type: ignore[union-attr]
    assert window.editor.scenario() is not None  # the run's setup is loaded


def test_export_and_import_are_live(qtbot, make_window, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from thermosim.gui import recording_adapter as rec

    assert rec.export_csv_available()
    assert rec.measured_available()
    assert rec.compare_available()
    window = make_window()
    window.load_scenario(quick_scenario(max_sim_time_s=120.0))
    assert window.act_import.isEnabled()
    assert not window.act_export.isEnabled()  # nothing to export yet
    window.start_run(RunMode.FAST)
    wait_finished(qtbot, window)
    assert window.act_export.isEnabled()
    out = tmp_path / "out.csv"
    assert window.export_csv(out)
    lines = out.read_text().splitlines()
    assert len(lines) == len(window.samples) + 1

    # measured data round trip: the run's own mean temperature, in degC and minutes
    measured = tmp_path / "meas.csv"
    rows = ["time_min,temp_C"] + [
        f"{s.t / 60:.4f},{s.T_mean - 273.15:.3f}" for s in window.samples[::5]
    ]
    measured.write_text("\n".join(rows))
    series = rec.load_measured(measured, "time_min", "temp_C", "C", "min")
    window.apply_measured(series)
    assert window.chart.has_measured
    assert "rmse" in window.summary.comparison_label.text().lower()
