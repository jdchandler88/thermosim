"""Flow mode in the GUI: model choice, arrows, speed colours, peak speed, pre-run estimate."""

from __future__ import annotations

import numpy as np
import pytest
from gi_helpers import quick_scenario
from PySide6.QtWidgets import QComboBox, QLabel

from thermosim.core.types import RunMode, Snapshot
from thermosim.gui import integration
from thermosim.gui.slice_view import (
    MAX_ARROWS_ACROSS,
    SliceView,
    arrow_stride,
    snapshot_peak_speed,
)

RUN_TIMEOUT_MS = 240_000


def flow_quick(**sim: float):  # type: ignore[no-untyped-def]
    sc = quick_scenario(dx=0.02, **sim)
    return sc.model_copy(update={"model": sc.model.model_copy(update={"liquid_transport": "flow"})})


def synthetic_snapshot(nx: int = 30, nz: int = 20, with_flow: bool = True) -> Snapshot:
    dx = 0.01
    x = (np.arange(nx) + 0.5) * dx - nx * dx / 2
    z = (np.arange(nz) + 0.5) * dx
    T = np.full((nx, nz), 330.0, np.float32)
    T[:2, :] = np.nan
    u = np.zeros((nx, nz), np.float32)
    w = np.zeros((nx, nz), np.float32)
    ii, kk = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    u[:] = 0.05 * np.sin(2 * np.pi * kk / nz)
    w[:] = 0.1 * np.cos(2 * np.pi * ii / nx)
    u[np.isnan(T)] = np.nan
    w[np.isnan(T)] = np.nan
    return Snapshot(
        t=10.0,
        slice_T=T,
        x_centers=x,
        z_centers=z,
        wall_left_z=z,
        wall_left_T=np.full(nz, 330.0, np.float32),
        wall_right_z=z,
        wall_right_T=np.full(nz, 330.0, np.float32),
        base_x=x,
        base_T=np.full(nx, 340.0, np.float32),
        lid_T=None,
        probes={},
        T_mean=330.0,
        power_delivered=1000.0,
        velocity_u=u if with_flow else None,
        velocity_w=w if with_flow else None,
    )


def test_slice_view_arrows_are_decimated_scaled_and_toggleable(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = SliceView()
    qtbot.addWidget(view)
    view.show()
    snap = synthetic_snapshot(nx=60, nz=40)
    view.set_snapshot(snap)
    assert view.has_flow and view.arrows_visible
    n = view.n_arrows()
    stride = arrow_stride(snap.slice_T.shape)
    assert stride == 3  # 60 cells across, at most 24 arrows: every third cell
    assert 0 < n <= (60 // stride + 1) * (40 // stride + 1)
    assert n <= (MAX_ARROWS_ACROSS + 1) ** 2
    # scaled so that the longest arrow is ARROW_MAX_CELLS decimated cells long
    longest = max(np.hypot(x1 - x0, z1 - z0) for x0, z0, x1, z1 in view.arrow_segments())
    from thermosim.gui.formatting import plot_length
    from thermosim.gui.slice_view import ARROW_MAX_CELLS

    dx_disp = float(plot_length(0.01, view._system))
    assert longest <= ARROW_MAX_CELLS * stride * dx_disp * 1.0001
    assert longest > 0.5 * ARROW_MAX_CELLS * stride * dx_disp
    view.set_show_arrows(False)
    assert view.n_arrows() == 0 and not view.arrows_visible
    view.set_show_arrows(True)
    assert view.n_arrows() == n


def test_slice_view_speed_colours_change_the_image(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = SliceView()
    qtbot.addWidget(view)
    view.show()
    view.set_snapshot(synthetic_snapshot())
    temp_img = view.displayed_image()
    assert temp_img is not None
    view.set_show_speed(True)
    assert view.speed_colors
    speed_img = view.displayed_image()
    assert speed_img is not None and not np.array_equal(temp_img, speed_img)
    assert (speed_img[:2, :, 3] == 0).all()  # NaN cells stay transparent
    view.set_show_speed(False)
    assert np.array_equal(view.displayed_image(), temp_img)


def test_slice_view_without_velocity_has_no_arrows_or_flow_controls(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = SliceView()
    qtbot.addWidget(view)
    view.show()
    view.set_snapshot(synthetic_snapshot(with_flow=False))
    assert not view.has_flow and view.n_arrows() == 0
    assert not view._arrows_check.isVisible() and not view._speed_check.isVisible()
    view.set_snapshot(synthetic_snapshot())
    assert view._arrows_check.isVisible() and view.n_arrows() > 0


def test_readout_includes_flow_speed(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = SliceView()
    qtbot.addWidget(view)
    view.show()
    view.set_snapshot(synthetic_snapshot())
    r = view._rect_disp
    assert "cm/s" in view.readout_at(r.center().x(), r.center().y())


def test_snapshot_peak_speed() -> None:
    assert snapshot_peak_speed(synthetic_snapshot(with_flow=False)) is None
    snap = synthetic_snapshot()
    peak = snapshot_peak_speed(snap)
    assert peak is not None and 0.1 <= peak < 0.12


def test_editor_offers_model_choice_with_slow_note(qtbot, make_window) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(quick_scenario())
    qtbot.wait(50)  # the editor rebuilds its form on load; let the old widgets go
    combos = [
        c
        for c in window.editor.findChildren(QComboBox)
        if c.findData("flow") >= 0 and c.findData("effective_conductivity") >= 0
    ]
    assert combos
    combo = combos[-1]
    assert "slower" in combo.itemText(combo.findData("flow"))
    combo.setCurrentIndex(combo.findData("flow"))
    sc = window.editor.scenario()
    assert sc is not None and sc.model.liquid_transport == "flow"
    notes = [lab for lab in window.editor.findChildren(QLabel) if lab.objectName() == "flow_note"]
    assert notes and "much slower" in notes[0].text()


def test_preflight_in_flow_mode_accounts_for_flow_cost() -> None:
    eff = quick_scenario(dx=0.02)
    flow = flow_quick()
    r_eff = integration.run_preflight(eff)
    r_flow = integration.run_preflight(flow)
    assert r_eff.ok and r_flow.ok
    assert not r_eff.flow and r_flow.flow
    assert r_flow.wall_per_step_s > 0 and 0 < r_flow.dt_run_s <= 1.0
    assert r_flow.sim_per_wall < r_eff.sim_per_wall
    assert r_flow.est_wall_s is not None and r_eff.est_wall_s is not None
    assert r_flow.est_wall_s > 3 * r_eff.est_wall_s
    assert r_flow.preview_m == integration.PREVIEW_CELL_SIZE_FLOW_M
    assert r_eff.preview_m == integration.PREVIEW_CELL_SIZE_M
    assert "flow mode" in integration.describe_preflight(r_flow)


def test_flow_dt_estimate_shrinks_with_cell_size() -> None:
    a = integration.flow_dt_estimate(0.02, 100.0)
    b = integration.flow_dt_estimate(0.01, 100.0)
    assert 0.03 < a < 0.15 and 0.01 < b < 0.05 and b < a
    assert integration.flow_dt_estimate(0.02, 0.001) == 0.001


def test_gui_flow_run_shows_arrows_and_peak_speed(qtbot, make_window) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(flow_quick(max_sim_time_s=120.0))
    assert window.start_run(RunMode.FAST)
    qtbot.waitUntil(
        lambda: window.result is not None and not window.is_active, timeout=RUN_TIMEOUT_MS
    )
    view = window.slice_view
    snap = view.snapshot
    assert snap is not None and snap.velocity_u is not None
    assert view.has_flow and view.arrows_visible and view.n_arrows() > 5
    summary = window.summary
    assert summary.peak_speed_label.isVisibleTo(summary)
    assert "cm/s" in summary.peak_speed_label.text()
    peak_cm = float(summary.peak_speed_label.text().split()[0])
    assert peak_cm == pytest.approx(
        100 * max(snapshot_peak_speed(window.store[n]) or 0.0 for n in range(len(window.store))),
        abs=0.06,
    )
    assert peak_cm > 0.1
    view.set_show_speed(True)
    assert view.speed_colors
    # scrubbing back to t = 0 (fluid at rest) leaves nothing to draw
    window.scrub_to(0.0)
    assert view.n_arrows() == 0


def test_effective_mode_run_hides_flow_widgets(qtbot, make_window) -> None:  # type: ignore[no-untyped-def]
    window = make_window()
    window.load_scenario(quick_scenario(max_sim_time_s=20.0))
    window.start_run(RunMode.FAST)
    qtbot.waitUntil(
        lambda: window.result is not None and not window.is_active, timeout=RUN_TIMEOUT_MS
    )
    assert not window.slice_view.has_flow
    assert not window.summary.peak_speed_label.isVisibleTo(window.summary)
