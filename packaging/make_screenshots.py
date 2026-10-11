"""Generate the user-guide screenshots with the offscreen Qt renderer and the REAL runner.

Usage (from the repository root):

    QT_QPA_PLATFORM=offscreen uv run python packaging/make_screenshots.py [OUT_DIR]

OUT_DIR defaults to docs/images. Takes about a minute.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from thermosim.config.io import example_scenarios
from thermosim.config.schema import Scenario
from thermosim.core.types import RunMode
from thermosim.gui import recording_adapter as rec
from thermosim.gui.app import create_application
from thermosim.gui.main_window import MainWindow

CELL_SIZE_M = 0.01


def pump(app: QApplication, seconds: float = 0.2) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait_for(app: QApplication, condition, timeout: float = 300.0) -> None:  # type: ignore[no-untyped-def]
    end = time.time() + timeout
    while not condition():
        if time.time() > end:
            raise TimeoutError("timed out waiting for the GUI")
        pump(app, 0.05)


def scenario(name: str = "small_pot_stove", **sim: float) -> Scenario:
    base = example_scenarios()[name]
    update = {"cell_size_m": CELL_SIZE_M, **sim}
    return base.model_copy(update={"simulation": base.simulation.model_copy(update=update)})


def shot(window: MainWindow, app: QApplication, path: Path) -> None:
    pump(app, 0.3)
    window.grab().save(str(path))
    print(f"wrote {path}")


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    app = create_application(["thermosim"])
    with tempfile.TemporaryDirectory() as tmp:
        settings = QSettings(str(Path(tmp) / "s.ini"), QSettings.Format.IniFormat)
        window = MainWindow(settings=settings)  # the real runner and the pre-run checks
        window.resize(1500, 900)
        window.show()
        window.load_scenario(scenario())
        pump(app)

        # 1. the setup and the quick estimate, before any run
        window.quick_panel.button.click()
        shot(window, app, out_dir / "setup.png")
        window.editor._tabs.setCurrentIndex(3)  # Heat source
        shot(window, app, out_dir / "setup-heat-source.png")
        window.editor._tabs.setCurrentIndex(0)

        # 2. a fast run, to the end
        window.start_run(RunMode.FAST)
        wait_for(app, lambda: window.result is not None and not window.is_active)
        window.scrub_to(window.store.times()[-1] * 0.6)
        shot(window, app, out_dir / "main-window-fast-run.png")

        # 3. measured data and the comparison (illustrative data: the run plus an offset and
        # some noise, in degrees F and minutes as a user's log might be)
        rng = np.random.default_rng(1)
        csv = Path(tmp) / "measured.csv"
        rows = ["time_min,temp_F"]
        for s in window.samples[::6]:
            temp_f = (s.T_mean - 273.15) * 1.8 + 32.0 - 1.5 + rng.normal(0.0, 0.4)
            rows.append(f"{s.t / 60:.3f},{temp_f:.2f}")
        csv.write_text("\n".join(rows) + "\n")
        window.apply_measured(rec.load_measured(csv, "time_min", "temp_F", "F", "min"))
        window.scrub_to(window.store.times()[-1])
        shot(window, app, out_dir / "measured-comparison.png")

        # 4. real time at 10x, power turned down part-way
        window.chart.clear_measured()
        window._measured = None
        window.new_scenario(force=True)
        window.load_scenario(scenario(max_sim_time_s=3600.0))
        window.speed_spin.setValue(10.0)
        window.start_run(RunMode.REALTIME)
        wait_for(app, lambda: window.is_active and len(window.samples) > 40)
        window.power_slider.setValue(40)
        t_change = window.samples[-1].t
        wait_for(app, lambda: window.samples[-1].t > t_change + 120.0)
        shot(window, app, out_dir / "realtime-run.png")
        window.stop_run()
        wait_for(app, lambda: window.result is not None and not window.is_active)

        # 5. a geometry error shown on its field
        window._result_saved = True  # no unsaved-run prompt in a script
        window.chart.clear_measured()
        window._measured = None
        window.load_example("kettle_electric_insulated")
        window.editor._tabs.setCurrentIndex(2)  # Liquid
        window.editor.set_field_text("liquid.fill_height_m", "7.9")
        window.act_run_fast.trigger()
        wait_for(app, lambda: not window.is_preparing)
        window.editor._tabs.setCurrentIndex(3)  # Heat source
        shot(window, app, out_dir / "geometry-error.png")

        window.editor.mark_clean()
        window._result_saved = True
        window.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/images"))
