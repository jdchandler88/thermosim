# WP08: GUI shell (against a fake runner)

**Wave:** 1 · **Size:** L · **Depends on:** WP00. It builds forms from the
WP01 schema. Until WP01 lands, develop against the field list in CONTRACTS §5,
using a minimal local copy of the models under `tests/gui/` only, and switch
to the real import when WP01 merges.

## Goal

A complete desktop GUI that works end to end against a **fake runner**:
editing setups, controlling runs, a live slice heat map, charts, playback
scrubbing, export and measured-data overlay. WP11 later swaps in the real
runner.

## Owns

- `src/thermosim/gui/**` *except* `gui/integration.py` (WP11)
- `tests/gui/**`

## Tech

- PySide6 and pyqtgraph.
- No Qt Designer `.ui` files; build the layout in code.
- Use `pyqtgraph.ImageItem` for the slice and `PlotWidget` for the charts.
- Colormap: `pyqtgraph.colormap.get("inferno")`, falling back to "viridis".

## Layout

```
┌ Toolbar: New · Open · Save · Save As │ Run ▾(Fast | Real-time) · Pause · Stop │ Speed [1×] │ Power [====|  ] 75% │ Units (Metric|US) │ Export CSV · Import measured ┐
├───────────────┬───────────────────────────────────────────────┬───────────────────┤
│ Setup editor  │  Slice view (heat map, y=0 plane)             │ Summary           │
│ (tabs)        │   + vessel wall/base/lid lines colored by T   │  status / reason  │
│  Vessel       │   + probe markers + heat source drawn         │  time to target   │
│  Lid & insul. │   + color bar, hover readout "x, z: 63.2 °C"  │  sim t / wall t   │
│  Liquid       │   + optional velocity arrows (Phase 2 hook)   │  energy breakdown │
│  Heat source  │  Playback scrubber ◀ ▶ ━━━━●━━━━ t = 12:35     │  (table + bar)    │
│  Atmosphere   ├───────────────────────────────────────────────┤  fuel / kWh       │
│  Probes/target│  Chart: probe temps, mean, min/max band,      │  measured-vs-sim  │
│  Simulation   │  target line, power (2nd axis), measured data │  comparison       │
└───────────────┴───────────────────────────────────────────────┴───────────────────┘
```

## Requirements

1. **Setup editor** (`gui/editor/`), generated from the pydantic schema:
   - Use `quantity_of()` for units. Show values in the current unit system and
     store them in SI.
   - Discriminated unions (heat source type, actuator type) become a combo box
     that swaps the sub-form.
   - Lists (probes, element path points, power schedule) become editable
     tables with add and remove buttons.
   - Validation happens live. Invalid fields get a red border and a tooltip
     with the message, and the Run buttons are disabled while the setup is
     invalid. Show cross-field errors in a banner.
   - Dirty-state tracking: an asterisk in the title, and a prompt to save on
     close.
2. **Slice view** (`gui/slice_view.py`):
   - Shows the latest snapshot while running, or the scrubbed one.
   - The image is scaled to physical coordinates (m in metric, in in US) and
     uses a fixed aspect ratio.
   - Walls are drawn as thick polylines colored by temperature with the same
     colormap. The lid is a top bar when present.
   - Burner footprint: a hatched bar under the base. Element: its path
     projected onto the slice, as a dashed line, drawn faded where it's off
     the plane.
   - Color range: "Auto" (the current min and max) or "Fixed" (from the
     initial temperature to the target plus 2 K, the default). User-editable.
   - Hovering shows the coordinates and the temperature in display units.
3. **Chart** (`gui/chart.py`):
   - Lines for each probe and the mean, a min–max band, a dashed target line,
     and power on a right-hand axis.
   - The measured series appears as dots.
   - The time axis is mm:ss, switching to h:mm when longer than an hour.
   - A vertical cursor follows the scrubber.
4. **Run controls:**
   - Run in Fast or Real-time mode; Pause/Resume; Stop.
   - The Speed spinbox (0.1× to 100×) and the Power slider are enabled **only
     in real-time mode while running**. They call `runner.set_speed` and
     `runner.set_power_command`.
   - The status bar shows the state, the sim time, the wall time and the
     real-time lag.
5. **Event loop:** a `QTimer` at 30 Hz calls `runner.poll_events()`, then
   updates the views with the newest snapshot and the batch of samples.
   Never block the UI thread.
6. **Playback:** the scrubber is enabled during and after a run. Dragging it
   shows `store.nearest(t)`. Play/pause plays back at a chosen speed. A
   "Follow live" toggle is on by default while running.
7. **Export and import:**
   - Export CSV goes through `recording.export_csv` (via an adapter that is a
     no-op stub until WP07 lands) in the current unit system.
   - Import measured opens a dialog with a preview of the parsed columns and
     dropdowns for the time and temperature columns and their units, then
     overlays the data and fills in the comparison panel.
8. **Units toggle:** switches every displayed value immediately, without
   changing the stored SI values. The choice is persisted with `QSettings`.
9. **`gui/fake_runner.py`:** `FakeRunner(scenario, mode)` implements the
   `Runner` protocol in a background thread.
   - It generates plausible synthetic data: a 2D temperature field that warms
     from the heat source with diffusion, using the real slice dimensions
     computed from the scenario (nx = ceil(D/dx), nz = ceil(h/dx)). It also
     produces probe and mean temperatures following
     T_target − (T_target − T0)·e^(−t/τ), and synthetic energy flows.
   - It honors pause, stop, speed and power commands. Real-time mode paces
     against the wall clock.
   - It emits a `FinishedEvent` with a `RunResult` whose snapshots use a
     simple list-backed store.
10. **Entry point:**
    - `gui/app.py: main(argv) -> int`. It accepts an optional scenario path,
      and `--fake` to force the fake runner (the default until WP11).
    - It sets the application name, organisation name and icon (a simple
      generated icon is fine).
    - High-DPI friendly; must look right on Windows at 150% scaling.
11. **Factory seam for WP11:** `gui/runner_factory.py: make_runner(scenario,
    mode) -> Runner`, which returns a `FakeRunner` for now. WP11 changes only
    this function and `integration.py`.

## Acceptance criteria (pytest-qt, offscreen)

- The window opens, loads each example scenario (or local fixtures) into the
  editor, edits a field, and saves; the saved TOML round-trips.
- Invalid input disables Run and shows a message.
- A fake run (fast mode, 60 sim-s at 100× speed) produces snapshot and chart
  updates and a summary. Stop works mid-run.
- In real-time mode, the power slider calls `set_power_command`. In fast mode
  it is disabled.
- After the run, scrubbing to t changes the displayed image to
  `store.nearest(t)`.
- The units toggle changes the editor and chart labels between °C and °F.
- Screenshot test: render the main window offscreen to a PNG (stored as a CI
  artifact, not compared).
