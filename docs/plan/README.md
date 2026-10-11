# ThermoSim Implementation Plan

This plan breaks [REQUIREMENTS.md](../../REQUIREMENTS.md) into **work packages
(WPs)**. Each work package can be handed to an independent agent or developer.

Read this file first, then [CONTRACTS.md](CONTRACTS.md), then your own
work-package file in [`wp/`](wp/).

---

## 1. Architecture overview

```
                    ┌────────────────────────────────────────────┐
                    │  GUI (PySide6 + pyqtgraph)     WP08, WP11   │
                    │  setup editor · slice view · charts · scrub │
                    └───────────────▲────────────────────────────┘
                                    │ Runner protocol (poll events)
 ┌──────────────┐   ┌───────────────┴────────────────────────────┐
 │ CLI   WP09   │──▶│  sim.runner  (thread, fast/real-time)  WP09 │
 └──────────────┘   │  sim.model   (one time step = orchestration)│
                    └──┬──────┬─────────┬─────────┬─────────┬────┘
                       │      │         │         │         │
          ┌────────────▼┐ ┌───▼──────┐ ┌▼───────┐ ┌▼───────┐ ┌▼──────────┐
          │ config WP01 │ │ geometry │ │ sources│ │ solver │ │ recording │
          │ Scenario,   │ │ WP04     │ │ WP05   │ │ WP06   │ │ WP07      │
          │ units, TOML │ │ Domain   │ │ power, │ │ thermal│ │ series,   │
          └─────────────┘ │ builder  │ │ actuat.│ │ network│ │ snapshots,│
                          └──────────┘ └────────┘ └────────┘ │ CSV, cmp  │
          ┌──────────────────────────────────────┐          └───────────┘
          │ physics: properties WP02,            │
          │          correlations WP03           │
          └──────────────────────────────────────┘
          core.types / testing.synthetic (WP00) — shared by everything
```

### What happens in one simulation step

The model loop lives in `sim.model` (WP09). Each step:

1. **Refresh coefficients**, every `coeff_refresh_s` of simulated time:
   - read material properties (WP02) at the current temperatures;
   - compute the heat-transfer coefficients and the effective conductivity
     (WP03);
   - pack them into `StepCoefficients`.
2. **Get the heat input.** Ask the heat source (WP05) for the delivered power,
   given the current command (from the schedule, the user, or later the PID
   controller), and turn it into per-cell and per-shell-node source terms.
3. **Advance the temperatures.** `ThermalSolver.step(state, coeffs, dt)` (WP06)
   advances the temperature field and returns the energy that flowed during
   the step, by category.
4. **Record.** Every `sample_interval_s` and `snapshot_interval_s`, build a
   `Sample` and a `Snapshot` and hand them to the recorder (WP07). In a run,
   the runner also publishes snapshots to the GUI.
5. **Check whether to stop:** target reached, temperature levelled off below the
   target, maximum time reached, or the user pressed stop.

### Key design decisions (do not change without the coordinator)

| # | Decision | Why |
|---|---|---|
| D1 | SI units internally, **temperature in kelvin** everywhere in code. Conversions happen only in `config.units` (for the GUI, input and output). | Avoids °C/K mix-ups, and radiation needs absolute temperature. |
| D2 | **Coordinates:** origin at the center of the inner base (the bottom of the liquid); z points up; x and y are horizontal. The slice is the plane y = 0. | One convention for geometry, probes, elements and the GUI. |
| D3 | The liquid is a uniform Cartesian grid of cubic cells of edge `dx`. Array index order is `[i, j, k]` = `[x, y, z]`. Flat index is `np.ravel_multi_index((i,j,k), shape)` (C order). | Simple, fast with Numba, and reusable by the Phase 2 flow solver. |
| D4 | The vessel walls and base are **shell nodes** on their own 2D surface mesh (base disk plus unrolled cylinder), coupled to the liquid cells next to them. The lid and the headspace air are each a single lumped node. | Walls are thinner than grid cells; see REQUIREMENTS P3. |
| D5 | The solver is a **finite-volume, explicit** scheme, compiled with Numba. The time step is chosen automatically within its stability limit, and the shell may take several smaller substeps per step. | Simple and conservative, and fast enough at these grid sizes. |
| D6 | The solver is physics-agnostic. It receives **linearized coefficients** (`h`, `T_ref`) and source powers. All physics lives in WP02, WP03 and WP05, and is assembled in WP09. | Lets the solver, physics and source work proceed in parallel and be tested on their own. |
| D7 | The GUI talks to the simulation only through the `Runner` protocol, by **polling events** from a thread-safe queue. | Works the same with the fake runner and the real one, and doesn't depend on Qt internals. |
| D8 | Setups (`Scenario`) are pydantic v2 models, stored as TOML. Every numeric field carries a `quantity` tag, so the GUI can build its forms and unit conversions automatically. | Lets WP01 and WP08 proceed in parallel. |
| D9 | **Heat flux sign convention:** positive means heat *leaving* the body, for every loss correlation. Sources are positive into the body. | One rule everywhere. |

## 2. Repository layout (target)

```
pyproject.toml            # WP00 (deps added by WP owners via coordinator)
uv.lock                   # WP00
.github/workflows/ci.yml  # WP00
src/thermosim/
  __init__.py             # WP00
  core/types.py           # WP00 — shared dataclasses (CONTRACTS §2)
  core/errors.py          # WP00
  core/protocols.py       # WP00 — Runner, HeatSource, Actuator protocols
  testing/synthetic.py    # WP00 — synthetic box Domain for tests
  config/                 # WP01 — schema.py, units.py, io.py, defaults.py
  scenarios/*.toml        # WP01 — example setups (package data)
  physics/properties.py   # WP02
  physics/correlations.py # WP03
  geometry/               # WP04 — mesh.py, shell.py, lid.py, probes.py, slice.py, build.py
  sources/                # WP05 — heat_sources.py, actuators.py, commands.py, build.py
  solver/                 # WP06 — thermal.py, kernels.py
  recording/              # WP07 — recorder.py, snapshots.py, summary.py, measured.py, export.py
  gui/                    # WP08 (shell) + WP11 (integration)
  sim/                    # WP09 — model.py, coefficients.py, runner.py
  cli.py                  # WP09
  validation/lumped.py    # WP10 — single-temperature reference model
tests/<area>/             # each WP owns tests/<its area>/
benchmarks/               # WP10
docs/plan/                # coordinator
```

**Ownership rule:** an agent edits only the paths its work-package file lists
under *Owns*. If it needs a change elsewhere, including the contracts, it stops
and reports the change it needs (see §6).

## 3. Work packages and parallelism

| WP | Name | Wave | Depends on (code) | Size |
|---|---|---|---|---|
| [WP00](wp/WP00-skeleton-contracts.md) | Project skeleton, contracts, CI, synthetic fixtures | 0 | — | S |
| [WP01](wp/WP01-scenario-config.md) | Scenario schema, units, TOML I/O, examples | 1 | WP00 | M |
| [WP02](wp/WP02-properties.md) | Material and fluid properties | 1 | WP00 | M |
| [WP03](wp/WP03-correlations.md) | Heat-transfer correlations and effective conductivity | 1 | WP00 (WP02 API only) | M |
| [WP04](wp/WP04-geometry.md) | Geometry: grid, shell, probes, slice builder | 1 | WP00 | L |
| [WP05](wp/WP05-heat-sources.md) | Heat sources, actuators, power schedule | 1 | WP00 | M |
| [WP06](wp/WP06-thermal-solver.md) | Thermal network solver (Numba) | 1 | WP00 | L |
| [WP07](wp/WP07-recording.md) | Recording, snapshots, summary, CSV, measured-data comparison | 1 | WP00 | M |
| [WP08](wp/WP08-gui-shell.md) | GUI shell against a fake runner | 1 | WP00 (WP01 schema shape) | L |
| [WP09](wp/WP09-assembly-runner-cli.md) | Model assembly, runner, CLI | 2 | WP01–WP07 | L |
| [WP10](wp/WP10-validation.md) | Validation suite, lumped reference model, benchmarks | 1 → 2 | WP00; WP09 for end-to-end parts | M |
| [WP11](wp/WP11-gui-integration.md) | GUI integration with the real runner, Windows launch | 3 | WP08, WP09 | M |
| [Phase 2](wp/PHASE2.md) | WP20 backend spike · WP21 flow solver · WP22 integration · WP23 calibration | after Phase 1 | — | XL |
| [Phase 3](wp/PHASE3.md) | WP30 PID controller · WP31 PID GUI | after Phase 2 | — | M |

```
Wave 0:  WP00
            │
Wave 1:  WP01  WP02  WP03  WP04  WP05  WP06  WP07  WP08  WP10(part A)   ← 9 in parallel
            └─────┴─────┴─────┴─────┴─────┴─────┘        │
Wave 2:                     WP09 ──── WP10(part B)        │
                              └───────────┬───────────────┘
Wave 3:                                 WP11
```

How wave 1 runs in parallel: every wave-1 package codes against the
dataclasses and protocols that WP00 commits from CONTRACTS.md, and tests
against `thermosim.testing.synthetic` or its own stubs. None imports another
wave-1 package's implementation, except that WP03 calls WP02's *functions*;
until WP02 lands, WP03's tests use the constant-property stub shipped by WP00.

## 4. Conventions for every agent

- **Python 3.12+.** `from __future__ import annotations` in every module.
  Type hints on all public functions.
- **Tooling:** `uv run pytest`, `uv run ruff check`, `uv run ruff format --check`
  and `uv run mypy src` must all pass. WP00 sets these up.
- **Docstrings:** give the units of every parameter and return value, the valid
  range, and the literature source for any correlation (author, year,
  equation).
- **Vectorized:** physics functions accept NumPy arrays or scalars and
  broadcast.
- **No global mutable state.** Numba functions use `cache=True`, and
  `parallel=True` only where it is measured to help.
- **Tests** go in `tests/<area>/`, named `test_*.py`. Every public function has
  at least one test. Physics tests compare against tabulated or analytic
  values and state the tolerance.
- **Cross-platform:** use `pathlib`. Don't assume a shell. CI runs on
  `windows-latest`, `macos-latest` and `ubuntu-latest`. GUI tests use
  `pytest-qt` with `QT_QPA_PLATFORM=offscreen`.
- **Dependencies:** only those listed in CONTRACTS §1. Anything new goes
  through the coordinator.
- **Commits:** one work package per branch and per PR, named
  `wp/NN-short-name`. The PR description lists what was done, anything not
  done, and any contract concerns.

## 5. Definition of done (every WP)

1. Everything in the work package's *Acceptance criteria* is met, with tests.
2. CI is green on all three operating systems.
3. No edits outside the work package's *Owns* paths.
4. Public APIs match CONTRACTS.md exactly (names, signatures, units).
5. A short `README` section or module docstring explains the design choices and
   known limitations.

## 6. Changing the contracts

The contracts in CONTRACTS.md are frozen once WP00 merges. If an agent finds
one wrong or insufficient:

1. **Don't** work around it silently, and don't edit the shared files.
2. Write the problem and the proposed change in the PR description under a
   `## Contract change request` heading, then stop or finish the parts that
   aren't affected.
3. The coordinator updates CONTRACTS.md and `core/`, and tells the affected
   work packages.

## 7. Coordinator checklist

- [ ] Merge WP00 and check that CONTRACTS.md matches `src/thermosim/core/`.
- [ ] Launch wave 1 (WP01–WP08, plus WP10 part A) in parallel, each in its own
      worktree or branch.
- [ ] Review contract change requests as they arrive.
- [ ] When WP01–WP07 have merged, launch WP09 and WP10 part B.
- [ ] When WP08 and WP09 have merged, launch WP11.
- [ ] Phase 1 sign-off: the user runs the GUI on Windows and checks a real
      heat-up if possible.
- [ ] Before Phase 2: refine PHASE2.md using the WP20 results (target GPU:
      GTX 1060).
