# WP00: Project skeleton, contracts, CI, synthetic fixtures

**Wave:** 0 (runs alone; blocks everything else) · **Size:** S

## Goal

Create the Python project so that wave-1 agents can start straight away:
packaging, tooling, CI, the frozen shared types (`core/`), empty module stubs
for every package, and a synthetic test domain.

## Owns

- `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore` (Python entries
  appended; keep the existing Java entries)
- `.github/workflows/ci.yml`
- `src/thermosim/__init__.py`, `src/thermosim/core/**`,
  `src/thermosim/testing/**`
- **Stub files only** (each must contain the signatures from CONTRACTS.md,
  with bodies `raise NotImplementedError`) for:
  - `config/{schema,io,units}.py`
  - `physics/{properties,correlations}.py`
  - `geometry/build.py`
  - `sources/build.py`
  - `solver/thermal.py`
  - `recording/__init__.py`
  - `sim/{model,runner}.py`
  - `cli.py`

  The owning work packages replace these stubs.
- `tests/core/**`, `tests/testing/**`, `tests/conftest.py`

**Do not touch** `ThermoSim/` (the legacy Java code).

## Tasks

1. **`pyproject.toml`**
   - Build backend `hatchling`, `src/` layout, project name `thermosim`, and
     the dependencies in CONTRACTS §1.
   - Console scripts: `thermosim = "thermosim.cli:main"`.
   - Package data: `thermosim/scenarios/*.toml`.
   - Settings for ruff (line length 100; rules E, F, I, UP, B, NPY, RUF), mypy
     (`ignore_missing_imports = true`, `check_untyped_defs = true`) and pytest
     (`testpaths = ["tests"]`).
   - Commit `uv.lock`.
2. **`core/`**: transcribe CONTRACTS §2–§4 *exactly*. Implement the
   `EnergyFlows` methods and `LiquidGrid.cell_centers`.
3. **`testing/synthetic.py`** provides:
   - `box_domain(nx, ny, nz, dx, *, shell=True, wall_thickness=0.002, wall=SolidProps-like, lid=False, fill_fraction_top=1.0) -> Domain`.
     The liquid is a rectangular box with every cell liquid.
     - If `shell` is set: one shell node per boundary face on the bottom
       (kind BASE, outer DOWN) and on the four sides (kind SIDE, outer SIDE).
       Each node gets `area = dx²`, and in-plane edges join neighboring nodes
       on the same face and across corners.
     - One probe called `"center"` sits at the box center.
     - The slice is at `j = ny // 2`.
     - Coordinates follow D2 (the origin is at the bottom center).
   - `uniform_coefficients(domain, *, k=0.6, rho_c=4.18e6, h_film=0.0, h_out=0.0, h_surface=0.0, T_amb=293.15) -> StepCoefficients`
     builds full-shape arrays, with every source zero and every loss
     coefficient set to the given values.
   - `ConstantProps`: a stub with the same function names as
     `physics.properties`, returning water at 20 °C constants. WP03 uses it
     until WP02 lands.
4. **Stubs** for every module listed above, with exact signatures and
   docstrings copied from CONTRACTS.md.
5. **CI** (`.github/workflows/ci.yml`):
   - Matrix over {ubuntu, windows, macos}-latest, Python 3.12.
   - Install with `astral-sh/setup-uv`.
   - Run `uv sync --all-extras`, `uv run ruff check`,
     `uv run ruff format --check`, `uv run mypy src` and `uv run pytest -q`.
   - Set `QT_QPA_PLATFORM=offscreen`. On Ubuntu, install the Qt runtime
     libraries (`libegl1 libxkbcommon0 libdbus-1-3` and friends) so PySide6
     imports.
   - Tests that hit `NotImplementedError` stubs must not fail CI: mark
     placeholder tests `xfail(strict=False)`, or add no tests for stubs.
6. **Tests:**
   - `EnergyFlows` arithmetic.
   - `box_domain` invariants: the volume sums correctly; `couple_area` per
     node is ≤ its area; the edge list has a < b and no duplicates; surface
     area sums to nx·ny·dx²; probe weights sum to 1.
   - `import thermosim.*` succeeds for every module.

## Acceptance criteria

- `uv run pytest`, ruff and mypy pass on all three operating systems in CI.
- `src/thermosim/core/` matches CONTRACTS.md §2–§4 (the reviewer diffs them).
- `box_domain(10, 8, 6, 0.01)` returns a valid `Domain` whose invariants are
  tested.
- `thermosim --help` runs (the stub CLI prints usage).

## Out of scope

Any physics, geometry for cylinders, or GUI.
