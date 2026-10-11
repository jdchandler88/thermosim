"""Command-line entry point: ``thermosim run | validate | examples | compare | gui``.

Exit codes: 0 success; 1 target not reached; 2 invalid scenario (or bad arguments);
3 solver error.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from thermosim.config.io import example_scenarios, load_scenario
from thermosim.config.schema import Scenario
from thermosim.config.units import UnitSystem, to_display, unit_label
from thermosim.core.errors import GeometryError, ScenarioError, SolverError, ThermoSimError
from thermosim.core.protocols import ProgressEvent
from thermosim.core.types import RunResult, StopReason, Summary

EXIT_OK = 0
EXIT_NOT_REACHED = 1
EXIT_INVALID = 2
EXIT_SOLVER = 3

_J_PER_BTU = 1055.05585
_J_PER_MJ = 1.0e6
_KG_PER_LB = 0.45359237


# ---------------------------------------------------------------------------- helpers
def _with_overrides(
    scenario: Scenario, cell_size: float | None, max_time: float | None
) -> Scenario:
    """Re-validate the scenario with command-line overrides of the simulation settings."""
    if cell_size is None and max_time is None:
        return scenario
    data = scenario.model_dump(mode="python")
    if cell_size is not None:
        data["simulation"]["cell_size_m"] = cell_size
    if max_time is not None:
        data["simulation"]["max_sim_time_s"] = max_time
    try:
        return Scenario.model_validate(data)
    except ValidationError as exc:
        raise ScenarioError(str(exc)) from exc


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "run"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "not reached"
    minutes, sec = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{sec:02d} ({seconds:.0f} s)"


def _temp(value_K: float, system: UnitSystem) -> str:
    value = float(to_display(value_K, "temperature", system))
    return f"{value:.1f} {unit_label('temperature', system)}"


def _energy(value_J: float, system: UnitSystem) -> str:
    if system == "us":
        return f"{value_J / _J_PER_BTU:,.0f} BTU"
    return f"{value_J / _J_PER_MJ:.3f} MJ"


def format_summary(summary: Summary, system: UnitSystem) -> str:
    """Plain-text summary table."""
    e = summary.energy
    rows: list[tuple[str, str]] = [
        ("Stop reason", summary.stop_reason.value),
        ("Simulated time", _fmt_duration(summary.sim_time_s)),
        ("Time to target", _fmt_duration(summary.time_to_target_s)),
        (
            "Target",
            f"{_temp(summary.target_temp_K, system)} at '{summary.target_probe}'",
        ),
        ("Final mean temperature", _temp(summary.final_T_mean, system)),
    ]
    rows += [(f"Final probe '{k}'", _temp(v, system)) for k, v in summary.final_probes.items()]
    fuel = (
        f"{summary.fuel_kg / _KG_PER_LB:.2f} lb propane"
        if system == "us"
        else f"{summary.fuel_kg:.3f} kg propane"
    )
    rows += [
        ("Energy delivered", _energy(e.total_in(), system)),
        ("Stored in liquid", _energy(e.stored_liquid, system)),
        ("Stored in vessel", _energy(e.stored_shell + e.stored_lid_headspace, system)),
        ("Loss: vessel outer surface", _energy(e.loss_shell_outer, system)),
        ("Loss: surface conv+rad", _energy(e.loss_surface_convrad, system)),
        ("Loss: evaporation", _energy(e.loss_surface_evap, system)),
        ("Loss: lid", _energy(e.loss_lid_outer, system)),
        ("Loss: dry wall (open)", _energy(e.loss_dry_wall_open, system)),
        ("Energy balance error", f"{summary.energy_balance_rel_error:.2e} (relative)"),
        ("Heating efficiency", f"{100.0 * summary.heating_efficiency:.1f} %"),
    ]
    if summary.fuel_kg > 0:
        rows.append(("Fuel used", fuel))
    if summary.electric_kwh > 0:
        rows.append(("Electricity used", f"{summary.electric_kwh:.3f} kWh"))
    rows.append(("Wall time", f"{summary.wall_time_s:.1f} s"))
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"{k:<{width}}  {v}" for k, v in rows)


class _ProgressBar:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.last = 0.0

    def __call__(self, ev: ProgressEvent) -> None:
        frac = min(1.0, ev.sim_time_s / ev.max_time_s) if ev.max_time_s > 0 else 0.0
        filled = round(30 * frac)
        self.stream.write(
            f"\r[{'#' * filled}{'.' * (30 - filled)}] {100 * frac:5.1f}%  "
            f"sim {ev.sim_time_s / 60:7.1f} min  wall {ev.wall_time_s:6.1f} s"
        )
        self.stream.flush()

    def done(self) -> None:
        self.stream.write("\n")
        self.stream.flush()


# ---------------------------------------------------------------------------- commands
def _cmd_run(args: argparse.Namespace) -> int:
    from thermosim.recording import save_run
    from thermosim.sim import run_headless

    scenario = _with_overrides(load_scenario(args.scenario), args.cell_size, args.max_time)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else Path("runs") / f"{_slug(scenario.name)}-{stamp}"
    bar = None if args.quiet else _ProgressBar(sys.stderr)
    try:
        result: RunResult = run_headless(scenario, bar)
    finally:
        if bar is not None:
            bar.done()
    save_run(result, out_dir)
    print(format_summary(result.summary, args.units))
    print(f"\nResults written to {out_dir}")
    return EXIT_OK if result.summary.stop_reason == StopReason.TARGET_REACHED else EXIT_NOT_REACHED


def _cmd_validate(args: argparse.Namespace) -> int:
    from thermosim.sim.model import Model
    from thermosim.sources.commands import ScheduleCommand

    scenario = load_scenario(args.scenario)
    t0 = time.perf_counter()
    model = Model(scenario, ScheduleCommand(scenario))
    build_s = time.perf_counter() - t0
    dom = model.domain
    n_liq = int(dom.grid.is_liquid.sum())
    n_total = int(dom.grid.is_liquid.size)
    # 4 float64 coefficient/state arrays on the grid + solver scratch (about 10 more).
    mem_mb = (14 * 8 * n_total + 8 * 20 * dom.shell.n) / 1e6
    model.advance(0.0)
    t1 = time.perf_counter()
    t_start = model.state.t
    model.advance(scenario.simulation.max_sim_time_s, max_steps=50)
    wall = max(time.perf_counter() - t1, 1e-9)
    sim_per_wall = (model.state.t - t_start) / wall
    est = scenario.simulation.max_sim_time_s / sim_per_wall if sim_per_wall > 0 else float("inf")
    print(f"Scenario '{scenario.name}' is valid.")
    cell_mm = scenario.simulation.cell_size_m * 1000
    print(f"  Liquid cells:       {n_liq:,} (cell size {cell_mm:.1f} mm)")
    print(f"  Shell nodes:        {dom.shell.n:,}")
    print(f"  Estimated memory:   {mem_mb:.0f} MB")
    print(f"  Build time:         {build_s:.1f} s")
    print(f"  Speed (50 steps):   {sim_per_wall:,.1f} simulated s per wall s (early in the run)")
    print(
        f"  Estimated run time: up to {est:.0f} s if run to the maximum simulated time "
        f"({scenario.simulation.max_sim_time_s / 3600:.1f} h); usually stops earlier"
    )
    return EXIT_OK


def _cmd_examples(args: argparse.Namespace) -> int:
    folder = files("thermosim") / "scenarios"
    if args.write is None:
        for name, sc in example_scenarios().items():
            print(f"{name:<28} {sc.name}")
        return EXIT_OK
    out = Path(args.write)
    out.mkdir(parents=True, exist_ok=True)
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".toml"):
            (out / entry.name).write_text(entry.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"wrote {out / entry.name}")
    return EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    from thermosim.config.io import scenario_from_toml
    from thermosim.recording import compare, load_measured_csv, load_run

    run = load_run(Path(args.run_dir))
    scenario = scenario_from_toml(run.scenario_toml)
    probe = args.probe or scenario.target.probe
    measured = load_measured_csv(
        Path(args.measured), temp_unit=args.temp_unit, time_unit=args.time_unit
    )
    c = compare(run.samples, probe, measured, scenario.target.temp_K)
    print(f"Probe: {probe}   Measured: {measured.label}")
    print(f"  RMSE:               {c.rmse_K:.2f} K ({c.rmse_K * 1.8:.2f} F)")
    print(f"  Max abs error:      {c.max_abs_K:.2f} K ({c.max_abs_K * 1.8:.2f} F)")
    print(f"  Bias (sim - meas):  {c.bias_K:+.2f} K ({c.bias_K * 1.8:+.2f} F)")
    print(f"  Time to target sim: {_fmt_duration(c.time_to_target_sim_s)}")
    print(f"  Time to target meas:{_fmt_duration(c.time_to_target_meas_s)}")
    if c.time_to_target_error_rel is not None:
        print(f"  Time to target err: {100 * c.time_to_target_error_rel:+.1f} %")
    return EXIT_OK


def _cmd_gui(args: argparse.Namespace) -> int:
    from thermosim.gui.app import main as gui_main

    return int(gui_main([str(args.scenario)] if args.scenario else []))


# ---------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermosim",
        description="ThermoSim: thermal simulation of a heated vessel.",
        epilog="Exit codes: 0 ok, 1 target not reached, 2 invalid scenario, 3 solver error.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a scenario headless and print the summary")
    run.add_argument("scenario", type=Path, help="scenario TOML file")
    run.add_argument("--out", type=Path, help="output directory (default runs/<name>-<timestamp>)")
    run.add_argument("--cell-size", type=float, metavar="M", help="override cell size, metres")
    run.add_argument("--max-time", type=float, metavar="S", help="override max simulated time, s")
    run.add_argument("--units", choices=["us", "metric"], default="us", help="output units")
    run.add_argument("--quiet", action="store_true", help="no progress bar")
    run.set_defaults(func=_cmd_run)

    val = sub.add_parser("validate", help="check a scenario and estimate memory and run time")
    val.add_argument("scenario", type=Path, help="scenario TOML file")
    val.set_defaults(func=_cmd_validate)

    ex = sub.add_parser("examples", help="list the example scenarios, or write them out")
    ex.add_argument("--write", type=Path, metavar="DIR", help="write the example TOML files here")
    ex.set_defaults(func=_cmd_examples)

    cmp_ = sub.add_parser("compare", help="compare a saved run with measured data")
    cmp_.add_argument("run_dir", type=Path, help="directory written by 'thermosim run'")
    cmp_.add_argument("measured", type=Path, help="measured CSV file")
    cmp_.add_argument("--probe", help="probe name or 'mean' (default: the scenario's target)")
    cmp_.add_argument("--temp-unit", choices=["C", "F"], default="C")
    cmp_.add_argument("--time-unit", choices=["s", "min"], default="s")
    cmp_.set_defaults(func=_cmd_compare)

    gui = sub.add_parser("gui", help="launch the graphical interface")
    gui.add_argument("scenario", nargs="?", type=Path, help="scenario TOML file to open")
    gui.set_defaults(func=_cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the requested action. Returns the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_OK
    try:
        return int(args.func(args))
    except (ScenarioError, GeometryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except SolverError as exc:
        print(f"solver error: {exc}", file=sys.stderr)
        return EXIT_SOLVER
    except ThermoSimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    sys.exit(main())
