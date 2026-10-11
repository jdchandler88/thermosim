"""Full runs of the example scenarios, and the requirement R2 benchmark.

    uv run python benchmarks/bench_examples.py                      # all examples, 10 mm and 5 mm
    uv run python benchmarks/bench_examples.py --cell-sizes 0.01    # preview size only
    uv run python benchmarks/bench_examples.py --r2                 # 50 L, 5 mm, 45 sim-min

R2 (REQUIREMENTS.md): at the standard cell size (5 mm) a 45-minute simulated heat-up of a
50 L kettle finishes in under 5 minutes on a typical 4-core machine. The R2 case is the propane
kettle with 0.05 m^3 of water and ``max_sim_time_s = 2700``; it runs the full 45 simulated
minutes (the water does not reach 99 C in that time).
"""

from __future__ import annotations

import argparse
import time

from machine import describe

from thermosim.config.io import example_scenarios
from thermosim.core.types import ShellKind
from thermosim.sim.model import Model
from thermosim.sources.commands import ScheduleCommand


def run(name: str, dx: float, *, volume_m3: float | None = None, max_time: float | None = None):  # type: ignore[no-untyped-def]
    sc = example_scenarios()[name].model_copy(deep=True)
    sc.simulation.cell_size_m = dx
    if volume_m3 is not None:
        sc.liquid.volume_m3 = volume_m3
    if max_time is not None:
        sc.simulation.max_sim_time_s = max_time
    t0 = time.perf_counter()
    m = Model(sc, ScheduleCommand(sc))
    build = time.perf_counter() - t0
    base = m.domain.shell.kind == int(ShellKind.BASE)
    base_max = 0.0
    t0 = time.perf_counter()
    while m.stop_reason is None:
        m.advance(m.state.t + 5.0)
        base_max = max(base_max, float(m.state.T_shell[base].max()))
    wall = time.perf_counter() - t0
    res = m.build_result(wall)
    s = res.summary
    near = [x.probes["near_wall"] for x in res.samples if "near_wall" in x.probes]
    return {
        "name": name,
        "dx": dx,
        "cells": int(m.solver.net.n_liq),
        "build_s": build,
        "wall_s": wall,
        "sim_s": m.state.t,
        "steps": m.step_count,
        "refresh_frac": m.refresh_fraction,
        "ttt": s.time_to_target_s,
        "stop": s.stop_reason.value,
        "ebal": s.energy_balance_rel_error,
        "base_max_C": base_max - 273.15,
        "near_max_C": (max(near) - 273.15) if near else float("nan"),
    }


def show(r: dict) -> None:  # type: ignore[type-arg]
    ttt = f"{r['ttt'] / 60:.2f} min" if r["ttt"] else "-"
    print(
        f"{r['name']:<28}{r['dx'] * 1e3:>6.0f}{r['cells']:>9}{r['wall_s']:>9.1f}"
        f"{r['sim_s'] / 60:>9.1f}{r['steps']:>9}{100 * r['refresh_frac']:>7.0f}%"
        f"{ttt:>12}{r['base_max_C']:>9.0f}{r['near_max_C']:>8.0f}  {r['stop']}"
        f"  ebal {r['ebal']:.1e}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-sizes", default="0.01,0.005")
    ap.add_argument("--examples", default=",".join(example_scenarios()))
    ap.add_argument("--r2", action="store_true", help="only the 50 L, 5 mm, 45 min case")
    args = ap.parse_args()
    print(describe())
    print()
    print(
        f"{'example':<28}{'dx mm':>6}{'cells':>9}{'wall s':>9}{'sim min':>9}{'steps':>9}"
        f"{'refr':>8}{'to target':>12}{'baseTmax':>9}{'nearC':>8}"
    )
    if args.r2:
        r = run("kettle_propane_40L", 0.005, volume_m3=0.05, max_time=2700.0)
        show(r)
        verdict = "MET" if r["wall_s"] < 300.0 else "NOT MET"
        print(f"\nR2 (< 300 s): {r['wall_s']:.0f} s -> {verdict}")
        return
    for dx in (float(s) for s in args.cell_sizes.split(",")):
        for name in args.examples.split(","):
            show(run(name, dx))
    print("(baseTmax: hottest base shell node; nearC: hottest 'near_wall' probe reading, C)")


if __name__ == "__main__":
    main()
