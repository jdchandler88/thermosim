"""Cost of one coefficient refresh (build_coefficients + stable_dt) at the examples' sizes.

    uv run python benchmarks/bench_coefficients.py [--cell-sizes 0.01,0.005]

The propane kettle, 50 L, advanced to t = 120 s so that the base is boiling, then the refresh is
repeated 20 times. Also prints the shell and liquid sizes.
"""

from __future__ import annotations

import argparse
import statistics
import time

from machine import describe

from thermosim.config.io import example_scenarios
from thermosim.sim.coefficients import build_coefficients
from thermosim.sim.model import Model
from thermosim.sources.commands import ScheduleCommand


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-sizes", default="0.01,0.005")
    args = ap.parse_args()
    print(describe())
    print()
    head = ("dx (mm)", "liquid cells", "shell nodes", "refresh ms", "stable_dt ms")
    print("".join(f"{h:>14}" for h in head))
    for dx in (float(s) for s in args.cell_sizes.split(",")):
        sc = example_scenarios()["kettle_propane_40L"].model_copy(deep=True)
        sc.simulation.cell_size_m = dx
        sc.liquid.volume_m3 = 0.05
        m = Model(sc, ScheduleCommand(sc))
        m.advance(120.0)
        ref, sdt = [], []
        for _ in range(20):
            t0 = time.perf_counter()
            build_coefficients(m.domain, m.state, sc, m._last_sp, m._cache)
            t1 = time.perf_counter()
            m.solver.stable_dt(m._coeffs)
            t2 = time.perf_counter()
            ref.append(t1 - t0)
            sdt.append(t2 - t1)
        row = (
            f"{dx * 1e3:.0f}",
            f"{m.solver.net.n_liq}",
            f"{m.domain.shell.n}",
            f"{statistics.median(ref) * 1e3:.1f}",
            f"{statistics.median(sdt) * 1e3:.1f}",
        )
        print("".join(f"{c:>14}" for c in row))


if __name__ == "__main__":
    main()
