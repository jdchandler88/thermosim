"""Solver step time against the number of liquid cells (64^3, 96^3, 128^3).

    uv run python benchmarks/bench_solver_step.py [--sizes 64,96,128] [--threads 1,2,4,auto]

A box of ``n^3`` cells of 5 mm with a convective top and wall losses, uniform conductivity
(the model's case) and a point source. Prints the median milliseconds per step over 30 steps
and the cell updates per second.
"""

from __future__ import annotations

import argparse
import inspect
import statistics
import time

import numba
from machine import describe

from thermosim.solver import ThermalSolver
from thermosim.testing.synthetic import box_domain, uniform_coefficients


def measure(n: int, threads: str, steps: int = 30) -> tuple[float, int, str]:
    domain = box_domain(n, n, n, 0.005)
    kwargs = {}
    if "threads" in inspect.signature(ThermalSolver.__init__).parameters and threads != "auto":
        kwargs["threads"] = int(threads)
    solver = ThermalSolver(domain, **kwargs)
    coeffs = uniform_coefficients(
        domain, k=300.0, h_film=300.0, h_out=10.0, h_surface=10.0, T_amb=293.15
    )
    coeffs.q_cells[n // 2, n // 2, n // 2] = 1000.0
    state = solver.initial_state(300.0, 300.0, 293.15)
    dt = 0.9 * solver.stable_dt(coeffs)
    step_kwargs = (
        {"reuse_conductances": True}
        if "reuse_conductances" in inspect.signature(ThermalSolver.step).parameters
        else {}
    )
    for _ in range(40):  # warm-up; also lets the thread tuner settle
        solver.step(state, coeffs, dt, **step_kwargs)
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        solver.step(state, coeffs, dt, **step_kwargs)
        times.append(time.perf_counter() - t0)
    tuner = getattr(solver, "_tuner", None)
    used = f"{tuner.best}" if tuner is not None else f"{numba.get_num_threads()} (fixed)"
    return statistics.median(times) * 1e3, domain.grid.is_liquid.sum(), used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="64,96,128")
    ap.add_argument("--threads", default="auto")
    args = ap.parse_args()
    print(describe())
    print()
    print(f"{'cells':>10}{'threads':>9}{'used':>10}{'ms/step':>10}{'Mcell/s':>10}")
    for threads in args.threads.split(","):
        for n in (int(s) for s in args.sizes.split(",")):
            ms, cells, used = measure(n, threads)
            print(
                f"{cells:>10}{threads:>9}{used:>10}{ms:>10.2f}{cells / ms / 1e3:>10.0f}", flush=True
            )


if __name__ == "__main__":
    main()
