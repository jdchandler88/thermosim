"""Timing of the flow solver (not a test; run it by hand).

    uv run python tests/flow/perf_flow.py [--threads 1 4] [--seconds 60] [--cases box32 box64 ...]

Each case heats the base over a disk (the 40 L kettle: 6.2 kW into the liquid, as in
``scenarios/kettle_propane_40L.toml``; the boxes: 2 kW) and advances the flow from rest with
the largest stable time step (``0.9 * max_dt``, capped at 1 s). The temperature is updated with
the advective source and the heating only (conduction is negligible for timing). Reports the
wall time per ``FlowSolver.step`` call, the CG iterations per step and the mean time step, then
the wall time a 45-minute heat-up would need if the time step stayed at the measured mean.
"""

from __future__ import annotations

import argparse
import time

import numba
import numpy as np
from flow_harness import box_grid

from thermosim.core.types import LiquidGrid
from thermosim.flow import FlowProps, FlowSolver
from thermosim.geometry.mesh import build_grid

KETTLE_R = 0.18
KETTLE_H = 0.393  # 40 L


def make_case(name: str) -> tuple[LiquidGrid, float]:
    """Grid and heating power (W)."""
    if name == "box32":
        return box_grid(32, 32, 32, 0.01), 2000.0
    if name == "box64":
        return box_grid(64, 64, 64, 0.01), 2000.0
    if name == "kettle20":
        return build_grid(KETTLE_R, KETTLE_H, 0.020), 6200.0
    if name == "kettle10":
        return build_grid(KETTLE_R, KETTLE_H, 0.010), 6200.0
    raise SystemExit(f"unknown case {name}")


def run(name: str, threads: int, seconds: float, warm: float = 20.0) -> dict[str, float]:
    numba.set_num_threads(threads)
    grid, power = make_case(name)
    fs = FlowSolver(grid, threads=threads)
    st = fs.initial_state()
    props = FlowProps(rho=985.0, beta=4.5e-4, nu=5.0e-7, cp=4180.0, T_ref=None)
    xc, yc, _ = grid.cell_centers()
    disk = (xc[:, None] ** 2 + yc[None, :] ** 2) < 0.1**2
    base = (disk & grid.is_liquid[:, :, 0]).astype(float)
    vol = np.where(grid.volume > 0, grid.volume, 1.0)
    q_heat = np.zeros(grid.shape)
    q_heat[:, :, 0] = base * power / base.sum()
    T = np.where(grid.is_liquid, 300.0, 0.0)
    rcp_v = props.rho * props.cp * vol
    t = 0.0
    wall = 0.0
    n = 0
    iters = 0
    subs = 0
    t_meas = None
    while t < warm + seconds:
        dt = min(0.9 * fs.max_dt(st, props.nu), 1.0)
        t0 = time.perf_counter()
        res = fs.step(st, T, props, dt)
        el = time.perf_counter() - t0
        T = T + dt * (res.q_adv + q_heat) / rcp_v
        t += dt
        if t >= warm:
            if t_meas is None:
                t_meas = t - dt
            wall += el
            n += 1
            iters += res.poisson_iterations
            subs += res.substeps
    sim = t - t_meas  # type: ignore[operator]
    out = {
        "cells": float(grid.is_liquid.sum()),
        "ms_per_step": 1e3 * wall / n,
        "iters_per_step": iters / n,
        "mean_dt": sim / n,
        "substeps": subs / n,
        "umax": fs.max_speed(st),
        "wall_per_sim_s": wall / sim,
    }
    out["wall_45min_min"] = out["wall_per_sim_s"] * 2700.0 / 60.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    pool = int(numba.config.NUMBA_NUM_THREADS)  # type: ignore[attr-defined]
    ap.add_argument("--threads", type=int, nargs="+", default=[1, pool])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--cases", nargs="+", default=["box32", "box64", "kettle20", "kettle10"])
    a = ap.parse_args()
    print(f"{'case':9s} {'thr':>3s} {'cells':>8s} {'ms/step':>8s} {'CG it':>6s} {'dt s':>7s} "
          f"{'umax':>6s} {'wall/sim-s':>10s} {'45 min -> min':>13s}")  # fmt: skip
    for case in a.cases:
        for thr in a.threads:
            r = run(case, thr, a.seconds)
            print(
                f"{case:9s} {thr:3d} {r['cells']:8.0f} {r['ms_per_step']:8.2f} "
                f"{r['iters_per_step']:6.1f} {r['mean_dt']:7.4f} {r['umax']:6.3f} "
                f"{r['wall_per_sim_s']:10.4f} {r['wall_45min_min']:13.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
