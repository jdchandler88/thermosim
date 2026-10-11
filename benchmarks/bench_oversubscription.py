"""Solver speed while other processes compete for the CPU (the WP10 "known issue").

Usage::

    uv run python benchmarks/bench_oversubscription.py            # compare configurations
    uv run python benchmarks/bench_oversubscription.py --worker   # one measurement, JSON out

The default mode first measures every configuration on an idle machine, then starts
``--burners`` busy-loop processes (plain ``python -c`` children, killed at the end) and
measures again. Every configuration runs in a fresh interpreter, because the Numba threading
layer and thread count are fixed at the first parallel call. The table shows milliseconds per
solver step; a configuration that takes longer than ``--timeout`` seconds is reported as
``timeout``.

Configurations: ``default`` (what a user gets), ``threads=N`` (``NUMBA_NUM_THREADS``),
``layer=NAME`` (``NUMBA_THREADING_LAYER``), ``env=NAME=VALUE`` (any environment variable).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

BURNER = "while True:\n    pass\n"


def worker(cell_size: float, sim_s: float, volume_m3: float) -> None:
    """Advance the propane-kettle example by ``sim_s`` and print the timing as JSON."""
    import numba

    from thermosim.config.io import example_scenarios
    from thermosim.sim.model import Model
    from thermosim.sources.commands import ScheduleCommand

    sc = example_scenarios()["kettle_propane_40L"].model_copy(deep=True)
    sc.simulation.cell_size_m = cell_size
    sc.liquid.volume_m3 = volume_m3
    model = Model(sc, ScheduleCommand(sc))
    model.advance(2.0)  # warm up (first refresh, caches)
    steps0, t0 = model.step_count, time.perf_counter()
    model.advance(2.0 + sim_s)
    wall = time.perf_counter() - t0
    steps = model.step_count - steps0
    print(
        json.dumps(
            {
                "wall_s": wall,
                "steps": steps,
                "ms_per_step": 1e3 * wall / max(steps, 1),
                "cells": int(model.solver.net.n_liq),
                "numba_threads": numba.get_num_threads(),
                "layer": numba.threading_layer(),
            }
        )
    )


def run_worker(args: argparse.Namespace, env_extra: dict[str, str]) -> dict[str, object] | None:
    env = dict(os.environ, **env_extra)
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--cell-size",
        str(args.cell_size),
        "--sim-s",
        str(args.sim_s),
        "--volume",
        str(args.volume),
    ]
    try:
        out = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=args.timeout, check=True
        )
    except subprocess.TimeoutExpired:
        return None
    return json.loads(out.stdout.strip().splitlines()[-1])  # type: ignore[no-any-return]


def config_env(cfg: str) -> dict[str, str]:
    if cfg.startswith("threads="):
        return {"NUMBA_NUM_THREADS": cfg.split("=", 1)[1]}
    if cfg.startswith("layer="):
        return {"NUMBA_THREADING_LAYER": cfg.split("=", 1)[1]}
    if cfg.startswith("env="):
        _, name, value = cfg.split("=", 2)
        return {name: value}
    return {}


def run_command(cmd: str, n_burn: int, timeout: float, idle: bool = True) -> None:
    """Time ``cmd`` on an idle machine and next to ``n_burn`` busy-loop processes."""
    print(f"cpus={os.cpu_count()}  command: {cmd}")
    for burners in (0, n_burn) if idle else (n_burn,):
        procs = [subprocess.Popen([sys.executable, "-c", BURNER]) for _ in range(burners)]
        try:
            time.sleep(1.0)
            t0 = time.perf_counter()
            try:
                done = subprocess.run(
                    shlex.split(cmd), capture_output=True, text=True, timeout=timeout
                )
                status = f"exit {done.returncode}"
                if done.returncode != 0:
                    status += "\n" + "\n".join(done.stdout.splitlines()[-25:])
            except subprocess.TimeoutExpired:
                status = f"TIMEOUT after {timeout:.0f} s"
            print(f"burners={burners:>3}: {time.perf_counter() - t0:7.1f} s  {status}", flush=True)
        finally:
            for p in procs:
                p.kill()
            for p in procs:
                p.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--burners", type=int, default=None, help="default: number of CPUs")
    ap.add_argument("--cell-size", type=float, default=0.005)
    ap.add_argument("--sim-s", type=float, default=10.0)
    ap.add_argument("--volume", type=float, default=0.05)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument(
        "--configs",
        default="default,threads=1,threads=2",
        help="comma list of configurations: default, threads=N, layer=NAME, env=NAME=VALUE",
    )
    ap.add_argument("--no-idle", action="store_true", help="with --cmd: skip the idle run")
    ap.add_argument(
        "--cmd",
        default=None,
        help="instead of the table: run this command (e.g. a pytest invocation) once idle and "
        "once with the burners, and report the durations",
    )
    args = ap.parse_args()
    if args.worker:
        worker(args.cell_size, args.sim_s, args.volume)
        return

    n_burn = args.burners if args.burners is not None else (os.cpu_count() or 1)
    if args.cmd:
        run_command(args.cmd, n_burn, args.timeout, idle=not args.no_idle)
        return
    print(f"cpus={os.cpu_count()}  load before: {os.getloadavg()}")
    print(f"{'config':<28}{'burners':>8}{'ms/step':>12}{'threads':>9}{'layer':>11}")
    for burners in (0, n_burn):
        procs = [subprocess.Popen([sys.executable, "-c", BURNER]) for _ in range(burners)]
        try:
            time.sleep(1.0)
            for cfg in args.configs.split(","):
                res = run_worker(args, config_env(cfg))
                if res is None:
                    print(f"{cfg:<28}{burners:>8}{'timeout':>12}")
                else:
                    print(
                        f"{cfg:<28}{burners:>8}{res['ms_per_step']:>12.2f}"
                        f"{res['numba_threads']:>9}{res['layer']!s:>11}",
                        flush=True,
                    )
        finally:
            for p in procs:
                p.kill()
            for p in procs:
                p.wait()


if __name__ == "__main__":
    main()
