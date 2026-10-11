"""Print the tables quoted in docs/validation.md (section 6).

    uv run python benchmarks/validation_report.py [--lumped] [--convergence] [--examples]

* ``--lumped``: 3-D versus lumped time to target for every example at k_eff_scale = 100 (well
  mixed; the test criterion is 3 %) and at k_eff_scale = 1 (information only);
* ``--convergence``: the 40 L propane example at dx = 20, 10 and 5 mm (about 5 minutes);
* ``--examples``: energy balance, time to target and rule-of-thumb check of every example at
  20 mm.

With no option everything is printed.
"""

from __future__ import annotations

import argparse

from machine import describe

from thermosim.config.io import example_scenarios
from thermosim.sim.runner import run_headless
from thermosim.validation.crosscheck import compare_with_lumped, richardson

EXAMPLES = example_scenarios()
LUMPED_CELL_M = {
    "small_pot_stove": 0.02,
    "kettle_electric_insulated": 0.02,
    "kettle_propane_40L": 0.04,
}


def lumped() -> None:
    print("\n3-D versus lumped (time to target, liquid-mean target)")
    print("example                     dx mm  k_eff_scale  3-D (s)  lumped (s)  error")
    for name, dx in LUMPED_CELL_M.items():
        for scale, cell in ((100.0, dx), (1.0, 0.01)):
            c = compare_with_lumped(EXAMPLES[name], k_eff_scale=scale, cell_size_m=cell)
            print(
                f"{name:<27}{c.cell_size_m * 1e3:>5.0f}{scale:>13.0f}{c.time_3d_s:>9.0f}"
                f"{c.time_lumped_s:>12.0f}{100 * c.rel_error:>+8.2f} %",
                flush=True,
            )


def examples() -> None:
    print("\nEnergy balance and time to target at dx = 20 mm (default k_eff_scale = 1)")
    print("example                     time to target (min)  energy_balance_rel_error  stop")
    for name, sc in EXAMPLES.items():
        s = sc.model_copy(deep=True)
        s.simulation.cell_size_m = 0.02
        r = run_headless(s).summary
        t = r.time_to_target_s / 60.0 if r.time_to_target_s else float("nan")
        print(f"{name:<27}{t:>20.2f}{r.energy_balance_rel_error:>26.1e}  {r.stop_reason.value}")


def convergence() -> None:
    print("\nGrid convergence, 40 L propane example")
    sc = EXAMPLES["kettle_propane_40L"]
    levels = (0.02, 0.01, 0.005)
    times, centre, wall = [], [], []
    for dx in levels:
        s = sc.model_copy(deep=True)
        s.simulation.cell_size_m = dx
        res = run_headless(s)
        assert res.summary.time_to_target_s is not None
        times.append(res.summary.time_to_target_s)
        smp = next(x for x in res.samples if abs(x.t - 1500.0) < 0.5)
        centre.append(smp.probes["center"])
        wall.append(smp.probes["near_wall"])
        print(f"  dx = {dx * 1e3:.0f} mm done ({res.summary.wall_time_s:.0f} s)", flush=True)
    t0 = sc.liquid.initial_temp_K
    print(f"time to target (s): {', '.join(f'{t:.2f}' for t in times)}")
    c = richardson(times)
    print(f"  monotone {c.monotone}; spread {max(times) - min(times):.2f} s")
    for name, vals in (("center at 1500 s", centre), ("near_wall at 1500 s", wall)):
        c = richardson([v - t0 for v in vals])
        print(
            f"{name}: temperature rise (K) {', '.join(f'{v - t0:.4f}' for v in vals)}; "
            f"monotone {c.monotone}; order {c.order:.2f}; "
            f"extrapolated rise {c.extrapolated:.4f} K; "
            f"5 mm error {100 * c.fine_error:.3f} %"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lumped", action="store_true")
    ap.add_argument("--convergence", action="store_true")
    ap.add_argument("--examples", action="store_true")
    args = ap.parse_args()
    everything = not (args.lumped or args.convergence or args.examples)
    print(describe())
    if args.lumped or everything:
        lumped()
    if args.examples or everything:
        examples()
    if args.convergence or everything:
        convergence()


if __name__ == "__main__":
    main()
