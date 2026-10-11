"""Cross-checks of the full 3-D model: against the lumped model, and under grid refinement.

Both are used by the automated validation (``tests/validation``) and by
``benchmarks/validation_report.py``, which prints the tables quoted in ``docs/validation.md``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from thermosim.config.schema import Scenario
from thermosim.sim.runner import run_headless
from thermosim.validation.lumped import LumpedModel


@dataclass(frozen=True)
class LumpedComparison:
    """3-D versus lumped time to target for one scenario."""

    cell_size_m: float
    k_eff_scale: float
    time_3d_s: float | None
    time_lumped_s: float | None
    energy_balance_rel_error: float
    wall_s: float

    @property
    def rel_error(self) -> float:
        """(3-D - lumped) / lumped; NaN if either run did not reach the target."""
        if self.time_3d_s is None or self.time_lumped_s is None:
            return math.nan
        return self.time_3d_s / self.time_lumped_s - 1.0


def compare_with_lumped(
    scenario: Scenario, *, k_eff_scale: float, cell_size_m: float | None = None
) -> LumpedComparison:
    """Run the 3-D model and the lumped model on the same scenario.

    The lumped model heats the liquid mean, so the 3-D run's target probe is set to ``mean``.
    ``k_eff_scale`` is applied to both (the lumped model has no use for it: its liquid is
    perfectly mixed); a large value (100) makes the 3-D liquid well mixed.
    """
    sc = scenario.model_copy(deep=True)
    sc.target.probe = "mean"
    sc.model.k_eff_scale = k_eff_scale
    if cell_size_m is not None:
        sc.simulation.cell_size_m = cell_size_m
    lumped = LumpedModel(sc).run()
    t0 = time.perf_counter()
    res = run_headless(sc)
    wall = time.perf_counter() - t0
    s = res.summary
    return LumpedComparison(
        cell_size_m=float(sc.simulation.cell_size_m),
        k_eff_scale=k_eff_scale,
        time_3d_s=s.time_to_target_s,
        time_lumped_s=lumped.time_to_target_s,
        energy_balance_rel_error=float(s.energy_balance_rel_error),
        wall_s=wall,
    )


@dataclass(frozen=True)
class Convergence:
    """Three-level grid convergence of one scalar (cell size halved at each level).

    ``values`` are ordered coarse, medium, fine.
    """

    values: tuple[float, float, float]
    monotone: bool
    order: float  # observed order of convergence (NaN if not monotone)
    extrapolated: float  # Richardson-extrapolated value (the fine value if not monotone)

    @property
    def fine_error(self) -> float:
        """|fine - extrapolated| / |extrapolated|."""
        return abs(self.values[2] - self.extrapolated) / abs(self.extrapolated)


def richardson(values: Sequence[float], ratio: float = 2.0) -> Convergence:
    """Observed order and Richardson extrapolation from three levels with refinement ratio r.

    With f1 (coarse), f2 (medium), f3 (fine): p = ln((f1 - f2)/(f2 - f3)) / ln r and
    f* = f3 + (f3 - f2) / (r^p - 1)   (Roache, *Verification and Validation in Computational
    Science and Engineering*, 1998, Eqs. 5.6.x). Defined only when the sequence converges
    monotonically (the two differences have the same sign and the second is smaller).
    """
    f1, f2, f3 = (float(v) for v in values)
    d1, d2 = f1 - f2, f2 - f3
    mono = d1 * d2 > 0.0 and abs(d2) < abs(d1)
    if not mono:
        return Convergence((f1, f2, f3), False, math.nan, f3)
    p = math.log(d1 / d2) / math.log(ratio)
    return Convergence((f1, f2, f3), True, p, f3 - d2 / (ratio**p - 1.0))
