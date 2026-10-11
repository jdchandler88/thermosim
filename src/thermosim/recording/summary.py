"""Run summary: time-to-target, energy totals, efficiency and balance error."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from thermosim.core.types import F64, EnergyFlows, Sample, StopReason, Summary


def sample_times(samples: list[Sample]) -> F64:
    return np.array([s.t for s in samples], dtype=np.float64)


def probe_series(samples: list[Sample], name: str) -> F64:
    """Temperature (K) of a probe over the samples; "mean" selects the volume-weighted mean."""
    if name == "mean":
        return np.array([s.T_mean for s in samples], dtype=np.float64)
    try:
        return np.array([s.probes[name] for s in samples], dtype=np.float64)
    except KeyError as exc:
        available = sorted(samples[0].probes) if samples else []
        raise ValueError(f"unknown probe {name!r}; available: {available} or 'mean'") from exc


def first_crossing_time(t: F64, v: F64, target: float) -> float | None:
    """First time v >= target, linearly interpolated between samples; None if never reached.

    Non-finite points are ignored.
    """
    keep = np.isfinite(t) & np.isfinite(v)
    t = t[keep]
    v = v[keep]
    hits = np.flatnonzero(v >= target)
    if hits.size == 0:
        return None
    i = int(hits[0])
    if i == 0:
        return float(t[0])
    t0, t1 = float(t[i - 1]), float(t[i])
    v0, v1 = float(v[i - 1]), float(v[i])
    # v0 < target <= v1 here, so the denominator is positive.
    return t0 + (target - v0) * (t1 - t0) / (v1 - v0)


def _gross_energy_J(samples: list[Sample]) -> float:
    """Trapezoid integral of power_gross over time (J)."""
    if len(samples) < 2:
        return 0.0
    t = sample_times(samples)
    p = np.array([s.power_gross for s in samples], dtype=np.float64)
    return float(np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(t)))


def summarize(
    samples: list[Sample],
    target_temp_K: float,
    target_probe: str,
    stop_reason: StopReason,
    wall_time_s: float,
) -> Summary:
    """Build the run summary from the time series."""
    if not samples:
        return Summary(
            stop_reason=stop_reason,
            sim_time_s=0.0,
            wall_time_s=wall_time_s,
            time_to_target_s=None,
            target_temp_K=target_temp_K,
            target_probe=target_probe,
            final_probes={},
            final_T_mean=math.nan,
            energy=EnergyFlows(),
            energy_balance_rel_error=0.0,
            fuel_kg=0.0,
            electric_kwh=0.0,
            heating_efficiency=0.0,
        )

    last = samples[-1]
    energy = replace(last.cumulative)
    time_to_target = first_crossing_time(
        sample_times(samples), probe_series(samples, target_probe), target_temp_K
    )
    total_in = energy.total_in()
    balance_rel = abs(energy.balance_error()) / max(total_in, 1.0)
    gross = _gross_energy_J(samples)
    efficiency = energy.stored_liquid / gross if gross > 0.0 else 0.0

    return Summary(
        stop_reason=stop_reason,
        sim_time_s=last.t,
        wall_time_s=wall_time_s,
        time_to_target_s=time_to_target,
        target_temp_K=target_temp_K,
        target_probe=target_probe,
        final_probes=dict(last.probes),
        final_T_mean=last.T_mean,
        energy=energy,
        energy_balance_rel_error=balance_rel,
        fuel_kg=last.fuel_kg,
        electric_kwh=last.electric_kwh,
        heating_efficiency=efficiency,
    )
