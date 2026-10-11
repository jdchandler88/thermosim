"""Acceptance tests of the thermal network solver (WP06)."""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pytest
from scipy.special import erfc

from thermosim.core.errors import SolverError
from thermosim.core.types import Domain, EnergyFlows, LidSpec, StepCoefficients
from thermosim.solver import ThermalSolver
from thermosim.testing.synthetic import box_domain, uniform_coefficients

T_AMB = 293.15


def _random_coefficients(domain: Domain, rng: np.random.Generator) -> StepCoefficients:
    """Coefficients with every feature on: random k, rho_c, films, losses, sources, evaporation."""
    c = uniform_coefficients(domain, T_amb=T_AMB)
    shape = domain.grid.shape
    n = domain.shell.n
    n_c = len(domain.shell.couple_shell)
    n_s = len(domain.grid.surface_cells)
    c.k_liquid = rng.uniform(0.3, 1.0, shape)
    c.rho_c_liquid = rng.uniform(3.0e6, 4.5e6, shape)
    c.h_film = rng.uniform(100.0, 800.0, n_c)
    c.h_out_shell = rng.uniform(5.0, 30.0, n)
    c.h_dry = rng.uniform(2.0, 15.0, n)
    c.h_surface_open = rng.uniform(5.0, 25.0, n_s)
    c.q_evap_open = rng.uniform(0.0, 300.0, n_s)
    c.h_surface_covered = rng.uniform(5.0, 40.0, n_s)
    c.h_lid_inner = 8.0
    c.h_lid_outer = 12.0
    c.q_cells = rng.uniform(0.0, 5.0, shape)
    c.q_shell = rng.uniform(0.0, 2.0, n)
    return c


def _with_lid(domain: Domain, covered_fraction: float) -> Domain:
    area = covered_fraction * math.pi * domain.inner_radius**2
    lid = LidSpec(
        present=True,
        covered_fraction=covered_fraction,
        area=area,
        heat_capacity=8000.0 * 500.0 * area * 1e-3,
        emissivity=0.3,
        insulation_r=0.05,
        headspace_volume=area * 0.01,
    )
    return dataclasses.replace(domain, lid=lid)


# ------------------------------------------------------------------ 1. insulated heating
def test_insulated_uniform_heating_is_exact() -> None:
    domain = box_domain(6, 5, 4, 0.01)
    solver = ThermalSolver(domain)
    coeffs = uniform_coefficients(domain)
    power = 250.0
    coeffs.q_cells[:] = power / coeffs.q_cells.size
    state = solver.initial_state(300.0, 300.0, T_AMB)
    state.T_liquid[:] = 300.0 + np.random.default_rng(1).uniform(0.0, 5.0, state.T_liquid.shape)
    vol = domain.grid.volume
    c_total = float((coeffs.rho_c_liquid * vol).sum())
    mean0 = float((state.T_liquid * coeffs.rho_c_liquid * vol).sum() / c_total)
    dt = 0.5 * solver.stable_dt(coeffs)
    n_steps = 200
    for _ in range(n_steps):
        solver.step(state, coeffs, dt)
    mean = float((state.T_liquid * coeffs.rho_c_liquid * vol).sum() / c_total)
    expected_rise = power * dt * n_steps / c_total
    assert (mean - mean0) == pytest.approx(expected_rise, rel=1e-10)
    assert state.t == pytest.approx(dt * n_steps)


# ------------------------------------------------------------------ 2. 1-D conduction
def test_one_dimensional_conduction_matches_erfc() -> None:
    nz, dx = 200, 0.005
    domain = box_domain(1, 1, nz, dx, shell=False)
    solver = ThermalSolver(domain)
    k, rho_c = 0.6, 4.18e6
    coeffs = uniform_coefficients(domain, k=k, rho_c=rho_c, h_surface=1.0e12, T_amb=400.0)
    state = solver.initial_state(300.0, 300.0, 400.0)
    dt = solver.stable_dt(coeffs)
    t_end = 20000.0
    n = math.ceil(t_end / dt)
    dt = t_end / n
    for _ in range(n):
        solver.step(state, coeffs, dt)
    alpha = k / rho_c
    depth = nz * dx - (np.arange(nz) + 0.5) * dx  # distance below the surface
    exact = 300.0 + 100.0 * erfc(depth / (2.0 * math.sqrt(alpha * t_end)))
    got = state.T_liquid[0, 0, :]
    assert exact[0] < 300.0 + 1e-3  # the front has not reached the bottom
    assert np.max(np.abs(got - exact)) < 0.01 * 100.0


# ------------------------------------------------------------------ 3. Newton cooling
def test_newton_cooling_time_constant() -> None:
    domain = box_domain(4, 4, 4, 0.01)
    solver = ThermalSolver(domain)
    h_film, h_out = 1.0e5, 20.0
    coeffs = uniform_coefficients(domain, k=1.0e3, h_film=h_film, h_out=h_out, T_amb=T_AMB)
    # Only the shell outer side loses heat: no surface and no dry-wall coefficients.
    coeffs.h_dry[:] = 0.0
    shell = domain.shell
    c_liq = float((coeffs.rho_c_liquid * domain.grid.volume).sum())
    c_shell = float((shell.rho_c * shell.thickness * shell.area).sum())
    g_out = float(
        (
            1.0 / (shell.thickness / (2 * shell.k * shell.area) + 1.0 / (h_out * shell.outer_area))
        ).sum()
    )
    tau = (c_liq + c_shell) / g_out

    state = solver.initial_state(350.0, 350.0, T_AMB)
    dt = 0.9 * solver.stable_dt(coeffs)
    t = 0.0
    checkpoints = [tau, 2.0 * tau]
    for target in checkpoints:
        n = math.ceil((target - t) / dt)
        step = (target - t) / n
        for _ in range(n):
            solver.step(state, coeffs, step)
        t = target
        lumped = solver.internal_energy(state, coeffs) / (c_liq + c_shell)
        expected = T_AMB + (350.0 - T_AMB) * math.exp(-t / tau)
        assert (lumped - T_AMB) == pytest.approx(expected - T_AMB, rel=5e-3)


# ------------------------------------------------------------------ 4. energy balance
@pytest.mark.parametrize("cover", [None, 1.0, 0.6], ids=["no_lid", "full_lid", "partial_lid"])
def test_energy_balance_every_step(cover: float | None) -> None:
    domain = box_domain(6, 5, 4, 0.02, lid=cover == 1.0, fill_fraction_top=0.7)
    if cover is not None and cover < 1.0:
        domain = _with_lid(domain, cover)
    rng = np.random.default_rng(42)
    solver = ThermalSolver(domain)
    coeffs = _random_coefficients(domain, rng)
    state = solver.initial_state(320.0, 310.0, T_AMB)
    state.T_liquid[:] = 320.0 + rng.uniform(-5.0, 5.0, state.T_liquid.shape)
    state.T_shell[:] = 310.0 + rng.uniform(-5.0, 5.0, state.T_shell.shape)
    dt = solver.stable_dt(coeffs)
    total = EnergyFlows()
    e0 = solver.internal_energy(state, coeffs)
    for _ in range(1000):
        flows = solver.step(state, coeffs, dt)
        assert flows.total_in() > 0.0
        assert abs(flows.balance_error()) <= 1e-9 * flows.total_in()
        total.add(flows)
    # Stored energy agrees with the change of internal energy of the state.
    de = solver.internal_energy(state, coeffs) - e0
    assert de == pytest.approx(total.total_stored(), rel=1e-9)
    if cover is None:
        assert total.loss_dry_wall_open > 0.0
        assert total.loss_lid_outer == 0.0
    else:
        assert total.loss_dry_wall_open == 0.0
        assert total.loss_lid_outer > 0.0
    if cover == 1.0:  # a full lid leaves no open surface
        assert total.loss_surface_evap == 0.0 and total.loss_surface_convrad == 0.0
    else:
        assert total.loss_surface_evap > 0.0 and total.loss_surface_convrad > 0.0


# ------------------------------------------------------------------ 5. symmetry
def test_symmetric_setup_gives_symmetric_field() -> None:
    domain = box_domain(7, 5, 6, 0.01)
    solver = ThermalSolver(domain)
    coeffs = uniform_coefficients(domain, h_film=300.0, h_out=10.0, h_surface=15.0, T_amb=T_AMB)
    coeffs.q_cells[3, 2, 2] = 80.0
    state = solver.initial_state(300.0, 300.0, T_AMB)
    dt = solver.stable_dt(coeffs)
    for _ in range(300):
        solver.step(state, coeffs, dt)
    t = state.T_liquid
    assert t.max() - t.min() > 0.1  # something happened
    assert np.max(np.abs(t - t[::-1, :, :])) < 1e-12
    assert np.max(np.abs(t - t[:, ::-1, :])) < 1e-12


# ------------------------------------------------------------------ 6. stability
def test_stability_at_the_limit_stays_bounded() -> None:
    domain = _with_lid(box_domain(5, 5, 5, 0.01), 0.5)
    rng = np.random.default_rng(7)
    solver = ThermalSolver(domain)
    coeffs = _random_coefficients(domain, rng)
    coeffs.q_cells[:] = 0.0
    coeffs.q_shell[:] = 0.0
    coeffs.q_evap_open[:] = 0.0
    state = solver.initial_state(300.0, 300.0, T_AMB)
    state.T_liquid[:] = rng.uniform(300.0, 350.0, state.T_liquid.shape)
    state.T_shell[:] = rng.uniform(300.0, 350.0, state.T_shell.shape)
    lo, hi = T_AMB - 1e-9, 350.0 + 1e-9
    dt = solver.stable_dt(coeffs)
    for _ in range(10_000):
        solver.step(state, coeffs, dt)
    for arr in (state.T_liquid, state.T_shell):
        assert arr.min() >= lo and arr.max() <= hi
    assert lo <= state.T_lid <= hi and lo <= state.T_headspace <= hi


def test_step_above_stable_dt_raises_and_leaves_state_untouched() -> None:
    domain = box_domain(5, 4, 3, 0.01, lid=True)
    solver = ThermalSolver(domain)
    coeffs = _random_coefficients(domain, np.random.default_rng(3))
    state = solver.initial_state(330.0, 320.0, T_AMB)
    before = (state.T_liquid.copy(), state.T_shell.copy(), state.T_lid, state.t)
    with pytest.raises(SolverError):
        solver.step(state, coeffs, 1.1 * solver.stable_dt(coeffs))
    assert np.array_equal(state.T_liquid, before[0])
    assert np.array_equal(state.T_shell, before[1])
    assert state.T_lid == before[2] and state.t == before[3]
    solver.step(state, coeffs, solver.stable_dt(coeffs))  # the limit itself is fine


# ------------------------------------------------------------------ 7. substeps
def test_shell_substeps_converge_first_order() -> None:
    base = box_domain(4, 4, 3, 0.01)
    shell = dataclasses.replace(base.shell, edge_conductance=base.shell.edge_conductance * 2000.0)
    domain = dataclasses.replace(base, shell=shell)
    solver = ThermalSolver(domain)
    coeffs = uniform_coefficients(domain, h_film=500.0, h_out=10.0, h_surface=10.0, T_amb=T_AMB)
    t_end = 8.0
    rng = np.random.default_rng(5)
    t_shell0 = 350.0 + rng.uniform(0.0, 20.0, shell.n)

    def run(dt: float) -> float:
        state = solver.initial_state(300.0, 350.0, T_AMB)
        state.T_shell[:] = t_shell0
        for _ in range(round(t_end / dt)):
            solver.step(state, coeffs, dt)
        assert solver.last_substeps > 1
        return float(state.T_liquid.mean())

    results = [run(t_end / (2**m)) for m in range(2, 7)]
    diffs = [a - b for a, b in itertools.pairwise(results)]
    assert all(d != 0.0 for d in diffs)
    for d_coarse, d_fine in itertools.pairwise(diffs):
        assert d_coarse / d_fine == pytest.approx(2.0, rel=0.25)


# ------------------------------------------------------------------ extras
def test_non_finite_values_raise_with_index() -> None:
    domain = box_domain(4, 3, 3, 0.01)
    solver = ThermalSolver(domain)
    coeffs = uniform_coefficients(domain, h_film=100.0, h_out=5.0, h_surface=5.0)
    state = solver.initial_state(300.0, 300.0, T_AMB)
    state.T_liquid[2, 1, 1] = np.nan
    with pytest.raises(SolverError, match=r"non-finite liquid temperature at cell \(\d, \d, \d\)"):
        solver.step(state, coeffs, 0.5 * solver.stable_dt(coeffs))
    state = solver.initial_state(300.0, 300.0, T_AMB)
    state.T_shell[5] = np.inf
    with pytest.raises(SolverError):
        solver.step(state, coeffs, 0.5 * solver.stable_dt(coeffs))


def test_stable_dt_is_infinite_without_conduction() -> None:
    domain = box_domain(1, 1, 1, 0.01, shell=False)
    solver = ThermalSolver(domain)
    coeffs = uniform_coefficients(domain)
    assert solver.stable_dt(coeffs) == math.inf
    state = solver.initial_state(300.0, 300.0, T_AMB)
    flows = solver.step(state, coeffs, 10.0)
    assert flows.balance_error() == 0.0


def test_initial_state_and_internal_energy() -> None:
    domain = box_domain(3, 3, 3, 0.01, lid=True)
    solver = ThermalSolver(domain)
    coeffs = uniform_coefficients(domain)
    state = solver.initial_state(350.0, 340.0, T_AMB)
    assert state.t == 0.0 and state.T_lid == T_AMB and state.T_headspace == T_AMB
    shell = domain.shell
    expected = (
        float((coeffs.rho_c_liquid * domain.grid.volume).sum()) * 350.0
        + float((shell.rho_c * shell.thickness * shell.area).sum()) * 340.0
        + domain.lid.heat_capacity * T_AMB
    )
    assert solver.internal_energy(state, coeffs) == pytest.approx(expected, rel=1e-12)


def test_lid_headspace_reaches_quasi_steady_value() -> None:
    """A covered pot with a lid warms the lid; the headspace stays between lid and liquid."""
    domain = box_domain(4, 4, 3, 0.02, lid=True)
    solver = ThermalSolver(domain)
    coeffs = uniform_coefficients(domain, h_film=500.0, h_out=8.0, h_surface=10.0, T_amb=T_AMB)
    state = solver.initial_state(350.0, 350.0, T_AMB)
    dt = 0.5 * solver.stable_dt(coeffs)
    for _ in range(300):
        solver.step(state, coeffs, dt)
    assert T_AMB < state.T_lid < 350.0
    assert state.T_lid < state.T_headspace < 350.0


# ------------------------------------------------------------------ 8. performance
@pytest.mark.slow
def test_performance_512k_cells() -> None:
    import time
    import warnings

    import numba

    domain = box_domain(80, 80, 80, 0.005)
    assert domain.shell.n >= 20_000
    solver = ThermalSolver(domain)
    coeffs = uniform_coefficients(domain, h_film=300.0, h_out=10.0, h_surface=10.0, T_amb=T_AMB)
    coeffs.q_cells[40, 40, 40] = 1000.0
    state = solver.initial_state(300.0, 300.0, T_AMB)
    dt = 0.5 * solver.stable_dt(coeffs)
    for _ in range(3):
        solver.step(state, coeffs, dt)
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        solver.step(state, coeffs, dt)
        times.append(time.perf_counter() - t0)
    ms = 1e3 * float(np.median(times))
    print(
        f"\n512k cells, {domain.shell.n} shell nodes, {numba.get_num_threads()} threads: "
        f"{ms:.2f} ms/step ({solver.last_substeps} substeps)"
    )
    if ms > 10.0:
        warnings.warn(f"solver step took {ms:.1f} ms (> 10 ms target)", stacklevel=1)
    assert ms < 50.0
