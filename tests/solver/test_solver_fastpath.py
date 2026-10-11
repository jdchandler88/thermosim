"""Fast interior path, conductance reuse and threading of the thermal solver (WP10 fixes B, C).

The interior fast path evaluates the same expressions as the general path, so the two must agree
to round-off; the thread count must not change the result at all (fixed reduction blocks).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from thermosim.core.types import Domain, StepCoefficients
from thermosim.solver import ThermalSolver
from thermosim.solver import thermal as th
from thermosim.testing.synthetic import box_domain, uniform_coefficients

T_AMB = 293.15


def _setup(
    domain: Domain, seed: int = 3
) -> tuple[StepCoefficients, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    c = uniform_coefficients(domain, T_amb=T_AMB)
    shape = domain.grid.shape
    c.k_liquid = np.full(shape, 0.8)  # uniform: the fast path is eligible
    c.rho_c_liquid = rng.uniform(3.5e6, 4.5e6, shape)
    c.h_film = rng.uniform(100.0, 800.0, len(domain.shell.couple_shell))
    c.q_cells = rng.uniform(0.0, 50.0, shape)
    t0 = 300.0 + rng.uniform(0.0, 30.0, shape)
    return c, t0, c.h_film.copy(), c.rho_c_liquid.copy()


def _run(
    solver: ThermalSolver,
    domain: Domain,
    c: StepCoefficients,
    t0: np.ndarray,
    steps: int = 12,
    **kw: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    state = solver.initial_state(300.0, 300.0, T_AMB)
    state.T_liquid[:] = np.where(domain.grid.is_liquid, t0, 0.0)
    dt = 0.9 * solver.stable_dt(c)
    imbalance = 0.0
    for _ in range(steps):
        f = solver.step(state, c, dt, **kw)
        imbalance = max(imbalance, abs(f.balance_error()))
    return state.T_liquid.copy(), state.T_shell.copy(), imbalance


@pytest.fixture(scope="module")
def domain() -> Domain:
    return box_domain(24, 20, 18, 0.01)


def test_interior_path_matches_general_path(domain: Domain) -> None:
    c, t0, _, _ = _setup(domain)
    fast = ThermalSolver(domain, threads=1)
    assert fast._interior is not None and fast._interior.run_len.sum() > 0.5 * fast.net.n_liq
    slow = ThermalSolver(domain, threads=1)
    slow._interior = None  # force the general path everywhere
    a = _run(fast, domain, c, t0)
    assert fast._layout.run_len.size > 0  # the fast layout was used
    b = _run(slow, domain, c, t0)
    assert slow._layout.run_len.size == 0
    np.testing.assert_allclose(a[0], b[0], rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-13, atol=0.0)
    assert a[2] < 1e-9 and b[2] < 1e-9


def test_non_uniform_conductivity_uses_general_path(domain: Domain) -> None:
    c, t0, _, _ = _setup(domain)
    c.k_liquid[3, 3, 3] *= 1.0000001
    solver = ThermalSolver(domain, threads=1)
    solver.stable_dt(c)
    assert solver._layout.run_len.size == 0
    _, _, imbalance = _run(solver, domain, c, t0)
    assert imbalance < 1e-9


def test_stable_dt_agrees_between_paths(domain: Domain) -> None:
    c, _, _, _ = _setup(domain)
    fast = ThermalSolver(domain, threads=1)
    slow = ThermalSolver(domain, threads=1)
    slow._interior = None
    assert fast.stable_dt(c) == pytest.approx(slow.stable_dt(c), rel=1e-12)


def test_thread_count_does_not_change_the_result(domain: Domain) -> None:
    c, t0, _, _ = _setup(domain)
    results = []
    for n in (1, 2, 4):
        solver = ThermalSolver(domain, threads=n)
        results.append(_run(solver, domain, c, t0))
    for r in results[1:]:
        np.testing.assert_array_equal(r[0], results[0][0])
        np.testing.assert_array_equal(r[1], results[0][1])


def test_reuse_conductances_gives_identical_steps(domain: Domain) -> None:
    c, t0, _, _ = _setup(domain)
    plain = _run(ThermalSolver(domain, threads=1), domain, c, t0)
    solver = ThermalSolver(domain, threads=1)
    reused = _run(solver, domain, c, t0, reuse_conductances=True)
    np.testing.assert_array_equal(plain[0], reused[0])
    np.testing.assert_array_equal(plain[1], reused[1])


def test_reuse_is_ignored_when_the_coefficient_arrays_change(domain: Domain) -> None:
    """Swapping an array (as the burner on/off switch does) forces a rebuild."""
    c, _, _, _ = _setup(domain)
    solver = ThermalSolver(domain, threads=1)
    dt = 0.5 * solver.stable_dt(c)
    state = solver.initial_state(300.0, 300.0, T_AMB)
    solver.step(state, c, dt)
    c2 = dataclasses.replace(c, h_out_shell=np.zeros_like(c.h_out_shell))
    ref = solver.initial_state(300.0, 300.0, T_AMB)
    solver2 = ThermalSolver(domain, threads=1)
    solver2.step(ref, c, dt)
    a = solver.step(state, c2, dt, reuse_conductances=True)
    b = solver2.step(ref, c2, dt)
    assert a.loss_shell_outer == b.loss_shell_outer == 0.0
    np.testing.assert_array_equal(state.T_shell, ref.T_shell)


def test_small_grids_never_use_the_threading_layer(domain: Domain) -> None:
    solver = ThermalSolver(domain)
    assert solver.net.n_liq < th.SERIAL_BELOW_CELLS
    assert solver._tuner.best == 1
    assert solver._tuner.pick() == 1


def test_thermosim_threads_environment_variable(
    domain: Domain, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("THERMOSIM_THREADS", "2")
    solver = ThermalSolver(domain)
    assert solver._tuner.best == min(2, solver._tuner.n_max)
    assert solver._tuner.pick() == solver._tuner.best  # fixed: no trials


def test_tuner_falls_back_to_fewer_threads_when_parallel_is_slower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(th, "_RETUNE_S", 0.0)
    tuner = th._ThreadTuner(n_max=4, fixed=None, n_cells=10 * th.SERIAL_BELOW_CELLS)
    assert tuner.best == 4
    cost = {1: 1.0, 2: 2.0, 4: 5.0}  # oversubscribed: more threads is slower
    for _ in range(40):
        n = tuner.pick()
        tuner.record(n, cost[n] * 1e-3)
    assert tuner.best == 1
    cost = {1: 5.0, 2: 3.0, 4: 1.0}  # idle machine: all threads is fastest
    for _ in range(40):
        n = tuner.pick()
        tuner.record(n, cost[n] * 1e-3)
    assert tuner.best == 4
