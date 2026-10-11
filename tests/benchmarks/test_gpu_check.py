"""Tests for benchmarks/gpu/gpu_check.py (WP20).

The GPU code runs under Numba's CUDA simulator. NUMBA_ENABLE_CUDASIM must be set before numba
is imported, so the simulator checks run in a subprocess. The script itself is not a package,
so it is loaded from its file path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "gpu" / "gpu_check.py"
TIMEOUT_S = 600

# Loads the script, runs its core functions at 16^3 and prints one JSON line. If numba.cuda
# cannot be imported even in simulator mode, prints {"skip": reason} instead.
SIM_PROBE = r"""
import importlib.util, json, sys
from numba import config
if not config.ENABLE_CUDASIM:
    print(json.dumps({"skip": "NUMBA_ENABLE_CUDASIM was not seen by numba"}))
    raise SystemExit(0)
try:
    from numba import cuda  # noqa: F401
except Exception as exc:  # pragma: no cover - only without numba.cuda at all
    print(json.dumps({"skip": "numba.cuda cannot be imported: " + repr(exc)}))
    raise SystemExit(0)
spec = importlib.util.spec_from_file_location("gpu_check", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
out = {"mode": mod.device_report()["mode"]}
out["errors"] = mod.check_correctness(16, use_gpu=True)
out["jacobi_cpu_s"] = mod.bench_case("jacobi", 16, "cpu", 1)
out["cg_cpu_s"] = mod.bench_case("cg", 16, "cpu", 1)
out["jacobi_gpu_s"] = mod.bench_case("jacobi", 16, "gpu", 1)
mask, b = mod.make_problem(16)
x, r, p, ap, rr0 = mod._cg_state_cpu(b)
_, hist = mod.cg_iterations_cpu(x, r, p, ap, mask, rr0, 5)
out["cg_rr0"] = rr0
out["cg_hist"] = hist
print(json.dumps(out))
"""


def _env(simulator: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("NUMBA_ENABLE_CUDASIM", None)
    if simulator:
        env["NUMBA_ENABLE_CUDASIM"] = "1"
    return env


def test_script_file_exists() -> None:
    assert SCRIPT.is_file()


def test_core_functions_under_cuda_simulator_at_16_cubed() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", SIM_PROBE, str(SCRIPT)],
        env=_env(simulator=True),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    if "skip" in result:
        pytest.skip(result["skip"])

    assert result["mode"] == "simulator"
    errors = result["errors"]
    # Same tolerances as the script (TOL_STENCIL and TOL_CG).
    for key in ("cpu_apply", "cpu_jacobi", "gpu_apply", "gpu_jacobi"):
        assert errors[key] <= 1e-5, (key, errors[key])
    assert errors["cg_residual_gpu_vs_cpu"] <= 1e-3

    for key in ("jacobi_cpu_s", "cg_cpu_s", "jacobi_gpu_s"):
        assert result[key] > 0.0

    # CPU CG on a symmetric positive definite operator: the residual shrinks.
    assert result["cg_hist"][-1] < result["cg_rr0"]


def test_script_runs_without_cuda_and_prints_cpu_timings() -> None:
    """With no simulator, the script must still run and report why the GPU part is skipped.

    On a machine with a real GPU it also passes, since the GPU part then runs.
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--sizes", "16", "--repeats", "1"],
        env=_env(simulator=False),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "== 1. Device ==" in proc.stdout
    assert "== 3. Timings" in proc.stdout
    assert "jacobi" in proc.stdout and "cg" in proc.stdout
    assert "--- end ---" in proc.stdout
