"""WP20 GPU check: can this PC run the Numba CUDA backend, and how fast is it?

Run on the target Windows PC (NVIDIA GTX 1060 expected):

    uv sync --extra gpu
    uv run --extra gpu python benchmarks/gpu/gpu_check.py

The script prints, in order:

1. a device report (driver, toolkit, GPU, memory), with advice if CUDA is not usable;
2. a correctness check of a float32 masked 3-D 7-point operator against NumPy, on the GPU and
   the CPU, plus a short CG run compared between them;
3. timings, GPU vs CPU (Numba prange), of a masked Jacobi sweep and of a masked CG iteration at
   64^3, 96^3 and 128^3, with a cylinder mask;
4. a short summary to copy back.

Without CUDA it says why and still runs the CPU correctness and timings. With
NUMBA_ENABLE_CUDASIM=1 the GPU code runs in Numba's simulator: it is correct but very slow and
is not a GPU, so GPU timings are skipped.

Only numpy, numba, numba.cuda and the standard library are used. Nothing is imported from
thermosim, so the file can be copied anywhere.

Timing notes: GPU arrays stay on the device during timing, so host-device copies are not
included. A GPU CG iteration does two dot products and copies their per-block partial sums to
the host, so it includes that synchronisation. The CPU baseline uses Numba prange for the
stencil and NumPy for the vector operations.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from numba import config, cuda, njit, prange

SHIFT = np.float32(0.01)  # diagonal shift: makes the masked operator positive definite
BLOCK3 = (8, 8, 8)  # threads per block for the 3-D stencil
BLOCK1 = 64  # threads per block for the vector kernels (must be a power of two)
# Dot products use a grid-stride loop over at most this many blocks. The simulator runs one
# Python thread per CUDA thread, so it gets few blocks; the GPU gets many.
DOT_BLOCKS = 16 if config.ENABLE_CUDASIM else 512
DEFAULT_SIZES = (64, 96, 128)
CHECK_N = 32  # grid size of the correctness check
CHECK_CG_ITERS = 2
CHECK_CG_N = 16  # grid size of the CG check (kept small for the simulator)
TOL_STENCIL = 1e-5  # max error relative to max |reference|
TOL_CG = 1e-3  # relative difference of the CG residual norm, GPU vs CPU
OPS = ("jacobi", "cg")


# ---------------------------------------------------------------------------
# Masked 7-point operator. One function, compiled twice: for the CPU and as a device function.
# ---------------------------------------------------------------------------


def _point(u, b, mask, shift, jacobi, i, j, k):
    """One cell of the masked operator, float32 throughout.

    The mask is 1 in liquid and 0 outside. Neighbours count only if they are inside the mask,
    so the operator is a graph Laplacian on the masked cells plus ``shift``.

    jacobi=False returns (A u)[i, j, k] = (deg + shift) u - sum(neighbours).
    jacobi=True returns the Jacobi update (b + sum(neighbours)) / (deg + shift).
    Both return 0 outside the mask.
    """
    if mask[i, j, k] == 0:
        return np.float32(0.0)
    nx = mask.shape[0]
    ny = mask.shape[1]
    nz = mask.shape[2]
    deg = 0
    s = np.float32(0.0)
    if i > 0 and mask[i - 1, j, k] != 0:
        deg += 1
        s += u[i - 1, j, k]
    if i < nx - 1 and mask[i + 1, j, k] != 0:
        deg += 1
        s += u[i + 1, j, k]
    if j > 0 and mask[i, j - 1, k] != 0:
        deg += 1
        s += u[i, j - 1, k]
    if j < ny - 1 and mask[i, j + 1, k] != 0:
        deg += 1
        s += u[i, j + 1, k]
    if k > 0 and mask[i, j, k - 1] != 0:
        deg += 1
        s += u[i, j, k - 1]
    if k < nz - 1 and mask[i, j, k + 1] != 0:
        deg += 1
        s += u[i, j, k + 1]
    diag = np.float32(deg) + shift
    if jacobi:
        return np.float32((b[i, j, k] + s) / diag)
    return np.float32(diag * u[i, j, k] - s)


_point_cpu = njit(_point)
_point_dev = cuda.jit(device=True)(_point)


@cuda.jit
def _stencil_kernel(u, b, mask, shift, jacobi, out):
    i, j, k = cuda.grid(3)
    if i < out.shape[0] and j < out.shape[1] and k < out.shape[2]:
        out[i, j, k] = _point_dev(u, b, mask, shift, jacobi, i, j, k)


@njit(parallel=True)
def _stencil_cpu(u, b, mask, shift, jacobi, out):
    nx, ny, nz = out.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = _point_cpu(u, b, mask, shift, jacobi, i, j, k)


@cuda.jit
def _axpy_kernel(alpha, x, y):
    """y += alpha * x (flat arrays)."""
    idx = cuda.grid(1)
    if idx < y.size:
        y[idx] += alpha * x[idx]


@cuda.jit
def _xpay_kernel(beta, r, p):
    """p = r + beta * p (flat arrays)."""
    idx = cuda.grid(1)
    if idx < p.size:
        p[idx] = r[idx] + beta * p[idx]


@cuda.jit
def _dot_partial_kernel(a, b, partial):
    """Per-block partial sums of a * b (float32): grid-stride loop, then a shared-memory tree."""
    cache = cuda.shared.array(BLOCK1, dtype=np.float32)
    tid = cuda.threadIdx.x
    idx = cuda.grid(1)
    stride_total = cuda.gridsize(1)
    acc = np.float32(0.0)
    while idx < a.size:
        acc += a[idx] * b[idx]
        idx += stride_total
    cache[tid] = acc
    cuda.syncthreads()
    stride = BLOCK1 // 2
    while stride > 0:
        if tid < stride:
            cache[tid] += cache[tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        partial[cuda.blockIdx.x] = cache[0]


def _blocks1(size: int) -> int:
    return (size + BLOCK1 - 1) // BLOCK1


def _stencil_gpu(u, b, mask, shift, jacobi: bool, out) -> None:
    grid = tuple(-(-s // t) for s, t in zip(out.shape, BLOCK3, strict=True))
    _stencil_kernel[grid, BLOCK3](u, b, mask, shift, jacobi, out)


def _axpy_gpu(alpha: float, x, y) -> None:
    _axpy_kernel[_blocks1(y.size), BLOCK1](np.float32(alpha), x, y)


def _xpay_gpu(beta: float, r, p) -> None:
    _xpay_kernel[_blocks1(p.size), BLOCK1](np.float32(beta), r, p)


def _to_host(arr: Any) -> np.ndarray:
    """Copy a device array to the host. Under the simulator the arrays are already NumPy."""
    if hasattr(arr, "copy_to_host"):
        return np.asarray(arr.copy_to_host())
    return np.asarray(arr)


def _dot_gpu(a, b, partial) -> float:
    blocks = max(1, min(DOT_BLOCKS, _blocks1(a.size)))
    _dot_partial_kernel[blocks, BLOCK1](a, b, partial)
    return float(_to_host(partial[:blocks]).sum(dtype=np.float64))


# ---------------------------------------------------------------------------
# CG iteration, CPU and GPU. Both return the squared residual norm and its history.
# ---------------------------------------------------------------------------


def cg_iterations_cpu(
    x: np.ndarray,
    r: np.ndarray,
    p: np.ndarray,
    ap: np.ndarray,
    mask: np.ndarray,
    rr: float,
    iters: int,
) -> tuple[float, list[float]]:
    """Run ``iters`` CG iterations on the masked operator, in place. ``rr`` = |r|^2 on entry."""
    history: list[float] = []
    for _ in range(iters):
        _stencil_cpu(p, p, mask, SHIFT, False, ap)
        alpha = rr / float(np.dot(p.ravel(), ap.ravel()))
        x += np.float32(alpha) * p
        r -= np.float32(alpha) * ap
        rr_new = float(np.dot(r.ravel(), r.ravel()))
        p *= np.float32(rr_new / rr)
        p += r
        rr = rr_new
        history.append(rr)
    return rr, history


def cg_iterations_gpu(
    d_x: Any,
    d_r: Any,
    d_p: Any,
    d_ap: Any,
    d_mask: Any,
    d_partial: Any,
    rr: float,
    iters: int,
) -> tuple[float, list[float]]:
    """GPU version of :func:`cg_iterations_cpu`, arrays on the device."""
    history: list[float] = []
    x, r, p, ap = (a.reshape(-1) for a in (d_x, d_r, d_p, d_ap))
    for _ in range(iters):
        _stencil_gpu(d_p, d_p, d_mask, SHIFT, False, d_ap)
        alpha = rr / _dot_gpu(p, ap, d_partial)
        _axpy_gpu(alpha, p, x)
        _axpy_gpu(-alpha, ap, r)
        rr_new = _dot_gpu(r, r, d_partial)
        _xpay_gpu(rr_new / rr, r, p)
        rr = rr_new
        history.append(rr)
    return rr, history


# ---------------------------------------------------------------------------
# Problem set-up and NumPy reference
# ---------------------------------------------------------------------------


def cylinder_mask(n: int) -> np.ndarray:
    """Upright cylinder in an n^3 grid: radius 0.45 n, filling the lower 90 % of the height.

    Returns uint8, 1 inside the cylinder.
    """
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    radial = (x[:, None] ** 2 + x[None, :] ** 2) <= (0.45 * n) ** 2
    height = np.arange(n) < round(0.9 * n)
    return (radial[:, :, None] & height[None, None, :]).astype(np.uint8)


def make_problem(n: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Cylinder mask and a random float32 right-hand side, zero outside the mask."""
    mask = cylinder_mask(n)
    rng = np.random.default_rng(seed)
    b = rng.standard_normal((n, n, n)).astype(np.float32) * mask
    return mask, b


def _neighbour_terms(u: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-mask neighbour count and sum of u over in-mask neighbours (float64, NumPy only)."""
    m = np.pad(mask.astype(np.float64), 1)
    v = np.pad(u.astype(np.float64) * mask, 1)
    core = tuple(slice(1, 1 + size) for size in u.shape)
    deg = np.zeros(u.shape)
    s = np.zeros(u.shape)
    for axis in range(3):
        for step in (-1, 1):
            idx = list(core)
            idx[axis] = slice(1 + step, 1 + step + u.shape[axis])
            sel = tuple(idx)
            deg += m[sel]
            s += v[sel]
    return deg, s


def reference_apply(u: np.ndarray, mask: np.ndarray, shift: float) -> np.ndarray:
    """NumPy reference for the masked operator A u (float32 result)."""
    deg, s = _neighbour_terms(u, mask)
    out = mask * ((deg + shift) * u.astype(np.float64) - s)
    return out.astype(np.float32)


def reference_jacobi(u: np.ndarray, b: np.ndarray, mask: np.ndarray, shift: float) -> np.ndarray:
    """NumPy reference for one masked Jacobi sweep (float32 result)."""
    deg, s = _neighbour_terms(u, mask)
    out = mask * ((b.astype(np.float64) + s) / (deg + shift))
    return out.astype(np.float32)


def _rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    scale = max(float(np.abs(ref).max()), 1e-30)
    return float(np.abs(got.astype(np.float64) - ref).max() / scale)


def _cg_state_cpu(b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.zeros_like(b)
    r = b.copy()
    p = b.copy()
    ap = np.zeros_like(b)
    return x, r, p, ap, float(np.dot(r.ravel(), r.ravel()))


def _cg_state_gpu(b: np.ndarray) -> tuple[Any, Any, Any, Any, Any, float]:
    d_r = cuda.to_device(b)
    d_p = cuda.to_device(b)
    d_x = cuda.to_device(np.zeros_like(b))
    d_ap = cuda.device_array(b.shape, dtype=np.float32)
    d_partial = cuda.device_array(DOT_BLOCKS, dtype=np.float32)
    rr0 = float(np.dot(b.ravel(), b.ravel()))
    return d_x, d_r, d_p, d_ap, d_partial, rr0


def check_correctness(n: int = CHECK_N, use_gpu: bool = True, seed: int = 0) -> dict[str, float]:
    """Relative errors against NumPy for the stencil (CPU, GPU) and the CG residual (GPU vs CPU).

    The caller decides pass/fail with TOL_STENCIL and TOL_CG.
    """
    mask, rhs = make_problem(n, seed)
    rng = np.random.default_rng(seed + 1)
    u = (rng.standard_normal(rhs.shape).astype(np.float32) * mask).astype(np.float32)
    ref_apply = reference_apply(u, mask, float(SHIFT))
    ref_jacobi = reference_jacobi(u, rhs, mask, float(SHIFT))
    errors: dict[str, float] = {}

    out = np.zeros_like(u)
    _stencil_cpu(u, rhs, mask, SHIFT, False, out)
    errors["cpu_apply"] = _rel_err(out, ref_apply)
    _stencil_cpu(u, rhs, mask, SHIFT, True, out)
    errors["cpu_jacobi"] = _rel_err(out, ref_jacobi)

    cg_mask, cg_rhs = make_problem(CHECK_CG_N, seed)
    x, r, p, ap, rr0 = _cg_state_cpu(cg_rhs)
    rr_cpu, _ = cg_iterations_cpu(x, r, p, ap, cg_mask, rr0, CHECK_CG_ITERS)

    if use_gpu:
        d_mask = cuda.to_device(mask)
        d_u = cuda.to_device(u)
        d_rhs = cuda.to_device(rhs)
        d_out = cuda.device_array(u.shape, dtype=np.float32)
        _stencil_gpu(d_u, d_rhs, d_mask, SHIFT, False, d_out)
        errors["gpu_apply"] = _rel_err(_to_host(d_out), ref_apply)
        _stencil_gpu(d_u, d_rhs, d_mask, SHIFT, True, d_out)
        errors["gpu_jacobi"] = _rel_err(_to_host(d_out), ref_jacobi)

        d_cg_mask = cuda.to_device(cg_mask)
        d_x, d_r, d_p, d_ap, d_partial, rr0_gpu = _cg_state_gpu(cg_rhs)
        rr_gpu, _ = cg_iterations_gpu(
            d_x, d_r, d_p, d_ap, d_cg_mask, d_partial, rr0_gpu, CHECK_CG_ITERS
        )
        errors["cg_residual_gpu_vs_cpu"] = abs(np.sqrt(rr_gpu) - np.sqrt(rr_cpu)) / np.sqrt(rr_cpu)
    return errors


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _nosync() -> None:
    return None


def _time_calls(step: Callable[[], None], repeats: int, sync: Callable[[], None]) -> float:
    """Median seconds per call, after one untimed warm-up call (which compiles the kernels)."""
    step()
    sync()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        step()
        sync()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _jacobi_cpu_step(b: np.ndarray, mask: np.ndarray) -> Callable[[], None]:
    u = b.copy()
    out = np.zeros_like(b)

    def step() -> None:
        _stencil_cpu(u, b, mask, SHIFT, True, out)

    return step


def _cg_cpu_step(b: np.ndarray, mask: np.ndarray) -> Callable[[], None]:
    x, r, p, ap, rr0 = _cg_state_cpu(b)
    state = [rr0]

    def step() -> None:
        state[0], _ = cg_iterations_cpu(x, r, p, ap, mask, state[0], 1)

    return step


def _jacobi_gpu_step(b: np.ndarray, mask: np.ndarray) -> Callable[[], None]:
    d_mask = cuda.to_device(mask)
    d_b = cuda.to_device(b)
    d_u = cuda.to_device(b)
    d_out = cuda.device_array(b.shape, dtype=np.float32)

    def step() -> None:
        _stencil_gpu(d_u, d_b, d_mask, SHIFT, True, d_out)

    return step


def _cg_gpu_step(b: np.ndarray, mask: np.ndarray) -> Callable[[], None]:
    d_mask = cuda.to_device(mask)
    d_x, d_r, d_p, d_ap, d_partial, rr0 = _cg_state_gpu(b)
    state = [rr0]

    def step() -> None:
        state[0], _ = cg_iterations_gpu(d_x, d_r, d_p, d_ap, d_mask, d_partial, state[0], 1)

    return step


def bench_case(op: str, n: int, backend: str, repeats: int) -> float:
    """Median seconds for one masked Jacobi sweep or one masked CG iteration at n^3."""
    mask, b = make_problem(n)
    if backend == "gpu":
        step = _jacobi_gpu_step(b, mask) if op == "jacobi" else _cg_gpu_step(b, mask)
        return _time_calls(step, repeats, cuda.synchronize)
    step = _jacobi_cpu_step(b, mask) if op == "jacobi" else _cg_cpu_step(b, mask)
    return _time_calls(step, repeats, _nosync)


# ---------------------------------------------------------------------------
# Device report
# ---------------------------------------------------------------------------


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _fmt_version(v: Any) -> str:
    return ".".join(str(x) for x in v) if isinstance(v, tuple) else str(v)


def _advice_for_error(reason: str) -> list[str]:
    text = reason.lower()
    if "no cuda-capable device" in text or "cuda_error_no_device" in text:
        return [
            "Windows sees no NVIDIA GPU for CUDA. In Device Manager (Display adapters) check "
            "that the GeForce GTX 1060 is listed and has no warning icon.",
            "On a laptop with two GPUs, set this app to use the NVIDIA GPU "
            "(NVIDIA Control Panel > Manage 3D settings, or Windows Graphics settings).",
        ]
    if "driver" in text or "libcuda" in text or "nvcuda" in text or "cannot be found" in text:
        return [
            "The NVIDIA driver is missing or too old. Install the latest Game Ready or Studio "
            "driver for the GTX 1060 from nvidia.com, reboot, and run this again.",
        ]
    if "nvrtc" in text or "nvvm" in text or "toolkit" in text:
        return [
            "The CUDA 12 toolkit wheels are missing or do not load. Run `uv sync --extra gpu` "
            "and try again.",
        ]
    return ["Unrecognised error. Paste the message above back so it can be diagnosed."]


def device_report() -> dict[str, Any]:
    """Describe the CUDA situation.

    Returns ``mode`` ('gpu', 'simulator' or 'none'), ``lines`` (text for the report),
    ``advice`` (list of strings), and ``device`` (short GPU name, or '').
    """
    info: dict[str, Any] = {"mode": "none", "lines": [], "advice": [], "device": ""}
    lines: list[str] = info["lines"]
    advice: list[str] = info["advice"]
    lines.append(
        f"numba-cuda: {_package_version('numba-cuda')}   numba: {_package_version('numba')}"
    )

    if config.ENABLE_CUDASIM:
        info["mode"] = "simulator"
        info["device"] = "simulator"
        lines.append("mode: CUDA SIMULATOR (NUMBA_ENABLE_CUDASIM=1). Not a GPU.")
        advice.append(
            "Unset NUMBA_ENABLE_CUDASIM to use the real GPU. Simulator results are "
            "correct but slow, and its timings are not GPU timings."
        )
        return info

    if _package_version("numba-cuda") == "not installed":
        advice.append("numba-cuda is not installed. Run with `uv run --extra gpu python ...`.")

    try:
        available = bool(cuda.is_available())
    except Exception as exc:  # pragma: no cover - depends on the machine
        available = False
        reason = repr(exc)
    else:
        reason = ""
        if not available:
            error = cuda.cuda_error()
            reason = (
                " ".join(str(error).split())
                if error is not None
                else "no CUDA device or driver found"
            )

    if not available:
        lines.append("mode: no usable CUDA GPU. The CPU timings still run.")
        lines.append(f"reason: {reason}")
        advice.extend(_advice_for_error(reason))
        return info

    try:
        dev = cuda.get_current_device()
        name = dev.name
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        cc = tuple(dev.compute_capability)
        free, total = cuda.current_context().get_memory_info()
    except Exception as exc:
        lines.append(f"mode: CUDA reported available but device query failed: {exc!r}")
        advice.append("Unexpected failure querying the GPU. Paste this message back.")
        return info

    driver_v: Any
    runtime_v: Any
    try:
        driver_v = cuda.cudadrv.driver.get_version()
    except Exception:
        driver_v = (0, 0)
    try:
        runtime_v = cuda.cudadrv.runtime.get_version()
    except Exception:
        runtime_v = (0, 0)

    info["mode"] = "gpu"
    info["device"] = str(name)
    total_gib = total / 2**30
    lines.extend(
        [
            "mode: GPU (CUDA)",
            f"device: {name}",
            f"compute capability: {_fmt_version(cc)}",
            f"memory: {free / 2**30:.2f} GiB free of {total_gib:.2f} GiB",
            f"driver supports CUDA: {_fmt_version(driver_v)}",
            f"toolkit (NVRTC) version: {_fmt_version(runtime_v)}",
        ]
    )
    if cc < (6, 0):
        advice.append(f"Compute capability {_fmt_version(cc)} is older than Pascal (6.0).")
    if tuple(driver_v) < (12, 0):
        advice.append(
            f"The driver supports CUDA {_fmt_version(driver_v)}, but the CUDA 12 wheels need "
            "12.0 or newer. Update the NVIDIA driver."
        )
    if tuple(runtime_v)[:1] != (12,):
        advice.append(
            f"Toolkit is CUDA {_fmt_version(runtime_v)}. Use the cu12 wheels: "
            "`uv sync --extra gpu`. CUDA 13 dropped Pascal."
        )
    if total_gib < 2.5:
        advice.append("Less than 2.5 GiB of GPU memory. The 128^3 cases may not fit.")
    if not advice:
        advice.append("No problems found.")
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _fmt_ms(seconds: float | None) -> str:
    return "n/a" if seconds is None else f"{seconds * 1e3:.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ThermoSim WP20 GPU check.")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    print("== 1. Device ==")
    report = device_report()
    for line in report["lines"]:
        print(f"  {line}")
    print("  Advice:")
    for line in report["advice"]:
        print(f"   - {line}")
    mode = report["mode"]
    use_gpu = mode in ("gpu", "simulator")

    print(f"\n== 2. Correctness (float32 masked 7-point, n={CHECK_N}) ==")
    if not use_gpu:
        print("  GPU part skipped (no usable CUDA). CPU part still runs.")
    errors = check_correctness(CHECK_N, use_gpu=use_gpu)
    passed = True
    for key, value in errors.items():
        tol = TOL_CG if key.startswith("cg") else TOL_STENCIL
        ok = value <= tol
        passed = passed and ok
        print(f"  {key:<24} max rel err {value:.2e}  (tol {tol:.0e})  {'PASS' if ok else 'FAIL'}")

    print("\n== 3. Timings (median seconds per call, ms) ==")
    if mode == "simulator":
        print("  GPU timings skipped under the simulator (it runs Python per thread).")
    rows: list[tuple[str, int, float, float | None]] = []
    for n in args.sizes:
        for op in OPS:
            cpu = bench_case(op, n, "cpu", args.repeats)
            gpu = bench_case(op, n, "gpu", args.repeats) if mode == "gpu" else None
            rows.append((op, n, cpu, gpu))
            print(f"  {op:<8} {n:>4}^3  CPU {_fmt_ms(cpu):>10}  GPU {_fmt_ms(gpu):>10}")

    print("\n== 4. Summary (copy and paste) ==")
    print("--- ThermoSim WP20 GPU check ---")
    print(
        f"python {platform.python_version()} on {sys.platform}; numpy {np.__version__}; "
        f"numba {_package_version('numba')}; numba-cuda {_package_version('numba-cuda')}"
    )
    print(f"gpu: {report['device'] or 'none'} (mode: {mode})")
    print(f"correctness: {'PASS' if passed else 'FAIL'} ({len(errors)} checks, n={CHECK_N})")
    print(f"{'op':<8}{'n^3':>6}{'CPU ms':>10}{'GPU ms':>10}{'speedup':>10}")
    for op, n, cpu, gpu in rows:
        speed = f"{cpu / gpu:.1f}x" if gpu else "n/a"
        print(f"{op:<8}{n:>6}{_fmt_ms(cpu):>10}{_fmt_ms(gpu):>10}{speed:>10}")
    print("--- end ---")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
