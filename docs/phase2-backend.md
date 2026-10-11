# Phase 2 backend: Numba CPU and Numba CUDA

This records the backend decision for Phase 2 (fluid flow, see `docs/plan/wp/PHASE2.md`) and
explains how to run the GPU check on the Windows PC with the NVIDIA GTX 1060.

## Decision

- **Backend: Numba.** Numba CPU (`prange`) first, then Numba CUDA (`numba-cuda`, CUDA 12.x
  wheels) for the GTX 1060. This is the same language and toolchain as Phase 1. The kernels are
  point-wise, so the CPU `prange` kernels port to `@cuda.jit` almost line for line.
- CuPy, Taichi and PyTorch are dropped. None is needed, and each adds a heavy dependency with
  uncertain Pascal support.
- **Target hardware:** GTX 1060 (Pascal, compute capability 6.1, 3 or 6 GB).
- **Use CUDA 12.x only.** CUDA 13 dropped Pascal.
- **Float32** for flow fields. Consumer Pascal cards run float64 at 1/32 speed. Energy
  accumulators may stay float64.
- **CPU fallback always.** The program must still run without an NVIDIA GPU.
- **Development container has no GPU.** GPU code is developed and tested with Numba's CUDA
  simulator (`NUMBA_ENABLE_CUDASIM=1`). Real GPU timing is done on the user's Windows PC with
  `benchmarks/gpu/gpu_check.py`.

## Pinned versions

The optional `gpu` extra in `pyproject.toml` pins:

| Package | Spec | Why |
|---|---|---|
| `numba-cuda[cu12]` | `==0.30.4` (Linux and Windows only) | Latest release (2026-07-03). Pascal is verified below. The `cu12` extra gives CUDA 12.9 wheels. The marker `sys_platform != 'darwin'` is there because `cuda-bindings` has no macOS wheels, and `uv sync --all-extras` must keep working on macOS CI. |
| `numpy` | `>=2.0,<2.5` | `numba-cuda` calls `np.row_stack`, which NumPy 2.5 removed. Kernel compilation fails with NumPy 2.5 (verified). Older `numba-cuda` releases (0.26.0, 0.28.2) also call `row_stack`. |

Verified on the development container (no GPU), with `numba-cuda` 0.30.4:

- Numba compiles a kernel to PTX for `sm_61` (`cc=(6, 1)`) without a device.
- The NVRTC library in the `cu12` wheels (version 12.9) lists architecture 61 among its supported
  architectures.

The real GPU run on the GTX 1060 has not been done yet. The Windows run is the first real test.

The `numpy<2.5` cap applies to the whole lock file, so a plain `uv sync` also gets NumPy 2.4.x
until the cap is removed. Everything else in the project is unaffected.

## Windows: step by step

Run these in PowerShell or Command Prompt, in the ThermoSim project folder.

1. **Check the driver.** Open Command Prompt and run:

   ```
   nvidia-smi
   ```

   Look at the top of the output. "Driver Version" is the driver. "CUDA Version" must be **12.0
   or higher**. If `nvidia-smi` is not found, or the CUDA version is below 12.0, go to step 2.

2. **Install or update the NVIDIA driver.** Go to nvidia.com, then Drivers, and choose GeForce,
   GeForce 10 Series, GTX 1060 and Windows. Install the latest **Game Ready** or **Studio**
   driver. Reboot afterwards, then run `nvidia-smi` again.

3. **Install the GPU extra.**

   ```
   uv sync --extra gpu
   ```

   This downloads the CUDA 12 pip wheels (a few hundred MB, once). It also removes the dev tools
   (pytest, ruff, mypy) from the environment. Run `uv sync --all-extras` to restore them.

4. **Run the check.**

   ```
   uv run --extra gpu python benchmarks/gpu/gpu_check.py
   ```

   The first run compiles the kernels, so it is slower than later runs. Expect a few minutes at most.

5. **Copy the output.** Paste back the block between `--- ThermoSim WP20 GPU check ---` and
   `--- end ---`. If the Advice section is not "No problems found.", paste that too.

## Reading the output

The script has four sections.

1. **Device.** Look for:
   - `mode: GPU (CUDA)`. Anything else means the GPU is not being used. The Advice lines say
     what to do.
   - `device: NVIDIA GeForce GTX 1060 ...` and `compute capability: 6.1`.
   - `driver supports CUDA: 12.x`, which must be 12.0 or higher.
   - `memory: ... free of ... GiB`. The GTX 1060 has 3 or 6 GiB.
2. **Correctness.** Each line is a relative error against a NumPy reference, with a `PASS` or
   `FAIL`. The stencil checks (`gpu_apply`, `gpu_jacobi`) must pass with errors near 1e-7. The
   check `cg_residual_gpu_vs_cpu` compares the GPU and CPU CG residuals; it must be below 1e-3.
   Any `FAIL` makes the script exit with status 1.
3. **Timings.** Median time per call, in milliseconds, for one masked Jacobi sweep and one
   masked CG iteration, at 64^3, 96^3 and 128^3. The GPU arrays stay on the device during timing,
   so copies between host and device are not included. A GPU CG iteration includes two small
   copies of partial sums to the host, so its speedup is smaller than the Jacobi speedup.
4. **Summary.** The copy-paste block. The `speedup` column is CPU time divided by GPU time.
   Values above 1 mean the GPU is faster.

The script exits with status 0 when the correctness checks pass, even when there is no GPU.
Without a GPU, the GPU columns say `n/a` and the Advice section says why.

## Common Advice messages

| Message | What to do |
|---|---|
| "Windows sees no NVIDIA GPU for CUDA" | Check Device Manager for the GTX 1060. On a laptop, set the app to use the NVIDIA GPU. |
| "The NVIDIA driver is missing or too old" | Install the latest Game Ready or Studio driver, reboot. |
| "The CUDA 12 toolkit wheels are missing" | Run `uv sync --extra gpu` again. |
| "numba-cuda is not installed" | Run with `uv run --extra gpu python ...`. |
| "driver supports CUDA x.y, but the CUDA 12 wheels need 12.0" | Update the driver. |
| "Less than 2.5 GiB of GPU memory" | The 128^3 cases may not fit; the rest still works. |

## What to paste back

1. The summary block (`--- ThermoSim WP20 GPU check --- ... --- end ---`).
2. The Device section, if the mode is not GPU or the Advice is not "No problems found.".
3. Any traceback, in full.
