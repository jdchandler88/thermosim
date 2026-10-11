"""Machine description shared by the benchmark scripts (plain functions, no dependencies)."""

from __future__ import annotations

import os
import platform
import subprocess


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def describe() -> str:
    """CPU, memory, load and library versions, as a short multi-line string."""
    import numba
    import numpy

    import thermosim

    lines = [f"platform: {platform.platform()}", f"python: {platform.python_version()}"]
    lscpu = {}
    for line in _run(["lscpu"]).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            lscpu[k.strip()] = v.strip()
    for key in ("Model name", "CPU(s)", "Thread(s) per core", "Core(s) per socket", "L2 cache"):
        if key in lscpu:
            lines.append(f"{key}: {lscpu[key]}")
    mem = [line for line in _run(["free", "-m"]).splitlines() if line.startswith("Mem")]
    if mem:
        lines.append(f"memory: {mem[0].split()[1]} MB")
    lines.append(f"load average (1, 5, 15 min): {tuple(round(x, 2) for x in os.getloadavg())}")
    lines.append(f"numpy {numpy.__version__}, numba {numba.__version__}")
    lines.append(f"thermosim: {os.path.dirname(thermosim.__file__)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
