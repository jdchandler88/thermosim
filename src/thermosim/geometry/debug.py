"""Debug plotting for reviewing a Domain (WP04). matplotlib is an optional, dev-only import."""

from __future__ import annotations

from typing import Any

import numpy as np

from thermosim.core.types import Domain, ShellKind


def plot_domain(domain: Domain, ax: Any = None) -> Any:
    """Draw the x-z slice cells, the shell nodes near y = 0 and the shell-to-liquid couplings.

    Args:
        domain: the Domain to draw.
        ax: optional matplotlib Axes; a new figure is created if None.

    Returns:
        The matplotlib Axes.

    Raises:
        ImportError: if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("plot_domain needs matplotlib (dev-only dependency)") from exc

    g, sh, sl = domain.grid, domain.shell, domain.slice
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    nx, _, nz = g.shape
    x0, z0 = g.origin[0], g.origin[2]
    mask = g.is_liquid[:, sl.j, :].T.astype(float)
    ax.imshow(
        mask,
        origin="lower",
        extent=(x0, x0 + nx * g.dx, z0, z0 + nz * g.dx),
        cmap="Blues",
        alpha=0.5,
        vmin=0,
        vmax=1.5,
    )
    tol = 0.5 * g.dx
    near = np.abs(sh.position[:, 1]) <= tol + 1e-12
    for kind, colour in ((ShellKind.BASE, "k"), (ShellKind.SIDE, "r")):
        m = near & (sh.kind == int(kind))
        ax.plot(sh.position[m, 0], sh.position[m, 2], ".", color=colour, ms=3)
    cell_xyz = np.stack(np.unravel_index(sh.couple_cell, g.shape), axis=1)
    sel = np.isin(sh.couple_shell, np.flatnonzero(near)) & (cell_xyz[:, 1] == sl.j)
    for node, (i, _, k) in zip(sh.couple_shell[sel], cell_xyz[sel], strict=True):
        cx = g.origin[0] + (i + 0.5) * g.dx
        cz = g.origin[2] + (k + 0.5) * g.dx
        ax.plot([sh.position[node, 0], cx], [sh.position[node, 2], cz], "g-", lw=0.4)
    for p in domain.probes:
        ax.plot(p.position[0], p.position[2], "m*", ms=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    return ax
