"""Numba kernels of the thermal solver. Pure array code, no physics correlations.

All kernels are ``cache=True, nogil=True``; only the two per-liquid-cell kernels use
``parallel=True`` (the shell, surface and lid parts are small and run serially).
Reductions over liquid cells use a fixed number of blocks (``part`` rows), so results are
deterministic and independent of the thread count.

Layout of ``extra`` (N, 2): column 0 is energy in J added to a liquid cell during the step
by everything that is not liquid-liquid conduction or the source (shell coupling, covered
surface, open surface, evaporation); column 1 is the sum of the non-neighbour conductances
(W/K) touching the cell, used for the stability limit.

Layout of ``flows`` (10,): source_liquid, source_shell, loss_shell_outer,
loss_surface_convrad, loss_surface_evap, loss_lid_outer, loss_dry_wall_open,
stored_liquid, stored_shell, stored_lid_headspace.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

F_SRC_LIQ, F_SRC_SHELL, F_OUT, F_CONV, F_EVAP, F_LID, F_DRY, F_ST_LIQ, F_ST_SHELL, F_ST_LID = range(
    10
)
N_FLOWS = 10
N_BLOCKS = 256


@njit(cache=True, nogil=True)
def reset_extra(extra: np.ndarray, special: np.ndarray) -> None:
    """Zero the rows of ``extra`` that the previous step may have written."""
    for s in range(special.shape[0]):
        n = special[s]
        extra[n, 0] = 0.0
        extra[n, 1] = 0.0


@njit(cache=True, nogil=True)
def shell_conductances(
    area: np.ndarray,
    thick: np.ndarray,
    rho_c: np.ndarray,
    k: np.ndarray,
    outer_area: np.ndarray,
    ins_r: np.ndarray,
    dry_area: np.ndarray,
    edge_sum: np.ndarray,
    couple_shell: np.ndarray,
    couple_n: np.ndarray,
    couple_area: np.ndarray,
    h_film: np.ndarray,
    h_out: np.ndarray,
    h_dry: np.ndarray,
    cs: np.ndarray,
    g_out: np.ndarray,
    g_dry: np.ndarray,
    g_c: np.ndarray,
    g_sum: np.ndarray,
    extra: np.ndarray,
) -> float:
    """Shell capacities and conductances. Returns min C/sum(G) over shell nodes (inf if none)."""
    n = area.shape[0]
    for s in range(n):
        cs[s] = rho_c[s] * thick[s] * area[s]
        if h_out[s] > 0.0:
            res = (
                thick[s] / (2.0 * k[s] * area[s])
                + ins_r[s] / outer_area[s]
                + 1.0 / (h_out[s] * outer_area[s])
            )
            g_out[s] = 1.0 / res
        else:
            g_out[s] = 0.0
        if h_dry[s] > 0.0 and dry_area[s] > 0.0:
            g_dry[s] = dry_area[s] / (1.0 / h_dry[s] + thick[s] / (2.0 * k[s]))
        else:
            g_dry[s] = 0.0
        g_sum[s] = edge_sum[s] + g_out[s] + g_dry[s]
    for c in range(couple_shell.shape[0]):
        s = couple_shell[c]
        if h_film[c] > 0.0:
            g = couple_area[c] / (1.0 / h_film[c] + thick[s] / (2.0 * k[s]))
        else:
            g = 0.0
        g_c[c] = g
        g_sum[s] += g
        extra[couple_n[c], 1] += g
    lim = math.inf
    for s in range(n):
        if g_sum[s] > 0.0:
            r = cs[s] / g_sum[s]
            if r < lim:
                lim = r
    return lim


@njit(cache=True, nogil=True)
def surface_conductances(
    surf_n: np.ndarray,
    surf_flat: np.ndarray,
    surf_area: np.ndarray,
    k: np.ndarray,
    vol: np.ndarray,
    h_open: np.ndarray,
    h_cov: np.ndarray,
    f: float,
    g_open: np.ndarray,
    g_cov: np.ndarray,
    extra: np.ndarray,
) -> None:
    """Free-surface conductances G_open (to ambient) and G_cov (to headspace)."""
    for s in range(surf_n.shape[0]):
        a = surf_area[s]
        kp = k[surf_flat[s]]
        g_open[s] = 0.0
        g_cov[s] = 0.0
        if a > 0.0 and kp > 0.0:
            delta = vol[surf_n[s]] / (2.0 * a)
            if h_open[s] > 0.0 and f < 1.0:
                g_open[s] = a * (1.0 - f) / (delta / kp + 1.0 / h_open[s])
            if h_cov[s] > 0.0 and f > 0.0:
                g_cov[s] = a * f / (delta / kp + 1.0 / h_cov[s])
        extra[surf_n[s], 1] += g_open[s] + g_cov[s]


@njit(cache=True, nogil=True)
def surface_open_energy(
    surf_n: np.ndarray,
    surf_flat: np.ndarray,
    surf_area: np.ndarray,
    T: np.ndarray,
    g_open: np.ndarray,
    q_evap: np.ndarray,
    f: float,
    t_amb: float,
    dt: float,
    extra: np.ndarray,
    flows: np.ndarray,
) -> None:
    """Energy lost through the uncovered surface (convection + radiation, evaporation)."""
    for s in range(surf_n.shape[0]):
        tp = T[surf_flat[s]]
        e_conv = g_open[s] * (tp - t_amb) * dt
        e_evap = q_evap[s] * surf_area[s] * (1.0 - f) * dt
        extra[surf_n[s], 0] -= e_conv + e_evap
        flows[F_CONV] += e_conv
        flows[F_EVAP] += e_evap


@njit(cache=True, nogil=True)
def shell_substeps(
    nsub: int,
    dt_sub: float,
    ts: np.ndarray,
    t_lid: float,
    t_hs: float,
    lid_present: bool,
    t_amb: float,
    cs: np.ndarray,
    g_out: np.ndarray,
    g_dry: np.ndarray,
    g_c: np.ndarray,
    q_shell: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    edge_g: np.ndarray,
    couple_shell: np.ndarray,
    couple_n: np.ndarray,
    t_couple: np.ndarray,
    surf_n: np.ndarray,
    g_cov: np.ndarray,
    t_surf: np.ndarray,
    c_lid: float,
    g_hl: float,
    g_lo: float,
    g_opn: float,
    net: np.ndarray,
    extra: np.ndarray,
    flows: np.ndarray,
) -> tuple[float, float]:
    """Advance shell, lid and (quasi-steady) headspace by ``nsub`` substeps of ``dt_sub``.

    The liquid is held at its start-of-step temperatures (``t_couple`` and ``t_surf``);
    the energy exchanged with it is accumulated in ``extra[:, 0]``. ``ts`` is updated in
    place. Returns the new (T_lid, T_headspace).
    """
    n = ts.shape[0]
    n_edges = edge_a.shape[0]
    n_couple = couple_shell.shape[0]
    n_surf = surf_n.shape[0]

    g_cov_sum = 0.0
    s_cov = 0.0
    for s in range(n_surf):
        g_cov_sum += g_cov[s]
        s_cov += g_cov[s] * t_surf[s]

    for _ in range(nsub):
        if lid_present:
            num = s_cov + g_hl * t_lid + g_opn * t_amb
            den = g_cov_sum + g_hl + g_opn
            for s in range(n):
                num += g_dry[s] * ts[s]
                den += g_dry[s]
            if den > 0.0:
                t_hs = num / den
        else:
            t_hs = t_amb
        t_wall_ref = t_hs

        for s in range(n):
            e_out = g_out[s] * (ts[s] - t_amb)
            e_dry = g_dry[s] * (ts[s] - t_wall_ref)
            net[s] = q_shell[s] - e_out - e_dry
            flows[F_OUT] += e_out * dt_sub
            if not lid_present:
                flows[F_DRY] += e_dry * dt_sub
        for e in range(n_edges):
            fe = edge_g[e] * (ts[edge_b[e]] - ts[edge_a[e]])
            net[edge_a[e]] += fe
            net[edge_b[e]] -= fe
        for c in range(n_couple):
            s = couple_shell[c]
            fc = g_c[c] * (t_couple[c] - ts[s])
            net[s] += fc
            extra[couple_n[c], 0] -= fc * dt_sub
        for s in range(n_surf):
            extra[surf_n[s], 0] += g_cov[s] * (t_hs - t_surf[s]) * dt_sub

        if lid_present:
            q_in = g_hl * (t_hs - t_lid)
            q_out = g_lo * (t_lid - t_amb)
            flows[F_LID] += (q_out + g_opn * (t_hs - t_amb)) * dt_sub
            t_lid_new = t_lid + dt_sub * (q_in - q_out) / c_lid
        else:
            t_lid_new = t_lid
        for s in range(n):
            ts[s] += dt_sub * net[s] / cs[s]
        t_lid = t_lid_new
    return t_lid, t_hs


@njit(cache=True, nogil=True, parallel=True)
def liquid_step(
    T: np.ndarray,
    k: np.ndarray,
    rho_c: np.ndarray,
    q: np.ndarray,
    vol: np.ndarray,
    liq_flat: np.ndarray,
    nbr: np.ndarray,
    nbr_area: np.ndarray,
    extra: np.ndarray,
    dx: float,
    dt: float,
    rate_limit: float,
    t_new: np.ndarray,
    part: np.ndarray,
) -> int:
    """Forward-Euler update of every liquid cell, double buffered through ``t_new``.

    ``part`` (N_BLOCKS, 3) receives per-block sums of stored energy and source energy, and
    the per-block maximum of sum(G)/C. The new temperatures are written into ``T`` only if
    ``max(sum(G)/C) * dt <= rate_limit`` and the energy sums are finite; the return value
    is 1 in that case and 0 otherwise (``T`` is then untouched). Doing the check and the
    write-back in one call keeps the parallel regions back to back, which matters because
    idle worker threads are slow to wake up.
    """
    n_liq = liq_flat.shape[0]
    nb = part.shape[0]
    chunk = (n_liq + nb - 1) // nb
    for b in prange(nb):
        lo = b * chunk
        hi = min(lo + chunk, n_liq)
        stored = 0.0
        src = 0.0
        rmax = 0.0
        for n in range(lo, hi):
            p = liq_flat[n]
            tp = T[p]
            kp = k[p]
            cap = rho_c[p] * vol[n]
            flux = 0.0
            sg = extra[n, 1]
            for ax in range(3):
                tot = 0.0
                for side in range(2):
                    d = 2 * ax + side
                    qi = nbr[n, d]
                    if qi >= 0:
                        kq = k[qi]
                        s2 = kp + kq
                        if s2 > 0.0:
                            g = nbr_area[n, d] * (2.0 * kp * kq / s2) / dx
                            tot += g * (T[qi] - tp)
                            sg += g
                flux += tot
            qp = q[p]
            de = dt * (flux + qp) + extra[n, 0]
            tn = tp + de / cap
            t_new[n] = tn
            stored += cap * (tn - tp)
            src += qp * dt
            r = sg / cap
            if r > rmax:
                rmax = r
        part[b, 0] = stored
        part[b, 1] = src
        part[b, 2] = rmax
    total = 0.0
    worst = 0.0
    for b in range(nb):
        total += part[b, 0] + part[b, 1]
        if part[b, 2] > worst:
            worst = part[b, 2]
    if not math.isfinite(total) or worst * dt > rate_limit:
        return 0
    for b in prange(nb):
        lo = b * chunk
        hi = min(lo + chunk, n_liq)
        for n in range(lo, hi):
            T[liq_flat[n]] = t_new[n]
    return 1


@njit(cache=True, nogil=True, parallel=True)
def liquid_max_rate(
    k: np.ndarray,
    rho_c: np.ndarray,
    vol: np.ndarray,
    liq_flat: np.ndarray,
    nbr: np.ndarray,
    nbr_area: np.ndarray,
    extra: np.ndarray,
    dx: float,
    part: np.ndarray,
) -> None:
    """Per-block maximum of sum(G)/C over liquid cells (``part[:, 2]``)."""
    n_liq = liq_flat.shape[0]
    nb = part.shape[0]
    chunk = (n_liq + nb - 1) // nb
    for b in prange(nb):
        lo = b * chunk
        hi = min(lo + chunk, n_liq)
        rmax = 0.0
        for n in range(lo, hi):
            p = liq_flat[n]
            kp = k[p]
            sg = extra[n, 1]
            for d in range(6):
                qi = nbr[n, d]
                if qi >= 0:
                    kq = k[qi]
                    s2 = kp + kq
                    if s2 > 0.0:
                        sg += nbr_area[n, d] * (2.0 * kp * kq / s2) / dx
            r = sg / (rho_c[p] * vol[n])
            if r > rmax:
                rmax = r
        part[b, 2] = rmax
