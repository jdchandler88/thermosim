"""Numba kernels of the thermal solver. Pure array code, no physics correlations.

All kernels are ``cache=True, nogil=True``; only the two per-liquid-cell kernels use
``parallel=True`` (the shell, surface and lid parts are small and run serially).
Reductions over liquid cells use a fixed number of blocks (``part`` rows), so results are
deterministic and independent of the thread count.

Liquid update: cells whose six neighbours are all liquid with full-size faces ("interior" cells,
the vast majority) are stored as runs of consecutive flat indices and updated with direct
index arithmetic when the conductivity is uniform (the model's case): about a third of the
memory traffic of the general path, which handles every other cell (cut faces at the wall,
cells with shell/surface couplings) through the neighbour tables and also handles a
non-uniform conductivity. Both paths evaluate the same expressions in the same order. Each
liquid kernel exists in a serial and a ``parallel=True`` variant (the serial one is used for
small grids and when only one thread is wanted).

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
import os

# OpenMP threads that spin while they wait for work keep burning CPU when other processes
# compete for it, which made the parallel kernels 10-1000x slower on an oversubscribed machine.
# Passive waiting costs a few percent when the machine is idle. Must be set before the first
# parallel kernel launches (libgomp reads it when it loads); a value set by the user wins.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
from numba import njit, prange

F_SRC_LIQ, F_SRC_SHELL, F_OUT, F_CONV, F_EVAP, F_LID, F_DRY, F_ST_LIQ, F_ST_SHELL, F_ST_LID = range(
    10
)
N_FLOWS = 10
N_INTERIOR_BLOCKS = 192  # blocks over the "interior" cell runs (fast path)
N_GENERAL_BLOCKS = 64  # blocks over all other cells (general path)
# The drivers loop over N_BLOCKS indices i; i % 4 == 3 selects general block i // 4, any other i
# interior block 3 * (i // 4) + i % 4, so that a static split of the loop over threads gives
# every thread the same mix of the cheap and the expensive blocks.
N_BLOCKS = N_INTERIOR_BLOCKS + N_GENERAL_BLOCKS


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
    t_liq: np.ndarray,
    couple_flat: np.ndarray,
    surf_n: np.ndarray,
    g_cov: np.ndarray,
    surf_flat: np.ndarray,
    c_lid: float,
    g_hl: float,
    g_lo: float,
    g_opn: float,
    net: np.ndarray,
    extra: np.ndarray,
    flows: np.ndarray,
) -> tuple[float, float]:
    """Advance shell, lid and (quasi-steady) headspace by ``nsub`` substeps of ``dt_sub``.

    The liquid is held at its start-of-step temperatures (``t_liq``, read at ``couple_flat``
    and ``surf_flat``); the energy exchanged with it is accumulated in ``extra[:, 0]``. ``ts``
    is updated in place. Returns the new (T_lid, T_headspace).
    """
    n = ts.shape[0]
    n_edges = edge_a.shape[0]
    n_couple = couple_shell.shape[0]
    n_surf = surf_n.shape[0]

    g_cov_sum = 0.0
    s_cov = 0.0
    for s in range(n_surf):
        g_cov_sum += g_cov[s]
        s_cov += g_cov[s] * t_liq[surf_flat[s]]

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
            fc = g_c[c] * (t_liq[couple_flat[c]] - ts[s])
            net[s] += fc
            extra[couple_n[c], 0] -= fc * dt_sub
        for s in range(n_surf):
            extra[surf_n[s], 0] += g_cov[s] * (t_hs - t_liq[surf_flat[s]]) * dt_sub

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


@njit(cache=True, nogil=True)
def shell_energy_change(
    cs: np.ndarray, ts_new: np.ndarray, ts_old: np.ndarray
) -> tuple[float, bool]:
    """Shell stored-energy change sum(C (T_new - T_old)) and whether every T_new is finite."""
    de = 0.0
    finite = True
    for s in range(ts_new.shape[0]):
        de += cs[s] * (ts_new[s] - ts_old[s])
        if not math.isfinite(ts_new[s]):
            finite = False
    return de, finite


@njit(cache=True, nogil=True)
def reset_extra_energy(extra: np.ndarray, special: np.ndarray) -> None:
    """Zero only the energy column of the rows of ``extra`` that the last step wrote."""
    for s in range(special.shape[0]):
        extra[special[s], 0] = 0.0


@njit(cache=True, nogil=True)
def make_capacity(
    rho_c: np.ndarray, vol: np.ndarray, liq_flat: np.ndarray, cap: np.ndarray
) -> None:
    """cap[n] = rho_c[p] * V[n] for every liquid cell (compact index n, flat index p)."""
    for n in range(liq_flat.shape[0]):
        cap[n] = rho_c[liq_flat[n]] * vol[n]


@njit(cache=True, nogil=True)
def _interior_block(
    b: int,
    T: np.ndarray,
    cap: np.ndarray,
    q: np.ndarray,
    t_new: np.ndarray,
    part: np.ndarray,
    run_start: np.ndarray,
    run_comp: np.ndarray,
    run_len: np.ndarray,
    blk: np.ndarray,
    gx: float,
    gy: float,
    gz: float,
    sg6: float,
    sx: int,
    sy: int,
    dt: float,
) -> None:
    """Update the interior runs of block ``b`` (uniform conductivity, six full faces)."""
    stored = 0.0
    src = 0.0
    rmax = 0.0
    for r in range(blk[b], blk[b + 1]):
        p0 = run_start[r]
        n0 = run_comp[r]
        for i in range(run_len[r]):
            p = p0 + i
            n = n0 + i
            tp = T[p]
            c = cap[n]
            fx = 0.0
            fx += gx * (T[p - sx] - tp)
            fx += gx * (T[p + sx] - tp)
            fy = 0.0
            fy += gy * (T[p - sy] - tp)
            fy += gy * (T[p + sy] - tp)
            fz = 0.0
            fz += gz * (T[p - 1] - tp)
            fz += gz * (T[p + 1] - tp)
            flux = 0.0
            flux += fx
            flux += fy
            flux += fz
            qp = q[p]
            de = dt * (flux + qp) + 0.0
            tn = tp + de / c
            t_new[p] = tn
            stored += c * (tn - tp)
            src += qp * dt
            rr = sg6 / c
            if rr > rmax:
                rmax = rr
    part[b, 0] = stored
    part[b, 1] = src
    part[b, 2] = rmax


@njit(cache=True, nogil=True)
def _general_block(
    b: int,
    row: int,
    T: np.ndarray,
    k: np.ndarray,
    cap: np.ndarray,
    q: np.ndarray,
    liq_flat: np.ndarray,
    nbr: np.ndarray,
    nbr_area: np.ndarray,
    extra: np.ndarray,
    gen_n: np.ndarray,
    dx: float,
    dt: float,
    t_new: np.ndarray,
    part: np.ndarray,
) -> None:
    """Update the general cells of block ``b`` through the neighbour tables (any ``k``)."""
    n_gen = gen_n.shape[0]
    chunk = (n_gen + N_GENERAL_BLOCKS - 1) // N_GENERAL_BLOCKS
    lo = b * chunk
    hi = min(lo + chunk, n_gen)
    stored = 0.0
    src = 0.0
    rmax = 0.0
    for m in range(lo, hi):
        n = gen_n[m]
        p = liq_flat[n]
        tp = T[p]
        kp = k[p]
        c = cap[n]
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
        tn = tp + de / c
        t_new[p] = tn
        stored += c * (tn - tp)
        src += qp * dt
        r = sg / c
        if r > rmax:
            rmax = r
    part[row, 0] = stored
    part[row, 1] = src
    part[row, 2] = rmax


@njit(cache=True, nogil=True)
def _reduce_ok(part: np.ndarray, dt: float, rate_limit: float) -> bool:
    total = 0.0
    worst = 0.0
    for b in range(part.shape[0]):
        total += part[b, 0] + part[b, 1]
        if part[b, 2] > worst:
            worst = part[b, 2]
    return math.isfinite(total) and not worst * dt > rate_limit


@njit(cache=True, nogil=True)
def copy_liquid(T: np.ndarray, out: np.ndarray, liq_flat: np.ndarray) -> None:
    """T[p] = out[p] for every liquid cell p (the non-swapping way to commit a step)."""
    for n in range(liq_flat.shape[0]):
        p = liq_flat[n]
        T[p] = out[p]


@njit(cache=True, nogil=True, parallel=True)
def liquid_step_par(
    T: np.ndarray,
    k: np.ndarray,
    cap: np.ndarray,
    q: np.ndarray,
    liq_flat: np.ndarray,
    nbr: np.ndarray,
    nbr_area: np.ndarray,
    extra: np.ndarray,
    gen_n: np.ndarray,
    run_start: np.ndarray,
    run_comp: np.ndarray,
    run_len: np.ndarray,
    blk: np.ndarray,
    gx: float,
    gy: float,
    gz: float,
    sg6: float,
    sx: int,
    sy: int,
    dx: float,
    dt: float,
    rate_limit: float,
    t_new: np.ndarray,
    part: np.ndarray,
) -> int:
    """Forward-Euler update of every liquid cell from ``T`` into the flat buffer ``t_new``.

    ``t_new`` has the size of the grid (liquid cells at their flat index); ``T`` itself is never
    written, so a failed step leaves the state untouched and a successful one is committed by
    swapping the buffers or by `copy_liquid`. ``part`` (N_BLOCKS, 3) receives per-block sums of
    stored energy and source energy, and the per-block maximum of sum(G)/C. The return value is
    1 if ``max(sum(G)/C) * dt <= rate_limit`` and the energy sums are finite, 0 otherwise (the
    contents of ``t_new`` are then meaningless).
    """
    nbi = N_INTERIOR_BLOCKS
    for i in prange(N_BLOCKS):
        if i % 4 != 3:
            _interior_block(
                (i // 4) * 3 + i % 4,
                T,
                cap,
                q,
                t_new,
                part,
                run_start,
                run_comp,
                run_len,
                blk,
                gx,
                gy,
                gz,
                sg6,
                sx,
                sy,
                dt,
            )
        else:
            _general_block(
                i // 4, nbi + i // 4, T, k, cap, q, liq_flat, nbr, nbr_area, extra, gen_n, dx, dt,
                t_new, part,
            )  # fmt: skip
    if not _reduce_ok(part, dt, rate_limit):
        return 0
    return 1


@njit(cache=True, nogil=True)
def liquid_step_ser(
    T: np.ndarray,
    k: np.ndarray,
    cap: np.ndarray,
    q: np.ndarray,
    liq_flat: np.ndarray,
    nbr: np.ndarray,
    nbr_area: np.ndarray,
    extra: np.ndarray,
    gen_n: np.ndarray,
    run_start: np.ndarray,
    run_comp: np.ndarray,
    run_len: np.ndarray,
    blk: np.ndarray,
    gx: float,
    gy: float,
    gz: float,
    sg6: float,
    sx: int,
    sy: int,
    dx: float,
    dt: float,
    rate_limit: float,
    t_new: np.ndarray,
    part: np.ndarray,
) -> int:
    """Serial twin of `liquid_step_par` (no threading layer involved)."""
    nbi = N_INTERIOR_BLOCKS
    for i in range(N_BLOCKS):
        if i % 4 != 3:
            _interior_block(
                (i // 4) * 3 + i % 4,
                T,
                cap,
                q,
                t_new,
                part,
                run_start,
                run_comp,
                run_len,
                blk,
                gx,
                gy,
                gz,
                sg6,
                sx,
                sy,
                dt,
            )
        else:
            _general_block(
                i // 4, nbi + i // 4, T, k, cap, q, liq_flat, nbr, nbr_area, extra, gen_n, dx, dt,
                t_new, part,
            )  # fmt: skip
    if not _reduce_ok(part, dt, rate_limit):
        return 0
    return 1


@njit(cache=True, nogil=True)
def _rate_interior_block(
    b: int,
    cap: np.ndarray,
    part: np.ndarray,
    run_comp: np.ndarray,
    run_len: np.ndarray,
    blk: np.ndarray,
    sg6: float,
) -> None:
    rmax = 0.0
    for r in range(blk[b], blk[b + 1]):
        n0 = run_comp[r]
        for i in range(run_len[r]):
            rr = sg6 / cap[n0 + i]
            if rr > rmax:
                rmax = rr
    part[b, 2] = rmax


@njit(cache=True, nogil=True)
def _rate_general_block(
    b: int,
    row: int,
    k: np.ndarray,
    cap: np.ndarray,
    liq_flat: np.ndarray,
    nbr: np.ndarray,
    nbr_area: np.ndarray,
    extra: np.ndarray,
    gen_n: np.ndarray,
    dx: float,
    part: np.ndarray,
) -> None:
    n_gen = gen_n.shape[0]
    chunk = (n_gen + N_GENERAL_BLOCKS - 1) // N_GENERAL_BLOCKS
    lo = b * chunk
    hi = min(lo + chunk, n_gen)
    rmax = 0.0
    for m in range(lo, hi):
        n = gen_n[m]
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
        r = sg / cap[n]
        if r > rmax:
            rmax = r
    part[row, 2] = rmax


@njit(cache=True, nogil=True, parallel=True)
def liquid_max_rate_par(
    k: np.ndarray,
    cap: np.ndarray,
    liq_flat: np.ndarray,
    nbr: np.ndarray,
    nbr_area: np.ndarray,
    extra: np.ndarray,
    gen_n: np.ndarray,
    run_comp: np.ndarray,
    run_len: np.ndarray,
    blk: np.ndarray,
    sg6: float,
    dx: float,
    part: np.ndarray,
) -> None:
    """Per-block maximum of sum(G)/C over liquid cells (``part[:, 2]``)."""
    nbi = N_INTERIOR_BLOCKS
    for i in prange(N_BLOCKS):
        if i % 4 != 3:
            _rate_interior_block((i // 4) * 3 + i % 4, cap, part, run_comp, run_len, blk, sg6)
        else:
            g = i // 4
            _rate_general_block(g, nbi + g, k, cap, liq_flat, nbr, nbr_area, extra, gen_n, dx, part)


@njit(cache=True, nogil=True)
def liquid_max_rate_ser(
    k: np.ndarray,
    cap: np.ndarray,
    liq_flat: np.ndarray,
    nbr: np.ndarray,
    nbr_area: np.ndarray,
    extra: np.ndarray,
    gen_n: np.ndarray,
    run_comp: np.ndarray,
    run_len: np.ndarray,
    blk: np.ndarray,
    sg6: float,
    dx: float,
    part: np.ndarray,
) -> None:
    """Serial twin of `liquid_max_rate_par`."""
    nbi = N_INTERIOR_BLOCKS
    for i in range(N_BLOCKS):
        if i % 4 != 3:
            _rate_interior_block((i // 4) * 3 + i % 4, cap, part, run_comp, run_len, blk, sg6)
        else:
            g = i // 4
            _rate_general_block(g, nbi + g, k, cap, liq_flat, nbr, nbr_area, extra, gen_n, dx, part)
