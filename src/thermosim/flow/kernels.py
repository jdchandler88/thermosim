"""Numba CPU kernels of the flow solver (WP21).

Style (so that the kernels port to ``@numba.cuda.jit`` almost line for line):
  * arrays and scalars only, no Python objects, no allocation inside a kernel;
  * one loop body per cell or face, written point-wise: ``for i in prange(n): for j: for k:``
    becomes one CUDA thread per ``(i, j, k)``;
  * every kernel reads its inputs and writes a *different* array (no in-place stencils), except
    the red-black Gauss-Seidel sweep, which is race-free within one colour;
  * reductions write one partial result per ``i`` plane into a ``part`` array; the caller sums
    it (deterministic, independent of the thread count).

Layout (MAC staggered grid, all index order ``[i, j, k] = [x, y, z]``):
  * ``u[i, j, k]``: x-velocity on the face between cells ``i-1`` and ``i``; shape
    ``(nx+1, ny, nz)``;
  * ``v``: shape ``(nx, ny+1, nz)``; ``w``: shape ``(nx, ny, nz+1)``;
  * ``p``, ``T``, ``nut``: cell centres, shape ``(nx, ny, nz)``.
  * ``mu, mv, mw``: boolean, True on *open* faces (both neighbouring cells are liquid). Velocity
    on a closed face is kept at exactly 0.
  * ``wx, wy, wz``: face open fractions ``area / dx^2`` (0 on closed faces, 1 on full faces; the
    partial top layer and the staircase footprint correction of ``LiquidGrid`` make them differ
    from 1). They enter the continuity equation and the pressure operator, so the discrete flow
    is divergence free in exactly the sense in which the thermal solver sees the cell volumes.

Boundary treatment of the tangential velocity at a wall (the wall is on the cell face, half a cell
from the velocity sample): the ghost value is ``-c`` (no slip) for a solid neighbour, and for
a neighbour outside the grid ``f * c + v`` with ``(f, v) = (-1, 0)`` no slip, ``(+1, 0)`` free
slip, ``(-1, 2 U)`` no slip on a wall moving at ``U``.

Advection schemes: 0 = central, 1 = first-order upwind, 2 = van Leer TVD (2nd order).
"""

from __future__ import annotations

import os

# Spinning OpenMP threads burn CPU when other processes compete for it (see solver.kernels).
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
from numba import njit, prange

SCHEME_CENTRAL = 0
SCHEME_UPWIND = 1
SCHEME_VANLEER = 2


# ----------------------------------------------------------------------------- point helpers
@njit(cache=True, nogil=True, inline="always")
def _ld(a, i, j, k, fb):
    """a[i, j, k], or ``fb`` if the index is outside the array."""
    if i < 0 or j < 0 or k < 0 or i >= a.shape[0] or j >= a.shape[1] or k >= a.shape[2]:
        return fb
    return a[i, j, k]


@njit(cache=True, nogil=True, inline="always")
def _lim(d1, d2):
    """van Leer limited slope: ``psi(r) * d2`` with ``r = d1 / d2`` (0 if the slopes disagree)."""
    pr = d1 * d2
    if pr <= 0.0:
        return 0.0
    return 2.0 * pr / (d1 + d2)


@njit(cache=True, nogil=True, inline="always")
def _face(mm, m, p, pp, vel, scheme):
    """Value of a transported quantity on the face between the samples ``m`` (low side) and
    ``p`` (high side); ``mm`` and ``pp`` are the next samples outward. ``vel`` is the velocity
    through the face (its sign picks the upwind side)."""
    if scheme == 0:
        return 0.5 * (m + p)
    if vel >= 0.0:
        if scheme == 1:
            return m
        return m + 0.5 * _lim(m - mm, p - m)
    if scheme == 1:
        return p
    return p + 0.5 * _lim(p - pp, m - p)


@njit(cache=True, nogil=True, inline="always")
def _tg(a, m, i, j, k, c, fo, vo):
    """Tangential neighbour ``a[i, j, k]`` of a sample with value ``c``, with wall ghost values."""
    if i < 0 or j < 0 or k < 0 or i >= a.shape[0] or j >= a.shape[1] or k >= a.shape[2]:
        return fo * c + vo
    if m[i, j, k]:
        return a[i, j, k]
    return -c


@njit(cache=True, nogil=True, inline="always")
def _gd(a, m, i1, j1, k1, i0, j0, k0, f_lo, v_lo, f_hi, v_hi):
    """Undivided difference ``a(1) - a(0)`` of two adjacent samples (1 = high side) with wall
    ghost values: no slip for a closed sample, ``(f, v)`` for one outside the grid."""
    oor1 = i1 >= a.shape[0] or j1 >= a.shape[1] or k1 >= a.shape[2]
    oor0 = i0 < 0 or j0 < 0 or k0 < 0
    o1 = (not oor1) and m[i1, j1, k1]
    o0 = (not oor0) and m[i0, j0, k0]
    if o1 and o0:
        return a[i1, j1, k1] - a[i0, j0, k0]
    if o0:
        c = a[i0, j0, k0]
        if oor1:
            return f_hi * c + v_hi - c
        return -2.0 * c
    if o1:
        c = a[i1, j1, k1]
        if oor0:
            return c - (f_lo * c + v_lo)
        return 2.0 * c
    return 0.0


# ----------------------------------------------------------------------------- Smagorinsky
@njit(cache=True, nogil=True, parallel=True)
def smag_nut(nut, u, v, w, mu, mv, mw, liq, cs2dx2, idx, fx, fy, top_fo, top_vo_u, top_vo_v):
    """Smagorinsky eddy viscosity ``nut = (Cs dx)^2 |S|`` (m^2/s) in the liquid cells.

    ``|S| = sqrt(2 S_ij S_ij)``; the diagonal strain rates sit at the cell centre, the three
    shear strain rates at the cell edges (four per cell, averaged as squares). ``fx, fy`` are the
    ghost factors of the grid-boundary walls in x and y (-1 no slip, +1 free slip); ``top_*``
    describe the upper boundary (z) for the u and v components.
    """
    nx, ny, nz = nut.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if not liq[i, j, k]:
                    nut[i, j, k] = 0.0
                    continue
                sxx = (u[i + 1, j, k] - u[i, j, k]) * idx
                syy = (v[i, j + 1, k] - v[i, j, k]) * idx
                szz = (w[i, j, k + 1] - w[i, j, k]) * idx
                s2 = 2.0 * (sxx * sxx + syy * syy + szz * szz)
                axy = 0.0
                axz = 0.0
                ayz = 0.0
                for da in range(2):
                    for db in range(2):
                        # xy edge (a, b) at cell row k
                        a = i + da
                        b = j + db
                        dudy = _gd(u, mu, a, b, k, a, b - 1, k, fy, 0.0, fy, 0.0)
                        dvdx = _gd(v, mv, a, b, k, a - 1, b, k, fx, 0.0, fx, 0.0)
                        sxy = 0.5 * (dudy + dvdx) * idx
                        axy += sxy * sxy
                        # xz edge (a, c) at cell column j
                        c = k + db
                        dudz = _gd(u, mu, a, j, c, a, j, c - 1, -1.0, 0.0, top_fo, top_vo_u)
                        dwdx = _gd(w, mw, a, j, c, a - 1, j, c, fx, 0.0, fx, 0.0)
                        sxz = 0.5 * (dudz + dwdx) * idx
                        axz += sxz * sxz
                        # yz edge (b, c) at cell column i
                        b2 = j + da
                        dvdz = _gd(v, mv, i, b2, c, i, b2, c - 1, -1.0, 0.0, top_fo, top_vo_v)
                        dwdy = _gd(w, mw, i, b2, c, i, b2 - 1, c, fy, 0.0, fy, 0.0)
                        syz = 0.5 * (dvdz + dwdy) * idx
                        ayz += syz * syz
                s2 += axy + axz + ayz
                nut[i, j, k] = cs2dx2 * np.sqrt(s2)


# ----------------------------------------------------------------------------- momentum
@njit(cache=True, nogil=True, parallel=True)
def rhs_u(us, u, v, w, p, mu, nut, nu, dt, dx, scheme, fy, top_fo, top_vo, use_nut):
    """Predictor ``us = u + dt (-adv + diff - dp/dx)`` on the open x-faces (flux form)."""
    nx1, ny, nz = u.shape
    idx = 1.0 / dx
    idx2 = idx * idx
    for i in prange(nx1):
        for j in range(ny):
            for k in range(nz):
                if not mu[i, j, k]:
                    us[i, j, k] = 0.0
                    continue
                c = u[i, j, k]
                ue = u[i + 1, j, k]
                uw = u[i - 1, j, k]
                # advection: x
                ra = 0.5 * (c + ue)
                la = 0.5 * (uw + c)
                fr = _face(uw, c, ue, _ld(u, i + 2, j, k, ue), ra, scheme)
                fl = _face(_ld(u, i - 2, j, k, uw), uw, c, ue, la, scheme)
                adv = ra * fr - la * fl
                # y
                vn = 0.5 * (v[i - 1, j + 1, k] + v[i, j + 1, k])
                vs = 0.5 * (v[i - 1, j, k] + v[i, j, k])
                un = _ld(u, i, j + 1, k, c)
                usv = _ld(u, i, j - 1, k, c)
                fn = _face(usv, c, un, _ld(u, i, j + 2, k, un), vn, scheme)
                fs = _face(_ld(u, i, j - 2, k, usv), usv, c, un, vs, scheme)
                adv += vn * fn - vs * fs
                # z
                wt = 0.5 * (w[i - 1, j, k + 1] + w[i, j, k + 1])
                wb = 0.5 * (w[i - 1, j, k] + w[i, j, k])
                ut = _ld(u, i, j, k + 1, c)
                ub = _ld(u, i, j, k - 1, c)
                ft = _face(ub, c, ut, _ld(u, i, j, k + 2, ut), wt, scheme)
                fb = _face(_ld(u, i, j, k - 2, ub), ub, c, ut, wb, scheme)
                adv += wt * ft - wb * fb
                # diffusion
                if use_nut:
                    nr = nu + nut[i, j, k]
                    nl = nu + nut[i - 1, j, k]
                    nn = nu + 0.25 * (
                        nut[i - 1, j, k]
                        + nut[i, j, k]
                        + _ld(nut, i - 1, j + 1, k, 0.0)
                        + _ld(nut, i, j + 1, k, 0.0)
                    )
                    ns = nu + 0.25 * (
                        nut[i - 1, j, k]
                        + nut[i, j, k]
                        + _ld(nut, i - 1, j - 1, k, 0.0)
                        + _ld(nut, i, j - 1, k, 0.0)
                    )
                    nt = nu + 0.25 * (
                        nut[i - 1, j, k]
                        + nut[i, j, k]
                        + _ld(nut, i - 1, j, k + 1, 0.0)
                        + _ld(nut, i, j, k + 1, 0.0)
                    )
                    nb = nu + 0.25 * (
                        nut[i - 1, j, k]
                        + nut[i, j, k]
                        + _ld(nut, i - 1, j, k - 1, 0.0)
                        + _ld(nut, i, j, k - 1, 0.0)
                    )
                else:
                    nr = nu
                    nl = nu
                    nn = nu
                    ns = nu
                    nt = nu
                    nb = nu
                gn = _tg(u, mu, i, j + 1, k, c, fy, 0.0)
                gs = _tg(u, mu, i, j - 1, k, c, fy, 0.0)
                gt = _tg(u, mu, i, j, k + 1, c, top_fo, top_vo)
                gb = _tg(u, mu, i, j, k - 1, c, -1.0, 0.0)
                dif = (
                    nr * (ue - c)
                    - nl * (c - uw)
                    + nn * (gn - c)
                    - ns * (c - gs)
                    + nt * (gt - c)
                    - nb * (c - gb)
                ) * idx2
                dp = (p[i, j, k] - p[i - 1, j, k]) * idx
                us[i, j, k] = c + dt * (-adv * idx + dif - dp)


@njit(cache=True, nogil=True, parallel=True)
def rhs_v(vs_, u, v, w, p, mv, nut, nu, dt, dx, scheme, fx, top_fo, top_vo, use_nut):
    """Predictor on the open y-faces; see :func:`rhs_u`."""
    nx, ny1, nz = v.shape
    idx = 1.0 / dx
    idx2 = idx * idx
    for i in prange(nx):
        for j in range(ny1):
            for k in range(nz):
                if not mv[i, j, k]:
                    vs_[i, j, k] = 0.0
                    continue
                c = v[i, j, k]
                vn_ = v[i, j + 1, k]
                vs = v[i, j - 1, k]
                # y
                na = 0.5 * (c + vn_)
                sa = 0.5 * (vs + c)
                fn = _face(vs, c, vn_, _ld(v, i, j + 2, k, vn_), na, scheme)
                fs = _face(_ld(v, i, j - 2, k, vs), vs, c, vn_, sa, scheme)
                adv = na * fn - sa * fs
                # x
                ue = 0.5 * (u[i + 1, j - 1, k] + u[i + 1, j, k])
                uw = 0.5 * (u[i, j - 1, k] + u[i, j, k])
                ve = _ld(v, i + 1, j, k, c)
                vw = _ld(v, i - 1, j, k, c)
                fe = _face(vw, c, ve, _ld(v, i + 2, j, k, ve), ue, scheme)
                fw = _face(_ld(v, i - 2, j, k, vw), vw, c, ve, uw, scheme)
                adv += ue * fe - uw * fw
                # z
                wt = 0.5 * (w[i, j - 1, k + 1] + w[i, j, k + 1])
                wb = 0.5 * (w[i, j - 1, k] + w[i, j, k])
                vt = _ld(v, i, j, k + 1, c)
                vb = _ld(v, i, j, k - 1, c)
                ft = _face(vb, c, vt, _ld(v, i, j, k + 2, vt), wt, scheme)
                fb = _face(_ld(v, i, j, k - 2, vb), vb, c, vt, wb, scheme)
                adv += wt * ft - wb * fb
                if use_nut:
                    nnn = nu + nut[i, j, k]
                    nss = nu + nut[i, j - 1, k]
                    nee = nu + 0.25 * (
                        nut[i, j - 1, k]
                        + nut[i, j, k]
                        + _ld(nut, i + 1, j - 1, k, 0.0)
                        + _ld(nut, i + 1, j, k, 0.0)
                    )
                    nww = nu + 0.25 * (
                        nut[i, j - 1, k]
                        + nut[i, j, k]
                        + _ld(nut, i - 1, j - 1, k, 0.0)
                        + _ld(nut, i - 1, j, k, 0.0)
                    )
                    ntt = nu + 0.25 * (
                        nut[i, j - 1, k]
                        + nut[i, j, k]
                        + _ld(nut, i, j - 1, k + 1, 0.0)
                        + _ld(nut, i, j, k + 1, 0.0)
                    )
                    nbb = nu + 0.25 * (
                        nut[i, j - 1, k]
                        + nut[i, j, k]
                        + _ld(nut, i, j - 1, k - 1, 0.0)
                        + _ld(nut, i, j, k - 1, 0.0)
                    )
                else:
                    nnn = nu
                    nss = nu
                    nee = nu
                    nww = nu
                    ntt = nu
                    nbb = nu
                ge = _tg(v, mv, i + 1, j, k, c, fx, 0.0)
                gw = _tg(v, mv, i - 1, j, k, c, fx, 0.0)
                gt = _tg(v, mv, i, j, k + 1, c, top_fo, top_vo)
                gb = _tg(v, mv, i, j, k - 1, c, -1.0, 0.0)
                dif = (
                    nnn * (vn_ - c)
                    - nss * (c - vs)
                    + nee * (ge - c)
                    - nww * (c - gw)
                    + ntt * (gt - c)
                    - nbb * (c - gb)
                ) * idx2
                dp = (p[i, j, k] - p[i, j - 1, k]) * idx
                vs_[i, j, k] = c + dt * (-adv * idx + dif - dp)


@njit(cache=True, nogil=True, parallel=True)
def rhs_w(ws, u, v, w, p, mw, T, nut, nu, dt, dx, scheme, fx, fy, use_nut, gbeta, tref):
    """Predictor on the open z-faces, with the Boussinesq buoyancy ``g beta (T - tref)``."""
    nx, ny, nz1 = w.shape
    idx = 1.0 / dx
    idx2 = idx * idx
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz1):
                if not mw[i, j, k]:
                    ws[i, j, k] = 0.0
                    continue
                c = w[i, j, k]
                wn = w[i, j, k + 1]
                wsb = w[i, j, k - 1]
                # z
                ta = 0.5 * (c + wn)
                ba = 0.5 * (wsb + c)
                ft = _face(wsb, c, wn, _ld(w, i, j, k + 2, wn), ta, scheme)
                fb = _face(_ld(w, i, j, k - 2, wsb), wsb, c, wn, ba, scheme)
                adv = ta * ft - ba * fb
                # x
                ue = 0.5 * (u[i + 1, j, k - 1] + u[i + 1, j, k])
                uw = 0.5 * (u[i, j, k - 1] + u[i, j, k])
                we = _ld(w, i + 1, j, k, c)
                ww = _ld(w, i - 1, j, k, c)
                fe = _face(ww, c, we, _ld(w, i + 2, j, k, we), ue, scheme)
                fw = _face(_ld(w, i - 2, j, k, ww), ww, c, we, uw, scheme)
                adv += ue * fe - uw * fw
                # y
                vn = 0.5 * (v[i, j + 1, k - 1] + v[i, j + 1, k])
                vs = 0.5 * (v[i, j, k - 1] + v[i, j, k])
                wnn = _ld(w, i, j + 1, k, c)
                wss = _ld(w, i, j - 1, k, c)
                fn = _face(wss, c, wnn, _ld(w, i, j + 2, k, wnn), vn, scheme)
                fs = _face(_ld(w, i, j - 2, k, wss), wss, c, wnn, vs, scheme)
                adv += vn * fn - vs * fs
                if use_nut:
                    nt = nu + nut[i, j, k]
                    nb = nu + nut[i, j, k - 1]
                    ne = nu + 0.25 * (
                        nut[i, j, k - 1]
                        + nut[i, j, k]
                        + _ld(nut, i + 1, j, k - 1, 0.0)
                        + _ld(nut, i + 1, j, k, 0.0)
                    )
                    nw = nu + 0.25 * (
                        nut[i, j, k - 1]
                        + nut[i, j, k]
                        + _ld(nut, i - 1, j, k - 1, 0.0)
                        + _ld(nut, i - 1, j, k, 0.0)
                    )
                    nnn = nu + 0.25 * (
                        nut[i, j, k - 1]
                        + nut[i, j, k]
                        + _ld(nut, i, j + 1, k - 1, 0.0)
                        + _ld(nut, i, j + 1, k, 0.0)
                    )
                    nss = nu + 0.25 * (
                        nut[i, j, k - 1]
                        + nut[i, j, k]
                        + _ld(nut, i, j - 1, k - 1, 0.0)
                        + _ld(nut, i, j - 1, k, 0.0)
                    )
                else:
                    nt = nu
                    nb = nu
                    ne = nu
                    nw = nu
                    nnn = nu
                    nss = nu
                ge = _tg(w, mw, i + 1, j, k, c, fx, 0.0)
                gw = _tg(w, mw, i - 1, j, k, c, fx, 0.0)
                gn = _tg(w, mw, i, j + 1, k, c, fy, 0.0)
                gs = _tg(w, mw, i, j - 1, k, c, fy, 0.0)
                dif = (
                    nt * (wn - c)
                    - nb * (c - wsb)
                    + ne * (ge - c)
                    - nw * (c - gw)
                    + nnn * (gn - c)
                    - nss * (c - gs)
                ) * idx2
                dp = (p[i, j, k] - p[i, j, k - 1]) * idx
                buoy = gbeta * (0.5 * (T[i, j, k - 1] + T[i, j, k]) - tref)
                ws[i, j, k] = c + dt * (-adv * idx + dif - dp + buoy)


# ----------------------------------------------------------------------------- projection
@njit(cache=True, nogil=True, parallel=True)
def div_rhs(b, us, vs, ws, wx, wy, wz, scale):
    """``b = scale * sum_faces(outward sign * w * u)`` per cell; ``scale = -dx / dt`` gives the
    right-hand side of ``A phi = b`` with A the (positive) weighted Laplacian."""
    nx, ny, nz = b.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                s = (
                    wx[i + 1, j, k] * us[i + 1, j, k]
                    - wx[i, j, k] * us[i, j, k]
                    + wy[i, j + 1, k] * vs[i, j + 1, k]
                    - wy[i, j, k] * vs[i, j, k]
                    + wz[i, j, k + 1] * ws[i, j, k + 1]
                    - wz[i, j, k] * ws[i, j, k]
                )
                b[i, j, k] = scale * s


@njit(cache=True, nogil=True, parallel=True)
def div_max(part, u, v, w, wx, wy, wz):
    """Per-plane maximum of ``|sum_faces(outward sign * w * u)|`` (m/s)."""
    nx, ny, nz = part.shape[0], u.shape[1], u.shape[2]
    for i in prange(nx):
        m = 0.0
        for j in range(ny):
            for k in range(nz):
                s = (
                    wx[i + 1, j, k] * u[i + 1, j, k]
                    - wx[i, j, k] * u[i, j, k]
                    + wy[i, j + 1, k] * v[i, j + 1, k]
                    - wy[i, j, k] * v[i, j, k]
                    + wz[i, j, k + 1] * w[i, j, k + 1]
                    - wz[i, j, k] * w[i, j, k]
                )
                if abs(s) > m:
                    m = abs(s)
        part[i] = m


@njit(cache=True, nogil=True, parallel=True)
def project_u(u, us, phi, mu, dtdx):
    nx1, ny, nz = u.shape
    for i in prange(nx1):
        for j in range(ny):
            for k in range(nz):
                if mu[i, j, k]:
                    u[i, j, k] = us[i, j, k] - dtdx * (phi[i, j, k] - phi[i - 1, j, k])
                else:
                    u[i, j, k] = 0.0


@njit(cache=True, nogil=True, parallel=True)
def project_v(v, vs, phi, mv, dtdx):
    nx, ny1, nz = v.shape
    for i in prange(nx):
        for j in range(ny1):
            for k in range(nz):
                if mv[i, j, k]:
                    v[i, j, k] = vs[i, j, k] - dtdx * (phi[i, j, k] - phi[i, j - 1, k])
                else:
                    v[i, j, k] = 0.0


@njit(cache=True, nogil=True, parallel=True)
def project_w(w, ws, phi, mw, dtdx):
    nx, ny, nz1 = w.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz1):
                if mw[i, j, k]:
                    w[i, j, k] = ws[i, j, k] - dtdx * (phi[i, j, k] - phi[i, j, k - 1])
                else:
                    w[i, j, k] = 0.0


@njit(cache=True, nogil=True, parallel=True)
def add_mean_free(p, phi, liq, mean):
    """``p += phi - mean`` in the liquid cells."""
    nx, ny, nz = p.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if liq[i, j, k]:
                    p[i, j, k] += phi[i, j, k] - mean


@njit(cache=True, nogil=True, parallel=True)
def plane_sum(part, a):
    """Per-plane sum of a cell array."""
    nx, ny, nz = a.shape
    for i in prange(nx):
        s = 0.0
        for j in range(ny):
            for k in range(nz):
                s += a[i, j, k]
        part[i] = s


@njit(cache=True, nogil=True, parallel=True)
def max_abs(part, a):
    """Per-plane maximum of ``|a|``."""
    nx, ny, nz = a.shape
    for i in prange(nx):
        m = 0.0
        for j in range(ny):
            for k in range(nz):
                x = abs(a[i, j, k])
                if x > m:
                    m = x
        part[i] = m


# ----------------------------------------------------------------------------- heat advection
@njit(cache=True, nogil=True, parallel=True)
def flux_x(F, T, u, mu, wx, liq, rcp_a, scheme):
    """Advective heat flux (W, +x) through the open x-faces: ``rho cp u (w dx^2) T_face``.
    ``rcp_a = rho cp dx^2``. The face temperature is van Leer limited (scheme 2)."""
    nx1, ny, nz = u.shape
    nx = nx1 - 1
    for i in prange(nx1):
        for j in range(ny):
            for k in range(nz):
                if not mu[i, j, k]:
                    continue
                vel = u[i, j, k]
                tm = T[i - 1, j, k]
                tp = T[i, j, k]
                mm = tm
                if i - 2 >= 0 and liq[i - 2, j, k]:
                    mm = T[i - 2, j, k]
                pp = tp
                if i + 1 < nx and liq[i + 1, j, k]:
                    pp = T[i + 1, j, k]
                F[i, j, k] = rcp_a * wx[i, j, k] * vel * _face(mm, tm, tp, pp, vel, scheme)


@njit(cache=True, nogil=True, parallel=True)
def flux_y(F, T, v, mv, wy, liq, rcp_a, scheme):
    nx, ny1, nz = v.shape
    ny = ny1 - 1
    for i in prange(nx):
        for j in range(ny1):
            for k in range(nz):
                if not mv[i, j, k]:
                    continue
                vel = v[i, j, k]
                tm = T[i, j - 1, k]
                tp = T[i, j, k]
                mm = tm
                if j - 2 >= 0 and liq[i, j - 2, k]:
                    mm = T[i, j - 2, k]
                pp = tp
                if j + 1 < ny and liq[i, j + 1, k]:
                    pp = T[i, j + 1, k]
                F[i, j, k] = rcp_a * wy[i, j, k] * vel * _face(mm, tm, tp, pp, vel, scheme)


@njit(cache=True, nogil=True, parallel=True)
def flux_z(F, T, w, mw, wz, liq, rcp_a, scheme):
    nx, ny, nz1 = w.shape
    nz = nz1 - 1
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz1):
                if not mw[i, j, k]:
                    continue
                vel = w[i, j, k]
                tm = T[i, j, k - 1]
                tp = T[i, j, k]
                mm = tm
                if k - 2 >= 0 and liq[i, j, k - 2]:
                    mm = T[i, j, k - 2]
                pp = tp
                if k + 1 < nz and liq[i, j, k + 1]:
                    pp = T[i, j, k + 1]
                F[i, j, k] = rcp_a * wz[i, j, k] * vel * _face(mm, tm, tp, pp, vel, scheme)


@njit(cache=True, nogil=True, parallel=True)
def flux_accumulate(q, Fx, Fy, Fz, frac):
    """``q += frac * (inflow - outflow)`` per cell from the stored face fluxes, so that every flux
    is subtracted from one cell and added to its neighbour as the same number."""
    nx, ny, nz = q.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                q[i, j, k] += frac * (
                    Fx[i, j, k]
                    - Fx[i + 1, j, k]
                    + Fy[i, j, k]
                    - Fy[i, j + 1, k]
                    + Fz[i, j, k]
                    - Fz[i, j, k + 1]
                )


# ----------------------------------------------------------------------------- Poisson kernels
@njit(cache=True, nogil=True, parallel=True)
def apply_A(y, x, wx, wy, wz):
    """``y = A x``, A the positive semi-definite weighted Laplacian ``sum w (x_c - x_nb)``."""
    nx, ny, nz = x.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                c = x[i, j, k]
                s = 0.0
                a = wx[i, j, k]
                if a != 0.0:
                    s += a * (c - x[i - 1, j, k])
                a = wx[i + 1, j, k]
                if a != 0.0:
                    s += a * (c - x[i + 1, j, k])
                a = wy[i, j, k]
                if a != 0.0:
                    s += a * (c - x[i, j - 1, k])
                a = wy[i, j + 1, k]
                if a != 0.0:
                    s += a * (c - x[i, j + 1, k])
                a = wz[i, j, k]
                if a != 0.0:
                    s += a * (c - x[i, j, k - 1])
                a = wz[i, j, k + 1]
                if a != 0.0:
                    s += a * (c - x[i, j, k + 1])
                y[i, j, k] = s


@njit(cache=True, nogil=True, parallel=True)
def apply_A_dot(part, y, x, wx, wy, wz):
    """``y = A x`` and per-plane partial sums of ``x . y``."""
    nx, ny, nz = x.shape
    for i in prange(nx):
        acc = 0.0
        for j in range(ny):
            for k in range(nz):
                c = x[i, j, k]
                s = 0.0
                a = wx[i, j, k]
                if a != 0.0:
                    s += a * (c - x[i - 1, j, k])
                a = wx[i + 1, j, k]
                if a != 0.0:
                    s += a * (c - x[i + 1, j, k])
                a = wy[i, j, k]
                if a != 0.0:
                    s += a * (c - x[i, j - 1, k])
                a = wy[i, j + 1, k]
                if a != 0.0:
                    s += a * (c - x[i, j + 1, k])
                a = wz[i, j, k]
                if a != 0.0:
                    s += a * (c - x[i, j, k - 1])
                a = wz[i, j, k + 1]
                if a != 0.0:
                    s += a * (c - x[i, j, k + 1])
                y[i, j, k] = s
                acc += c * s
        part[i] = acc


@njit(cache=True, nogil=True, parallel=True)
def residual(r, b, x, wx, wy, wz):
    """``r = b - A x``."""
    nx, ny, nz = x.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                c = x[i, j, k]
                s = 0.0
                a = wx[i, j, k]
                if a != 0.0:
                    s += a * (c - x[i - 1, j, k])
                a = wx[i + 1, j, k]
                if a != 0.0:
                    s += a * (c - x[i + 1, j, k])
                a = wy[i, j, k]
                if a != 0.0:
                    s += a * (c - x[i, j - 1, k])
                a = wy[i, j + 1, k]
                if a != 0.0:
                    s += a * (c - x[i, j + 1, k])
                a = wz[i, j, k]
                if a != 0.0:
                    s += a * (c - x[i, j, k - 1])
                a = wz[i, j, k + 1]
                if a != 0.0:
                    s += a * (c - x[i, j, k + 1])
                r[i, j, k] = b[i, j, k] - s


@njit(cache=True, nogil=True, parallel=True)
def rbgs(x, b, wx, wy, wz, color):
    """One Gauss-Seidel sweep over the cells of one colour ``(i + j + k) % 2 == color``."""
    nx, ny, nz = x.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if ((i + j + k) & 1) != color:
                    continue
                d = 0.0
                s = 0.0
                a = wx[i, j, k]
                if a != 0.0:
                    d += a
                    s += a * x[i - 1, j, k]
                a = wx[i + 1, j, k]
                if a != 0.0:
                    d += a
                    s += a * x[i + 1, j, k]
                a = wy[i, j, k]
                if a != 0.0:
                    d += a
                    s += a * x[i, j - 1, k]
                a = wy[i, j + 1, k]
                if a != 0.0:
                    d += a
                    s += a * x[i, j + 1, k]
                a = wz[i, j, k]
                if a != 0.0:
                    d += a
                    s += a * x[i, j, k - 1]
                a = wz[i, j, k + 1]
                if a != 0.0:
                    d += a
                    s += a * x[i, j, k + 1]
                if d > 0.0:
                    x[i, j, k] = (b[i, j, k] + s) / d


@njit(cache=True, nogil=True, parallel=True)
def restrict_sum(rc, rf, rx, ry, rz):
    """Coarse residual = sum of the fine residuals of the (up to 2x2x2) children."""
    nxc, nyc, nzc = rc.shape
    nxf, nyf, nzf = rf.shape
    for i in prange(nxc):
        for j in range(nyc):
            for k in range(nzc):
                s = 0.0
                for di in range(rx):
                    for dj in range(ry):
                        for dk in range(rz):
                            fi = i * rx + di
                            fj = j * ry + dj
                            fk = k * rz + dk
                            if fi < nxf and fj < nyf and fk < nzf:
                                s += rf[fi, fj, fk]
                rc[i, j, k] = s


@njit(cache=True, nogil=True, parallel=True)
def prolong_add(xf, xc, act, rx, ry, rz):
    """``xf += xc[parent]`` (piecewise constant) on the active fine cells."""
    nxf, nyf, nzf = xf.shape
    for i in prange(nxf):
        for j in range(nyf):
            for k in range(nzf):
                if act[i, j, k]:
                    xf[i, j, k] += xc[i // rx, j // ry, k // rz]


@njit(cache=True, nogil=True, parallel=True)
def dot(part, a, b):
    nx, ny, nz = a.shape
    for i in prange(nx):
        s = 0.0
        for j in range(ny):
            for k in range(nz):
                s += a[i, j, k] * b[i, j, k]
        part[i] = s


@njit(cache=True, nogil=True, parallel=True)
def cg_update(part, x, r, p, ap, alpha):
    """``x += alpha p; r -= alpha Ap`` and per-plane partial sums of ``r . r``."""
    nx, ny, nz = x.shape
    for i in prange(nx):
        s = 0.0
        for j in range(ny):
            for k in range(nz):
                x[i, j, k] += alpha * p[i, j, k]
                rr = r[i, j, k] - alpha * ap[i, j, k]
                r[i, j, k] = rr
                s += rr * rr
        part[i] = s


@njit(cache=True, nogil=True, parallel=True)
def update_p(p, z, beta):
    """``p = z + beta p``."""
    nx, ny, nz = p.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                p[i, j, k] = z[i, j, k] + beta * p[i, j, k]
