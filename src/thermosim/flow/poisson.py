"""Matrix-free pressure solve: conjugate gradients with a geometric multigrid preconditioner (WP21).

The operator is ``(A x)_c = sum_faces w_f (x_c - x_nb)`` over the open faces of cell ``c``, with
``w_f`` the face open fractions of :class:`~thermosim.flow.grid.FlowGrid`. This is the weighted
7-point Laplacian with homogeneous Neumann conditions on walls, base and free surface. It is
symmetric positive semi-definite with the constants as null space; the right-hand side must
sum to zero (the solver removes the round-off mean).

Preconditioner: one V-cycle, symmetric (red-black Gauss-Seidel red/black before, black/red after,
sum-restriction, piecewise-constant prolongation), so that CG applies. Coarse levels aggregate
2 x 2 x 2 cells (an axis of length 1 is not coarsened); a coarse cell is active if any child is.
The coarse face weight is the sum of the fine weights of the faces it covers divided by 2 along
the coarsened normal axis. This is the geometric re-discretisation (it equals the Galerkin
operator divided by the over-correction factor 2 that piecewise-constant transfer needs), and it
keeps cut-cell walls sealed (a coarse face is open if any fine face under it is). The coarsest
level (<= ``DENSE_MAX`` active cells) is solved directly with a pseudo-inverse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import DTypeLike

from thermosim.flow import kernels as K

DENSE_MAX = 600


@dataclass(eq=False)
class _Level:
    shape: tuple[int, int, int]
    wx: np.ndarray
    wy: np.ndarray
    wz: np.ndarray
    act: np.ndarray
    x: np.ndarray
    b: np.ndarray
    r: np.ndarray
    part: np.ndarray
    ratio: tuple[int, int, int] = (1, 1, 1)  # fine -> next coarser
    idx: np.ndarray | None = None  # flat active indices (coarsest level only)
    inv: np.ndarray | None = None  # pseudo-inverse of A on the active cells (coarsest only)


def _pad(a: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=a.dtype)
    out[tuple(slice(0, n) for n in a.shape)] = a
    return out


def _new_level(
    shape: tuple[int, int, int],
    wx: np.ndarray,
    wy: np.ndarray,
    wz: np.ndarray,
    act: np.ndarray,
    dtype: DTypeLike,
) -> _Level:
    return _Level(
        shape=shape,
        wx=np.ascontiguousarray(wx, dtype=dtype),
        wy=np.ascontiguousarray(wy, dtype=dtype),
        wz=np.ascontiguousarray(wz, dtype=dtype),
        act=np.ascontiguousarray(act, dtype=bool),
        x=np.zeros(shape, dtype=dtype),
        b=np.zeros(shape, dtype=dtype),
        r=np.zeros(shape, dtype=dtype),
        part=np.zeros(shape[0], dtype=np.float64),
    )


def _coarsen(lv: _Level, dtype: DTypeLike) -> _Level:
    nx, ny, nz = lv.shape
    rx, ry, rz = (2 if n > 1 else 1 for n in lv.shape)
    cx, cy, cz = math.ceil(nx / rx), math.ceil(ny / ry), math.ceil(nz / rz)
    wx = _pad(lv.wx.astype(np.float64), (rx * cx + 1, ry * cy, rz * cz))
    wy = _pad(lv.wy.astype(np.float64), (rx * cx, ry * cy + 1, rz * cz))
    wz = _pad(lv.wz.astype(np.float64), (rx * cx, ry * cy, rz * cz + 1))
    wxc = np.zeros((cx + 1, cy, cz))
    wyc = np.zeros((cx, cy + 1, cz))
    wzc = np.zeros((cx, cy, cz + 1))
    for a in range(ry):
        for b in range(rz):
            wxc += wx[0::rx, a::ry, b::rz][: cx + 1, :cy, :cz]
    for a in range(rx):
        for b in range(rz):
            wyc += wy[a::rx, 0::ry, b::rz][:cx, : cy + 1, :cz]
    for a in range(rx):
        for b in range(ry):
            wzc += wz[a::rx, b::ry, 0::rz][:cx, :cy, : cz + 1]
    wxc /= rx
    wyc /= ry
    wzc /= rz
    act = _pad(lv.act, (rx * cx, ry * cy, rz * cz))
    actc = act.reshape(cx, rx, cy, ry, cz, rz).any(axis=(1, 3, 5))
    lv.ratio = (rx, ry, rz)
    return _new_level((cx, cy, cz), wxc, wyc, wzc, actc, dtype)


def _dense_inverse(lv: _Level) -> None:
    """Pseudo-inverse of the operator on the active cells of a (small) level."""
    n_cells = lv.shape[0] * lv.shape[1] * lv.shape[2]
    flat = np.flatnonzero(lv.act.ravel())
    pos = np.full(n_cells, -1, dtype=np.int64)
    pos[flat] = np.arange(len(flat))
    ids = np.arange(n_cells).reshape(lv.shape)
    mat = np.zeros((len(flat), len(flat)))
    for w, axis in ((lv.wx, 0), (lv.wy, 1), (lv.wz, 2)):
        n = lv.shape[axis]
        lo: list[slice] = [slice(None)] * 3
        hi: list[slice] = [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, n - 1), slice(1, n)
        a = ids[tuple(lo)].ravel()
        b = ids[tuple(hi)].ravel()
        ww = w.astype(np.float64)[tuple(hi)].ravel()
        keep = ww > 0
        a, b, ww = pos[a[keep]], pos[b[keep]], ww[keep]
        np.add.at(mat, (a, a), ww)
        np.add.at(mat, (b, b), ww)
        np.add.at(mat, (a, b), -ww)
        np.add.at(mat, (b, a), -ww)
    lv.idx = flat
    lv.inv = np.linalg.pinv(mat, hermitian=True) if len(flat) else mat


@dataclass(eq=False)
class PoissonSolver:
    """CG + V-cycle solver for ``A phi = b`` on the open faces of a :class:`FlowGrid`.

    Args:
        wx, wy, wz: face open fractions (see ``FlowGrid``); dtype of the solver.
        liq: liquid-cell mask.
        tol: relative residual ``|r|_2 / |b|_2`` at which CG stops (default 1e-9 for float64,
            1e-5 for float32).
        max_iter: iteration cap.
    """

    wx: np.ndarray
    wy: np.ndarray
    wz: np.ndarray
    liq: np.ndarray
    tol: float | None = None
    max_iter: int = 200
    levels: list[_Level] = field(init=False, default_factory=list)
    last_iterations: int = field(init=False, default=0)
    last_relres: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        dtype = self.wx.dtype
        if self.tol is None:
            self.tol = 1e-9 if dtype == np.float64 else 1e-5
        lv = _new_level(self.liq.shape, self.wx, self.wy, self.wz, self.liq, dtype)  # type: ignore[arg-type]
        self.levels = [lv]
        while int(lv.act.sum()) > DENSE_MAX and any(n > 1 for n in lv.shape):
            lv = _coarsen(lv, dtype)
            self.levels.append(lv)
        _dense_inverse(self.levels[-1])
        shape = self.liq.shape
        self._z = np.zeros(shape, dtype=dtype)
        self._p = np.zeros(shape, dtype=dtype)
        self._ap = np.zeros(shape, dtype=dtype)
        self._r = np.zeros(shape, dtype=dtype)

    # ------------------------------------------------------------------ V-cycle
    def _vcycle(self, n: int) -> None:
        lv = self.levels[n]
        if n == len(self.levels) - 1:
            assert lv.idx is not None and lv.inv is not None
            lv.x[...] = 0.0
            lv.x.reshape(-1)[lv.idx] = lv.inv @ lv.b.reshape(-1)[lv.idx].astype(np.float64)
            return
        lv.x[...] = 0.0
        K.rbgs(lv.x, lv.b, lv.wx, lv.wy, lv.wz, 0)
        K.rbgs(lv.x, lv.b, lv.wx, lv.wy, lv.wz, 1)
        K.residual(lv.r, lv.b, lv.x, lv.wx, lv.wy, lv.wz)
        nxt = self.levels[n + 1]
        K.restrict_sum(nxt.b, lv.r, *lv.ratio)
        self._vcycle(n + 1)
        K.prolong_add(lv.x, nxt.x, lv.act, *lv.ratio)
        K.rbgs(lv.x, lv.b, lv.wx, lv.wy, lv.wz, 1)
        K.rbgs(lv.x, lv.b, lv.wx, lv.wy, lv.wz, 0)

    def precondition(self, r: np.ndarray, z: np.ndarray) -> None:
        """``z = M^-1 r`` with one V-cycle."""
        l0 = self.levels[0]
        l0.b[...] = r
        self._vcycle(0)
        z[...] = l0.x

    # ------------------------------------------------------------------ CG
    def solve(self, b: np.ndarray, x: np.ndarray, atol: float = 0.0) -> int:
        """Solve ``A x = b`` (x starts at 0 and is overwritten). Returns the CG iterations.

        CG stops when ``|r|_2 <= max(tol |b|_2, atol)``.

        ``b`` is modified (its mean over the liquid cells is removed). The relative residual
        reached is stored in ``last_relres``.
        """
        assert self.tol is not None
        l0 = self.levels[0]
        part = l0.part
        n_act = int(l0.act.sum())
        x[...] = 0.0
        self.last_iterations, self.last_relres = 0, 0.0
        if n_act == 0:
            return 0
        K.plane_sum(part, b)
        mean = float(part.sum()) / n_act
        if mean != 0.0:
            b[l0.act] -= mean
        r = self._r
        r[...] = b
        K.dot(part, r, r)
        bnorm = math.sqrt(float(part.sum()))
        if bnorm == 0.0 or bnorm <= atol:
            return 0
        z, p, ap = self._z, self._p, self._ap
        self.precondition(r, z)
        p[...] = z
        K.dot(part, r, z)
        rz = float(part.sum())
        rel = 1.0
        it = 0
        for it in range(1, self.max_iter + 1):
            self.last_iterations = it
            K.apply_A_dot(part, ap, p, l0.wx, l0.wy, l0.wz)
            pap = float(part.sum())
            if pap <= 0.0:
                break
            K.cg_update(part, x, r, p, ap, rz / pap)
            rel = math.sqrt(float(part.sum())) / bnorm
            if rel <= self.tol or rel * bnorm <= atol:
                break
            self.precondition(r, z)
            K.dot(part, r, z)
            rz_new = float(part.sum())
            beta = rz_new / rz
            rz = rz_new
            K.update_p(p, z, beta)
        self.last_iterations, self.last_relres = it, rel
        return it
