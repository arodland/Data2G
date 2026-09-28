"""Constellations: 2^m complex points indexed by their bit label.

Point i carries the bits of i, most significant first. A learned
constellation moves the points and so learns the labelling too; the
index-to-bits rule never changes. Points are unit average power.

Frozen point sets live in data2g/constellations/<name>.npy (complex128).
`gray-qamN` names are built in: Gray-labelled square QAM, the baseline
every learned set is judged against.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

DIR = Path(__file__).parent / "constellations"


def label_bits(m: int) -> np.ndarray:
    """(2^m, m) array of 0/1: row i is i's bits, MSB first."""
    return (np.arange(2**m)[:, None] >> np.arange(m - 1, -1, -1)) & 1


def gray_qam(m: int) -> np.ndarray:
    """Square QAM, Gray-labelled per axis: first m/2 bits pick I, rest Q."""
    if m % 2:
        raise ValueError("square QAM needs even m")
    k = m // 2
    n = 2**k
    levels = 2 * np.arange(n) - (n - 1)  # -(n-1)..(n-1), step 2
    gray = np.arange(n) ^ (np.arange(n) >> 1)
    amp = np.empty(n)
    amp[gray] = levels  # label g sits at the g-th Gray position
    idx = np.arange(2**m)
    pts = amp[idx >> k] + 1j * amp[idx & (n - 1)]
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


_QPSK = gray_qam(2)


@lru_cache(maxsize=None)
def load(name: str) -> np.ndarray:
    if name.startswith("gray-qam"):
        return gray_qam(int(np.log2(int(name[len("gray-qam"):]))))
    pts = np.load(DIR / f"{name}.npy")
    pts.setflags(write=False)
    return pts


@lru_cache(maxsize=None)
def ace_dirs(name: str) -> np.ndarray:
    """(2^m, 2) complex unit directions each point may move along, outward
    only (active constellation extension; zero: none). Moving along them
    only takes a point further from its decision boundaries.
    - Square QAM (gray-qam*): an outer level may move out along its axis; a
      corner along both.
    - Learned sets: a point on the convex hull may move radially out, the
      rest not. Conservative: its true region (the outer part of its
      Voronoi cell) is wider."""
    pts = load(name)
    d = np.zeros((len(pts), 2), dtype=complex)
    if name.startswith("gray-qam"):
        top = np.max(np.abs(pts.real))
        re, im = np.isclose(np.abs(pts.real), top), np.isclose(np.abs(pts.imag), top)
        d[re, 0] = np.sign(pts.real[re])
        d[im, 1] = 1j * np.sign(pts.imag[im])
    else:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(np.c_[pts.real, pts.imag]).vertices
        d[hull, 0] = pts[hull] / np.abs(pts[hull])
    d.setflags(write=False)
    return d


def ace_project(got: np.ndarray, want: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """ACE's projection: `got` (received cells) onto the region around
    `want` (the sent points, scaled) that `dirs` (per cell, (..., 2)) allow:
    want plus the outward part of the error along each direction. Works
    for numpy and torch alike."""
    e = got - want
    out = want
    for i in range(2):
        di = dirs[..., i]
        a = (e * di.conj()).real  # component along di (unit or zero)
        out = out + di * (a * (a > 0))
    return out


def bits_per_symbol(points: np.ndarray) -> int:
    return int(np.log2(len(points)))


def modulate(bits: np.ndarray, points: np.ndarray) -> np.ndarray:
    m = bits_per_symbol(points)
    idx = bits.reshape(-1, m) @ (1 << np.arange(m - 1, -1, -1))
    return points[idx]


def llr(y: np.ndarray, h: np.ndarray, var: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Exact bit LLRs (log P0/P1) for y = h x + n, n ~ CN(0, var), every
    point equally likely. y, h, var share a shape; returns shape + (m,)
    flattened to (..., m) then to one row of bits in symbol order."""
    m = bits_per_symbol(points)
    if m == 2 and np.allclose(points, _QPSK):
        # Gray QPSK: each bit rides one axis and the other axis's terms cancel,
        # so the exact LLR is linear: -4a Re|Im(y conj h) / var (a = 1/sqrt 2).
        # The general path below took 13% of a Pat exchange's CPU.
        u = y * np.conj(h) * (-2 * np.sqrt(2) / var)
        return np.stack([u.real, u.imag], axis=-1).reshape(-1)
    if m % 2 == 0 and m <= 8 and np.allclose(points, _square(m)):
        return _llr_square(y, h, var, m)
    d = -np.abs(y[..., None] - h[..., None] * points) ** 2 / var[..., None]  # (..., 2^m)
    lb = label_bits(m).astype(bool)  # (2^m, m)
    l0 = _lse(np.where(~lb.T, d[..., None, :], -np.inf))
    l1 = _lse(np.where(lb.T, d[..., None, :], -np.inf))
    return (l0 - l1).reshape(-1)


@lru_cache(maxsize=None)
def _square(m: int) -> np.ndarray:
    return gray_qam(m)


def _llr_square(y, h, var, m):
    """Exact LLRs for Gray square QAM, one axis at a time: -|y - h x|^2 / var
    = -(|h|^2 / var) |y / h - x|^2, which splits into I and Q terms, and the
    first m/2 bits ride I, the rest Q, so each bit's LLR needs its own axis's
    2^(m/2) levels (16-QAM: 4 levels, not 16 points)."""
    k = m // 2
    pts = _square(m)
    amp = pts[:: 2**k].real  # I level of each I label (index = I label << k)
    g2 = np.abs(h) ** 2
    w = g2 / var
    z = np.where(g2 > 0, y * np.conj(h) / np.where(g2 > 0, g2, 1), 0)
    lb = label_bits(k).astype(bool)  # (levels, k)
    out = []
    for t in (z.real, z.imag):
        d = -w[..., None] * (t[..., None] - amp) ** 2  # (..., levels)
        l0 = _lse(np.where(~lb.T, d[..., None, :], -np.inf))
        l1 = _lse(np.where(lb.T, d[..., None, :], -np.inf))
        out.append(l0 - l1)  # (..., k)
    return np.concatenate(out, axis=-1).reshape(-1)


def _lse(a: np.ndarray) -> np.ndarray:
    mx = np.max(a, axis=-1, keepdims=True)
    return (mx + np.log(np.sum(np.exp(a - mx), axis=-1, keepdims=True)))[..., 0]
