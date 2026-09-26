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
    d = -np.abs(y[..., None] - h[..., None] * points) ** 2 / var[..., None]  # (..., 2^m)
    lb = label_bits(m).astype(bool)  # (2^m, m)
    l0 = _lse(np.where(~lb.T, d[..., None, :], -np.inf))
    l1 = _lse(np.where(lb.T, d[..., None, :], -np.inf))
    return (l0 - l1).reshape(-1)


def _lse(a: np.ndarray) -> np.ndarray:
    mx = np.max(a, axis=-1, keepdims=True)
    return (mx + np.log(np.sum(np.exp(a - mx), axis=-1, keepdims=True)))[..., 0]
