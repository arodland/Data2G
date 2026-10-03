"""CRC-aided polar code for short PDUs (the ACK/ping submode).

Mother length N = 2^n >= the E transmitted bits; the surplus N - E is
punctured quasi-uniformly (the first N - E positions in bit-reversed
order), and the punctured bits reach the decoder as LLR 0. The frozen
set is whatever `design` says: here Gaussian-approximation density
evolution at a design SNR with the punctured bits carrying nothing, a
starting point for the genetic search in scripts/design_polar.py.

u (N bits, frozen = 0, info in the rest in increasing index) -> x = u G,
G = F^(kron n), F = [[1, 0], [1, 1]], natural order. G is its own
inverse over GF(2), so a decoder's re-encoded x gives u back.

Decoder: successive-cancellation list, batched in numpy, with the CRC
picking among the survivors (CA-SCL).
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np


def transform(u: np.ndarray) -> np.ndarray:
    """(..., N) -> u G mod 2."""
    x = u.copy()
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        x[..., 0, :] ^= x[..., 1, :]
        x = x.reshape(*x.shape[:-3], n)
        h *= 2
    return x


def bitrev(n_bits: int) -> np.ndarray:
    i = np.arange(2**n_bits)
    r = np.zeros_like(i)
    for b in range(n_bits):
        r |= ((i >> b) & 1) << (n_bits - 1 - b)
    return r


def _phi(m):
    m = np.maximum(m, 1e-12)
    return np.where(m < 10, np.exp(-0.4527 * m**0.86 + 0.0218), np.sqrt(np.pi / m) * np.exp(-m / 4) * (1 - 10 / (7 * m)))


def _phi_inv(y):
    # bisection; phi is decreasing
    lo, hi = np.zeros_like(y), np.full_like(y, 1e4)
    for _ in range(80):
        mid = (lo + hi) / 2
        big = _phi(mid) > y
        lo, hi = np.where(big, mid, lo), np.where(big, hi, mid)
    return (lo + hi) / 2


def ga_reliability(mean_llr: np.ndarray) -> np.ndarray:
    """Mean LLR of each u bit under SC decoding (Trifonov's Gaussian
    approximation), given the mean LLR of each coded bit x."""
    m = mean_llr.astype(float)
    n = len(m)
    h = n // 2
    while h >= 1:
        m = m.reshape(-1, 2, h)
        a, b = m[:, 0, :], m[:, 1, :]
        f = _phi_inv(1 - (1 - _phi(a)) * (1 - _phi(b)))
        f = np.where((a <= 0) | (b <= 0), 0.0, f)
        m = np.stack([f, a + b], axis=1).reshape(n)
        h //= 2
    return m


@dataclass
class PolarCode:
    k: int  # info bits incl. CRC
    e: int  # transmitted bits
    design_snr_db: float = 0.0  # per coded bit, BPSK-equivalent Es/N0
    frozen_override: tuple | None = None  # info positions, from a search

    @property
    def n(self) -> int:
        return 1 << int(np.ceil(np.log2(self.e)))

    @cached_property
    def punctured(self) -> np.ndarray:
        return np.sort(bitrev(int(np.log2(self.n)))[: self.n - self.e])

    @cached_property
    def sent(self) -> np.ndarray:
        mask = np.ones(self.n, bool)
        mask[self.punctured] = False
        return np.flatnonzero(mask)

    @cached_property
    def info_pos(self) -> np.ndarray:
        if self.frozen_override is not None:
            return np.array(sorted(self.frozen_override))
        mean = np.full(self.n, 4 * 10 ** (self.design_snr_db / 10))
        mean[self.punctured] = 0.0
        rel = ga_reliability(mean)
        return np.sort(np.argsort(rel)[-self.k :])

    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.atleast_2d(bits).astype(np.uint8)
        u = np.zeros((bits.shape[0], self.n), dtype=np.uint8)
        u[:, self.info_pos] = bits
        return transform(u)[:, self.sent]


def _softplus(x):
    return np.logaddexp(np.float32(0), x)


class SCLDecoder:
    """Batched SCL in numpy. decode() returns every list path's info bits
    and path metrics, best first; the caller's CRC picks.
    decoders_torch.SCLDecoder is the same on torch, for GPU studies."""

    def __init__(self, code: PolarCode, list_size: int = 8):
        self.code, self.L = code, list_size
        self.frozen = np.ones(code.n, bool)
        self.frozen[code.info_pos] = False

    @staticmethod
    def _f(a, b):
        return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def _node(self, alpha, lo, pm):
        """alpha (B, L, n) for leaves lo..lo+n. Returns (beta, perm, pm)."""
        n = alpha.shape[-1]
        B, L = alpha.shape[:2]
        if n == 1:
            a = alpha[..., 0]
            if self.frozen[lo]:
                return np.zeros(alpha.shape, np.uint8), np.broadcast_to(np.arange(L), (B, L)), pm + _softplus(-a)
            cand = np.concatenate([pm + _softplus(-a), pm + _softplus(a)], axis=1)
            idx = np.argsort(cand, axis=1, kind="stable")[:, :L]
            return (idx // L).astype(np.uint8)[..., None], idx % L, np.take_along_axis(cand, idx, 1)
        h = n // 2
        a, b = alpha[..., :h], alpha[..., h:]
        bl, p1, pm = self._node(self._f(a, b), lo, pm)
        a = np.take_along_axis(a, p1[..., None], 1)
        b = np.take_along_axis(b, p1[..., None], 1)
        br, p2, pm = self._node(b + (1 - 2 * bl.astype(a.dtype)) * a, lo + h, pm)
        bl = np.take_along_axis(bl, p2[..., None], 1)
        return np.concatenate([bl ^ br, br], axis=-1), np.take_along_axis(p1, p2, 1), pm

    def decode(self, llr_sent):
        """(B, e) LLRs -> (info bits (B, L, k) uint8, path metric (B, L)), best first."""
        llr_sent = np.asarray(llr_sent, np.float32)
        B = llr_sent.shape[0]
        alpha = np.zeros((B, self.code.n), np.float32)
        alpha[:, self.code.sent] = llr_sent
        alpha = np.repeat(alpha[:, None, :], self.L, axis=1)
        pm = np.full((B, self.L), np.inf, np.float32)
        pm[:, 0] = 0.0
        x, _, pm = self._node(alpha, 0, pm)
        order = pm.argsort(axis=1, kind="stable")
        x = np.take_along_axis(x, order[..., None], 1)
        return transform(x)[..., self.code.info_pos], np.take_along_axis(pm, order, 1)
