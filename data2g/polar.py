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

Decoder: successive-cancellation list, batched in torch, with the CRC
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


class SCLDecoder:
    """Batched SCL in torch. decode() returns every list path's info bits
    and path metrics, best first; the caller's CRC picks."""

    def __init__(self, code: PolarCode, list_size: int = 8, device="cpu"):
        import torch

        self.t, self.code, self.L, self.device = torch, code, list_size, device
        frozen = np.ones(code.n, bool)
        frozen[code.info_pos] = False
        self.frozen = frozen
        self.sent = torch.tensor(code.sent, device=device)
        self.info = torch.tensor(code.info_pos, device=device)

    def _f(self, a, b):
        t = self.t
        return t.sign(a) * t.sign(b) * t.minimum(a.abs(), b.abs())

    def _node(self, alpha, lo, pm):
        """alpha (B, L, n) for leaves lo..lo+n. Returns (beta, perm, pm)."""
        t = self.t
        n = alpha.shape[-1]
        B, L = alpha.shape[:2]
        if n == 1:
            a = alpha[..., 0]
            if self.frozen[lo]:
                pm = pm + t.nn.functional.softplus(-a)
                return t.zeros_like(alpha, dtype=t.uint8), t.arange(L, device=self.device).expand(B, L), pm
            cand = t.cat([pm + t.nn.functional.softplus(-a), pm + t.nn.functional.softplus(a)], dim=1)
            best = cand.topk(L, dim=1, largest=False)
            perm = best.indices % L
            bit = (best.indices // L).to(t.uint8)
            return bit[..., None], perm, best.values
        h = n // 2
        a, b = alpha[..., :h], alpha[..., h:]
        bl, p1, pm = self._node(self._f(a, b), lo, pm)
        a = a.gather(1, p1[..., None].expand(-1, -1, h))
        b = b.gather(1, p1[..., None].expand(-1, -1, h))
        br, p2, pm = self._node(b + (1 - 2 * bl.to(a.dtype)) * a, lo + h, pm)
        bl = bl.gather(1, p2[..., None].expand(-1, -1, h))
        return t.cat([bl ^ br, br], dim=-1), p1.gather(1, p2), pm

    def decode(self, llr_sent):
        """(B, e) LLRs -> (info bits (B, L, k) uint8, path metric (B, L)), best first."""
        t = self.t
        B = llr_sent.shape[0]
        alpha = t.zeros(B, self.code.n, device=self.device, dtype=llr_sent.dtype)
        alpha[:, self.sent] = llr_sent
        alpha = alpha[:, None, :].expand(B, self.L, -1).contiguous()
        pm = t.full((B, self.L), float("inf"), device=self.device, dtype=llr_sent.dtype)
        pm[:, 0] = 0.0
        x, _, pm = self._node(alpha, 0, pm)
        order = pm.argsort(dim=1)
        x = x.gather(1, order[..., None].expand(-1, -1, self.code.n))
        u = t.as_tensor(transform(x.cpu().numpy()), device=self.device)
        return u[..., self.info], pm.gather(1, order)
