"""The FEC decoders in torch, for studies that decode big batches on the
GPU (`codes.decode_llrs(..., device=...)`). The modem decodes with the
numpy ones in ldpc.py and polar.py; tests/test_decoders_torch.py pins
these against them.
"""

import numpy as np
import torch

from .ldpc import CH_CLAMP, QCLDPC
from .polar import PolarCode, transform


def _phi(x):
    x = x.clamp(1e-7, 30.0)
    return -torch.log(torch.tanh(x / 2))


class MinSumDecoder:
    """Batched flooding decoder on a QCLDPC, in torch: sum-product (BP,
    `alpha=None`, the default) or normalized min-sum with `alpha`.

    BP is the default because this modem's bit rate (a few kb/s) makes
    its cost irrelevant even on a phone; min-sum exists for hardware.
    Stops early once every codeword in the batch satisfies H.
    """

    BIG = 1e4  # LLR for filler bits: known zeros

    def __init__(self, code: QCLDPC, device="cpu"):
        self.code, self.device, self.torch = code, device, torch
        r, c = code.edges
        order = np.argsort(r, kind="stable")
        r, c = r[order], c[order]
        self.n_edges = len(r)
        self.var = torch.tensor(c, device=device)
        deg = np.bincount(r, minlength=code.mb * code.z)
        dmax = deg.max()
        # (checks, dmax) edge ids, padded with n_edges (a slot that reads +BIG).
        start = np.concatenate([[0], np.cumsum(deg)[:-1]])
        j = np.arange(dmax)
        idx = start[:, None] + j
        idx[j[None, :] >= deg[:, None]] = self.n_edges
        self.chk = torch.tensor(idx, device=device)
        self.pad = torch.tensor(j[None, :] >= deg[:, None], device=device)
        self.sent = torch.tensor(code.sent, device=device)
        self.filler = torch.arange(code.k, code.kb * code.z, device=device)

    def channel_llrs(self, llr_sent):
        """(B, n) LLRs of transmitted bits -> (B, n_cols) for every column."""
        t = self.torch
        full = t.zeros(llr_sent.shape[0], self.code.n_cols, device=self.device, dtype=llr_sent.dtype)
        full[:, self.sent] = llr_sent
        full[:, self.filler] = self.BIG
        return full

    def decode(self, llr_sent, iters: int = 30, alpha=None, posterior: bool = False):
        """-> (info bit decisions (B, k) uint8, converged (B,) bool), and
        with `posterior` the a-posteriori LLRs of the sent bits (B, n)."""
        t = self.torch
        ch = self.channel_llrs(llr_sent.clamp(-CH_CLAMP, CH_CLAMP))
        b = ch.shape[0]
        c2v = t.zeros(b, self.n_edges, device=self.device, dtype=ch.dtype)
        for it in range(iters):
            a = alpha[it] if isinstance(alpha, t.Tensor) and alpha.dim() else alpha
            tot = ch.index_add(1, self.var, c2v)
            v2c = tot[:, self.var] - c2v
            v2c = t.cat([v2c, t.full((b, 1), self.BIG, device=self.device, dtype=ch.dtype)], 1)
            m = v2c[:, self.chk]  # (B, checks, dmax)
            sgn = t.where(m < 0, -1.0, 1.0).to(ch.dtype)
            mag = m.abs()
            sprod = sgn.prod(dim=-1, keepdim=True)
            if a is None:
                # BP in the log domain: phi(x) = -log tanh(x/2), its own
                # inverse; padding (|m| = BIG) contributes phi = 0.
                ph = _phi(mag)
                out = sprod * sgn * _phi(ph.sum(dim=-1, keepdim=True) - ph)
            else:
                two = mag.topk(2, dim=-1, largest=False)
                min1, min2 = two.values[..., :1], two.values[..., 1:]
                is_min = t.arange(mag.shape[-1], device=self.device) == two.indices[..., :1]
                out = a * sprod * sgn * t.where(is_min, min2, min1)
            out = out.masked_fill(self.pad, 0.0)
            c2v = t.zeros(b, self.n_edges + 1, device=self.device, dtype=ch.dtype)
            c2v.scatter_(1, self.chk.reshape(1, -1).expand(b, -1), out.reshape(b, -1))
            c2v = c2v[:, :-1]
            tot = ch.index_add(1, self.var, c2v)
            hard = (tot < 0).to(t.uint8)
            ok = self.syndrome(hard)
            if bool(ok.all()):
                break
        if posterior:
            return hard[:, : self.code.k], ok, tot[:, self.sent]
        return hard[:, : self.code.k], ok

    def syndrome(self, hard):
        t = self.torch
        h = t.cat([hard[:, self.var], t.zeros(hard.shape[0], 1, dtype=hard.dtype, device=self.device)], 1)
        return (h[:, self.chk].sum(dim=-1) % 2 == 0).all(dim=1)


class SCLDecoder:
    """Batched SCL in torch. decode() returns every list path's info bits
    and path metrics, best first; the caller's CRC picks."""

    def __init__(self, code: PolarCode, list_size: int = 8, device="cpu"):
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
