"""Quasi-cyclic LDPC on two base graphs, any (K, N).

A base matrix holds, per (row, column), a circulant shift or -1. Base
graph 1 is (46, 68) with 22 info columns, base graph 2 (42, 52) with 10.
The block masks and one shift table per lifting size Z a submode uses
are in codes_data/ldpc_shifts.npz (scripts/own_shifts.py generated and
screened them). Rows 0..3 and parity columns kb..kb+3 form a
dual-diagonal core (as in 802.11n/802.16e), which is invertible and fixes
the first 4Z parities; every later row r adds one parity column kb+r
with an identity. That makes systematic encoding cheap.

Rate matching: the first two info columns are punctured (never sent), K
below kb*Z is padded with filler zeros (known to the decoder, never
sent), and only as many base rows are kept as the N transmitted bits
reach; H is truncated to them.

Circulant convention: row i of a block with shift s connects to column
(i + s) mod Z.

Encoder and decoder: numpy. Decoder: batched BP, or normalized min-sum
with an optional per-iteration normalization (neural min-sum);
decoders_torch has it on torch for GPU studies.
"""

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "codes_data"
SHIFTS = DATA / "ldpc_shifts.npz"
CORE = 4
KB = {1: 22, 2: 10}  # info columns per base graph

# Lifting sizes: a * 2^j up to 384, a in {2, 3, 5, ..., 15}.
LIFTING_SIZES = sorted({a << j for a in (2, 3, 5, 7, 9, 11, 13, 15) for j in range(8) if a << j <= 384})


@lru_cache(maxsize=None)
def tables() -> dict[str, np.ndarray]:
    """"mask_bg1", "mask_bg2" (bool) and "bg<b>_z<Z>" shift tables."""
    with np.load(SHIFTS) as d:
        return {key: d[key].astype(bool if key.startswith("mask") else np.int64) for key in d.files}


def mask(bg: int) -> np.ndarray:
    return tables()[f"mask_bg{bg}"].copy()


def layout(k: int, n: int, bg: int | None = None) -> tuple[int, int]:
    """(base graph, Z) for k bits in n: graph 2 for short blocks and low
    rates, and the smallest lifting size whose (graph 2: k-dependent)
    info columns hold k."""
    rate = k / n
    if bg is None:
        bg = 2 if (k <= 292 or (k <= 3824 and rate <= 0.67) or rate <= 0.25) else 1
    kb_z = 22 if bg == 1 else 10 if k > 640 else 9 if k > 560 else 8 if k > 192 else 6
    return bg, next(z for z in LIFTING_SIZES if kb_z * z >= k)


def qc_code(k: int, n: int, bg: int | None = None) -> "QCLDPC":
    """The code carrying k bits in n, on its shift table."""
    bg, z = layout(k, n, bg)
    try:
        base = tables()[f"bg{bg}_z{z}"].copy()
    except KeyError:
        raise KeyError(f"no shift table for base graph {bg}, Z={z} (k={k}): "
                       "add one with scripts/own_shifts.py (gen, screen, PICKS, export)") from None
    return QCLDPC(base=base, z=z, kb=KB[bg], k=k, n=n)


def _gf2_inv(a: np.ndarray) -> np.ndarray:
    n = len(a)
    m = np.concatenate([a.astype(bool), np.eye(n, dtype=bool)], axis=1)
    for c in range(n):
        piv = c + np.flatnonzero(m[c:, c])
        if len(piv) == 0:
            raise ValueError("core not invertible")
        m[[c, piv[0]]] = m[[piv[0], c]]
        rows = np.flatnonzero(m[:, c])
        rows = rows[rows != c]
        m[rows] ^= m[c]
    return m[:, n:].astype(np.uint8)


@dataclass
class QCLDPC:
    base: np.ndarray  # (mb, nb) shifts, -1 = zero block
    z: int
    kb: int  # info columns in the base matrix
    k: int  # info bits actually carried (<= kb*z; the rest are filler)
    n: int  # transmitted bits
    mb: int = field(init=False)  # base rows kept for this n
    full_base: np.ndarray = field(init=False, repr=False)  # every row, before truncation (mother())

    def __post_init__(self):
        self.full_base = self.base
        if not 0 < self.k <= self.kb * self.z:
            raise ValueError(f"k={self.k} does not fit kb*z={self.kb * self.z}")
        # Transmitted: columns from 2z on, fillers skipped. Keep rows until
        # their parity columns cover n.
        sent_info = self.k - 2 * self.z
        need_parity = self.n - sent_info
        if need_parity <= 0:
            raise ValueError("n too small: not even the info bits fit")
        self.mb = max(CORE, -(-need_parity // self.z))
        if self.mb > self.base.shape[0]:
            raise ValueError(f"rate {self.k}/{self.n} is below this base graph's mother rate")
        self.base = self.base[: self.mb, : self.kb + self.mb]

    def mother(self, n: int | None = None) -> "QCLDPC":
        """The same code with more of its base rows: n transmitted bits
        (default all: the whole circular buffer, docs/arq.md §5). Its first
        self.n sent bits are this code's codeword (the extension rows each
        add one parity column and leave earlier columns alone), which is
        what makes incremental redundancy possible without a format change."""
        n_all = (self.full_base.shape[1] - 2) * self.z - (self.kb * self.z - self.k)
        return QCLDPC(base=self.full_base, z=self.z, kb=self.kb, k=self.k, n=min(n or n_all, n_all))

    @property
    def n_cols(self) -> int:
        return (self.kb + self.mb) * self.z

    @cached_property
    def sent(self) -> np.ndarray:
        """Codeword positions transmitted, in order (length n)."""
        cols = np.arange(2 * self.z, self.n_cols)
        cols = cols[(cols < self.k) | (cols >= self.kb * self.z)]
        return cols[: self.n]

    @cached_property
    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        """(check index, variable index) for every 1 in H."""
        r, c = np.nonzero(self.base >= 0)
        i = np.arange(self.z)
        rows = (r[:, None] * self.z + i).reshape(-1)
        cols = (c[:, None] * self.z + (i + self.base[r, c][:, None]) % self.z).reshape(-1)
        return rows, cols

    # --- encoding -----------------------------------------------------------
    def _blocks_mul(self, blk: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Lifted `blk` (base sub-matrix) times x (B, ncols_blk, z) mod 2."""
        out = np.zeros((x.shape[0], blk.shape[0], self.z), dtype=np.uint8)
        i = np.arange(self.z)
        for r, c in zip(*np.nonzero(blk >= 0)):
            out[:, r] ^= x[:, c, (i + blk[r, c]) % self.z]
        return out

    @cached_property
    def _core_inv(self) -> np.ndarray:
        core = self.base[:CORE, self.kb : self.kb + CORE]
        dense = np.zeros((CORE * self.z, CORE * self.z), dtype=np.uint8)
        i = np.arange(self.z)
        for r, c in zip(*np.nonzero(core >= 0)):
            dense[r * self.z + i, c * self.z + (i + core[r, c]) % self.z] = 1
        return _gf2_inv(dense)

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """(B, k) info bits -> (B, n) transmitted bits."""
        return self.encode_full(bits)[:, self.sent]

    def encode_full(self, bits: np.ndarray) -> np.ndarray:
        """(B, k) info bits -> (B, n_cols): every column, punctured,
        filler and untransmitted parity included."""
        bits = np.atleast_2d(bits).astype(np.uint8)
        b = bits.shape[0]
        s = np.zeros((b, self.kb * self.z), dtype=np.uint8)
        s[:, : self.k] = bits
        s3 = s.reshape(b, self.kb, self.z)
        a_s = self._blocks_mul(self.base[:CORE, : self.kb], s3).reshape(b, -1)
        p_a = (a_s.astype(np.int32) @ self._core_inv.T.astype(np.int32)) % 2
        known = np.concatenate([s3, p_a.astype(np.uint8).reshape(b, CORE, self.z)], axis=1)
        p_b = self._blocks_mul(self.base[CORE:, : self.kb + CORE], known)
        return np.concatenate([s, p_a.astype(np.uint8), p_b.reshape(b, -1)], axis=1)

    def syndrome_ok(self, cw_full: np.ndarray) -> np.ndarray:
        """(B, n_cols) full codewords -> (B,) all checks satisfied."""
        r, c = self.edges
        syn = np.zeros((cw_full.shape[0], self.mb * self.z), dtype=np.uint8)
        np.add.at(syn.T, r, cw_full[:, c].T)
        return np.all(syn % 2 == 0, axis=1)


# Channel LLR magnitude cap at the decoder input: below the largest message
# a check can send (~16.8, _phi's floor in float32), so no channel value
# outvotes every check. At 50, combined IR buffers whose info bits were
# already right never satisfied H: 256-QAM r5/8 at 28 dB AWGN fell from 0.99
# decoded to 0.70 over five transmissions, 1.00 throughout at 16; unchanged at
# threshold (scripts/llr_clamp_study.py, runs/llr_clamp_study.csv).
CH_CLAMP = 16.0


def _phi(x):
    x = np.clip(x, 1e-7, 30.0)
    return -np.log(np.tanh(x / 2))


class MinSumDecoder:
    """Batched flooding decoder on a QCLDPC, in numpy: sum-product (BP,
    `alpha=None`, the default) or normalized min-sum with `alpha`.

    BP is the default because this modem's bit rate (a few kb/s) makes
    its cost irrelevant even on a phone; min-sum exists for hardware.
    Stops early once every codeword in the batch satisfies H.
    decoders_torch.MinSumDecoder is the same on torch, for GPU studies.
    """

    BIG = 1e4  # LLR for filler bits: known zeros

    def __init__(self, code: QCLDPC):
        import scipy.sparse

        self.code = code
        r, c = code.edges
        order = np.argsort(r, kind="stable")
        r, c = r[order], c[order]
        self.n_edges = len(r)
        self.var = c
        deg = np.bincount(r, minlength=code.mb * code.z)
        dmax = deg.max()
        # (checks, dmax) edge ids, padded with n_edges (a slot that reads +BIG).
        # Edges are in check order, so out[:, ~pad] lists them in edge order.
        start = np.concatenate([[0], np.cumsum(deg)[:-1]])
        j = np.arange(dmax)
        self.pad = j[None, :] >= deg[:, None]
        self.chk = start[:, None] + j
        self.chk[self.pad] = self.n_edges
        # (n_cols, n_edges): sums each variable's incoming messages
        self.gather = scipy.sparse.csr_matrix(
            (np.ones(len(c), np.float32), (c, np.arange(len(c)))), shape=(code.n_cols, len(c)))
        self.sent = code.sent
        self.filler = np.arange(code.k, code.kb * code.z)

    def channel_llrs(self, llr_sent):
        """(B, n) LLRs of transmitted bits -> (B, n_cols) for every column."""
        full = np.zeros((llr_sent.shape[0], self.code.n_cols), np.float32)
        full[:, self.sent] = llr_sent
        full[:, self.filler] = self.BIG
        return full

    def decode(self, llr_sent, iters: int = 30, alpha=None, posterior: bool = False):
        """-> (info bit decisions (B, k) uint8, converged (B,) bool), and
        with `posterior` the a-posteriori LLRs of the sent bits (B, n)."""
        ch = self.channel_llrs(np.clip(np.asarray(llr_sent, np.float32), -CH_CLAMP, CH_CLAMP))
        b = ch.shape[0]
        c2v = np.zeros((b, self.n_edges), np.float32)
        v2c = np.full((b, self.n_edges + 1), self.BIG, np.float32)
        for it in range(iters):
            a = alpha[it] if np.ndim(alpha) else alpha
            tot = ch + (self.gather @ c2v.T).T
            v2c[:, :-1] = tot[:, self.var] - c2v
            m = v2c[:, self.chk]  # (B, checks, dmax)
            neg = m < 0
            sgn = np.where(neg, np.float32(-1), np.float32(1))
            sprod = np.where(np.logical_xor.reduce(neg, axis=-1, keepdims=True), np.float32(-1), np.float32(1))
            mag = np.abs(m)
            if a is None:
                # BP in the log domain: phi(x) = -log tanh(x/2), its own
                # inverse; padding (|m| = BIG) contributes phi = 0.
                ph = _phi(mag)
                out = sprod * sgn * _phi(ph.sum(axis=-1, keepdims=True) - ph)
            else:
                i1 = mag.argmin(axis=-1)[..., None]
                min1 = np.take_along_axis(mag, i1, -1)
                np.put_along_axis(mag, i1, np.inf, -1)
                min2 = mag.min(axis=-1, keepdims=True)
                out = np.float32(a) * sprod * sgn * np.where(np.arange(mag.shape[-1]) == i1, min2, min1)
            c2v = out[:, ~self.pad]
            tot = ch + (self.gather @ c2v.T).T
            hard = (tot < 0).astype(np.uint8)
            ok = self.syndrome(hard)
            if ok.all():
                break
        if posterior:
            return hard[:, : self.code.k], ok, tot[:, self.sent]
        return hard[:, : self.code.k], ok

    def syndrome(self, hard):
        h = np.concatenate([hard[:, self.var], np.zeros((hard.shape[0], 1), hard.dtype)], 1)
        return (h[:, self.chk].sum(axis=-1) % 2 == 0).all(axis=1)
