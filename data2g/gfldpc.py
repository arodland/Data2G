"""Nonbinary LDPC over GF(2^p), decoded on symbol likelihoods: the CPM
modes' data code (one code symbol per M-FSK tone, M = 2^p), which sees
each tone's likelihoods whole instead of their bit marginals (BICM).

Construction (design time, `design`): a Tanner graph by progressive edge
growth (Hu, Eleftheriou & Arnold 2005) for a degree profile of mostly
degree-2 columns and a few heavy ones (Davey & MacKay 1998; Poulliat,
Fossorier & Declercq 2008: ultra-sparse codes over high-order fields),
coefficients pseudo-random non-zero, systematic encoding through H's
reduced row-echelon form. The profile shape (degree 2 plus a few heavy
columns) is ion-hf's GF(32) code's (Jesse Forkum, N0JLF; ideas adapted,
relicensed for Data2G by the author); the graphs and tables here are ours.

Decoder: belief propagation on probability vectors, the check node's
convolution in the Walsh-Hadamard domain (FFT-BP: Davey & MacKay 1998;
Barnault & Declercq 2003), flooding, early stop once every check holds.
"""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import hadamard

POLY = {3: 0b1011, 4: 0b10011, 5: 0b100101}  # x^3+x+1, x^4+x+1, x^5+x^2+1


class GF:
    """GF(2^p) by exp/log tables; mul (q, q), inv (q,)."""

    def __init__(self, p: int):
        self.p, self.q = p, 1 << p
        q = self.q
        exp = np.zeros(2 * q, np.int64)
        log = np.zeros(q, np.int64)
        x = 1
        for i in range(q - 1):
            exp[i] = x
            log[x] = i
            x <<= 1
            if x & q:
                x ^= POLY[p]
        exp[q - 1:2 * q - 2] = exp[:q - 1]
        a = np.arange(q)
        self.mul = np.where((a[:, None] == 0) | (a[None, :] == 0), 0, exp[(log[:, None] + log[None, :]) % (q - 1)])
        self.inv = np.zeros(q, np.int64)
        self.inv[1:] = exp[(q - 1 - log[1:]) % (q - 1)]


@dataclass(frozen=True, eq=False)
class GFCode:
    """H as rows: chk_var (m, dmax) variable per slot (-1 padding, at the
    end), chk_h (m, dmax) coefficients (0 padding). The core code is the
    first n0 symbols and the rows within them; every other row is an
    extension row (extend): one symbol of its own past n0, in order. k:
    information symbols (positions `info`, the rest of the core `parity`,
    by the core's echelon form: parity[i] = sum_j A[i, j] info[j])."""
    p: int
    n: int
    chk_var: np.ndarray
    chk_h: np.ndarray
    n0: int = 0  # 0: n

    @property
    def core_n(self) -> int:
        return self.n0 or self.n

    @cached_property
    def _rows(self):
        """(core rows, extension rows sorted by their own symbol, those symbols)."""
        top = np.where(self.chk_var < 0, -1, self.chk_var).max(axis=1)
        core = np.flatnonzero(top < self.core_n)
        ext = np.flatnonzero(top >= self.core_n)
        ext = ext[np.argsort(top[ext])]
        assert (top[ext] == np.arange(self.core_n, self.n)).all(), "an extension row per symbol past the core"
        return core, ext, top[ext]

    @cached_property
    def gf(self) -> GF:
        return GF(self.p)

    @property
    def q(self) -> int:
        return 1 << self.p

    @property
    def m(self) -> int:
        return len(self.chk_var)

    @cached_property
    def _echelon(self):
        gf, n = self.gf, self.core_n
        rows = self._rows[0]
        m = len(rows)
        h = np.zeros((m, n), np.int64)
        for r, row in enumerate(rows):
            for v, c in zip(self.chk_var[row], self.chk_h[row]):
                if v >= 0:
                    h[r, v] ^= c
        piv, r = [], 0
        for col in range(n - 1, -1, -1):  # pivots from the last column: information first
            if r == m:
                break
            nz = np.flatnonzero(h[r:, col])
            if not len(nz):
                continue
            h[[r, r + nz[0]]] = h[[r + nz[0], r]]
            h[r] = gf.mul[gf.inv[h[r, col]], h[r]]
            for i in np.flatnonzero(h[:, col]):
                if i != r:
                    h[i] ^= gf.mul[h[i, col], h[r]]
            piv.append(col)
            r += 1
        used = set(piv)
        info = np.array([c for c in range(n) if c not in used])
        return info, np.array(piv), h[:r][:, info]

    @property
    def info(self) -> np.ndarray:
        return self._echelon[0]

    @property
    def parity(self) -> np.ndarray:
        return self._echelon[1]

    @property
    def k(self) -> int:
        return len(self.info)

    def encode(self, u: np.ndarray) -> np.ndarray:
        """(B, k) information symbols -> (B, n) codewords."""
        u = np.atleast_2d(u)
        gf, A = self.gf, self._echelon[2]
        x = np.zeros((len(u), self.n), np.int64)
        x[:, self.info] = u
        # parity_i = sum_j A[i, j] u_j (char 2: minus is plus)
        prod = gf.mul[A[None], u[:, None, :]]  # (B, r, k)
        x[:, self.parity] = np.bitwise_xor.reduce(prod, axis=2)
        _, ext, own = self._rows
        for row, v in zip(ext, own):  # each extension symbol from its row, in order
            vs, hs = self.chk_var[row], self.chk_h[row]
            j = np.flatnonzero(vs == v)[0]
            sel = (vs >= 0) & (vs != v)
            acc = np.bitwise_xor.reduce(gf.mul[hs[sel][None], x[:, vs[sel]]], axis=1)
            x[:, v] = gf.mul[gf.inv[hs[j]], acc]
        return x

    def syndrome_ok(self, x: np.ndarray) -> np.ndarray:
        """(B, n) symbols -> (B,) every check holds."""
        xv = np.concatenate([x, np.zeros((len(x), 1), x.dtype)], 1)[:, self.chk_var]  # pad reads 0
        return (np.bitwise_xor.reduce(self.gf.mul[self.chk_h[None], xv], axis=2) == 0).all(axis=1)


def peg(n: int, m: int, degs, rng) -> list:
    """Progressive edge growth: each column's edges, lowest degrees first;
    each new edge to a check unreached from the column (else the deepest),
    the least loaded, ties at random. -> per check, its columns."""
    chk = [[] for _ in range(m)]
    var = [[] for _ in range(n)]
    load = np.zeros(m, int)

    def pick(cands):
        cands = np.asarray(sorted(cands))
        lo = load[cands].min()
        best = cands[load[cands] == lo]
        return int(best[rng.integers(len(best))])

    for v in sorted(range(n), key=lambda v: (degs[v], v)):
        for k in range(degs[v]):
            if k == 0:
                c = pick([c for c in range(m)])
            else:
                seen_c, seen_v, front = set(), {v}, [v]
                while True:
                    nc = {c for u in front for c in var[u]} - seen_c
                    seen_c |= nc
                    unreached = set(range(m)) - seen_c
                    if not nc:  # nothing new: anything unreached
                        c = pick(unreached - set(var[v]) or set(range(m)) - set(var[v]))
                        break
                    if not unreached:  # everything reachable: the newest layer
                        c = pick((nc - set(var[v])) or set(range(m)) - set(var[v]))
                        break
                    nv = {u for c in nc for u in chk[c]} - seen_v
                    seen_v |= nv
                    if not nv:
                        c = pick(unreached)
                        break
                    front = list(nv)
            chk[c].append(v)
            var[v].append(c)
            load[c] += 1
    return chk


def design(p: int, n: int, k: int, heavy: tuple, seed: int) -> GFCode:
    """A code of n symbols carrying (at least) k: n - k checks, columns of
    degree 2 but `heavy` ((count, degree), ...)."""
    rng = np.random.default_rng(seed)
    degs = [d for c, d in heavy for _ in range(c)]
    degs += [2] * (n - len(degs))
    rows = peg(n, n - k, degs, rng)
    dmax = max(len(r) for r in rows)
    cv = np.full((len(rows), dmax), -1, np.int64)
    ch = np.zeros((len(rows), dmax), np.int64)
    for i, r in enumerate(rows):
        cv[i, :len(r)] = r
        ch[i, :len(r)] = rng.integers(1, 1 << p, len(r))
    return GFCode(p, n, cv, ch)


class Decoder:
    """Batched FFT-BP over a GFCode. Channel input: (B, n, q) symbol
    log-likelihoods (any per-symbol offset)."""

    TINY = 1e-30

    def __init__(self, code: GFCode):
        self.code = code
        gf, q = code.gf, code.q
        self.had = hadamard(q).astype(np.float64)
        pad = code.chk_var < 0
        self.pad = pad
        hv = np.where(pad, 1, code.chk_h)
        # to the check domain: y = h x, P_y[y] = P_x[h^-1 y]; back: P_x[a] = R[h a]
        self.to_chk = gf.mul[gf.inv[hv]][..., :]  # (m, d, q): index h^-1 * y
        self.to_var = gf.mul[hv]  # (m, d, q): index h * a
        self.ev = np.where(pad, 0, code.chk_var)
        # per variable, its edges' flat ids (pad: the zero slot m * d)
        md = pad.size
        cols = [[] for _ in range(code.n)]
        for e, v in enumerate(code.chk_var.reshape(-1)):
            if v >= 0:
                cols[v].append(e)
        dv = max(len(c) for c in cols)
        self.col = np.full((code.n, dv), md, np.int64)
        for v, c in enumerate(cols):
            self.col[v, :len(c)] = c

    def decode(self, ll: np.ndarray, iters: int = 50):
        """-> (hard symbols (B, n), converged (B,), posterior log-probs (B, n, q))."""
        code, q = self.code, self.code.q
        ll = np.asarray(ll, np.float64)
        ll = ll - ll.max(axis=2, keepdims=True)
        B = len(ll)
        m, d = self.pad.shape
        c2v = np.zeros((B, m * d + 1, q))
        post = ll.copy()
        hard = post.argmax(axis=2)
        done = code.syndrome_ok(hard)
        live = np.flatnonzero(~done)
        delta = np.zeros(q)
        delta[0] = 1.0
        for _ in range(iters):
            if not len(live):
                break
            cl = c2v[live, :-1].reshape(len(live), m, d, q)
            v2c = post[live][:, self.ev] - cl  # (L, m, d, q)
            v2c = np.exp(v2c - v2c.max(axis=3, keepdims=True))
            v2c /= v2c.sum(axis=3, keepdims=True)
            pc = np.take_along_axis(v2c, np.broadcast_to(self.to_chk, v2c.shape), 3)
            pc[:, self.pad] = delta
            W = pc @ self.had
            fwd = np.cumprod(W, axis=2)
            bwd = np.cumprod(W[:, :, ::-1], axis=2)[:, :, ::-1]
            ex = np.ones_like(W)
            ex[:, :, 1:] *= fwd[:, :, :-1]
            ex[:, :, :-1] *= bwd[:, :, 1:]
            R = (ex @ self.had) / q
            pv = np.take_along_axis(R, np.broadcast_to(self.to_var, R.shape), 3)
            new = np.log(np.maximum(pv, self.TINY))
            new -= new.max(axis=3, keepdims=True)
            new[:, self.pad] = 0.0
            c2v[live, :-1] = new.reshape(len(live), m * d, q)
            post[live] = ll[live] + c2v[live][:, self.col].sum(axis=2)
            hard[live] = post[live].argmax(axis=2)
            ok = code.syndrome_ok(hard[live])
            done[live[ok]] = True
            live = live[~ok]
        return hard, done, post - post.max(axis=2, keepdims=True)


def extend(code: GFCode, n_ext: int, d: int, seed: int) -> GFCode:
    """Rate-compatible extension (raptor-like): n_ext more symbols, each
    in a row of its own with `d` of the code's symbols, the least used
    first, ties at random. The code is the first n symbols and m rows."""
    rng = np.random.default_rng(seed)
    use = np.zeros(code.n, int)
    dmax = max(code.chk_var.shape[1], d + 1)
    cv = np.full((code.m + n_ext, dmax), -1, np.int64)
    ch = np.zeros_like(cv)
    cv[:code.m, :code.chk_var.shape[1]] = code.chk_var
    ch[:code.m, :code.chk_var.shape[1]] = code.chk_h
    for j in range(n_ext):
        pick = []
        for _ in range(d):
            free = np.setdiff1d(np.arange(code.n), pick)
            lo = use[free].min()
            c = rng.choice(free[use[free] == lo])
            pick.append(int(c))
            use[c] += 1
        cv[code.m + j, :d + 1] = pick + [code.n + j]
        ch[code.m + j, :d + 1] = rng.integers(1, code.q, d + 1)
    return GFCode(code.p, code.n + n_ext, cv, ch, code.n)


def cut(code: GFCode, n: int) -> GFCode:
    """The code on its first n symbols: the rows that reach no further."""
    keep = np.flatnonzero(np.where(code.chk_var < 0, 0, code.chk_var).max(axis=1) < n)
    cv, ch = code.chk_var[keep], code.chk_h[keep]
    w = int((cv >= 0).sum(axis=1).max())
    return GFCode(code.p, n, cv[:, :w], ch[:, :w], min(code.core_n, n))


def pack(bits: np.ndarray, p: int, k: int) -> np.ndarray:
    """(B, nbits) bits -> (B, k) symbols, p bits each MSB first, zero filled."""
    bits = np.atleast_2d(bits)
    b = np.zeros((len(bits), k * p), np.int64)
    b[:, :bits.shape[1]] = bits
    return b.reshape(len(bits), k, p) @ (1 << np.arange(p)[::-1])


def unpack(x: np.ndarray, p: int) -> np.ndarray:
    """(B, n) symbols -> (B, n p) bits, MSB first."""
    x = np.asarray(x)
    return ((x[..., None] >> np.arange(p)[::-1]) & 1).reshape(len(x), -1).astype(np.uint8)


TABLES = Path(__file__).parent / "codes_data" / "gf_codes.npz"


def key(name: str) -> str:
    return name.replace("/", "_")


@lru_cache(maxsize=None)
def load(name: str) -> GFCode:
    """A spec's frozen mother code (scripts/gf_study.py write)."""
    with np.load(TABLES) as d:
        k = key(name)
        return GFCode(int(d[f"{k}_p"]), int(d[f"{k}_n"]), d[f"{k}_var"].astype(np.int64), d[f"{k}_h"].astype(np.int64),
                      int(d[f"{k}_n0"]))
