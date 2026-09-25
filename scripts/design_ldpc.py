"""Protograph search for NR-structured LDPC codes by PEXIT.

Fitness is the protograph EXIT threshold (Liva & Chiani 2007): the least
channel mutual information per coded bit, I_ch, at which every variable
node's APP information goes to 1. Under BICM the channel MI per coded
bit is BMI/m, so a threshold here maps onto the SNR axis through the
constellation's measured BMI curve (runs/constellations.png); lower is
better, and differences transfer as SNR differences via that curve's
slope.

The structure that makes encoding cheap (ldpc.py) is fixed: NR's 4x4
core in rows 0..3 / parity columns kb..kb+3, and one identity per
extension row. What the search moves is which other blocks exist: the
info columns of every row and the core columns of extension rows.
Columns 0 and 1 are punctured, as in NR.

    uv run python scripts/design_ldpc.py --kb 10 --rate 0.5 --pop 60 --gens 200
"""

import argparse

import numpy as np

from data2g import ldpc
from data2g.ldpc import embed, lift  # noqa: F401  (re-exported for callers)

H1, H2, H3 = 0.3073, 0.8935, 1.1064  # Brannstrom et al. J-function fit


def J(s):
    return (1 - 2 ** (-H1 * np.maximum(s, 0) ** (2 * H2))) ** H3


def Jinv(i):
    i = np.clip(i, 0, 1 - 1e-12)
    return (-np.log2(1 - i ** (1 / H3)) / H1) ** (1 / (2 * H2))


def pexit_converges(mask: np.ndarray, punct: np.ndarray, i_ch: float, iters: int = 400) -> bool:
    """mask (M, N) bool protograph (multiplicity 1), punct (N,) bool."""
    s_ch2 = np.where(punct, 0.0, Jinv(i_ch) ** 2)[None, :]
    iav = np.zeros(mask.shape)  # check -> var MI on each edge
    for _ in range(iters):
        a2 = np.where(mask, Jinv(iav) ** 2, 0.0)
        iev = np.where(mask, J(np.sqrt(a2.sum(0, keepdims=True) - a2 + s_ch2)), 0.0)
        b2 = np.where(mask, Jinv(1 - iev) ** 2, 0.0)
        iav = np.where(mask, 1 - J(np.sqrt(np.maximum(b2.sum(1, keepdims=True) - b2, 0))), 0.0)
        app = J(np.sqrt((np.where(mask, Jinv(iav) ** 2, 0.0)).sum(0) + s_ch2[0]))
        if np.all(app > 1 - 1e-5):
            return True
    return False


def threshold(mask: np.ndarray, punct: np.ndarray, tol: float = 1e-3) -> float:
    lo, hi = 0.0, 1.0
    if not pexit_converges(mask, punct, 1 - 1e-6):
        return 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if pexit_converges(mask, punct, mid) else (mid, hi)
    return hi


def nr_mask(bg: int, kb: int, mb: int) -> np.ndarray:
    return ldpc.nr_base_graph(bg, 0)[:mb, : kb + mb] >= 0


def fixed_part(kb: int, mb: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    """(forced value, free) masks: NR's core block and extension
    identities are fixed; everything else in rows is free, and the
    extension rows' own parity columns other than their identity are 0."""
    nr = nr_mask(bg, kb, mb)
    forced = np.zeros_like(nr)
    free = np.zeros_like(nr)
    forced[:4, kb : kb + 4] = nr[:4, kb : kb + 4]
    for r in range(4, mb):
        forced[r, kb + r] = True
    free[:, :kb] = True
    free[4:, kb : kb + 4] = True
    return forced, free


def valid(mask: np.ndarray, kb: int, max_deg2: int) -> bool:
    """Info columns of degree >= 2, and no more of degree 2 than NR has.
    Unconstrained, the search trades exactly those for threshold (rate
    1/2: two more degree-2 info columns, PEXIT gap 0.0225 -> 0.0068) and
    floors at finite length (K=1024: BLER 7.6e-3 at 1.6 dB vs NR 2e-4):
    degree-2 info columns chained through the degree-1 extension
    parities are low-weight codewords."""
    deg = mask.sum(0)[:kb]
    return bool(np.all(deg >= 2) and np.sum(deg == 2) <= max_deg2 and np.all(mask.sum(1) >= 2))


def search(kb, mb, bg, pop, gens, seed, verbose=True):
    rng = np.random.default_rng(seed)
    n = kb + mb
    punct = np.zeros(n, bool)
    punct[:2] = True
    forced, free = fixed_part(kb, mb, bg)
    nr = nr_mask(bg, kb, mb)
    base_thr = threshold(nr, punct)

    max_deg2 = int(np.sum(nr.sum(0)[:kb] == 2))

    def fit(m):
        return threshold(m, punct) if valid(m, kb, max_deg2) else 1.0

    def mutate(m, p=0.03):
        flip = (rng.random(m.shape) < p) & free
        return m ^ flip

    popn = [nr.copy()] + [mutate(nr, 0.08) for _ in range(pop - 1)]
    fits = np.array([fit(m) for m in popn])
    for g in range(gens):
        kids = []
        for _ in range(pop):
            i, j = rng.choice(pop, 2, replace=False), rng.choice(pop, 2, replace=False)
            a = popn[i[np.argmin(fits[i])]]
            b = popn[j[np.argmin(fits[j])]]
            rows = rng.random(mb) < 0.5  # row-wise crossover keeps checks intact
            kid = np.where(rows[:, None], a, b)
            kids.append(mutate(kid) | forced)
        kfit = np.array([fit(m) for m in kids])
        allp, allf = popn + kids, np.concatenate([fits, kfit])
        keep = np.argsort(allf)[:pop]
        popn, fits = [allp[i] for i in keep], allf[keep]
        if verbose and (g % 10 == 0 or g == gens - 1):
            print(f"gen {g}: best I_ch {fits[0]:.4f} (NR {base_thr:.4f}), "
                  f"edges {popn[0].sum()} (NR {nr.sum()})", flush=True)
    return popn[0], fits[0], base_thr


def count_short_cycles(base: np.ndarray, z: int) -> tuple[int, int]:
    """(4-cycles, 6-cycles) in the lifted graph, by the same shift-sum
    test over all block paths (each cycle counted once per start/direction
    it can be walked from, so compare counts, not absolutes)."""
    pos = np.argwhere(base >= 0)
    n4 = n6 = 0
    for r, c in pos:
        for r2 in np.flatnonzero(base[:, c] >= 0):
            if r2 == r:
                continue
            for c2 in np.flatnonzero(base[r2] >= 0):
                if c2 == c:
                    continue
                v = base[r, c] - base[r2, c] + base[r2, c2]
                if base[r, c2] >= 0 and (v - base[r, c2]) % z == 0:
                    n4 += 1
                for r3 in np.flatnonzero(base[:, c2] >= 0):
                    if r3 in (r, r2):
                        continue
                    for c3 in np.flatnonzero(base[r3] >= 0):
                        if c3 in (c, c2) or base[r, c3] < 0:
                            continue
                        if (v - base[r3, c2] + base[r3, c3] - base[r, c3]) % z == 0:
                            n6 += 1
    return n4, n6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=int, default=10)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--bg", type=int, default=2)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    mb = int(round(a.kb / a.rate - a.kb + 2))
    rate = a.kb / (a.kb + mb - 2)
    print(f"kb={a.kb} mb={mb}: design rate {rate:.3f} (capacity limit I_ch = {rate:.3f})", flush=True)
    best, thr, base = search(a.kb, mb, a.bg, a.pop, a.gens, a.seed)
    print(f"\nNR BG{a.bg} I_ch threshold {base:.4f}, searched {thr:.4f}, "
          f"gap to capacity {base - rate:.4f} -> {thr - rate:.4f}")
    if a.out:
        np.save(a.out, best)


if __name__ == "__main__":
    main()
