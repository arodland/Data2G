"""A non-periodic front for c8r50 (cpm.FRONTS, protocol 13). Protocol 12's
was a 6-symbol Costas array four times, so one period off still matched 18
of 24 symbols (sync ~0.76), and a search range ending before a burst's start
locked a period early and read another header (scripts/cpm_supersede_study.py:
24 of 223 fsk8r50 locks).

24 symbols over 8 tones, each tone 3 times (a flat spectrum; it also fixes
the bound: 24 equal-tone pairs over 23 positive shifts, so some shift has
2 coincidences). A candidate's cost is its worst score (coincidences over the template's
length) over
every time shift and whole-bin shift the detector searches (+-3 bins:
+-150 Hz at 50 Hz), (0, 0) excepted, against what a front search can
meet: the front followed by each valid header word (copy 1); the mid block
followed by each word (copy 2); and the mid detector over front + header.
Ties broken by how many shifts reach the worst, then the sum of squares.
Simulated annealing over swaps (tone counts kept), random restarts.

    uv run python scripts/c8r50_front_search.py --workers 4 --seconds 600 --out runs/c8r50_front.json
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

from data2g import cpm

G = cpm.GRIDS["c8r50"]
N, M, DK = 24, 8, 3
WORDS = np.array([w[3] for w in cpm.valid_words("c8r50")])  # (36, 12) header tones
MID = np.array(cpm.mid_block(G))  # 12, fixed (the mid detector's template)


def _shift_counts(tmpl: np.ndarray, seqs: np.ndarray, skip_zero: bool) -> np.ndarray:
    """Per (dk, tau): the most coincidences tmpl[r] + dk == seq[r + tau] over
    the rows both cover, worst over seqs (rows outside a seq: noise, none)."""
    n, L = len(tmpl), seqs.shape[1]
    taus = np.arange(-(n - 1), L)
    idx = np.arange(n)[:, None] + taus[None, :]  # (n, taus)
    ok = (idx >= 0) & (idx < L)
    got = seqs[:, np.clip(idx, 0, L - 1)]  # (S, n, taus)
    out = np.empty((2 * DK + 1, len(taus)), int)
    for j, dk in enumerate(range(-DK, DK + 1)):
        hit = (got == (tmpl[:, None] + dk)[None]) & ok[None]
        out[j] = hit.sum(axis=1).max(axis=0)
    if skip_zero:
        out[DK, n - 1] = 0  # (dk 0, tau 0): the pattern itself
    return out


def cost(p: np.ndarray) -> tuple:
    """(worst score, shifts at it, sum of squared scores), a score being a
    shift's coincidences over its template's length (the detector's mean
    share, on a clean signal): the front over itself and the header copy
    after it, the front over the mid block and header copy 2, and the mid
    template over the front (the mid template over a header is the front's
    business no more)."""
    front = np.concatenate([np.repeat(p[None], len(WORDS), 0), WORDS], axis=1)
    mid = np.concatenate([np.repeat(MID[None], len(WORDS), 0), WORDS], axis=1)
    c = np.concatenate([_shift_counts(p, front, True).ravel() / N, _shift_counts(p, mid, False).ravel() / N,
                        _shift_counts(MID, p[None], False).ravel() / len(MID)])
    worst = float(c.max())
    return round(worst, 4), int(np.isclose(c, worst).sum()), round(float((c ** 2).sum()), 4)


def anneal(args):
    seed, seconds = args
    rng = np.random.default_rng(seed)
    best = None
    t_end = time.time() + seconds
    while time.time() < t_end:
        p = rng.permutation(np.repeat(np.arange(M), N // M))
        c, temp = cost(p), 2.0
        for _ in range(4000):
            i, j = rng.integers(N, size=2)
            if p[i] == p[j]:
                continue
            q = p.copy()
            q[i], q[j] = q[j], q[i]
            cq = cost(q)
            d = (cq[0] - c[0]) * 2400 + (cq[1] - c[1]) + (cq[2] - c[2]) * 10
            if d <= 0 or rng.random() < np.exp(-d / temp):
                p, c = q, cq
            temp *= 0.999
            if best is None or c < best[0]:
                best = (c, p.tolist())
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--out", default="runs/c8r50_front.json")
    a = ap.parse_args()
    old = np.array(cpm.preamble_pattern(G))
    print("current front:", old.tolist(), "cost", cost(old))
    with Pool(a.workers) as pool:
        res = sorted(pool.map(anneal, [(s, a.seconds) for s in range(a.workers)]))
    for c, p in res:
        print("cost", c, p)
    json.dump([{"cost": list(c), "pattern": p} for c, p in res], open(a.out, "w"), indent=1)
