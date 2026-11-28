"""Pilot placement by a shared sequence (round 4): a full ordering of a
frame's (symbol, carrier) cells whose first n terms are the pilots at density
n / cells, every lower density a subset of every higher one, against the
staggered lattices of scripts/fast_fading_study.py (kKdD: every K-th carrier
of every D-th symbol) at the same density.

Orderings (all deterministic; a collision after quantising to cells takes
the sequence's next term):
  halton23 / halton32   2-D Halton, base 2 on symbols and 3 on carriers / the reverse
  sobol                 2-D Sobol, unscrambled
  r2                    the plastic-number Kronecker sequence
  gold-k / gold-s       i g mod N on the cells flattened carrier-major / symbol-major, g ~ N / golden ratio,
                        coprime to N (a rank-1 lattice taken in order: no collisions)
  vdc-k / vdc-s         the flattened cells by the base-2 radical inverse of their index
  maximin-A             farthest-point order on the frame's torus, a symbol A carriers long (ties: lowest cell)
Per frame the sequence restarts (`restart`: one pilot set, period 6 symbols),
or runs on (`run`: frame f takes terms f n .. (f + 1) n - 1 of the ordering,
wrapping: every cell is a pilot once every cells / gcd(n, cells) frames).

    python -m scripts.pilot_seq_study --grids            # the orderings' geometry on 24x6, 48x6, 10x6, 4x6
    python -m scripts.pilot_seq_study --out runs/sc4_screen.csv   # the screen (w band, QPSK, PEP 5 cells)
"""

from data2g import threads

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from functools import lru_cache  # noqa: E402
from math import gcd  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from data2g import constellation  # noqa: E402
from data2g.config import FS  # noqa: E402
from data2g.waveform import ofdm  # noqa: E402
from data2g.waveform.dsp import tx_condition  # noqa: E402
from scripts.fast_fading_study import (  # noqa: E402
    BACKOFF, BLOCK_S, CHANNELS, PAD, Tally, bit_info, cnoise, delay_corr, fade, peak_db, sigma)

S = 6  # symbols a frame
PILOT = ofdm.band("w").pilot
SPREAD_X = 1.25  # the receiver's spread over the true one (modem.SC_WIDEN); its delay support is 0.5 ms long
DENSITIES = {"1/12": 1 / 12, "1/8": 1 / 8, "1/6": 1 / 6, "1/4": 1 / 4, "1/3": 1 / 3, "1/2": 1 / 2}
LATTICES = {"1/12": [(6, 2), (4, 3), (12, 1), (2, 6)], "1/8": [(4, 2), (8, 1)], "1/6": [(3, 2), (6, 1), (2, 3), (1, 6)],
            "1/4": [(4, 1), (2, 2)], "1/3": [(3, 1), (1, 3)], "1/2": [(2, 1), (1, 2)]}
SEQS = ("halton23", "halton32", "sobol", "r2", "gold-k", "gold-s", "vdc-k", "vdc-s", "maximin-0.7", "maximin-1.5")


def _radinv(i: int, b: int) -> float:
    x, f = 0.0, 1.0 / b
    while i:
        x += f * (i % b)
        i //= b
        f /= b
    return x


def _points(name: str, n: int) -> np.ndarray:
    """n points of a 2-D sequence in [0, 1)^2: (symbol axis, carrier axis)."""
    i = np.arange(1, n + 1)
    if name.startswith("halton"):
        a, b = int(name[6]), int(name[7])
        return np.array([[_radinv(k, a), _radinv(k, b)] for k in i])
    if name == "sobol":
        from scipy.stats import qmc

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return qmc.Sobol(2, scramble=False).random(n)
    g = 1.32471795724474602596  # the plastic number
    return np.stack([(0.5 + i / g) % 1, (0.5 + i / g**2) % 1], axis=1)


@lru_cache(maxsize=None)
def ordering(name: str, nc: int) -> tuple:
    """Every cell of an S x nc frame once, as flat indices s * nc + k, in the sequence's order."""
    N = S * nc
    if name.startswith(("gold", "vdc")):
        flat = (lambda j: (j % S) * nc + j // S) if name.endswith("-k") else (lambda j: j)  # carrier-major | symbol-major
        if name.startswith("gold"):
            g = next(v for d in range(N) for v in (round(N / 1.6180339887) + d, round(N / 1.6180339887) - d) if gcd(v, N) == 1)
            return tuple(flat(i * g % N) for i in range(N))
        return tuple(flat(j) for j in sorted(range(N), key=lambda j: (_radinv(j, 2), j)))
    if name.startswith("maximin"):
        a = float(name[8:])
        s, k = np.divmod(np.arange(N), nc)
        ds = np.abs(s[:, None] - s[None, :])
        dk = np.abs(k[:, None] - k[None, :])
        d2 = (np.minimum(ds, S - ds) * a) ** 2 + np.minimum(dk, nc - dk) ** 2
        out, near = [0], d2[0].copy()
        while len(out) < N:
            j = int(np.argmax(np.round(near, 9)))  # ties: the lowest cell
            out.append(j)
            near = np.minimum(near, d2[j])
        return tuple(out)
    out, seen, n = [], set(), 4 * N
    while len(out) < N:
        for x, y in _points(name, n):
            c = min(int(x * S), S - 1) * nc + min(int(y * nc), nc - 1)
            if c not in seen:
                seen.add(c)
                out.append(c)
        n *= 2
        if n > 2 ** 20:  # a sequence that never reaches some cell: the rest in cell order
            out += [c for c in range(N) if c not in seen]
    return tuple(out[:N])


def mask(layout: str, nc: int, n: int) -> np.ndarray:
    """(period, nc) pilot mask of a layout: "kKdD" | "<sequence>:restart" | "<sequence>:run", n pilots a frame."""
    if layout[0] == "k" and layout[1].isdigit():
        K, D = (int(v) for v in layout[1:].split("d"))
        s = np.arange(S * K)[:, None]
        return (s % D == 0) & ((np.arange(nc)[None, :] - s // D) % K == 0)
    name, mode = layout.split(":")
    o = np.array(ordering(name, nc))
    frames = 1 if mode == "restart" else len(o) // gcd(n, len(o))
    m = np.zeros((frames, S * nc), dtype=bool)
    for f in range(frames):
        m[f, o[(f * n + np.arange(n)) % len(o)]] = True
    return m.reshape(frames * S, nc)


def geometry(m: np.ndarray, a: float = 0.7) -> dict:
    """A mask's evenness: pilots per symbol and per carrier (min-max over a period), and the farthest any cell
    is from a pilot (symbols * a and carriers, on the period's torus)."""
    per, nc = m.shape
    ps, pk = np.nonzero(m)
    s, k = np.divmod(np.arange(per * nc), nc)
    ds = np.abs(s[:, None] - ps[None, :])
    dk = np.abs(k[:, None] - pk[None, :])
    d = np.sqrt((np.minimum(ds, per - ds) * a) ** 2 + np.minimum(dk, nc - dk) ** 2).min(axis=1)
    return dict(sym=f"{m.sum(1).min()}-{m.sum(1).max()}", car=f"{m.sum(0).min() * S / per:.1f}-{m.sum(0).max() * S / per:.1f}",
                hole=round(float(d.max()), 2), period=per // S)


def wiener(Z, X, m, Ts, spread, tau, f, gain2, nvar, loo=False):
    """fast_fading_study.wiener for any periodic mask m (period symbols, nc); loo: every other cell as a pilot."""
    ns, nc = Z.shape
    per = 1 if loo else len(m)
    Rf = delay_corr(f[:, None] - f[None, :], tau)
    wt = max(6, int(np.ceil(0.7 / spread / Ts)))
    dens = 1.0 if loo else m.mean()
    wt = int(min(wt, max(3, 300 / (dens * nc))))  # ~600 pilots a window at most
    hls = Z * np.conj(X)
    est, mse = np.zeros_like(Z), np.zeros(Z.shape)
    for p in range(per):
        rows_m = np.ones((2 * wt + 1, nc), bool) if loo else m[(p + np.arange(-wt, wt + 1)) % per]
        ds, a = np.nonzero(rows_m)
        dt = (ds - wt) * Ts
        A = gain2 * np.exp(-(np.pi * spread * (dt[:, None] - dt[None, :])) ** 2 / 2) * Rf[a[:, None], a[None, :]]
        A[np.diag_indices_from(A)] += nvar
        B = gain2 * np.exp(-(np.pi * spread * dt[None, :]) ** 2 / 2) * Rf[:, a]
        Wm = np.linalg.solve(A, B.conj().T).conj().T
        e = gain2 * Rf.diagonal().real - np.real(np.sum(Wm * B.conj(), axis=1))
        rows = np.arange(p, ns, per)
        rows = rows[(rows >= wt) & (rows < ns - wt)]
        est[rows] = hls[rows[:, None] + (ds - wt)[None, :], a[None, :]] @ Wm.T
        mse[rows] = e
        if loo:
            own = np.flatnonzero(ds == wt)
            wii = np.real(Wm[a[own], own])
            est[rows] = (est[rows] - wii * hls[rows]) / (1 - wii)
            mse[rows] = nvar * wii / (1 - wii)
    return est, mse, wt


def job(args) -> list:
    layout, dens, chan, seeds, snrs, dur, nc = args
    pre = CHANNELS[chan]
    pts = constellation.load("gray-qam4")
    N, ncp = 160, 32
    T = N + ncp
    f = 1525 + 50.0 * (np.arange(nc) - (nc - 1) / 2)
    m = mask(layout, nc, round(DENSITIES[dens] * S * nc))
    ns = int(dur * FS / T)
    pil = m[np.arange(ns) % len(m)]
    tally = {(snr, arm): Tally() for snr in snrs for arm in ("wiener", "dd")}
    peaks = []
    for seed in seeds:
        rng = np.random.default_rng([seed, 7])
        bits = rng.integers(2, size=(ns, nc, 2))
        X = constellation.modulate(bits.reshape(-1), pts).reshape(ns, nc)
        X[pil] = np.broadcast_to(PILOT, X.shape)[pil]  # the modem's pilot phases: a pilot row keeps its low peak
        n = np.arange(T) - ncp
        x = np.real(np.exp(2j * np.pi * np.outer(n, f) / FS) @ X.T).T.reshape(-1)
        x = tx_condition(np.concatenate([np.zeros(PAD), x, np.zeros(PAD)]), 0.0, bandpass=(f[0] - 75, f[-1] + 75),
                         active=slice(PAD, -PAD))
        peaks.append(peak_db(x))
        y, g1, g2, d = fade(x, pre, seed)
        starts = PAD + np.arange(ns) * T + ncp - BACKOFF
        Y = (2.0 / N) * (y[starts[:, None] + np.arange(N)[None, :]] @ np.exp(-2j * np.pi * np.outer(np.arange(N) - BACKOFF, f) / FS))
        tc = starts + N // 2
        H = (g1[tc][:, None] + g2[tc][:, None] * np.exp(-2j * np.pi * f * d / FS)[None, :]) / np.sqrt(2)
        alpha = np.vdot(H * X, Y) / np.vdot(H * X, H * X)
        var_i = float(np.mean(np.abs(Y - alpha * H * X) ** 2))
        block = starts // int(BLOCK_S * FS)
        nb = int(block.max()) + 1
        for snr in snrs:
            nv = 4 * sigma(x, snr) ** 2 / N
            Zn = Y + cnoise(np.random.default_rng([seed, 11, int(snr * 10) + 1000]), Y.shape, nv)
            stats = (T / FS, pre.doppler_hz * SPREAD_X, (pre.delay_ms + 0.5) / 1000, f, abs(alpha) ** 2, nv + var_i)
            for arm, (est, mse, wt) in (("wiener", wiener(Zn, X, m, *stats)), ("dd", wiener(Zn, X, m, *stats, loo=True))):
                rows = np.zeros(ns, bool)
                rows[wt:ns - wt] = True
                data = ~pil & rows[:, None]
                llr = constellation.llr(Zn[data], est[data], (nv + var_i + mse)[data], pts)
                info, info1 = bit_info(llr, bits[data].reshape(-1), np.geomspace(0.02, 2, 20))
                tally[snr, arm].add(info, np.repeat(np.broadcast_to(block[:, None], data.shape)[data], 2), T / FS,
                                    np.bincount(block[rows], minlength=nb), info1)
    geo = geometry(m)
    base = dict(layout=layout, dens=dens, chan=chan, peak_db=round(float(np.mean(peaks)), 2), **geo)
    return [dict(base, snr=snr, arm=arm, **t.row(nc * 2 * (1 - m.mean()) / (T / FS))) for (snr, arm), t in tally.items()]


def grids():
    for nc in (24, 48, 10, 4):
        print(f"== {nc} x {S}: pilots per symbol min-max | per carrier and frame min-max | farthest cell from a pilot | period, frames")
        for dens, d in DENSITIES.items():
            n = round(d * S * nc)
            if n < 1:
                continue
            print(f" density {dens}: {n} pilots a frame ({S * nc - n} data cells)")
            lays = [f"k{K}d{D}" for K, D in LATTICES[dens] if nc % K == 0] + [f"{q}:{md}" for q in SEQS for md in ("restart", "run")]
            for lay in lays:
                g = geometry(mask(lay, nc, n))
                print(f"   {lay:22s} {g['sym']:>6s} | {g['car']:>9s} | {g['hole']:5.2f} | {g['period']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sc4_screen.csv")
    ap.add_argument("--grids", action="store_true")
    ap.add_argument("--chans", default="mpg,mpp,mpd,nvis-d,lowlat-d,hilat-m,hilat-d")
    ap.add_argument("--snrs", default="0,10,20")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--dur", type=float, default=30.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seqs", default=",".join(SEQS))
    a = ap.parse_args()
    if a.grids:
        return grids()
    nc = 24
    lays = [(f"k{K}d{D}", dens) for dens in DENSITIES for K, D in LATTICES[dens]]
    lays += [(f"{q}:{md}", dens) for dens in DENSITIES for q in a.seqs.split(",") for md in ("restart", "run")]
    jobs = [(lay, dens, ch, list(range(1, a.seeds + 1)), [float(s) for s in a.snrs.split(",")], a.dur, nc)
            for lay, dens in lays for ch in a.chans.split(",")]
    print(len(jobs), "jobs", file=sys.stderr)
    fields = ["layout", "dens", "chan", "snr", "arm", "bps", "p10_bps", "frac", "max_bps", "bps_unscaled", "peak_db", "sym",
              "car", "hole", "period"]
    with open(a.out, "w", newline="") as fh, Pool(a.jobs) as pool:
        wr = csv.DictWriter(fh, fields)
        wr.writeheader()
        for rows in pool.imap_unordered(job, jobs):
            wr.writerows(rows)
            fh.flush()


if __name__ == "__main__":
    main()
