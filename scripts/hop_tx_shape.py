"""Hopped fsk8r50 TX settings (docs/hopping-fsk.md), no decoding: per mode
and setting, the envelope peak over average (median and max of 24
codewords, dB) and the PSD at the band edges (dB against the mean between
the outer tones, max within +-2 Hz of: centre -+1200, -+600, -+250 Hz).
Prints TSV rows: mode bp clip passes glide dwell split pk_med pk_max e-1200
e+1200 e-600 e+600 e-250 e+250.

    uv run python scripts/hop_tx_shape.py k1,k4,g2-1000 > runs/hop_tx_shape.tsv
"""

import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import signal

from data2g.config import FS

sys.path.insert(0, str(Path(__file__).parent))
import hop_tx as H  # noqa: E402

N = 24
GRID = dict(bp=[50, 100, 150, 250, 350], clip=[0, -1, -2], passes=[3, 6], glide=[0, 0.1, 0.2], dwell=[1, 4],
            split=[0, 1])


def edges_db(f, P, edges, inband):
    ref = P[(f >= inband[0]) & (f <= inband[1])].mean()
    return [10 * np.log10(P[(f >= e - 2) & (f <= e + 2)].max() / ref) for e in edges]


def one(args):
    mode, p = args
    xs, pk = [], []
    for s in range(N):
        x, act, m = H.tx(mode, p, 1000 + s)
        pk.append(H.papr(x, act)[0])
        xs.append(x[act])
    P = 0
    for x in xs:
        f, q = signal.welch(x, FS, nperseg=4096, noverlap=2048)
        P = P + q
    lo = m["f0"]
    hi = lo + (m["span"] - 1) * H.G.rate
    cen = (lo + hi) / 2
    e = edges_db(f, P / N, [cen + d for d in (-1200, 1200, -600, 600, -250, 250)], (lo, hi))
    return mode, p, float(np.median(pk)), float(np.max(pk)), e


def main():
    jobs = []
    for mode in sys.argv[1].split(","):
        for vals in itertools.product(*GRID.values()):
            p = dict(zip(GRID, vals))
            if (p["split"] and not mode.startswith("g2")) or (p["dwell"] > 1 and mode == "k1"):
                continue
            jobs.append((mode, p))
    with Pool(4) as pool:
        for mode, p, med, mx, e in pool.imap(one, jobs, chunksize=2):
            print("\t".join([mode, *(str(p[k]) for k in GRID), f"{med:.2f}", f"{mx:.2f}", *(f"{v:.1f}" for v in e)]),
                  flush=True)


if __name__ == "__main__":
    main()
