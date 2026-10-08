"""Tables from hop_study.py's rows (docs/hopping-fsk.md): each variant's 10%
point (the 90th percentile of its per-draw thresholds; a draw that never
decoded counts as 40 dB) and its difference from the first variant, with
a 90% paired bootstrap interval (the draws resampled together).

    uv run python scripts/hop_report.py simple runs/hop_awgn.csv.gz k1,k4/bp=350/...
    uv run python scripts/hop_report.py ens runs/hop_ens.csv.gz K1W,K4,G2
    uv run python scripts/hop_report.py grid runs/hop_grid.csv.gz K1W,K4,G2

Variants may be written by their aliases (ALIAS).
"""

import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hop_tx as H  # noqa: E402

ALIAS = {"K1": "k1", "K1W": "k1/bp=350/glide=0.1/passes=6", "K4": "k4/bp=350/glide=0.1/passes=6",
         "G2": "g2-1000/bp=350/glide=0.1/passes=6/dwell=4"}


def load(path):
    """-> {(channel, seed): {variant: (threshold, peak dB)}}"""
    d = defaultdict(dict)
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as f:
        for v, ch, seed, t, pk in csv.reader(f):
            d[(ch, int(seed))][v] = (float(t), float(pk))
    return d


def table(rows, variants, q=90, B=2000, seed=0):
    """rows: [{variant: threshold}] -> (n, q-th percentiles, differences from
    the first, their 5th and 95th bootstrap percentiles)"""
    A = np.minimum(np.array([[r[v] for v in variants] for r in rows if all(v in r for v in variants)]), 40.0)
    p = np.percentile(A, q, axis=0)
    idx = np.random.default_rng(seed).integers(0, len(A), (B, len(A)))
    boot = np.percentile(A[idx], q, axis=1)
    lo, hi = np.percentile(boot - boot[:, :1], [5, 95], axis=0)
    return len(A), p, p - p[0], lo, hi


def line(n, p, dd, lo, hi, names):
    s = f"n={n:5d} " + " ".join(f"{k}={v:6.2f}" for k, v in zip(names, p))
    return s + " | vs " + names[0] + ": " + " ".join(
        f"{k} {x:+.2f} [{a:+.2f},{b:+.2f}]" for k, x, a, b in zip(names[1:], dd[1:], lo[1:], hi[1:]))


def main():
    which, path, names = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    variants = [ALIAS.get(k, k) for k in names]
    d = load(path)
    thr = {k: {v: r[0] for v, r in vs.items()} for k, vs in d.items()}
    if which == "simple":
        for q in (50, 90, 95):
            print(f"q{q}", line(*table(list(thr.values()), variants, q), names))
        print("peak dB", {k: round(float(np.median([r[v][1] for r in d.values() if v in r])), 2)
                          for k, v in zip(names, variants)})
    elif which == "ens":
        rows = []
        for (ch, s), r in thr.items():
            paths, dop = H.draw_paths(np.random.default_rng(s + 1))  # as hop_tx.channel drew them
            rows.append((r, dop, paths[1:]))

        def sel(label, f, q=90):
            print(f"{label:34s}", line(*table([r for r, dp, pa in rows if f(dp, pa)], variants, q), names))

        sel("all", lambda dp, pa: True)
        sel("all, 5% point", lambda dp, pa: True, 95)
        for a, b in [(0.05, 0.15), (0.15, 0.4), (0.4, 1.0), (1.0, 2.0)]:
            sel(f"Doppler {a}-{b} Hz", lambda dp, pa, a=a, b=b: a <= dp < b)
        for a, b in [(0.1, 0.4), (0.4, 0.8), (0.8, 1.5), (1.5, 3), (3, 5)]:
            sel(f"longest delay {a}-{b} ms, slow", lambda dp, pa, a=a, b=b: dp < 0.4 and a <= max(t for t, _ in pa) < b)
        for a, b in [(0.1, 0.8), (0.8, 5)]:
            sel(f"longest delay {a}-{b} ms, fast", lambda dp, pa, a=a, b=b: dp >= 0.4 and a <= max(t for t, _ in pa) < b)
        for k in (2, 3):
            sel(f"{k} paths", lambda dp, pa, k=k: len(pa) + 1 == k)
    elif which == "grid":
        cells = defaultdict(list)
        for (ch, s), r in thr.items():
            cells[ch].append(r)
        taus = sorted({float(c.split(":")[1]) for c in cells})
        dops = sorted({float(c.split(":")[2]) for c in cells})
        for k, v in zip(names[1:], variants[1:]):
            print(f"\n{k} minus {names[0]}, 10% point (dB, 90% interval); rows delay ms, columns Doppler Hz")
            print("delay  " + "".join(f"{x:>18}" for x in dops))
            for t in taus:
                out = f"{t:5}  "
                for dp in dops:
                    c = f"grid:{t:g}:{dp:g}"
                    if c not in cells:
                        out += f"{'':>18}"
                        continue
                    _, _, dd, lo, hi = table(cells[c], [variants[0], v], B=500)
                    out += f"  {dd[1]:+5.1f} [{lo[1]:+4.1f},{hi[1]:+4.1f}]"
                print(out)
        print(f"\n{names[0]}'s own 10% point")
        for t in taus:
            print(f"{t:5}  " + "".join(f"{table(cells[f'grid:{t:g}:{dp:g}'], variants[:1], B=10)[1][0]:8.1f}"
                                       for dp in dops if f"grid:{t:g}:{dp:g}" in cells))


if __name__ == "__main__":
    main()
