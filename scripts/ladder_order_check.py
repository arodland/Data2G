"""Did a receiver change reorder the ladder or shift pruning? Two seeded
ladder_study CSVs (before, after): per column and channel, the mode pairs
whose threshold order flipped by more than the 0.25 dB step; then
prune.py's rule (domination, near-ties, the <=500 Hz ladder) over the
current submodes on these end-to-end thresholds, PEP-referenced (+ the
post-clip peak PAPR; CPM, constant envelope: 0), before and after.

    uv run --no-sync python scripts/ladder_order_check.py runs/ladder_seeded_master_1pct.csv runs/ladder_seeded_pr6_1pct.csv
"""

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ladder_page as LP  # noqa: E402
import prune as P  # noqa: E402

from data2g.arq.modes import MODES, is_cpm  # noqa: E402
from data2g.config import clip_peak_db  # noqa: E402

CH = ("awgn", "mpg", "mpp", "mpd")
INF = float("inf")


def load(path) -> dict:
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["name"]] = {c: (float(r[c]) if r.get(c) not in (None, "", "nan") else INF) for c in CH}
    return out


def candidates(thr: dict) -> dict:
    """prune.py's form: name -> (payload bps, codeword length, PEP thresholds)."""
    c = {}
    for n, t in thr.items():
        s = MODES[n]
        peak = 0.0 if is_cpm(s) else clip_peak_db(s.band, s.headroom)
        c[n] = (LP.bps(s), s.coded_bits, {ch: t[ch] + peak for ch in CH})
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    a = ap.parse_args()
    b, f = load(a.before), load(a.after)
    names = [n for n in b if n in f]
    print("order flips (a more robust than b before, less after, each by > 0.25 dB):")
    flips = 0
    for ch in CH:
        for x, y in combinations(names, 2):
            d0, d1 = b[x][ch] - b[y][ch], f[x][ch] - f[y][ch]
            if abs(d0) > P.RESOLUTION and abs(d1) > P.RESOLUTION and (d0 > 0) != (d1 > 0):
                flips += 1
                lo0, lo1 = (x, y) if d0 < 0 else (y, x), (x, y) if d1 < 0 else (y, x)
                print(f"  {ch}: {lo0[0]} ({b[lo0[0]][ch]:.2f}) was ahead of {lo0[1]} ({b[lo0[1]][ch]:.2f}); "
                      f"now {lo1[0]} ({f[lo1[0]][ch]:.2f}) ahead of {lo1[1]} ({f[lo1[1]][ch]:.2f})")
    print(f"  {flips} flips")
    for tag, thr in (("before", b), ("after", f)):
        kept, why = P.prune(candidates({n: thr[n] for n in names}), keep=("ack-1f",))
        globals()[tag] = (set(kept), why)
    (k0, w0), (k1, w1) = globals()["before"], globals()["after"]
    print(f"prune.py's rule on these thresholds: {len(k0)} kept before, {len(k1)} after")
    for n in sorted(set(names)):
        if (n in k0) != (n in k1) or w0.get(n) != w1.get(n):
            print(f"  {n}: {'kept' if n in k0 else w0.get(n)} -> {'kept' if n in k1 else w1.get(n)}")


if __name__ == "__main__":
    main()
