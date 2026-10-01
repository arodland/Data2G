"""Paired ladder deltas: B minus A per mode and channel (dB; positive means
B needs more SNR), plus per-channel medians and the largest shifts.

    python -m scripts.compare_ladders runs/ladder_a_10pct.csv runs/ladder_b_10pct.csv
"""

import csv
import math
import sys

import numpy as np

CHANNELS = ("awgn", "mpg", "mpp", "mpd")


def load(path):
    return {r["name"]: {c: float(r[c]) for c in CHANNELS if r.get(c) not in (None, "")} for r in csv.DictReader(open(path))}


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    names = [n for n in a if n in b]
    print(f"{'mode':18s}" + "".join(f"{c:>8s}" for c in CHANNELS))
    d = {c: [] for c in CHANNELS}
    for n in names:
        row = []
        for c in CHANNELS:
            x = b[n].get(c, math.nan) - a[n].get(c, math.nan)
            row.append(x)
            if not math.isnan(x):
                d[c].append((x, n))
        print(f"{n:18s}" + "".join(f"{x:+8.2f}" for x in row))
    print(f"{'median':18s}" + "".join(f"{np.median([x for x, _ in d[c]]):+8.2f}" for c in CHANNELS))
    print(f"{'mean':18s}" + "".join(f"{np.mean([x for x, _ in d[c]]):+8.2f}" for c in CHANNELS))
    for c in CHANNELS:
        worst = sorted(d[c], reverse=True)[:3]
        print(f"{c}: largest B-A " + ", ".join(f"{n} {x:+.2f}" for x, n in worst))


if __name__ == "__main__":
    main()
