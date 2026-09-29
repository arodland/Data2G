"""Paired comparison of two loss_study CSVs run on the same seeds: per
cell, delivered bps of each arm and the per-seed paired difference.

    uv run python scripts/paired_loss.py runs/loss_dd0.csv runs/loss_dd1.csv
"""

import csv
import sys
from collections import defaultdict

import numpy as np


def delivered(path):
    out = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        out[(r["channel"], float(r["snr"]))][int(r["seed"])] = float(r["delivered_Bps"]) * 8
    return out


def main(a, b):
    A, B = delivered(a), delivered(b)
    print(f"{'cell':12s} {'A bps':>8s} {'B bps':>8s} {'B-A':>7s} {'%':>7s} {'+-SE %':>7s}  seeds B>A/B<A")
    for k in sorted(A):
        seeds = sorted(set(A[k]) & set(B.get(k, {})))
        x, y = np.array([A[k][s] for s in seeds]), np.array([B[k][s] for s in seeds])
        d = y - x
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"{k[0]:5s}{k[1]:+4.0f} dB {x.mean():8.0f} {y.mean():8.0f} {d.mean():+7.0f} "
              f"{100 * d.mean() / x.mean():+6.1f}% {100 * se / x.mean():6.1f}%  {np.sum(d > 0)}/{np.sum(d < 0)} of {len(d)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
