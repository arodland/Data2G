"""Read a cpm_floor_study e2e CSV with dd/gdd arms: 10% points (data and
control) and, per SNR, the lock diagnosis (lockcat: 0 none, 1 wrong word).

    python -m scripts.coh_diag runs/coh3/e2e.csv [--arms gdd8 dd8]
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def x10(snrs, ok, target=0.1):
    f = 1 - ok
    for i in range(len(snrs) - 1):
        if f[i] > target >= f[i + 1]:
            return round(float(snrs[i] + (f[i] - target) / (f[i] - f[i + 1]) * (snrs[i + 1] - snrs[i])), 2)
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--arms", nargs="+", default=["gdd8", "dd8"])
    a = ap.parse_args()
    trials = defaultdict(dict)  # (mode, chan, snr, seed) -> {arm: value}
    with open(a.csv) as f:
        for r in csv.DictReader(f):
            trials[(r["mode"], r["channel"], float(r["snr"]), r["seed"])][r["arm"]] = float(r["ok"])
    g0, d0 = a.arms[0], a.arms[-1]
    cells = sorted({k[:2] for k in trials})
    for m, c in cells:
        snrs = sorted({k[2] for k in trials if k[:2] == (m, c)})
        rows = {s: [v for k, v in trials.items() if k[:3] == (m, c, s)] for s in snrs}
        cols = [x for x in ["base", "genie", *a.arms, "basec", "geniec", *[y + "c" for y in a.arms]]
                if x in rows[snrs[0]][0]]
        mean = lambda s, fn: float(np.mean([fn(t) for t in rows[s]]))  # noqa: E731
        print(m, c, "10%:", {x: x10(snrs, np.array([mean(s, lambda t: t[x]) for s in snrs])) for x in cols})
        extra = {"nolock": lambda t: t["lockcat"] == 0, "wrong": lambda t: t["lockcat"] == 1,
                 f"{g0}&nolock": lambda t: t[g0] == 1 and t["lockcat"] < 2,
                 f"{g0}&!{d0}&lock": lambda t: t[g0] == 1 and t[d0] == 0 and t["lockcat"] == 2}
        print("  snr " + " ".join(f"{x:>9s}" for x in [*extra, *cols]))
        for s in snrs:
            vals = [mean(s, fn) for fn in extra.values()] + [mean(s, lambda t: t[x]) for x in cols]
            print(f"  {s:5.1f} " + " ".join(f"{v:9.2f}" for v in vals))


if __name__ == "__main__":
    main()
