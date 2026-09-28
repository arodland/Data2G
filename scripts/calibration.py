"""Outcome-model calibration per candidate mode on session rows: predicted
against actual P(burst usable) and P(codeword ok | usable), for a channel
kind and SNR window, alongside all channels. The basis for per-mode logit
offsets (data2g.arq.predictor.LOGIT_OFFSETS).

    uv run --no-sync python scripts/calibration.py runs/session_data_v8.csv runs/outcome_predictor_v7.npz \\
        --kind mpg --snr -2 2 --snr 6 10
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from data2g.arq import predictor as P

sys.path.insert(0, str(Path(__file__).parent))
import train_outcome as T  # noqa: E402

MIN_ROWS = 40


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def table(sel, z, mode, bok, dsent, dok, n_modes, title):
    print(f"\n{title}: mode | rows | P(usable) predicted vs actual (logit gap) | P(cw|usable) predicted vs actual (gap)")
    for m in sorted(set(mode[sel])):
        s = sel & (mode == m)
        if s.sum() < MIN_ROWS:
            continue
        pu = 1 / (1 + np.exp(-z[s, m]))
        pc = 1 / (1 + np.exp(-z[s, n_modes + m]))
        au = bok[s].mean()
        u = s & (bok > 0.5) & (dsent > 0)
        ac = dok[u].sum() / max(dsent[u].sum(), 1)
        pcu = (pc[(bok[s] > 0.5) & (dsent[s] > 0)] * dsent[u]).sum() / max(dsent[u].sum(), 1)
        print(f"  {T.MODES[m]:16s} {s.sum():5d} | {pu.mean():.3f} vs {au:.3f} ({logit(au) - logit(pu.mean()):+.2f}) | "
              f"{pcu:.3f} vs {ac:.3f} ({logit(ac) - logit(pcu):+.2f}) n_cw {int(dsent[u].sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("model")
    ap.add_argument("--kind", default="mpg")
    ap.add_argument("--snr", type=float, nargs=2, action="append", help="snr_next window (dB); repeatable")
    a = ap.parse_args()
    x, mode, bok, dsent, dok, rows, _ = T.load(a.data)
    model = P.outcome_model(a.model)
    z = np.concatenate([model(x[i:i + 20000]) for i in range(0, len(x), 20000)])
    n = len(model.modes)
    assert tuple(model.modes) == tuple(T.MODES)
    kind = np.array([r["kind"] for r in rows])
    snr = np.array([float(r["snr_next"]) for r in rows])
    for lo, hi in a.snr or [(-100, 100)]:
        w = (snr >= lo) & (snr < hi)
        table(w & (kind == a.kind), z, mode, bok, dsent, dok, n, f"{a.kind}, SNR {lo:g}..{hi:g} dB")
        table(w, z, mode, bok, dsent, dok, n, f"all channels, SNR {lo:g}..{hi:g} dB")


if __name__ == "__main__":
    main()
