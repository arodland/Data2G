"""Active constellation extension against plain clipping, on the submodes
that clip with headroom today, PEP-fair (clip_study.py's method, on the
production specs): per variant and headroom, the clip constants
(clip_constants.measure, with ACE) and the 1% codeword-failure threshold
(thresholds.py, 16-frame bursts) on AWGN and MPD. PEP-fair score:
threshold + mean burst peak (dB); lower is better.

Variants: plain; ACE closing with one plain pass (overshoot 1.0); ACE
closing with the stock three (1.0, 1.5, 2.0). Resumable CSV; submode
column "<name>" or "<name>+ace<closing>" for scripts/pick_headroom.py.

    PYTHONPATH=scripts uv run --no-sync python scripts/ace_study.py --out runs/ace_study.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import dataclasses
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from clip_constants import measure  # noqa: E402
from thresholds import Sim, threshold  # noqa: E402

from data2g.config import CLIP_OVERSHOOT, SUBMODES  # noqa: E402

NAMES = ("w48-16qam-r1/3", "w48-16qam-r1/2", "w48-16qam-r2/3", "w48-16qam-r3/4", "w48-16qam-r5/6",
         "w48-64l-r1/2", "w48-64l-r7/12", "w48-64l-r2/3", "w48-64l-r3/4", "w48-256l-r5/8",
         "n10-16qam-r2/3", "n10-16qam-r3/4")
VARIANTS = {"": (), "+ace1": (1.0,), "+ace1-1.5-2": (1.0, 1.5, 2.0)}
HEADROOMS = (0, 1, 2, 3, 4, 5, 6)
BURST_FRAMES = 16


def job(args):
    """One (mode, variant, headroom): its rows."""
    name, var, hr, channels, starts = args
    torch.set_num_threads(1)
    spec = dataclasses.replace(SUBMODES[name], clip_headroom_db=float(hr), ace=VARIANTS[var])
    c = measure(spec.band, hr, CLIP_OVERSHOOT, "cpu", const=spec.constellation, ace=spec.ace)
    (g1, _, _, _), (g8, sdr, papr, peak) = c[1], c[8]
    n_cw = max(1, BURST_FRAMES // spec.frames_per_cw)
    sim = Sim(spec, "cpu", n_cw, batch=max(8, 256 // n_cw), clip_setting=(hr, CLIP_OVERSHOOT, spec.ace),
              clip_consts=({1: g1}, g8, 10 ** (-sdr / 10)))
    rows = []
    for chan in channels:
        t0 = time.time()
        thr = threshold(sim, chan, round(starts[chan] * 4) / 4, verbose=False)
        rows.append([spec.band, hr, name + var, chan, thr, f"{papr:.2f}", f"{peak:.2f}", f"{sdr:.2f}", f"{g8:.3f}",
                     f"{g1:.3f}", f"{time.time() - t0:.0f}"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs="+", default=list(NAMES))
    ap.add_argument("--channels", nargs="+", default=["awgn", "mpd"])
    ap.add_argument("--ladder", default="runs/ladder_10pct.csv")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    p10 = {r["name"]: r for r in csv.DictReader(open(a.ladder))}
    seen = set()
    if os.path.exists(a.out) and os.path.getsize(a.out):
        seen = {(r["submode"], float(r["headroom"])) for r in csv.DictReader(open(a.out))}
    todo = []
    for name in a.names:
        # 1% sits above the 10% point: start there plus 1.5 dB (nan: 25 dB)
        starts = {c: (float(p10[name][c]) if p10[name][c] not in ("", "nan") else 23.5) + 1.5 for c in a.channels}
        h0 = SUBMODES[name].headroom
        for var in VARIANTS:
            # a threshold is 10-20 min of one core: plain at today's pick (the
            # baseline), ACE from 3 dB under it (it lowers the headroom needed)
            hrs = [h0] if not var else [h for h in HEADROOMS if h0 - 3 <= h <= h0]
            for hr in hrs:
                if (name + var, float(hr)) not in seen:
                    todo.append((name, var, hr, a.channels, starts))
    new = not seen
    with open(a.out, "a", newline="") as f, Pool(a.jobs) as pool:
        w = csv.writer(f)
        if new:
            w.writerow(["band", "headroom", "submode", "channel", "threshold_db", "papr_db", "peak_db", "sdr_db",
                        "gain", "gain_1f", "secs"])
        for rows in pool.imap_unordered(job, todo):
            w.writerows(rows)
            f.flush()
            for r in rows:
                print(" ".join(map(str, r)), flush=True)


if __name__ == "__main__":
    main()
