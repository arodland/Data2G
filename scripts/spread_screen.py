"""Thresholds with the per-burst leave-one-out Wiener spread
(equalizer.LOO_SPREAD) off and on, paired (seed 0 at every SNR, both
arms), per mode, channel and burst length: the spread estimate is worst
in 6-16 frame bursts. GPU path (scripts/thresholds.py: channel_torch,
numpy estimator, genie support, no DD). Average-power SNR; both arms
have the same peak, so the difference is PEP-fair too.

    PYTHONPATH=.:scripts python scripts/spread_screen.py --per 0.1 --out runs/spread_screen_10pct.csv
"""

import os

from data2g import threads  # noqa: E402

threads.limit(2)

import argparse
import csv
import time

import torch

torch.set_num_threads(4)

import thresholds  # noqa: E402
from data2g import equalizer  # noqa: E402
from data2g.config import SUBMODES  # noqa: E402
from llr_scale_sweep import POINTS  # noqa: E402
from thresholds import Sim, threshold  # noqa: E402

LENGTHS = {"w48-qpsk-r1/2": (4, 8, 16), "w48-16qam-r1/2": (4, 8, 16), "w48-64l-r1/2": (4, 8, 16),
           "w48-256l-r5/8": (4, 8, 16), "n10-qpsk-r1/2": (10, 20), "n10-64l-r3/4": (10, 20)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=float, default=0.1)
    ap.add_argument("--modes", nargs="+", default=list(LENGTHS))
    ap.add_argument("--channels", nargs="+", default=["awgn", "mpg", "mpp", "mpd"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    thresholds.PER_TARGET = a.per
    if a.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.12)
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    seen = set() if new else {(r["mode"], r["channel"], r["frames"], r["loo"]) for r in csv.DictReader(open(a.out))}
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["mode", "channel", "frames", "loo", "threshold_db", "per", "secs"])
        for name in a.modes:
            spec = SUBMODES[name]
            for frames in LENGTHS[name]:
                # bursts x frames per batch: the LLR tensor grows with the point count (GPU cap 12%)
                budget = 192 if "256" in spec.constellation else 512
                sim = Sim(spec, a.device, frames // spec.frames_per_cw, batch=max(8, budget // frames))
                for chan in a.channels:
                    prior = POINTS[name].get(chan)
                    if prior is None:
                        continue
                    for loo in (False, True):
                        if (name, chan, str(frames), str(loo)) in seen:
                            continue
                        equalizer.LOO_SPREAD = loo
                        t0 = time.time()
                        thr = threshold(sim, chan, round((prior - 1) * 4) / 4, verbose=False, step=0.5)
                        w.writerow([name, chan, frames, loo, thr, a.per, f"{time.time() - t0:.0f}"])
                        f.flush()
                        print(f"{name:15s} {chan:4s} {frames:2d}f loo={loo!s:5s} {thr:6.2f} dB", flush=True)
                if a.device.startswith("cuda"):
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
