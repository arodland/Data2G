"""Re-threshold ladder candidates with a different constellation, at the
same band, clip headroom and codeword: the check that decides whether a
retrained set is kept. Starts each search from the candidate's current
threshold (0.5 dB first step). Writes ladder-format rows, so the result
can go through prune.py alongside the old ones.

    PYTHONPATH=scripts uv run python scripts/retest_constellations.py runs/ladder_clip.csv \\
        --swap w48-ldpc-c64-snr18-f2-k1680@h5=c64-w48-r712 ... --out runs/ladder_retrain.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import csv
import dataclasses
import time

import torch

torch.set_num_threads(4)

from data2g.config import SubmodeSpec, clip_peak_db  # noqa: E402
from thresholds import Sim, threshold  # noqa: E402

BURST_FRAMES = 16
CHANNELS = ["awgn", "mpg", "mpp", "mpd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ladder")
    ap.add_argument("--swap", nargs="+", required=True, help="candidate=constellation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.ladder)))
    old = {(r["name"], r["channel"]): float(r["threshold_db"]) for r in rows}
    meta = {r["name"]: r for r in rows}
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    seen = set() if new else {(r["name"], r["channel"]) for r in csv.DictReader(open(a.out))}
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(list(rows[0].keys()))
        for pair in a.swap:
            cand, const = pair.split("=")
            r = meta[cand]
            spec = SubmodeSpec(15, cand, r["code"], const, int(r["frames"]), k=int(r["k"]),
                               band=r["band"], clip_headroom_db=float(r["headroom"]))
            name = f"{cand.split('@')[0].replace(r['constellation'], const)}@h{spec.headroom:g}"
            spec = dataclasses.replace(spec, name=name)
            burst = max(1, BURST_FRAMES // spec.frames_per_cw)
            sim = None
            for chan in CHANNELS:
                if (name, chan) in seen:
                    continue
                prior = old[(cand, chan)]
                t0 = time.time()
                if prior == float("inf"):
                    start, step = old[(cand, "awgn")] + 8, 2.0
                else:
                    start, step = prior, 0.5
                sim = sim or Sim(spec, a.device, burst, batch=max(8, 256 // burst))
                thr = threshold(sim, chan, round(start * 4) / 4, verbose=False, step=step)
                w.writerow([name, spec.code, const, spec.frames_per_cw, spec.k, spec.coded_bits,
                            r["bits_per_cu"], chan, thr, f"{time.time() - t0:.0f}", spec.band,
                            spec.headroom, clip_peak_db(spec.band, spec.headroom)])
                f.flush()
                print(f"{name:42s} {chan:4s} {prior:6.2f} -> {thr:6.2f} dB (avg power)", flush=True)


if __name__ == "__main__":
    main()
