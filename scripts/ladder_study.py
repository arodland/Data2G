"""The whole ladder (OFDM and CPM) on one footing: per mode and channel, the
SNR (2500 Hz, average power) where the smallest ARQ data burst (control +
one data codeword) fails FAIL of the time, end to end through the receiver
the ARQ uses (head-only search, as the streaming receiver; the burst's
header right; both codewords decoded). CFO +-50 Hz, 10 ppm, random lead.

    uv run python scripts/ladder_study.py --fail 0.1 --out runs/ladder_10pct.csv
    uv run python scripts/ladder_study.py --fail 0.01 --out runs/ladder_1pct.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import hfchannel
from data2g.arq import phy as PHY
from data2g.arq.modes import MODES
from data2g.config import FS
from data2g.tnc import receive_any

sys.path.insert(0, str(Path(__file__).parent))
import outcome_data as O  # noqa: E402

CHANNELS = ("awgn", "mpg", "mpp", "mpd")
TRIALS = {0.1: 200, 0.01: 400}  # per SNR point: fail <= FAIL of these


def trial(args):
    name, chan, snr, seed = args
    rng = np.random.default_rng(seed)
    b = O.burst(name, 2, rng)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    x = np.concatenate([np.zeros(lead), PHY.tx_audio(b), np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=float(rng.uniform(-50, 50)), ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    try:
        r = receive_any(y, lead=FS)
    except Exception:  # noqa: BLE001 (a header read past the audio: a miss)
        return False
    if r is None or r["spec"].name != name or r["n_cw"] != 2:
        return False
    rx = PHY.ModemRx(r, {})
    return all(rx.decode(i, s.mask_id, 0, None) == s.payload for i, s in enumerate(b.slots))


def passes(pool, name, chan, snr, fail):
    n_max = TRIALS[fail]
    fails = n = 0
    while n < n_max:
        r = pool.map(trial, [(name, chan, snr, 7919 * n + j + int((snr + 100) * 1000)) for j in range(100)])
        n += 100
        fails += r.count(False)
        if fails > fail * n_max:
            return False
    return True


def threshold(pool, name, chan, fail, start):
    """Lowest passing SNR to 0.25 dB, searched from `start` in 3 dB steps."""
    hi = start
    while not passes(pool, name, chan, hi, fail):
        hi += 3
        if hi > 40:
            return float("nan")
    lo = hi - 3
    while passes(pool, name, chan, lo, fail):
        hi, lo = lo, lo - 3
        if lo < -30:
            return hi
    while hi - lo > 0.25:
        mid = (hi + lo) / 2
        hi, lo = (mid, lo) if passes(pool, name, chan, mid, fail) else (hi, mid)
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail", type=float, choices=sorted(TRIALS), default=0.1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--channels", nargs="+", default=list(CHANNELS))
    ap.add_argument("--start", type=str, default=None, help="a previous run's csv: search from its thresholds")
    a = ap.parse_args()
    prev = {(r["name"], c): float(r[c]) for r in csv.DictReader(open(a.start)) for c in CHANNELS
            if r.get(c) not in (None, "", "nan")} if a.start else {}
    done = {r["name"]: r for r in csv.DictReader(open(a.out))} if os.path.exists(a.out) else {}
    with Pool(a.jobs) as pool:
        for name in a.modes:
            if name in done:
                continue
            row = dict(name=name)
            for c in a.channels:
                start = prev.get((name, c), O.threshold(name, c) if O.threshold(name, c) < 90 else 0.0) - 2
                row[c] = threshold(pool, name, c, a.fail, round(start * 4) / 4)
                print(name, c, row[c], flush=True)
            done[name] = row
            with open(a.out, "w", newline="") as f:
                w = csv.DictWriter(f, ["name", *CHANNELS])
                w.writeheader()
                w.writerows(done.values())


if __name__ == "__main__":
    main()
