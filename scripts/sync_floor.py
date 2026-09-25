"""Sync threshold per band and channel: the lowest SNR (average power,
0.25 dB grid) where acquisition + header fail in <= 1% of bursts,
through the full numpy receiver with every band's detector running.

Sync depends on the band (preamble, header), not on the submode behind
it, so this is measured once per band with one short burst, and a
submode's end-to-end threshold is max(its code threshold, its band's
sync threshold): at that point sync and code each fail <= 1%, so the
combined PER is <= ~2%.

A trial succeeds when receive() locks, reads the right submode and
codeword count, and puts the preamble within 2 NCP (64) samples of the truth, the reach of
the equalizer's re-timing from the pilots (equalizer._DELAYS; n4 on mpd
locks ~42 late and decodes).
Random start, CFO uniform +-50 Hz, 10 ppm.

    uv run python scripts/sync_floor.py --out runs/sync_floor.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
from multiprocessing import Pool

import numpy as np

from data2g import codes, hfchannel, modem
from data2g.config import FS, LEADIN_SAMPLES, NCP, SUBMODES

# the short burst each band is measured with (one codeword)
BURST = {"w": "ack-1f", "n10": "n10-ack-4f", "n4": "n4-ack-2f", "w48": "w48-qpsk-r1/2"}
CHANNELS = ["awgn", "mpg", "mpp", "mpd"]
MIN_TRIALS, CHUNK = 2000, 200


def trial(args):
    band, chan, snr, seed = args
    spec = SUBMODES[BURST[band]]
    rng = np.random.default_rng(seed)
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec))], spec)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    x = np.concatenate([np.zeros(lead), x, np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=rng.uniform(-50, 50), ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    try:
        r = modem.receive(y)
    except modem.SyncError:
        return False
    return r["spec"] == spec and r["n_cw"] == 1 and abs(r["preamble_start"] - (lead + LEADIN_SAMPLES)) <= 2 * NCP


def point(pool, band, chan, snr, seed0=0):
    n = fails = 0
    while True:
        res = pool.map(trial, [(band, chan, snr, seed0 + n + i) for i in range(CHUNK)])
        n += len(res)
        fails += len(res) - sum(res)
        if n >= 400 and fails >= 30 and fails / n > 3e-2:
            return False, fails / n, n
        if n >= MIN_TRIALS:
            return fails / n <= 1e-2, fails / n, n


def threshold(pool, band, chan, start, step=1.0):
    lo = hi = None
    snr = start
    while lo is None or hi is None:
        ok, rate, n = point(pool, band, chan, snr)
        print(f"   {band} {chan} {snr:6.2f} dB: fail {rate:.4f} ({n}) {'PASS' if ok else 'fail'}", flush=True)
        if ok:
            hi = snr
            if lo is None:
                snr -= step
        else:
            lo = snr
            if hi is None:
                snr += step
        step *= 2
        if snr > 40:
            return float("inf")
    while hi - lo > 0.25:
        mid = round((lo + hi) / 2 * 4) / 4
        if mid in (lo, hi):
            break
        ok, rate, n = point(pool, band, chan, mid)
        print(f"   {band} {chan} {mid:6.2f} dB: fail {rate:.4f} ({n}) {'PASS' if ok else 'fail'}", flush=True)
        lo, hi = (lo, mid) if ok else (mid, hi)
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="+", default=list(BURST))
    ap.add_argument("--channels", nargs="+", default=CHANNELS)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seen = set()
    if os.path.exists(a.out) and os.path.getsize(a.out):
        seen = {(r["band"], r["channel"]) for r in csv.DictReader(open(a.out))}
    new = not seen
    # starting points near the matched-filter detector's floors (runs/sync_diag_diff.txt)
    starts = {"awgn": -7.0, "mpg": 3.0, "mpp": 2.0, "mpd": 2.0}
    with Pool(a.jobs) as pool, open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["band", "channel", "sync_threshold_db"])
        for band in a.bands:
            for chan in a.channels:
                if (band, chan) in seen:
                    continue
                t = threshold(pool, band, chan, starts[chan], step=2.0)
                w.writerow([band, chan, t])
                f.flush()
                print(f"== {band} {chan}: sync threshold {t} dB", flush=True)


if __name__ == "__main__":
    main()
