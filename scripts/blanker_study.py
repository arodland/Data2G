"""The impulse blanker (data2g.tnc.Blanker) against clicks: the smallest
ARQ data burst (ladder_study's trial) at each mode's 10% point, with
Poisson clicks (hfchannel.clicks) added, received with and without the
blanker on the same audio. Then false locks: the streaming receiver on
noise plus clicks, with and without it.

    uv run --no-sync python scripts/blanker_study.py --ladder runs/ladder_10pct.csv
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import hfchannel, modem
from data2g.arq import phy as PHY
from data2g.arq.engine import MAX_BURST_S
from data2g.config import FS
from data2g.tnc import Blanker, Receiver, receive_any

sys.path.insert(0, str(Path(__file__).parent))
import outcome_data as O  # noqa: E402

MODES = ("qpsk-r1/5", "n10-qpsk-r1/5", "fsk16r25-r1/3")
CHANNELS = ("awgn", "mpg")
RATES = (0, 1, 5, 20)  # clicks per second
AMP_DB = 20.0  # click power over the signal's


def ok(y, b, name) -> bool:
    try:
        r = receive_any(y, lead=FS)
    except Exception:  # noqa: BLE001 (a header read past the audio: a miss)
        return False
    if r is None or r["spec"].name != name or r["n_cw"] != 2:
        return False
    rx = PHY.ModemRx(r, {})
    return all(rx.decode(i, s.mask_id, 0, None) == s.payload for i, s in enumerate(b.slots))


def trial(args) -> tuple[bool, bool]:
    """-> (decoded without the blanker, with it), the same audio."""
    name, chan, snr, rate, seed = args
    rng = np.random.default_rng(seed)
    b = O.burst(name, 2, rng)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    x = np.concatenate([np.zeros(lead), PHY.tx_audio(b), np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=float(rng.uniform(-50, 50)), ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    if rate:
        y = hfchannel.clicks(y, rate, AMP_DB, seed=seed + 2, s_power=hfchannel.active_power(x))
    return ok(y, b, name), ok(Blanker()(y), b, name)


def false_locks(args) -> tuple[int, float]:
    """Headers the streaming receiver reports on `minutes` of noise plus
    clicks, and its CPU seconds."""
    blank, rate, minutes, seed = args
    rng = np.random.default_rng(seed)
    y = rng.normal(size=int(minutes * 60 * FS)) * 0.01
    if rate:
        y = hfchannel.clicks(y, rate, AMP_DB, seed=seed + 1, s_power=1e-4)
    from data2g import cpm

    rx = Receiver(modem.Accept.of(None, MAX_BURST_S), cpm_grids=tuple(cpm.GRIDS), blank=blank)
    t0, n = time.process_time(), 0
    for i in range(0, len(y), FS // 50):
        n += sum(k == "header" for k, _ in rx.feed(y[i:i + FS // 50]))
    return n, time.process_time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True, help="ladder_study's 10% csv: the SNR per cell")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=10.0, help="per false-lock run; 0: none")
    ap.add_argument("--guard", type=int, default=Blanker.GUARD, help="Blanker.GUARD")
    a = ap.parse_args()
    Blanker.GUARD = a.guard
    p10 = {r["name"]: r for r in csv.DictReader(open(a.ladder))}
    with Pool(a.jobs) as pool:
        print("mode channel snr clicks/s | decoded: without with | only without, only with", flush=True)
        for name in MODES:
            for chan in CHANNELS:
                snr = float(p10[name][chan])
                for rate in RATES:
                    r = pool.map(trial, [(name, chan, snr, rate, 1000 * k + 17) for k in range(a.trials)])
                    wo, w = sum(x for x, _ in r), sum(y for _, y in r)
                    print(f"{name} {chan} {snr:.2f} {rate} | {wo} {w} | {sum(x and not y for x, y in r)} "
                          f"{sum(y and not x for x, y in r)}  (of {a.trials})", flush=True)
        if not a.minutes:
            return
        print("false locks: blanker clicks/s | headers per hour, receiver CPU s per audio min", flush=True)
        for rate in (0, 5, 20):
            for blank in (False, True):
                r = pool.map(false_locks, [(blank, rate, a.minutes, 31 * k + 5) for k in range(a.jobs)])
                mins = a.minutes * a.jobs
                print(f"{blank} {rate} | {sum(n for n, _ in r) * 60 / mins:.1f} {sum(t for _, t in r) / mins:.2f}",
                      flush=True)


if __name__ == "__main__":
    main()
