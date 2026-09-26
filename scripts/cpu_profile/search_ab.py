"""A/B: the receiver's old whole-buffer search vs the incremental one
(tnc.Receiver with and without its StreamDetector statistics): weak-burst
detection per cell on the same seeds, and header locks on noise. Edit
rx_for() to A/B another receiver change.

    uv run --no-sync python scripts/cpu_profile/search_ab.py 80
"""
import logging
import sys

import numpy as np

logging.disable(logging.WARNING)
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multiprocessing import Pool  # noqa: E402

from data2g import cpm, hfchannel, modem, tnc  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.config import FS  # noqa: E402

import outcome_data as O  # noqa: E402

ACC = modem.Accept.of(None, 16.0)


def rx_for(mode):
    r = tnc.Receiver(ACC, cpm_grids=tuple(cpm.GRIDS))
    if mode == "old":
        r._stats = lambda w0=0: None
        r._searched = lambda: None
    return r


def burst_trial(a):
    mode, name, chan, snr, seed = a
    rng = np.random.default_rng(seed)
    x = PHY.tx_audio(O.burst(name, 3, rng))
    y = np.concatenate([np.zeros(int(rng.uniform(2, 4) * FS)), x, np.zeros(FS)])
    y = hfchannel.apply_channel(y, snr_db=snr, freq_offset_hz=float(rng.uniform(-50, 50)), ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    r = rx_for(mode)
    for i in range(0, len(y), FS // 50):
        for k, ev in r.feed(y[i:i + FS // 50]):
            if k == "burst" and ev["rx"] is not None and ev["rx"]["spec"].name == name:
                return 1
    return 0


def noise_trial(a):
    mode, seed = a
    y = np.random.default_rng(seed).normal(0, 0.1, FS * 60)
    r = rx_for(mode)
    n = 0
    for i in range(0, len(y), FS // 50):
        n += sum(k == "header" for k, _ in r.feed(y[i:i + FS // 50]))
    return n


if __name__ == "__main__":
    cells = [("ack-4f", "awgn", -8), ("ack-4f", "mpp", -3), ("qpsk-r1/5", "mpg", -2), ("n10-ack-4f", "awgn", -9),
             ("n10-ack-4f", "mpp", -3), ("w48-qpsk-r1/5", "mpp", 2), ("fsk16r25-r1/2", "awgn", -11)]
    N = int(sys.argv[1])
    with Pool(8) as p:
        for c in cells:
            res = {m: sum(p.map(burst_trial, [(m,) + c + (5000 + s,) for s in range(N)])) for m in ("old", "new")}
            print(c, {m: f"{v}/{N}" for m, v in res.items()}, flush=True)
        for m in ("old", "new"):
            print("noise, 16 x 60 s:", m, "headers", sum(p.map(noise_trial, [(m, 900 + s) for s in range(16)])),
                  flush=True)
