"""What CRC32 would cost the CRC16 codes (companion to crc_study.py):
codeword failure vs SNR on channel_torch (thresholds.Sim, 16-frame bursts),
10% and 1% points, for three variants of each code:
- base: today (k, CRC16);
- same-k: k kept, CRC32 (the payload loses 2 bytes; polar's list selects
  with 32 bits);
- k+16: CRC32 with the payload kept (the code rate rises).
CPM codes aren't on channel_torch; their payload cost is the same 2 bytes.

    uv run --no-sync python scripts/crc_cost.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import dataclasses
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from thresholds import Sim  # noqa: E402

from data2g import codes  # noqa: E402
from data2g.arq.modes import MODES  # noqa: E402

NAMES = ("qpsk-r1/5", "w48-qpsk-r1/5", "n10-qpsk-r1/5", "n10-qpsk-r1/3", "n10-qpsk-r1/2", "n4-qpsk-r1/5",
         "n4-qpsk-r1/3", "n4-qpsk-r1/2", "ack-1f", "ack-4f", "n10-ack-4f", "n4-ack-8f", "n4-ack-2f",
         "polar-k96-f4", "polar-k96-f8", "polar-k192-f8")
VARIANTS = ("base", "same-k", "k+16")
CHANNELS = ("awgn", "mpd")
BURSTS = 512
_crc_bits = codes.crc_bits


def job(args):
    name, var, chan, snrs = args
    torch.set_num_threads(1)
    spec = MODES[name]
    if var == "k+16":
        spec = dataclasses.replace(spec, k=spec.k + 16)
    codes.crc_bits = _crc_bits if var == "base" else (lambda s: 32)
    n_cw = max(1, 16 // spec.frames_per_cw)
    sim = Sim(spec, "cpu", n_cw, batch=max(8, 256 // n_cw))
    rng, g = np.random.default_rng(1), torch.Generator().manual_seed(1)
    out = []
    for snr in snrs:
        cws = err = bursts = 0
        while bursts < BURSTS:
            _, _, c, e, _ = sim.run(chan, snr, rng, g)
            cws, err, bursts = cws + c, err + e, bursts + sim.batch
            if err > 400:  # clearly past 10%: enough
                break
        out.append((snr, err / cws))
    codes.crc_bits = _crc_bits
    return name, var, chan, out


def crossing(curve, p):
    """SNR where failure falls through p (log-linear between points)."""
    for (s0, f0), (s1, f1) in zip(curve, curve[1:]):
        if f0 > p >= f1:
            if f1 <= 0:
                return s1
            return s0 + (np.log(f0) - np.log(p)) / (np.log(f0) - np.log(f1)) * (s1 - s0)
    return float("nan")


def main():
    p10 = {r["name"]: r for r in csv.DictReader(open("runs/ladder_10pct.csv"))}
    jobs = []
    for name in NAMES:
        for chan in CHANNELS:
            c = float(p10[name][chan])
            snrs = list(np.arange(c - 4, c + 4.01, 0.5))
            jobs += [(name, var, chan, snrs) for var in VARIANTS]
    res = {}
    with Pool(8) as pool:
        for name, var, chan, curve in pool.imap_unordered(job, jobs):
            res[(name, var, chan)] = curve
    with open("runs/crc_cost.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "variant", "channel", "snr", "cw_fail"])
        for (name, var, chan), curve in sorted(res.items()):
            w.writerows([name, var, chan, s, e] for s, e in curve)
    print("mode channel | 10% point base, same-k, k+16 | 1% point base, same-k, k+16 | payload B base -> same-k")
    for name in NAMES:
        s = MODES[name]
        for chan in CHANNELS:
            t10 = [crossing(res[(name, v, chan)], 0.1) for v in VARIANTS]
            t1 = [crossing(res[(name, v, chan)], 0.01) for v in VARIANTS]
            print(f"{name} {chan} | " + " ".join(f"{x:6.2f}" for x in t10) + " | " + " ".join(f"{x:6.2f}" for x in t1)
                  + f" | {codes.payload_bytes(s)} -> {codes.payload_bytes(s) - 2}", flush=True)


if __name__ == "__main__":
    main()
