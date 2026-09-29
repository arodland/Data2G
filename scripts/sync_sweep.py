"""End-to-end burst success through the full numpy modem, with failures
split by stage: no preamble / header / CRC. Tells whether acquisition
and the header, not the code, set a submode's floor.

    DATA2G_PREAMBLE_REPEATS=16 \\
        uv run python scripts/sync_sweep.py --submode ack-2f --snr -6 -4 -2 0 2 --trials 200

Each trial: random payload, random start in 0.3-1 s of noise, CFO
uniform +-50 Hz, 10 ppm clock error. Preamble threshold is the value
config.PREAMBLE_THRESHOLDS' calibrated value for the repeat count.
"""

# One BLAS/OpenMP thread per process, set before numpy loads so forked
# workers inherit it: with a worker per core, numpy's default of a thread
# per core each put the load average past 300 on 24 cores (2026-09-23).
from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
from collections import Counter
from multiprocessing import Pool

import numpy as np

from data2g import codes, hfchannel, modem
from data2g.config import FS, PREAMBLE_REPEATS, PREAMBLE_THRESHOLDS, SUBMODES


def trial(args):
    sub, chan, snr, seed = args
    spec = SUBMODES[sub]
    rng = np.random.default_rng(seed)
    sent = [rng.bytes(codes.payload_bytes(spec))]
    x = modem.modulate(sent, spec)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    x = np.concatenate([np.zeros(lead), x, np.zeros(FS // 2)])
    y = hfchannel.apply_channel(
        x, snr_db=snr, freq_offset_hz=rng.uniform(-50, 50), ppm=10,
        fading_preset=None if chan == "awgn" else chan, seed=seed,
    )
    try:
        b = modem.demodulate(y)
    except modem.SyncError as e:
        return "header" if "header" in str(e) else "preamble" if "preamble" in str(e) else "sync-other"
    if b.submode != spec or len(b.payloads) != 1:
        return "wrong-header"
    return "ok" if b.crc_ok[0] and b.payloads[0] == sent[0] else "crc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submode", default="ack-1f")
    ap.add_argument("--channels", nargs="+", default=["awgn", "mpp", "mpd"])
    ap.add_argument("--snr", type=float, nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=8, help="worker processes (default 8)")
    a = ap.parse_args()
    print(f"submode {a.submode}, preamble repeats {PREAMBLE_REPEATS} (threshold {PREAMBLE_THRESHOLDS[PREAMBLE_REPEATS]})")
    with Pool(a.jobs) as pool:
        for chan in a.channels:
            for snr in a.snr:
                c = Counter(pool.map(trial, [(a.submode, chan, snr, s) for s in range(a.trials)]))
                fails = ", ".join(f"{k} {v}" for k, v in sorted(c.items()) if k != "ok")
                print(f"  {chan:5s} {snr:5.1f} dB: success {c['ok'] / a.trials:.3f}  ({fails})", flush=True)


if __name__ == "__main__":
    main()
