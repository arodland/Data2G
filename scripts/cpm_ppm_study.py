"""Sample-clock mismatch: coherent CPM receiver vs the square law, paired on
the same audio. Long bursts (fsk16r25 2 data codewords, fsk32r62 7), the
far end's clock off by `ppm` (hfchannel.sample_clock_offset), then the
channel. Per trial and arm: data codewords decoded of those sent (mask
known, unkeyed), through the real lock (find / late_lock / find_mid).

    python -m scripts.cpm_ppm_study --out runs/ppm/ppm.csv [--trials 60 --jobs 4]

Found (2026-11-12, 60 trials a point; main runs/coh/ppm): the coherent
receiver decodes at least as many as the square law at 0-200 ppm; both lose
on long fsk32r62 bursts (refine times the head only): awgn -11 at 100 ppm
0.93 / 0.31, 200 ppm 0.40 / 0.15 (0 ppm 1.00 / 0.82). Per-codeword
re-timing (refine's likelihood per data slot, moving only for >= 8 nats;
runs/coh/ppm/retime.patch) took 200 ppm back to 0.96 / 0.70 with no 0 ppm
loss, but cost 2.7-4x cpm.receive (48-60 ms in Python against refine's
12-20): dropped.
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from dataclasses import replace  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

CASES = {  # mode, slots (control + data), channel -> SNRs (dB, mean power in 2500 Hz)
    ("fsk16r25-r1/3", 3, "awgn"): (-16, -15, -14),
    ("fsk16r25-r1/3", 3, "mpp"): (-14, -13, -12),
    ("fsk32r62-r1/3", 8, "awgn"): (-12, -11, -10),
    ("fsk32r62-r1/3", 8, "mpp"): (-9, -8, -7),
}
PPMS = (0, 50, 100, 200)


ARMS = (("coh", True), ("sq", False))


def arm(coherent: bool):
    from data2g import cpm
    from data2g.arq import phy as PHY

    PHY.CPM_DD = coherent
    cpm.COH_CTL = coherent
    if not hasattr(arm, "lock"):
        arm.lock = dict(cpm.COH_LOCK)
    cpm.COH_LOCK.clear()
    if coherent:
        cpm.COH_LOCK.update(arm.lock)


def trial(args):
    import outcome_data as O
    import phy_session as PS
    from cpm_floor_study import CHANS, receive_traced
    from data2g import hfchannel
    from data2g.arq import phy as PHY

    mode, n, chan, snr, ppm, seed = args
    rng = np.random.default_rng(seed)
    b = replace(O.burst(mode, n, rng), cap=2)
    x = PHY.tx_audio(b)
    if ppm:
        x = hfchannel.sample_clock_offset(x, ppm)
    ch = PS.ContinuousChannel("awgn", snr, seed, 120.0, *(CHANS[chan] or (None, None)))
    y = ch.apply(x, float(rng.uniform(5, 100)))
    out = {}
    for name, coh in ARMS:
        arm(coh)
        r = receive_traced(y, mode)
        ok = 0
        if r is not None and r["spec"].name == mode and r["n_cw"] == n:
            r = dict(r)
            r.pop("_soft", None)
            rx = PHY.ModemRx(r, {})
            ok = sum(rx.decode(i, b.slots[i].mask_id, 0, None) == b.slots[i].payload for i in range(1, n))
        out[name] = ok
    return mode, chan, snr, ppm, seed, n - 1, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="runs/ppm/ppm.csv")
    ap.add_argument("--modes", nargs="+", default=None)
    ap.add_argument("--ppms", nargs="+", type=int, default=None)
    a = ap.parse_args()
    global PPMS
    PPMS = tuple(a.ppms or PPMS)
    jobs = [(m, n, c, s, p, 70000 + i) for (m, n, c), snrs in CASES.items() for s in snrs for p in PPMS
            for i in range(a.trials) if a.modes is None or m in a.modes]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    agg = defaultdict(lambda: [0] * (1 + len(ARMS)))
    with Pool(a.jobs) as pool, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "channel", "snr", "ppm", "seed", "sent", *(n for n, *_ in ARMS)])
        for m, c, s, p, seed, sent, o in pool.imap_unordered(trial, jobs, chunksize=2):
            w.writerow([m, c, s, p, seed, sent, *(o[n] for n, *_ in ARMS)])
            k = agg[(m, c, s, p)]
            k[0] += sent
            for i, (n, *_) in enumerate(ARMS):
                k[1 + i] += o[n]
    print("data codewords decoded, per ppm: " + " / ".join(n for n, *_ in ARMS))
    for (m, c) in sorted({k[:2] for k in agg}):
        for s in sorted({k[2] for k in agg if k[:2] == (m, c)}):
            cells = "  ".join(f"{p:3d} ppm " + "/".join(f"{v / agg[(m, c, s, p)][0]:.2f}" for v in agg[(m, c, s, p)][1:])
                              for p in PPMS)
            print(f"  {m:14s} {c:5s} {s:+4.0f} dB  {cells}")


if __name__ == "__main__":
    main()
