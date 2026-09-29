"""Decoder channel-LLR clamp (ldpc.CH_CLAMP), paired: the same soft bits
decoded at each clamp. Per trial: a burst at RV 0, then resends at RV 1,
2, 3 and 0 again, each through its own channel stretch; decoded after
each (codes.combine / decode_buffer, as the ARQ receiver does).

The n30-default session (w48-256l-r5/8, 30 dB AWGN) lost one codeword to
13 transmissions: with LLRs clamped at 50, a check's message (at most
~16.8: ldpc._phi) can't override the channel, and a combined buffer whose
info bits were already right never satisfied H. Cases: high SNR, where
64/256-QAM LLRs are overconfident (clip noise isn't in the demapper's
variance), and each ir_study mode at its threshold, where a lower clamp
could cost first-transmission decodes.

    python scripts/llr_clamp_study.py --jobs 4 --out runs/llr_clamp_study.csv
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, ldpc  # noqa: E402
from data2g.config import SUBMODES  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import ir_study as IR  # noqa: E402

CLAMPS = (50.0, 16.0, 12.0)
RVS = (0, 1, 2, 3, 0)
N_CW = 8
HIGH = [("w48-256l-r5/8", "awgn", s) for s in (26.0, 28.0, 30.0)] + \
       [("w48-64l-r3/4", "awgn", s) for s in (24.0, 28.0)] + [("w48-16qam-r5/6", "awgn", 24.0)]
NEAR_MODES = ("qpsk-r1/5", "n10-qpsk-r1/2", "w48-16qam-r2/3", "w48-64l-r1/2")
NEAR_OFFSETS_DB = (-3.0, 0.0)


def trial(args):
    name, chan, snr, seed = args
    spec = SUBMODES[name]
    rng = np.random.default_rng(seed)
    payloads = [bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8)) for _ in range(N_CW)]
    softs = []
    for j, rv in enumerate(RVS):
        s = IR.burst_soft(spec, payloads, [rv] * N_CW, chan, snr, seed + 1_000_003 * j)
        if s is None:
            return None  # sync lost: not what this measures
        softs.append((s, rv))
    out = dict(mode=name, channel=chan, snr=snr, seed=seed)
    for c in CLAMPS:
        ldpc.CH_CLAMP = c
        buf, top = None, 0
        for j, (s, rv) in enumerate(softs):
            buf = codes.combine(spec, buf, s, rv)
            top = max(top, rv)
            dec = codes.decode_buffer(spec, buf, max_rv=top)
            out[f"c{c:g}_tx{j + 1}"] = float(np.mean([ok and p == q for (p, ok), q in zip(dec, payloads)]))
    out["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return out


def jobs(trials):
    thr = IR.thresholds()
    js = [(n, ch, s, 1000 * i + j) for i, (n, ch, s) in enumerate(HIGH) for j in range(trials)]
    for name in NEAR_MODES:
        for chan in IR.CHANNELS:
            t = thr.get((IR.ladder_name(SUBMODES[name]), chan))
            if t is None:
                print("no threshold", name, chan)
                continue
            # common random numbers: the same seeds at each offset (ladder-seeded)
            js += [(name, chan, round(t + d, 2), 50_000 + j) for d in NEAR_OFFSETS_DB for j in range(trials)]
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/llr_clamp_study.csv")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="one job, print its result and peak RSS")
    a = ap.parse_args()
    js = jobs(a.trials)
    if a.smoke:
        print(trial(js[0]))
        return
    rows = []
    with Pool(a.jobs) as pool:
        for r in pool.imap_unordered(trial, js, chunksize=1):
            if r is not None:
                rows.append(r)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(a.out)


def summarize(path):
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        g[(r["mode"], r["channel"], float(r["snr"]))].append(r)
    txs = range(1, len(RVS) + 1)
    print("codewords decoded after each transmission (RVs " + " ".join(map(str, RVS)) + "), per clamp")
    for k in sorted(g):
        rs = g[k]
        line = f"{k[0]:15s} {k[1]:4s} {k[2]:5.1f} n={len(rs):2d}"
        for c in CLAMPS:
            line += f" | {c:>2g}: " + " ".join(f"{np.mean([float(r[f'c{c:g}_tx{t}']) for r in rs]):.2f}" for t in txs)
        print(line)
    print(f"peak RSS per worker: {max(float(r['rss_mb']) for rs in g.values() for r in rs):.0f} MB")


if __name__ == "__main__":
    main()
