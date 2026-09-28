"""Phase F: incremental redundancy through the real PHY.

Per submode, channel and SNR: one burst of codewords (RV 0), then a
resend of the same codewords through an independent channel stretch,
either at RV 1 (IR: fresh parity from the mother code) or at RV 0 again
(Chase). Soft bits combine in codes.combine, decode in codes.decode_buffer.
Reported beside the simulator's model of the same thing
(scripts/linksim.py SimRx: P = curve(MI_1 + MI_2), MI from the sim channel).

    uv run python scripts/ir_study.py --out runs/ir_study.csv
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import codes, hfchannel, modem
from data2g.config import SUBMODES

sys.path.insert(0, str(Path(__file__).parent))
import linksim as L  # noqa: E402

MODES = ("w48-qpsk-r1/2", "w48-16qam-r2/3", "w48-64l-r1/2", "n10-qpsk-r1/2", "qpsk-r1/5", "ack-4f")
CHANNELS = ("awgn", "mpp")
N_CW = 8
TRIALS = 12
OFFSETS_DB = (-7.5, -6.0, -4.5, -3.0, -1.5, 0.0)  # from the mode's single-shot 1% threshold


def thresholds() -> dict:
    out = {}
    for r in csv.DictReader(open(L.ROOT / "runs/ladder_final.csv")):
        out[(r["name"], r["channel"])] = float(r["threshold_db"])
    return out


def ladder_name(s) -> str:
    return f"{'' if s.band == 'w' else s.band + '-'}{s.code}-{s.constellation}-f{s.frames_per_cw}-k{s.k}@h{s.headroom:g}"


def burst_soft(spec, payloads, rvs, chan, snr, seed):
    x = modem.modulate(payloads, spec, rvs)
    x = np.concatenate([np.zeros(2400), x, np.zeros(2400)])
    y = hfchannel.apply_channel(x, snr_db=snr, fading_preset=None if chan == "awgn" else chan, seed=seed)
    try:
        b = modem.demodulate(y)
    except modem.SyncError:
        return None
    return b.soft if b.submode.name == spec.name else None


def trial(args):
    name, chan, snr, seed = args
    spec = SUBMODES[name]
    rng = np.random.default_rng(seed)
    payloads = [bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8)) for _ in range(N_CW)]
    s0 = burst_soft(spec, payloads, [0] * N_CW, chan, snr, seed)
    if s0 is None:
        return None  # sync lost: not what this measures
    one = codes.combine(spec, None, s0, 0)
    ok1 = [ok and p == q for (p, ok), q in zip(codes.decode_buffer(spec, one), payloads)]
    out = dict(mode=name, channel=chan, snr=snr, seed=seed, one=float(np.mean(ok1)))
    for kind, rv in (("ir", 1), ("chase", 0)):
        s1 = burst_soft(spec, payloads, [rv] * N_CW, chan, snr, seed + 1_000_003 * (rv + 1))
        if s1 is None:
            out[kind] = float("nan")
            continue
        both = codes.combine(spec, one.copy(), s1, rv)
        dec = codes.decode_buffer(spec, both, max_rv=rv)
        out[kind] = float(np.mean([ok and p == q for (p, ok), q in zip(dec, payloads)]))
    # the simulator's model at this SNR: two independent bursts' MI, summed
    dop, dly = L.PRESETS[chan]
    ch = L.Channel(dop, dly, snr, seed=seed, duration=400)
    t = rng.uniform(1, 300, 2)
    mi = [L.burst_mi(ch, spec, ti, N_CW * spec.frames_per_cw) for ti in t]
    out["mi0"], out["mi1"] = mi
    out["sim_one"] = L.p_decode(name, mi[0])
    out["sim_ir"] = L.p_decode(name, min(1.0, mi[0] + mi[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ir_study.csv")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    thr = thresholds()
    jobs = []
    for name in MODES:
        for chan in CHANNELS:
            t = thr.get((ladder_name(SUBMODES[name]), chan))
            if t is None:
                print("no threshold", name, chan)
                continue
            jobs += [(name, chan, t + d, 1000 * i + j) for i, d in enumerate(OFFSETS_DB) for j in range(TRIALS)]
    rows = []
    with Pool(a.jobs) as pool:
        for r in pool.imap_unordered(trial, jobs, chunksize=2):
            if r is not None:
                rows.append(r)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(a.out)


def summarize(path):
    from collections import defaultdict
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        g[(r["mode"], r["channel"], float(r["snr"]))].append(r)
    print(f"{'mode':16s} {'chan':5s} {'snr':>6s} {'n':>3s} | PHY: one    IR  Chase | sim: one    IR")
    for k in sorted(g):
        rs = g[k]
        m = {c: np.nanmean([float(r[c]) for r in rs]) for c in ("one", "ir", "chase", "sim_one", "sim_ir")}
        print(f"{k[0]:16s} {k[1]:5s} {k[2]:6.2f} {len(rs):3d} |     {m['one']:.2f}  {m['ir']:.2f}  {m['chase']:.2f} |"
              f"      {m['sim_one']:.2f}  {m['sim_ir']:.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(sys.argv[2])
    else:
        main()
