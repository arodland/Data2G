"""Streamed header detection of single bursts next to steady tones (PR #308 review, item 8): the receiver a session
runs (receiver.Receiver with the engine's accept rule and CPM grids), fed in 20 ms blocks, a fresh one per burst.
Each burst goes through phy_session's ContinuousChannel with 1 s of noise before and 2 s after, plus steady tones
at the receiver (interference.Qrm "carrier", +25 dB over the floor in 50 Hz, always on). Per burst:
  header    a header of the sent mode found within 0.5 s of the burst's start
  received  that burst handed on with a receive result
  false     headers of anything else seen

The pilot layout is read at import (DATA2G_PILOTS), so one process per arm; --native substitutes the native core as
a session endpoint does (tests/native_subs, the arq/gear/engine/host providers kept in Python).

Adapted from finitor's scripts/studies/ffi_qrm_carriers/stream_detect_tone.py (github.com/finitor/Data2G, 3b0f94b;
same project and licence).

    DATA2G_PEP_REF_DB=5 DATA2G_PILOTS=mix python -m scripts.sc_stream_tone --out runs/x_mix.csv
    python -m scripts.sc_stream_tone --compare runs/x_off.csv runs/x_mix.csv
"""

import os
import sys

from data2g import threads

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

if "--native" in sys.argv:
    sys.path.insert(0, "tests")
    import native_subs  # noqa: E402

    _n, _why = native_subs.import_native()
    assert _n is not None, _why
    _n.set_threads(1)
    native_subs.apply(native_subs.build(_n, skip={"arq", "gear", "engine", "host"}))

from data2g import codes, cpm, modem  # noqa: E402
from data2g import interference as I  # noqa: E402
from data2g.arq.engine import receive_accept  # noqa: E402
from data2g.arq.policy import slots_for  # noqa: E402
from data2g.config import FS, LEADIN_SAMPLES, SUBMODES  # noqa: E402
from data2g.receiver import Receiver  # noqa: E402
from scripts import phy_session as G  # noqa: E402

BLK = FS // 50
LEAD_S, TAIL_S = 1.0, 2.0
BURST_S = 4.0
TONE_INR_DB = 25.0
TONES = {"none": (), "800": (800.0,), "2200": (2200.0,), "both": (800.0, 2200.0)}


def one(args):
    name, chan, snr, tone, seed = args
    spec = SUBMODES[name]
    rng = np.random.default_rng([seed, 3])
    n_cw = slots_for(spec, BURST_S) if "ack" not in name else 1
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)], name)
    xs = np.concatenate([np.zeros(int(LEAD_S * FS)), x, np.zeros(int(TAIL_S * FS))])
    q = tuple(I.Qrm("carrier", f, 50.0, TONE_INR_DB, 600.0, 1.0) for f in TONES[tone])
    s = I.Spec(qrm=q)
    y = G.ContinuousChannel(chan, snr, seed, 60.0, interference=(s, s)).apply(xs, 5.0, rx=0)
    start = int(G.PAD_S * FS) + int(LEAD_S * FS) + LEADIN_SAMPLES
    r = Receiver(receive_accept(), cpm_grids=tuple(cpm.GRIDS))
    header = received = False
    false = 0
    for i in range(0, len(y) - BLK + 1, BLK):
        for kind, d in r.feed(y[i:i + BLK]):
            p = d if kind == "header" else d["header"]
            mine = p["spec"].name == name and abs(p["start"] - start) < FS // 2
            if kind == "header":
                header |= mine
                false += not mine
            elif kind == "burst" and mine and d["rx"] is not None:
                received = True
    return dict(pilots=os.environ.get("DATA2G_PILOTS", ""), mode=name, chan=chan, snr=snr, tone=tone, seed=seed,
                header=int(header), received=int(received), false=false)


def compare(a, b):
    def load(p):
        out = {}
        for r in csv.DictReader(open(p)):
            out.setdefault((r["mode"], r["chan"], float(r["snr"]), r["tone"]), []).append(r)
        return out

    A, B = load(a), load(b)
    print("| mode | chan | SNR | tone | header A / B | received A / B | false A / B |\n|---|---|---|---|---|---|---|")
    for k in sorted(A, key=lambda k: (k[0], k[1], ("none", "800", "2200", "both").index(k[3]), k[2])):
        x, y = A[k], B.get(k, [])
        f = lambda rows, c: sum(int(r[c]) for r in rows)  # noqa: E731
        print(f"| {k[0]} | {k[1]} | {k[2]:+.0f} | {k[3]} | {f(x, 'header')}/{len(x)} / {f(y, 'header')}/{len(y)} | "
              f"{f(x, 'received')} / {f(y, 'received')} | {f(x, 'false')} / {f(y, 'false')} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--modes", default="16qam-r1/3,qpsk-r1/2,ack-1f")
    ap.add_argument("--chans", default="mpp")
    ap.add_argument("--snrs", default="-2,0,2")
    ap.add_argument("--tones", default="none,both")
    ap.add_argument("--bursts", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    if G.PEP_REF_DB is None:
        ap.error("set DATA2G_PEP_REF_DB=5")
    jobs = [(m, c, float(s), t, seed) for m in a.modes.split(",") for c in a.chans.split(",")
            for s in a.snrs.split(",") for t in a.tones.split(",") for seed in range(a.bursts)]
    with Pool(a.jobs, maxtasksperchild=50) as pool, open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, ["pilots", "mode", "chan", "snr", "tone", "seed", "header", "received", "false"])
        w.writeheader()
        for row in pool.imap_unordered(one, jobs, chunksize=2):
            w.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()
