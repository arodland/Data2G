"""The OFDM expected slot, burst level: modem.prime_ofdm after
the blind receive fails, in a window like ReplySlot's (+-TOL_S about a
centre up to --slot-err off the true start), the CFO known to +-1 Hz.

  1. Noise: prime_ofdm's best header score over --noise windows of
     white noise with no floor; the floor at --fa false accepts per
     window, printed with how many noise windows pass PRIME_FLOOR.
  2. Bursts: blind ok, then blind-or-primed ok, wrong headers, and the
     data ceiling (receive's `known`).

    uv run --no-sync python scripts/expect_prime_study.py --jobs 2
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import codes, hfchannel, modem
from data2g.arq.engine import receive_accept
from data2g.config import FS, LEADIN_SAMPLES, SUBMODES
from data2g.receiver import ReplySlot, receive_any

sys.path.insert(0, str(Path(__file__).parent))
from expect_header_study import expected_accept  # noqa: E402

TOL = int(ReplySlot.TOL_S * FS)
SPAN = 2 * FS  # audio past the window's end: every header fits


def the_set(band, which):
    if which == "full":
        return modem.Accept(tuple((n, c) for n, c in receive_accept().max_cw if SUBMODES[n].sync_band == band), 0.0)
    return expected_accept(band)


def noise(args):
    band, seed, n, which = args
    rng = np.random.default_rng(seed)
    floor = dict(modem.PRIME_FLOOR)
    modem.PRIME_FLOOR.update({k: -1.0 for k in floor})
    out = []
    for _ in range(n):
        y = rng.standard_normal(2 * TOL + SPAN + FS // 2)
        lock = modem.prime_ofdm(y, FS // 4, FS // 4 + 2 * TOL, rng.uniform(-50, 50), the_set(band, which))
        out.append(lock["score"] if lock else 0.0)
    modem.PRIME_FLOOR.update(floor)
    return out


def trial(args):
    mode, n_cw, chan, snr, seed, slot_err, floors, which = args
    modem.PRIME_FLOOR.update(floors)
    spec = SUBMODES[mode]
    rng = np.random.default_rng(seed)
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)], spec)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    f0 = rng.uniform(-50, 50)
    x = np.concatenate([np.zeros(lead), x, np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=f0, ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    start = lead + LEADIN_SAMPLES
    row = {}
    try:
        k = modem.decode_received(modem.receive(y, known=dict(start=start, cfo=f0, spec=spec, n_cw=n_cw)))
        row["data"] = all(k.crc_ok)
    except modem.SyncError:
        row["data"] = False
    try:  # as the session sims hear (phy_session.RealPhy.hear)
        r = receive_any(y, lead=lead + FS // 4)
    except modem.SyncError:
        r = None
    if r is None:
        row["blind"] = "lost"
    else:
        row["blind"] = "ok" if (r["spec"].name, r["n_cw"]) == (mode, n_cw) else "wrong"
    row["primed"] = row["blind"]
    if row["blind"] == "lost":
        c = start + int(rng.uniform(-slot_err, slot_err) * FS)
        lock = modem.prime_ofdm(y, max(0, c - TOL), c + TOL, f0 + rng.uniform(-1, 1), the_set(spec.sync_band, which))
        if lock is not None:
            right = (lock["spec"].name, lock["n_cw"]) == (mode, n_cw)
            row["primed"] = "ok" if right else "wrong"
            if right:
                try:
                    b = modem.decode_received(modem.receive(y, known=dict(lock)))
                    row["primed_data"] = all(b.crc_ok)
                except modem.SyncError:
                    row["primed_data"] = False
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["ack-4f", "n10-ack-4f", "n4-ack-8f"])
    ap.add_argument("--n-cw", type=int, nargs="+", default=[1])
    ap.add_argument("--at", nargs="+", default=["awgn:-12", "awgn:-10", "mpp:-8", "mpp:-6", "mpd:-8", "mpd:-6"])
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--noise", type=int, default=6000)
    ap.add_argument("--fa", type=float, default=1e-3)
    ap.add_argument("--slot-err", type=float, default=0.1, help="s: the window centre's error, uniform +-")
    ap.add_argument("--set", default="expected", choices=("expected", "full"),
                    help="the words prime_ofdm reads: the expected ones, or every one the station takes")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", 2)))
    a = ap.parse_args()
    with Pool(a.jobs) as pool:
        floors = {}
        for band in sorted({SUBMODES[m].sync_band for m in a.modes}):
            parts = 4 * a.jobs
            s = np.array([v for p in pool.map(noise, [(band, 70_000 + i, a.noise // parts, a.set) for i in range(parts)])
                          for v in p])
            q = float(np.quantile(s, 1 - a.fa))
            floors[band] = max(q, modem.PRIME_FLOOR[band]) if a.set == "expected" else q
            print(f"{band}: {len(s)} noise windows, score at 1-{a.fa:g}: {q:.3f}, max {s.max():.3f}; "
                  f"over PRIME_FLOOR {modem.PRIME_FLOOR[band]:.2f}: {int(np.sum(s >= modem.PRIME_FLOOR[band]))}; "
                  f"floor used {floors[band]:.3f}", flush=True)
        for mode in a.modes:
            for n_cw in a.n_cw:
                for at in a.at:
                    chan, snr = at.split(":")
                    rows = pool.map(trial, [(mode, n_cw, chan, float(snr), 40_000 + i, a.slot_err, floors, a.set)
                                            for i in range(a.trials)], chunksize=4)
                    f = lambda k, v: np.mean([r[k] == v for r in rows])  # noqa: E731
                    pd = [r["primed_data"] for r in rows if "primed_data" in r]
                    print(f"{mode:11s} x{n_cw} {chan:4s} {float(snr):+5.1f} | data {np.mean([r['data'] for r in rows]):.2f}"
                          f" | blind ok {f('blind', 'ok'):.2f} wrong {f('blind', 'wrong'):.2f}"
                          f" | +primed ok {f('primed', 'ok'):.2f} wrong {f('primed', 'wrong'):.2f}"
                          f" | primed locks {len(pd)}, their data {sum(pd)}", flush=True)


if __name__ == "__main__":
    main()
