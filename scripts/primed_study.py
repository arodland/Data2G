"""Primed reception (docs/arq.md §3a): what a station waiting for a reply
gains from knowing much of it, at code and PHY level.

    DATA2G_PEP_REF_DB=5 python -m scripts.primed_study ctl --modes fsk16r25-r1/3 ack-4f --chans awgn mpp
    python -m scripts.primed_study summarize runs/primed_ctl.csv

`ctl`: bursts (a reply's control, then one data codeword unless --ack)
through phy_session's ContinuousChannel and the real receiver
(receiver.receive_any), the same trial seed at every SNR and in every arm.
The control is a reply as sessions send them: an ACK (core word, T_REPLY,
T_DUPCTL: 8 B) or, with --data, a data burst's (core, T_NEW, T_REPLY,
T_DUPCTL: 11 B). Arms, each a retry after the blind decode (phy.ModemRx.decode)
fails, with these payload bits known (codes.known_bits):
  blind    the shipped decode alone (plain, then the padding cuts or estimate)
  len      the padding from the control's exact end
  core     len, and the core word but its cumulative ACK (7 bits) and
           recommendation (6)
  tmpl     core, and the TLVs (types, lengths, the T_REPLY value)
  cum      everything but the cumulative ACK
  shipped  link.Station's own prediction (link.expect) for the reply
Per arm: control decoded (every control codeword) of all bursts sent
(`ok`), and of those whose header was read right (`hdr`, the code alone).
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import codes, cpm
from data2g.arq import frames as F
from data2g.arq import phy as PHY
from data2g.arq.burst import Slot, TxBurst, ctl_mask, data_mask
from data2g.arq.modes import MODES, ctl_spec, is_cpm
from data2g.receiver import receive_any

sys.path.insert(0, str(Path(__file__).parent))
import phy_session as PS  # noqa: E402

KEY = 0x5A5A
ARMS = ("blind", "len", "core", "tmpl", "cum", "free13", "free19", "free21", "all")
GRID = {"fsk16r25": -22.0, "fsk8r50": -19.0, "fsk32r62": -16.0, "n4-ack": -18.0, "ack": -16.0, "n10-ack": -16.0,
        "qpsk": -16.0, "n10-qpsk": -16.0, "n4-qpsk": -18.0}


def control(rng, data: bool) -> F.Control:
    core = F.Core(ftype=F.ARQ, burst_seq=int(rng.integers(8)), acted_on=int(rng.integers(8)), cum=int(rng.integers(128)),
                  recommend=int(rng.integers(64)), size_hint=int(rng.integers(4)))
    ext = {F.T_REPLY: bytes([int(rng.integers(64))]), F.T_DUPCTL: b""}
    if data:
        ext[F.T_NEW] = bytes([int(rng.integers(128))])
    return F.Control(core, ext)


def masks(ctl: F.Control, stream: bytes, arm: str) -> bytes:
    """The stream's known bits under `arm`, MSB first, as a byte mask."""
    n = len(stream)
    used = 4 + sum(2 + len(v) for v in ctl.ext.values())
    m = bytearray(n)
    m[used:] = b"\xff" * (n - used)  # len and on: the padding
    if arm == "len":
        return bytes(m)
    core = (1 << 32) - 1
    # core word fields, from the most significant: ftype 2, n_ctl 2, seq 3, acted 3, cum 7, lost 1, k 6, rec 6, hint 2
    cum = ((1 << 7) - 1) << (1 + 6 + 6 + 2)
    rec = ((1 << 6) - 1) << 2
    if arm != "all":
        core &= ~cum
    if arm in ("core", "free13", "free19", "free21"):
        core &= ~rec
    m[:4] = core.to_bytes(4, "big")
    if arm in ("tmpl", "cum", "all", "free13", "free19", "free21"):
        m[4:used] = b"\xff" * (used - 4)
    if arm == "free21":
        m[6] = 0  # the T_REPLY value (the first TLV of an ACK's)
    if arm == "free19":
        m[6] = 0xC0  # the T_REPLY value but its 2 top bits (a 6-bit recommendation code)
    return bytes(m)


PRIME_TOL_S = 0.15  # the expected start's uncertainty, either way (receiver.ReplySlot.TOL_S)


def trial(args):
    mode, chan, snr, seed, data, arms = args
    spec = MODES[mode]
    rng = np.random.default_rng(seed)
    ctl = control(rng, data)
    cpb = codes.payload_bytes(ctl_spec(spec))
    payloads = ctl.pack(cpb)
    stream = b"".join(payloads)
    slots = [Slot(ctl_mask(1, i, KEY), 0, p) for i, p in enumerate(payloads)]
    if "ack" not in mode:
        rand = bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8))
        slots.append(Slot(data_mask(1, 3, KEY), 0, rand))
    b = TxBurst(mode, slots, 0, 2)
    ch = PS.ContinuousChannel(chan, snr, seed, 120.0)
    y = ch.apply(PHY.tx_audio(b), float(rng.uniform(5, 100)))
    try:
        r = receive_any(y, lead=int(PS.PAD_S * PS.FS) + PS.FS // 2)
    except Exception:
        r = None
    hdr = r is not None and r["spec"].name == mode and r["n_cw"] == len(slots)
    rg = None
    if any(a.startswith("g_") for a in arms) and is_cpm(spec):
        # the genie lock: start, CFO and header known (cpm.receive refines them as it would a real lock's)
        rg = cpm.receive(y, dict(spec=spec, n_data=len(slots) - 1, dup=False, start=int(PS.PAD_S * PS.FS),
                                 cfo=PS.CFO_HZ, band=spec.grid))
    rp, psc = None, -np.inf
    if any(a.startswith("p") and "_" in a for a in arms) and is_cpm(spec):
        if hdr:
            rp, psc = r, np.inf
        else:  # blind acquisition failed: the expected slot, any word of the expected grid
            g = cpm.GRIDS[spec.grid]
            p0, tol = int(PS.PAD_S * PS.FS), int(PRIME_TOL_S * PS.FS)
            floor, cpm.PRIME_LLR = cpm.PRIME_LLR, -np.inf  # every arm's floor from the one search
            lock = cpm.prime_lock(g, y, p0 - tol, p0 + tol, PS.CFO_HZ)
            cpm.PRIME_LLR = floor
            psc = lock["score"]
            rp = cpm.receive(y, lock) if lock is not None else None
            if rp is not None and not (rp["spec"].name == mode and rp["n_cw"] == len(slots)):
                rp = None  # locked on the wrong word: the control's CRC would fail
    out = {}
    for arm in arms:
        ok = False
        primed = arm[0] == "p" and "_" in arm  # p<T>_<arm>: a primed lock scoring at least T
        genie = arm.startswith("g_") or primed
        base = arm.split("_", 1)[1] if genie else arm
        rr = rg if arm.startswith("g_") else (rp if psc >= float(arm[1:].split("_")[0]) else None) if primed else r
        h = rr is not None if genie else hdr
        if h:
            rx = PHY.ModemRx(dict(rr), {})
            ok = True
            for i, s in enumerate(slots[:len(payloads)]):
                got = rx.decode(i, s.mask_id, 0, None)
                if got is None and base != "blind":
                    m = masks(ctl, stream, base)
                    got = rx.decode_known(i, s.mask_id, stream[i * cpb:(i + 1) * cpb], m[i * cpb:(i + 1) * cpb])
                if got != s.payload:
                    ok = False
                    break
        out[arm] = (int(h), int(ok))
    return mode, chan, snr, seed, out


def grid(mode, chan):
    lo = GRID.get(mode.split("-")[0] if is_cpm(MODES[mode]) else mode.rsplit("-", 1)[0], -10.0)
    if chan != "awgn":
        lo += 2.0
    return np.arange(lo, lo + (10.0 if chan == "awgn" else 16.0), 1.0)


def crossing(xs, fail, target):
    """The lowest SNR from which failures stay under target (interpolated)."""
    xs, fail = np.asarray(xs, float), np.asarray(fail, float)
    above = np.flatnonzero(fail > target)
    if not len(above):
        return xs[0]
    i = above[-1]
    if i + 1 >= len(xs):
        return np.nan
    x0, x1, f0, f1 = xs[i], xs[i + 1], fail[i], fail[i + 1]
    return float(x0 + (f0 - target) / (f0 - f1) * (x1 - x0))


def summarize(path):
    rows = defaultdict(list)
    for r in csv.DictReader(open(path)):
        rows[(r["mode"], r["channel"], r["arm"])].append((float(r["snr"]), int(r["hdr"]), int(r["ok"])))
    print("SNR (cell dB) at 10% / 1% control failure: all bursts | given the header read; arm - blind in brackets")
    for mode, chan in sorted({(m, c) for m, c, _ in rows}):
        arms = [a for a in (*ARMS, "shipped", *("g_" + x for x in ARMS)) if (mode, chan, a) in rows]
        arms += sorted({a for m, c, a in rows if (m, c) == (mode, chan) and a[0] == "p" and "_" in a},
                       key=lambda a: float(a[1:].split("_")[0]))
        pts = {}
        for a in arms:
            v = rows[(mode, chan, a)]
            snrs = sorted({s for s, _, _ in v})
            fall = [1 - np.mean([o for s2, _, o in v if s2 == s]) for s in snrs]
            fh = [1 - np.mean([o for s2, h, o in v if s2 == s and h] or [0]) for s in snrs]
            hd = [1 - np.mean([h for s2, h, _ in v if s2 == s]) for s in snrs]
            pts[a] = (crossing(snrs, fall, 0.1), crossing(snrs, fall, 0.01), crossing(snrs, fh, 0.1), crossing(snrs, fh, 0.01),
                      crossing(snrs, hd, 0.1), crossing(snrs, hd, 0.01))
        b = pts["blind"]
        print(f"  {mode:15s} {chan:5s} header {b[4]:6.2f}/{b[5]:6.2f}")
        for a, p in pts.items():
            d = "" if a == "blind" else f"  ({p[0] - b[0]:+.2f}/{p[1] - b[1]:+.2f} | {p[2] - b[2]:+.2f}/{p[3] - b[3]:+.2f})"
            print(f"      {a:8s} {p[0]:6.2f}/{p[1]:6.2f} | {p[2]:6.2f}/{p[3]:6.2f}{d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ctl", "summarize"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--modes", nargs="+", default=["fsk16r25-r1/3", "fsk8r50-r1/3", "fsk32r62-r1/3", "ack-4f", "n10-ack-4f"])
    ap.add_argument("--chans", nargs="+", default=["awgn", "mpp"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--data", action="store_true", help="a data burst's control (T_NEW) in place of an ACK's")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default="runs/primed_ctl.csv")
    ap.add_argument("--primed-score", type=float, default=None, help="phy.PRIMED_SCORE for this run")
    ap.add_argument("--primed-paths", type=int, default=None, help="phy.PRIMED_PATHS for this run")
    a = ap.parse_args()
    if a.cmd == "summarize":
        return summarize(a.path)
    if a.primed_score is not None:
        PHY.PRIMED_SCORE = a.primed_score
    if a.primed_paths is not None:
        PHY.PRIMED_PATHS = a.primed_paths
    if PS.PEP_REF_DB is None:
        ap.error("set DATA2G_PEP_REF_DB (5, as the loss studies)")
    jobs = [(m, c, float(s), 70000 + i, a.data, a.arms) for m in a.modes for c in a.chans for s in grid(m, c)
            for i in range(a.trials)]
    with Pool(a.jobs) as p, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "channel", "snr", "seed", "arm", "hdr", "ok"])
        for m, c, s, seed, out in p.imap_unordered(trial, jobs, chunksize=4):
            for arm, (h, ok) in out.items():
                w.writerow([m, c, s, seed, arm, h, ok])
    summarize(a.out)


if __name__ == "__main__":
    main()
