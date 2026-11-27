"""Scattered pilots (DATA2G_PILOTS=k3d2, config.SC_BANDS) apart from the gear
shifter: fixed bursts of one mode through phy_session's channel (PEP-referenced
cells), received as a session's bursts are (receive_any, arq.phy.ModemRx with
its decision-directed passes). Per mode, channel and cell: bursts whose header
read right, and codewords decoded, of those sent. Run once per arm on the same
seeds and compare:

    DATA2G_PEP_REF_DB=5 python -m scripts.sc_phy_study --out runs/sc_phy_off.csv
    DATA2G_PEP_REF_DB=5 DATA2G_PILOTS=k3d2 python -m scripts.sc_phy_study --out runs/sc_phy_on.csv
    python -m scripts.sc_phy_study --compare runs/sc_phy_off.csv runs/sc_phy_on.csv
"""

from data2g import threads

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, modem  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.config import FS, SUBMODES  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import phy_session as G  # noqa: E402

BURST_S = 4.0  # codewords per burst: as many as fit


def one(args):
    name, chan, snr, seed = args
    spec = SUBMODES[name]
    rng = np.random.default_rng([seed, 3])
    n_cw = max(1, int((BURST_S - modem.burst_seconds(spec, 0)) / (modem.burst_seconds(spec, 1) - modem.burst_seconds(spec, 0))))
    pl = [rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)]
    x = modem.modulate(pl, name)
    ch = G.ContinuousChannel(chan, snr, seed, 60.0)
    y = ch.apply(x, 5.0)
    hdr = ok = 0
    try:
        r = G.receive_any(y, lead=int(G.PAD_S * FS) + FS // 2)
    except modem.SyncError:
        r = None
    if r is not None and r.get("family") != "cpm" and r["spec"].name == name and r["n_cw"] == n_cw:
        hdr = 1
        rx = PHY.ModemRx(r, {})
        ok = sum(rx.decode(i, (0, 0, 0), 0, None) == p for i, p in enumerate(pl))
    return dict(mode=name, chan=chan, snr=snr, seed=seed, n_cw=n_cw, hdr=hdr, ok=ok)


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        k = (r["mode"], r["chan"], float(r["snr"]))
        a = out.setdefault(k, [0, 0, 0, 0])
        a[0] += 1
        a[1] += int(r["hdr"])
        a[2] += int(r["n_cw"])
        a[3] += int(r["ok"])
    return out


def compare(a, b):
    A, B = load(a), load(b)
    print(f"{'mode':16s} {'chan':9s} {'snr':>4s}  header A / B    codewords A / B   (bursts)")
    for k in sorted(A):
        if k in B:
            x, y = A[k], B[k]
            print(f"{k[0]:16s} {k[1]:9s} {k[2]:+4.0f}  {x[1] / x[0]:5.2f} / {y[1] / y[0]:5.2f}    "
                  f"{x[3] / x[2]:5.2f} / {y[3] / y[2]:5.2f}      ({x[0]})")


def points(paths):
    """Per mode and channel, the cell where 10% of codewords fail (linear between the grid's cells), per file."""
    T = [load(p) for p in paths]
    keys = sorted({k[:2] for t in T for k in t})
    print(f"{'mode':16s} {'chan':9s} " + " ".join(f"{Path(p).stem[-12:]:>12s}" for p in paths))
    for m, c in keys:
        out = []
        for t in T:
            pts = sorted((k[2], v[3] / v[2]) for k, v in t.items() if k[:2] == (m, c))
            x = next((s0 + (0.9 - a0) * (s1 - s0) / (a1 - a0) for (s0, a0), (s1, a1) in zip(pts, pts[1:])
                      if a0 < 0.9 <= a1), None)
            out.append("    <" + f"{pts[0][0]:+.0f}".rjust(6) if pts and pts[0][1] >= 0.9 else "       never" if x is None else f"{x:+12.1f}")
        print(f"{m:16s} {c:9s} " + " ".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sc_phy.csv")
    ap.add_argument("--modes", default="qpsk-r1/5,qpsk-r1/2,16qam-r1/2,w48-qpsk-r1/2,w48-16qam-r1/2")
    ap.add_argument("--chans", default="hilat-m,lowlat-d,mpd,mpp,awgn")
    ap.add_argument("--snrs", default="0,5,10,15,20")
    ap.add_argument("--bursts", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--points", nargs="+")
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    if a.points:
        return points(a.points)
    if G.PEP_REF_DB is None:
        ap.error("set DATA2G_PEP_REF_DB=5")
    jobs = [(m, c, float(s), seed) for m in a.modes.split(",") for c in a.chans.split(",") for s in a.snrs.split(",")
            for seed in range(a.bursts)]
    with Pool(a.jobs) as pool, open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, ["mode", "chan", "snr", "seed", "n_cw", "hdr", "ok"])
        w.writeheader()
        for row in pool.imap_unordered(one, jobs, chunksize=4):
            w.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()
