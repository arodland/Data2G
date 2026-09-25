"""Where bursts die at low SNR: real-modem ARQ sessions (scripts/phy_session.py)
with every burst sent classified at its receiver:
  missed      no preamble + header found
  header      a header found, but not this burst's (submode or length wrong)
  ctl_lost    the right header, control codeword failed (the burst is wasted)
  ok          control decoded
plus, per burst, first-transmission data codewords sent / decoded against
their true identities (so data that decoded under a failed control shows).

    uv run python scripts/loss_study.py --out runs/loss_study.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import codes, modem
from data2g.arq import phy as PHY
from data2g.config import SUBMODES

sys.path.insert(0, str(Path(__file__).parent))
import linksim as L  # noqa: E402
import phy_session as G  # noqa: E402

CELLS = [("mpg", -4.0), ("mpp", -4.0), ("awgn", -4.0), ("mpg", 0.0), ("mpp", 0.0)]


class AuditPhy(G.RealPhy):
    def __init__(self, ch, rows, tag):
        super().__init__(ch)
        self.rows, self.tag = rows, tag

    def send(self, burst, t0):
        x = PHY.tx_audio(burst)
        end = t0 + len(x) / G.FS
        spec = SUBMODES[burst.submode]
        row = dict(self.tag, submode=spec.name, band=spec.band, n_cw=len(burst.slots),
                   ctl_slots=sum(1 for s in burst.slots if s.mask_id[2] >= 128))
        row["kind"] = "data" if row["n_cw"] > row["ctl_slots"] else "ctl"
        try:
            r = modem.receive(self.ch.apply(x, t0))
        except modem.SyncError:
            row.update(outcome="missed", data_sent=0, data_ok=0)
            self.rows.append(row)
            return end, None, None, None
        soft = PHY.soft_bits(r)
        right = r["spec"].name == spec.name and r["n_cw"] == len(burst.slots)
        ok = []
        if right:
            for i, s in enumerate(burst.slots):
                if s.rv:
                    ok.append(None)
                    continue
                p, good = codes.decode_many(spec, soft[i:i + 1], PHY.mask_value(s.mask_id), index=0)[0]
                ok.append(bool(good and p == s.payload))
        ctl_ok = right and all(ok[:row["ctl_slots"]])
        data = [o for o, s in zip(ok, burst.slots) if s.mask_id[2] < 128 and o is not None]
        row.update(outcome="ok" if ctl_ok else ("ctl_lost" if right else "header"),
                   data_sent=len([s for s in burst.slots if s.mask_id[2] < 128 and not s.rv]), data_ok=sum(data),
                   snr_est=round(float(PHY.measure(r)["snr_est"]), 1))
        self.rows.append(row)
        sb = modem.BANDS[r["spec"].sync_band]
        t_hdr = t0 + (G.LEADIN_SAMPLES + sb.preamble_samples + modem.header_samples(r["spec"].sync_band)) / G.FS

        def make_rx(store, stats, rng):
            return PHY.ModemRx(r, store)
        return end, (t_hdr, r["spec"].name, r["n_cw"]), make_rx, PHY.measure(r)


def one(args):
    chan, snr, seed, horizon = args
    rows = []
    ch = G.ContinuousChannel(chan, snr, seed, horizon)
    tag = dict(channel=chan, snr=snr, seed=seed)
    res = L.run(L.make_policy("shift"), L.make_policy("shift"), None, L.WORKLOADS["bulk"](random.Random(seed + 7)),
                seed=seed, horizon=horizon, phy=AuditPhy(ch, rows, tag))
    for r in rows:
        r["delivered_Bps"] = round(res["delivered"] / horizon, 1)
    return rows


def summarize(path):
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        g[(r["channel"], float(r["snr"]))].append(r)
    for k in sorted(g):
        rs = g[k]
        seeds = {r["seed"]: float(r["delivered_Bps"]) for r in rs}
        print(f"\n== {k[0]} {k[1]:+.0f} dB: {len(rs)} bursts, delivered {np.mean(list(seeds.values())) * 8:.0f} bps "
              f"(mean over {len(seeds)} seeds)")
        for kind in ("data", "ctl"):
            ks = [r for r in rs if r["kind"] == kind]
            if not ks:
                continue
            c = defaultdict(int)
            for r in ks:
                c[r["outcome"]] += 1
            ds, dk = sum(int(r["data_sent"]) for r in ks), sum(int(r["data_ok"]) for r in ks)
            wasted = sum(int(r["data_ok"]) for r in ks if r["outcome"] == "ctl_lost")
            print(f"  {kind:4s} bursts {len(ks):4d}: " + "  ".join(f"{o} {c[o] / len(ks):5.1%}" for o in
                                                           ("missed", "header", "ctl_lost", "ok"))
                  + (f" | first-tx data cw {dk}/{ds}, {wasted} of them in ctl-lost bursts" if ds else ""))
        by = defaultdict(lambda: defaultdict(int))
        for r in rs:
            by[r["submode"]][r["outcome"]] += 1
        top = sorted(by.items(), key=lambda kv: -sum(kv[1].values()))[:6]
        print("  by mode: " + "; ".join(f"{m} {sum(v.values())}: missed {v['missed']} hdr {v['header']} "
                                        f"ctl {v['ctl_lost']}" for m, v in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/loss_study.csv")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--horizon", type=float, default=300.0)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    jobs = [(c, s, seed, a.horizon) for c, s in CELLS for seed in range(a.seeds)]
    with Pool(a.jobs) as pool, open(a.out, "w", newline="") as f:
        w = None
        for rows in pool.imap_unordered(one, jobs):
            for r in rows:
                if w is None:
                    w = csv.DictWriter(f, ["channel", "snr", "seed", "submode", "band", "n_cw", "ctl_slots", "kind",
                                           "outcome", "data_sent", "data_ok", "snr_est", "delivered_Bps"])
                    w.writeheader()
                w.writerow({k: r.get(k, "") for k in w.fieldnames})
            f.flush()
    summarize(a.out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(sys.argv[2])
    else:
        main()
