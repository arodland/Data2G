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

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g.arq import phy as PHY
from data2g.arq.modes import MODES

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
        spec = MODES[burst.submode]
        row = dict(self.tag, submode=spec.name, band=spec.band, n_cw=len(burst.slots),
                   ctl_slots=sum(1 for s in burst.slots if s.mask_id[2] >= 128))
        row["kind"] = "data" if row["n_cw"] > row["ctl_slots"] else "ctl"
        rx_station = self.listen(burst, t0, end)
        r = self.hear(x, t0, rx_station)
        noise = self.heard_noise(rx_station, r, t0, end)
        if r is None:
            row.update(outcome="missed", data_sent=0, data_ok=0)
            self.rows.append(row)
            return end, None, None, None
        right = r["spec"].name == spec.name and r["n_cw"] == len(burst.slots)
        ok = []
        if right:
            rx = PHY.ModemRx(r, {})
            ok = [None if s.rv else rx.decode(i, s.mask_id, 0, None) == s.payload for i, s in enumerate(burst.slots)]
        # control: each codeword alone, or (ARQ_DUP: RV 0 then RV 1 of it) as a pair
        ctl_ok = right and ctl_decoded(r, burst.slots)
        row["dup"] = int(any(s.rv for s in burst.slots if s.mask_id[2] >= 128))
        data = [o for o, s in zip(ok, burst.slots) if s.mask_id[2] < 128 and o is not None]
        row.update(outcome="ok" if ctl_ok else ("ctl_lost" if right else "header"),
                   data_sent=len([s for s in burst.slots if s.mask_id[2] < 128 and not s.rv]), data_ok=sum(data),
                   snr_est=round(float(PHY.measure(r)["snr_est"]), 1))
        self.rows.append(row)
        t_hdr = G.header_time(r, t0)

        def make_rx(store, stats, rng):
            return PHY.ModemRx(r, store)
        return end, (t_hdr, r["spec"].name, r["n_cw"]), make_rx, dict(PHY.measure(r), noise=noise)


def ctl_decoded(r, slots) -> bool:
    """Every control codeword decoded, alone or combined with its RV 1 copy."""
    rx = PHY.ModemRx(r, {})
    i = 0
    while i < len(slots) and slots[i].mask_id[2] >= 128:
        s = slots[i]
        pair = i + 1 < len(slots) and slots[i + 1].mask_id == s.mask_id and slots[i + 1].rv == 1
        good = rx.decode(i, s.mask_id, 0, None) == s.payload
        if not good and pair:
            rx.decode(i, s.mask_id, 0, "ctl")  # RV 0 stored, then RV 1 combined with it
            good = rx.decode(i + 1, s.mask_id, 1, "ctl") == s.payload
        if not good:
            return False
        i += 2 if pair else 1
    return True


def one(args):
    chan, snr, seed, horizon, policy = args
    rows = []
    ch = G.ContinuousChannel(chan, snr, seed, horizon)
    tag = dict(channel=chan, snr=snr, seed=seed)
    res = L.run(L.make_policy(policy), L.make_policy(policy), None, L.WORKLOADS["bulk"](random.Random(seed + 7)),
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
            dups = [r for r in ks if r.get("dup") == "1" and r["outcome"] in ("ok", "ctl_lost")]
            singles = [r for r in ks if r.get("dup") != "1" and r["outcome"] in ("ok", "ctl_lost")]
            if kind == "data" and dups:
                lost = lambda v: sum(r["outcome"] == "ctl_lost" for r in v) / max(len(v), 1)  # noqa: E731
                print(f"       duplicated control in {len(dups)}/{len(dups) + len(singles)} heard data bursts: "
                      f"control lost {lost(dups):.0%} duplicated vs {lost(singles):.0%} single")
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
    ap.add_argument("--policy", default="shift", help='linksim.make_policy spec ("shift+cpm": CPM modes too)')
    ap.add_argument("--cells", default=None, help="channel:snr,... (default CELLS)")
    ap.add_argument("--average-snr", action="store_true",
                    help="allow SNR against each burst's average power (without DATA2G_PEP_REF_DB)")
    a = ap.parse_args()
    if G.PEP_REF_DB is None and not a.average_snr:
        ap.error("DATA2G_PEP_REF_DB is unset: set it (5: noise against each burst's peak, as data2g-host "
                 "transmits) or pass --average-snr")
    cells = [(c, float(s)) for c, s in (x.split(":") for x in a.cells.split(","))] if a.cells else CELLS
    jobs = [(c, s, seed, a.horizon, a.policy) for c, s in cells for seed in range(a.seeds)]
    with Pool(a.jobs) as pool, open(a.out, "w", newline="") as f:
        w = None
        for rows in pool.imap_unordered(one, jobs):
            for r in rows:
                if w is None:
                    w = csv.DictWriter(f, ["channel", "snr", "seed", "submode", "band", "n_cw", "ctl_slots", "dup", "kind",
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
