"""Idle-turn rules on the real modem (scripts/phy_session.py's RealPhy),
paired seeds: v1 (only the caller starts turns, polls 2 s doubling to
16 s) against the callee's wake bursts plus a 15-60 s jittered keepalive
(data2g/arq/session.py). Carrier sense as on air (linksim.CS_S): a burst
whose header was missed, or that started under CS_S earlier, is keyed
over, and an overlap loses both bursts.

Per session: each direction's mean step latency (write to delivery, an
undelivered step counted to the horizon), completion time, each
station's airtime, collisions.

    DATA2G_PEP_REF_DB=5 uv run python scripts/idle_study.py --out runs/idle_study.csv
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g.arq import session as S  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import linksim as L  # noqa: E402
import phy_session as G  # noqa: E402

VARIANTS = {
    "v1": dict(KEEPALIVE_S=(2.0, 16.0), CHAT_KEEPALIVE_S=(2.0, 4.0), KEEPALIVE_JITTER=0.0, WAKE_TRIES=0,
               CHAT_WAKE_TRIES=0, LINK_LOST_S=90.0),
    "wake": {},  # session.py as it stands
}
CELLS = [("awgn", 8.0), ("mpp", 8.0), ("mpp", 2.0), ("mpd", 4.0)]
WORKLOADS = ("chat", "winlink")
HORIZON = 1800.0


def one(args):
    variant, workload, chan, snr, seed, policy = args
    for k, v in VARIANTS[variant].items():
        setattr(S, k, v)
    ch = G.ContinuousChannel(chan, snr, seed, HORIZON)
    steps = L.WORKLOADS[workload](random.Random(seed + 7))
    res = L.run(L.make_policy(policy), L.make_policy(policy), None, steps, seed=seed, horizon=HORIZON,
                phy=G.RealPhy(ch), cs_s=L.CS_S)
    row = dict(variant=variant, workload=workload, channel=chan, snr=snr, seed=seed, complete=int(res["complete"]),
               t=round(res["t"], 1), collisions=res["collisions"], timeouts=res["timeouts"],
               reason="|".join(res["reason"])[:120])
    for w in "ab":
        lat = [(d if d is not None else HORIZON) - tw for (who, *_), tw, d in zip(steps, res["t_write"], res["t_done"])
               if who == w and tw is not None]
        row[f"lat_{w}"] = round(float(np.mean(lat)), 2) if lat else ""
        row[f"air_{w}"] = round(sum(v for k, v in res["time"].items() if k.startswith(w + "_")), 1)
    return row


def summarize(path):
    g = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        g[(r["workload"], r["channel"], float(r["snr"]))].setdefault(r["variant"], {})[int(r["seed"])] = r
    for k in sorted(g):
        v = g[k]
        seeds = sorted(set.intersection(*(set(x) for x in v.values())))
        done = ", ".join(f"{n} {sum(int(v[n][s]['complete']) for s in seeds)}" for n in v)
        print(f"\n== {k[0]} {k[1]} {k[2]:+.0f} dB, {len(seeds)} paired seeds (complete: {done})")
        for c in ("lat_a", "lat_b", "t", "air_a", "air_b", "collisions"):
            cells = []
            for n in VARIANTS:
                xs = [float(v[n][s][c]) for s in seeds if n in v and v[n][s][c] != ""]
                cells.append(f"{n} {np.mean(xs):8.1f}" if xs else f"{n} -")
            print(f"  {c:10s} " + "   ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/idle_study.csv")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--policy", default="shift+cpm")
    a = ap.parse_args()
    if G.PEP_REF_DB is None:
        ap.error("DATA2G_PEP_REF_DB is unset (5: noise against each burst's peak, as data2g-host transmits)")
    jobs = [(v, w, c, s, seed, a.policy) for w in WORKLOADS for c, s in CELLS for seed in range(a.seeds)
            for v in VARIANTS]
    # one task per worker: a variant patches session's constants for the process
    with Pool(a.jobs, maxtasksperchild=1) as pool, open(a.out, "w", newline="") as f:
        w = None
        for r in pool.imap_unordered(one, jobs):
            if w is None:
                w = csv.DictWriter(f, list(r))
                w.writeheader()
            w.writerow(r)
            f.flush()
    summarize(a.out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(sys.argv[2])
    else:
        main()
