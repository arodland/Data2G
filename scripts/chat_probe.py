"""Chat latency probe: per message, write -> delivery on the real modem (CHAT ON), and where the
time went. Write -> delivery per chat message (scripts/linksim.py chat workload, CHAT ON, real modem).

    DATA2G_PEP_REF_DB=5 python -m scripts.chat_probe 8 awgn:10 mpp:-6   # seeds, then channel:snr cells
"""
from data2g import threads

threads.limit(1)

import os  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import linksim as L  # noqa: E402
import phy_session as G  # noqa: E402

HORIZON = 900.0


def one(args):
    chan, snr, seed = args
    ch = G.ContinuousChannel(chan, snr, seed, HORIZON)
    steps = L.WORKLOADS["chat"](random.Random(seed + 7))
    res = L.run(L.make_policy("chat+cpm"), L.make_policy("chat+cpm"), None, steps, seed=seed, horizon=HORIZON,
                phy=G.RealPhy(ch), cs_s=L.CS_S)
    lat = [(d - tw) if d is not None else None for tw, d in zip(res["t_write"], res["t_done"])][1:]
    return chan, snr, seed, lat, res


if __name__ == "__main__":
    if sys.argv[1] == "keys":
        _, _, _, lat, res = one(("awgn", 10.0, 0))
        print(sorted(res.keys()))
        for k, v in res.items():
            if k not in ("t_write", "t_done"):
                print(k, str(v)[:400])
        print("lat", lat)
        sys.exit()
    cells = [(c, float(s)) for c, s in (x.split(":") for x in sys.argv[2:])]
    seeds = range(int(sys.argv[1]))
    with Pool(8) as p:
        out = p.map(one, [(c, s, k) for c, s in cells for k in seeds])
    if os.environ.get("CHAT_PROBE_OUT"):  # per message, for paired comparisons
        import json
        json.dump([(c, s, k, lat) for c, s, k, lat, _ in out], open(os.environ["CHAT_PROBE_OUT"], "w"))
    for c, s in cells:
        lats = [x for (cc, ss, _, lat, _) in out if (cc, ss) == (c, s) for x in lat]
        ok = [x for x in lats if x is not None]
        print(f"{c:5} {s:6.1f}: messages {len(lats)}, delivered {len(ok)}, latency median {np.median(ok):.1f} s, "
              f"p25 {np.percentile(ok, 25):.1f}, p75 {np.percentile(ok, 75):.1f}, p90 {np.percentile(ok, 90):.1f}")


def modes_report(cells, n):
    from collections import Counter
    with Pool(8) as p:
        out = p.map(one, [(c, s, k) for c, s in cells for k in range(n)])
    for c, s in cells:
        cnt = Counter()
        for (cc, ss, _, _, res) in out:
            if (cc, ss) == (c, s):
                cnt.update(res["modes"])
        tot = sum(cnt.values())
        print(f"{c} {s}: " + ", ".join(f"{m} {v / tot:.0%}" for m, v in cnt.most_common(6)))
