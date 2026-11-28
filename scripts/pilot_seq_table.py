"""pilot_seq_study's screen, read: per density and channel, each layout's bps (wiener arm) at 0 / 10 / 20 dB;
the best lattice against the best sequence; the best density per channel and cell, and 1/6's share of it."""
import csv
import sys
from collections import defaultdict

import numpy as np

R = list(csv.DictReader(open(sys.argv[1])))
arm = sys.argv[2] if len(sys.argv) > 2 else "wiener"
CH = ["mpg", "mpp", "mpd", "nvis-d", "lowlat-d", "hilat-m", "hilat-d"]
DN = ["1/12", "1/8", "1/6", "1/4", "1/3", "1/2"]
SN = [0.0, 10.0, 20.0]
I = {(r["layout"], r["dens"], r["chan"], float(r["snr"])): float(r["bps"]) for r in R if r["arm"] == arm}
G = {(r["layout"], r["dens"]): r for r in R}
lays = defaultdict(list)
for (l, d) in G:
    lays[d].append(l)
is_lat = lambda l: ":" not in l  # noqa: E731
score = lambda l, d: np.mean([I[l, d, c, s] / max(I[x, d, c, s] for x in lays[d]) for c in CH for s in SN])  # noqa: E731
print(f"arm {arm}. Per density: layouts by mean share of the best layout's bps over the 21 cells; bps at 10 dB per channel")
for d in DN:
    print(f"== density {d}")
    ranked = sorted(lays[d], key=lambda l: -score(l, d))
    for l in ranked[:6] + [x for x in ranked[6:] if is_lat(x)]:
        g = G[l, d]
        print(f"  {l:22s} {score(l, d):.3f} | " + " ".join(f"{c} {I[l, d, c, 10.0]:5.0f}" for c in CH)
              + f" | sym {g['sym']} car {g['car']} hole {g['hole']} peak {g['peak_db']}")
print("== best lattice vs best sequence (restart) vs best sequence (run), bps at 0/10/20 dB")
for d in DN:
    for c in CH:
        out = []
        for lab, f in (("lattice", is_lat), ("seq restart", lambda l: l.endswith(":restart")), ("seq run", lambda l: l.endswith(":run"))):
            best = max((l for l in lays[d] if f(l)), key=lambda l: sum(I[l, d, c, s] for s in SN))
            out.append(f"{lab} {best.split(':')[0]:12s} " + "/".join(f"{I[best, d, c, s]:.0f}" for s in SN))
        print(f"  {d:5s} {c:9s} " + " | ".join(out))
print("== best density per channel and cell (any layout), and the best 1/6 layout's share of it; k3d2's share")
for c in CH:
    out = []
    for s in SN:
        bd, bl = max(((d, l) for d in DN for l in lays[d]), key=lambda t: I[t[1], t[0], c, s])
        b6 = max(I[l, "1/6", c, s] for l in lays["1/6"])
        out.append(f"{s:+.0f} dB: {bd} {bl} {I[bl, bd, c, s]:.0f} bps; best 1/6 {b6 / I[bl, bd, c, s]:.2f}, k3d2 {I['k3d2', '1/6', c, s] / I[bl, bd, c, s]:.2f}")
    print(f"  {c:9s} " + " | ".join(out))
