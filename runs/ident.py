"""Hash today's grids' transmit and receive outputs (before/after check)."""
import hashlib, sys
import numpy as np
from data2g import cpm
h = hashlib.sha256()
def add(a): h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
for name, spec in list(cpm.SPECS.items()):
    if "w" in name.split("-")[0][3:]: continue
    g = cpm.GRIDS[spec.grid]
    rng = np.random.default_rng(len(name))
    coded = [rng.integers(0, 2, cpm.CTL[g.name].coded_bits)] + [rng.integers(0, 2, cpm.DATA_N)]
    for cap in (0, 2):
        x = cpm.modulate(spec, coded, False, cap); add(x)
    y = np.concatenate([np.zeros(3000), x, np.zeros(3000)]) + 0.5 * rng.normal(size=len(x) + 6000)
    add(cpm._energies(g, y, 2990, 40, 1.3, extra=2)); add(cpm.coarse_rows(g, y[:20000], 3))
    lock = cpm.find(g, y[:3000 + 4000 + cpm.hdr_span(g.name, 1) + g.T], wait=True)
    add([lock["start"], lock["cfo"], lock["score"], lock["header_score"]])
    r = cpm.receive(y, lock)
    for s in r["soft"]: add(s)
    add(r["E"]); m = cpm.measure(g, r["E"], len(r["E"])); add([m["snr_est"], m["spread_est"]]); add(m["snr"])
    add(cpm.find_mid(g, y)["start"]); add(cpm.confirm(g, y, lock))
print(h.hexdigest())
