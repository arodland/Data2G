"""Hash the wide grids' receive outputs, clean and with an in-grid carrier (fsk-wide-bins against this tree, leak off)."""
import hashlib, os
import numpy as np
from data2g import cpm, codes
h = hashlib.sha256()
def add(a): h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
flag = 0
for name in ("fsk8w50-r1/3", "fsk8w50-r1/2", "fsk16w125-r1/3"):
    spec = cpm.SPECS[name]; g = cpm.GRIDS[spec.grid]
    for db in (None, 12.0, 24.0):
        rng = np.random.default_rng([len(name), int(db or 0)])
        coded = [rng.integers(0, 2, cpm.CTL[g.name].coded_bits)] + [rng.integers(0, 2, cpm.DATA_N)]
        x = cpm.modulate(spec, coded, False, 2); add(x)
        n = len(x) + 6000
        y = np.concatenate([np.zeros(3000), x, np.zeros(3000)]) + 0.5 * rng.normal(size=n)
        if db is not None:
            y += np.sqrt(2 * 0.25 * 50 / 4000 * 10 ** (db / 10)) * np.cos(2 * np.pi * (g.f0 + 3 * g.rate + 4.0) * np.arange(n) / 8000)
        lock = cpm.find(g, y[:3000 + 4000 + cpm.hdr_span(g.name, 2) + g.T])
        add([lock["start"], lock["cfo"], lock["header_score"]])
        r = cpm.receive(y, lock)
        for s in r["soft"]: add(s)
        add(r["E"]); m = cpm.measure(g, r["E"], len(r["E"])); add([m["snr_est"], m["spread_est"]]); add(m["snr"])
        E = r["E"][:cpm.CTL[g.name].n_sym]; flag += cpm.tone_floors(E, cpm._n0(E, g.k), g.k) is not None
        p = codes.decode_posterior(spec, r["soft"][1])[2]
        add(cpm.dd_soft(r, cpm.dd_metric(r, {1: p}, 1), 1))
print(os.environ.get("DATA2G_CPM_LEAK", "-"), h.hexdigest()[:32], "floors flagged:", flag)
