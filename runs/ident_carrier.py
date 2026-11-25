"""As ident.py with a steady carrier on one of the grid's tones (master's per-tone floors, #295 / #296)."""
import hashlib
import numpy as np
from data2g import cpm, codes
h = hashlib.sha256()
def add(a): h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
flagged = 0
for name, spec in list(cpm.SPECS.items()):
    g = cpm.GRIDS[spec.grid]
    if getattr(g, "L", 1) > 1: continue
    for db in (12.0, 24.0):
        rng = np.random.default_rng([len(name), int(db)])
        coded = [rng.integers(0, 2, cpm.CTL[g.name].coded_bits)] + [rng.integers(0, 2, cpm.DATA_N)]
        x = cpm.modulate(spec, coded, False, 2)
        n = len(x) + 6000
        t = np.arange(n) / 8000
        y = np.concatenate([np.zeros(3000), x, np.zeros(3000)]) + 0.5 * rng.normal(size=n)
        y += np.sqrt(2 * 0.25 * 50 / 4000 * 10 ** (db / 10)) * np.cos(2 * np.pi * (g.f0 + 3 * g.rate + 4.0) * t)
        lock = dict(spec=spec, n_data=1, dup=False, start=3000, cfo=0.0, band=g.name)
        r = cpm.receive(y, lock)
        for s in r["soft"]: add(s)
        m = cpm.measure(g, r["E"], len(r["E"])); add([m["snr_est"], m["spread_est"]]); add(m["snr"])
        E = r["E"][:cpm.CTL[g.name].n_sym]
        flagged += cpm.tone_floors(E, cpm._n0(E)) is not None
        add(cpm.read_header(g, y, 3000, 0.0, 2)[3:5]); add(cpm.coh_read_header(g, y, 3000, 0.0, 2)[3:5])
        p = codes.decode_posterior(spec, r["soft"][1])[2]
        add(cpm.dd_soft(r, cpm.dd_metric(r, {1: p}, 1), 1))
        add(cpm.refine(g, y, 3004, 1.0, spec, 1, False))
print(h.hexdigest()[:32], "bursts with a floor flagged:", flagged)
