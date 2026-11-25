import hashlib, os, numpy as np
from data2g import cpm, hfchannel, codes
h = hashlib.sha256()
for name in ("fsk16r25-r1/2", "fsk8r50-r1/3", "fsk32r62-r1/2"):
    spec = cpm.SPECS[name]; g = cpm.GRIDS[spec.grid]
    for chan, snr in ((hfchannel.ITU_PRESETS["hilat-d"], 0.0), ("mpp", -6.0), (None, -8.0)):
        rng = np.random.default_rng([len(name), int(snr) + 50])
        coded = [rng.integers(0, 2, cpm.CTL[g.name].coded_bits), rng.integers(0, 2, cpm.DATA_N)]
        x = cpm.modulate(spec, coded, False, 2)
        y = hfchannel.apply_channel(np.concatenate([np.zeros(3000), x, np.zeros(3000)]), snr, 12.0, fading_preset=chan, seed=7)
        r = cpm.receive(y, dict(spec=spec, n_data=1, dup=False, start=3000, cfo=12.0, band=g.name))
        for s in r["soft"]: h.update(np.asarray(s, np.float64).tobytes())
        p = codes.decode_posterior(spec, r["soft"][1])[2]
        h.update(np.asarray(cpm.dd_soft(r, cpm.dd_metric(r, {1: p}, 1), 1), np.float64).tobytes())
print(os.environ.get("DATA2G_CPM_LEAK", "(off)"), h.hexdigest()[:24])
