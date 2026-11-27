"""Bussgang gain and distortion per cell kind of a clipped burst, no channel:
data cells, scattered pilots, the closing whole pilot symbol; flag off (frame
pilot rows) and on, plain and with the pilots restored (SC_RESTORE)."""

from data2g import threads

threads.limit(1)

import numpy as np  # noqa: E402

from data2g import codes, constellation, modem  # noqa: E402
from data2g.config import LEADIN_SAMPLES, SUBMODES  # noqa: E402
from data2g.waveform import dsp, ofdm  # noqa: E402
from scripts.sc_diag import cells  # noqa: E402


def fit(y, x, k, nc):
    """One complex gain per carrier (the TX filter and the window's phase are per carrier): (mean |gain|, SDR dB)."""
    num, den = np.zeros(nc, complex), np.zeros(nc)
    np.add.at(num, k, y * np.conj(x))
    np.add.at(den, k, np.abs(x) ** 2)
    g = num / den
    return float(np.mean(np.abs(g))), 10 * np.log10(np.sum(np.abs(g[k] * x) ** 2) / np.sum(np.abs(y - g[k] * x) ** 2))


for name in ("16qam-r1/2", "w48-qpsk-r1/2", "w48-16qam-r1/2", "w48-16qam-r3/4", "w48-64l-r3/4"):
    spec = SUBMODES[name]
    b = ofdm.band(spec.band)
    for lab, sc, restore in (("off", (), False), ("on", ("w", "w48"), False), ("on restore", ("w", "w48"), True)):
        modem.SC_BANDS, modem.SC_RESTORE_ACE, modem.SC_RESTORE_PLAIN = sc, (1.0 if restore else 0.0), (1.0 if restore else 0.0)
        rows = []
        for seed in range(6):
            rng = np.random.default_rng(seed)
            pl = [rng.bytes(codes.payload_bytes(spec)) for _ in range(4)]
            x = np.concatenate([np.zeros(2400), modem.modulate(pl, name), np.zeros(2400)])
            r = modem.receive(x, known=dict(start=2400 + LEADIN_SAMPLES, cfo=0.0, spec=spec, n_cw=4))
            bits = np.stack([codes.encode(spec, p, 0, index=i) for i, p in enumerate(pl)])
            X = constellation.modulate(codes.spread(bits, spec.bits_per_cu), constellation.load(spec.constellation))
            Y = modem.data_symbols(r["raw"]).ravel()
            d = fit(Y, X, cells(r)[1], b.nc)
            pk = dsp.papr_db(x[2400:-2400])
            if "sc" in r:
                ls = r["sc"]["ls"]
                p = fit(ls.ravel(), np.ones(ls.size), np.tile(modem.sc_cells(b.nc)[1], len(ls)), b.nc)
                w = (float(np.mean(np.abs(r["sc"]["full"][-1]))), 0.0)
            else:
                hp = r["hp"]
                p = fit(hp[:-1].ravel(), np.ones(hp[:-1].size), np.tile(np.arange(b.nc), len(hp) - 1), b.nc)
                w = (float(np.mean(np.abs(hp[-1]))), 0.0)
            rows.append((*d, *p, *w, pk))
        m = np.mean(rows, axis=0)
        print(f"{name:15s} {lab:10s} data/closing-pilot gain {m[0] / m[4]:.3f} sdr {m[1]:5.1f} | frame or scattered pilots/closing "
              f"{m[2] / m[4]:.3f} sdr {m[3]:5.1f} | peak {m[6]:.2f} dB | table gain {spec.clip.gain:.3f} sdr {-10 * np.log10(spec.clip.ratio):.1f}", flush=True)
