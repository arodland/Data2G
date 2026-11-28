"""Scattered pilots' decode CPU: modem.receive with the burst's timing known
(no search: demodulation and the channel estimate) and the plain decode,
per burst, flag off and on (modem.SC_BANDS patched in one process), clean
channel at 25 dB. `on cold`: the joint estimator's weights not yet cached."""

from data2g import threads

threads.limit(1)

import time  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, hfchannel, modem  # noqa: E402
from data2g.config import SUBMODES  # noqa: E402

from data2g.waveform import ofdm  # noqa: E402, F401

FADE = hfchannel.ITU_PRESETS["hilat-m"]
for name, n_cw, fade in (("ack-1f", 1, 0), ("qpsk-r1/2", 3, 0), ("qpsk-r1/2", 10, 0), ("qpsk-r1/2", 10, 1), ("w48-16qam-r1/2", 12, 0),
                         ("w48-qpsk-r1/2", 19, 1), ("w48-256l-r5/8", 40, 0), ("n10-qpsk-r1/2", 6, 0), ("n10-qpsk-r1/2", 6, 1),
                         ("n4-qpsk-r1/2", 3, 0)):
    spec = SUBMODES[name]
    line = f"{name:16s} {n_cw:2d} cw {modem.burst_seconds(spec, n_cw):5.2f} s air{' hilat-m' if fade else ''}:"
    for lab, sc in (("off", ()), ("on cold", ("w", "w48", "n10", "n4")), ("on", ("w", "w48", "n10", "n4"))):
        modem.SC_BANDS, modem.SC_LAYOUT = sc, "mix"
        if lab == "on cold":
            modem._SC_WEIGHTS.clear()
        rng = np.random.default_rng(1)
        ts, td = [], []
        for i in range(1 if lab == "on cold" else 5):
            pl = [rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)]
            x = np.concatenate([np.zeros(2000), modem.modulate(pl, name), np.zeros(2000)])
            y = hfchannel.awgn(hfchannel.fading(x, FADE, seed=i) if fade else x, 25, seed=i, s_power=1.0)
            from data2g.config import LEADIN_SAMPLES

            known = dict(start=2000 + LEADIN_SAMPLES, cfo=0.0, spec=spec, n_cw=n_cw)
            t = time.perf_counter()
            r = modem.receive(y, known=known)
            ts.append(time.perf_counter() - t)
            t = time.perf_counter()
            ok = modem.decode_received(r).crc_ok
            assert fade or all(ok)
            td.append(time.perf_counter() - t)
        line += f"  {lab} receive {1000 * np.median(ts):5.0f} ms decode {1000 * np.median(td):5.0f} ms;"
    print(line, flush=True)
