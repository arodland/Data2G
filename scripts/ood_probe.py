"""How confident is an outcome model where it has no data? Probe inputs
outside the training ranges (SNR, spread, delay, incoherent combinations)
and print, per model, each probe's highest P(burst usable) x P(codeword)
and its mode. A model that is honest there stays away from 0.99.

    uv run python scripts/ood_probe.py runs/outcome_predictor_v5a.npz ...
"""

import sys

import numpy as np

from data2g.arq import predictor as P
from data2g.arq.modes import MODES


def meas(snr, spread=0.3, delay=1.0, mi_shift=0.0, frames=16):
    m = dict(snr_est=snr, spread_est=spread, delay_est_ms=delay, headroom=0.0, frames=frames)
    for c in P.CONSTS:
        m[f"mi_{c}"] = float(np.clip(P.capacity(snr + mi_shift, c), 0, None))
    return m


PROBES = {
    "in range: 10 dB, mpg-like": (meas(10), "w"),
    "SNR 45 dB": (meas(45), "w48"),
    "SNR -25 dB": (meas(-25), "w"),
    "spread 8 Hz": (meas(10, spread=8.0), "w"),
    "delay 12 ms": (meas(10, delay=12.0), "w48"),
    "SNR 20, MI as if -5 dB": (meas(20, mi_shift=-25), "w48"),
    "SNR -5, MI as if 20 dB": (meas(-5, mi_shift=25), "w48"),
    "one-frame reply at 30 dB": (meas(30, frames=1), "n10"),
}


def main():
    for path in sys.argv[1:]:
        P.outcome_model.cache_clear()
        model = P.outcome_model(path)
        print(f"== {path.split('/')[-1]}")
        for name, (m, band) in PROBES.items():
            x = P.outcome_inputs(m, band, 2.5, 6.0, None, model.bands)
            z = model(x)
            n = len(model.modes)
            p = 1 / (1 + np.exp(-z))
            joint = p[:n] * p[n:]
            i = int(np.argmax(joint))
            w48 = [j for j, mm in enumerate(model.modes) if mm.startswith("w48-64l") or mm.startswith("w48-256l")]
            print(f"   {name:28s} best {model.modes[i]:16s} {joint[i]:.2f} | top-order w48 mean {joint[w48].mean():.2f}"
                  f" | P(usable) spread across modes {p[:n].min():.2f}-{p[:n].max():.2f}")


if __name__ == "__main__":
    main()
