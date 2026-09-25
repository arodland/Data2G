"""What a band's clipper setting does, as the receiver and a PEP-fair
comparison need to know it: data gain relative to the pilots (1-frame
bursts and longer), clip SDR on the data, and the envelope PAPR of the
transmitted burst. Prints a config.CLIP entry.

    uv run python scripts/clip_constants.py --band w --headroom 1.0 --overshoot 1.0 1.5 2.0
    uv run python scripts/clip_constants.py --headroom 0 1 2 3 4 5 6 \\
        --write data2g/codes_data/clip_constants.json   # the table the modem reads

PAPR matters because SNR here is average power after clipping, while a
transmitter is limited by peak (PEP): a setting with more headroom and a
higher PAPR sends less average power at the same PEP. A PEP-fair
threshold is threshold + PAPR (both dB), compared between settings.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse

import numpy as np
import torch

torch.set_num_threads(4)

from data2g import constellation  # noqa: E402
from data2g.channel_torch import CHANNELS, BurstChannel, _analytic  # noqa: E402
from data2g.config import LEADIN_SAMPLES, LEADOUT_SAMPLES, SubmodeSpec  # noqa: E402


def measure(band, headroom, overshoot, device, const="gray-qam16"):
    spec = SubmodeSpec(0, "clip", "ldpc", const, 1, band=band)
    pts = constellation.load(const)
    m = constellation.bits_per_symbol(pts)
    out = {}
    for n_f in (1, 8):
        # Unit gain in the RX model: what comes back is the raw ratio.
        ch = BurstChannel(spec, n_f, device=device, dtype=torch.float64,
                          clip_setting=(headroom, tuple(overshoot)), clip_consts=({}, 1.0, 0.0))
        b = max(16, 2000 // n_f)
        rng = np.random.default_rng(n_f)
        x = constellation.modulate(rng.integers(0, 2, b * n_f * 5 * ch.nc * m), pts).reshape(b, n_f, 5, ch.nc)
        tx = ch.transmit(torch.tensor(x, device=device))
        raw, h, _ = ch.receive(tx, CHANNELS["awgn"])
        raw, h = raw.cpu().numpy(), h.cpu().numpy()
        g = np.vdot(h * x, raw) / np.vdot(h * x, h * x)
        e = raw - g * h * x
        sdr = 10 * np.log10(np.mean(np.abs(g * h * x) ** 2) / np.mean(np.abs(e) ** 2))
        act = tx[:, LEADIN_SAMPLES : tx.shape[1] - LEADOUT_SAMPLES]
        env2 = _analytic(act).abs().pow(2).cpu().numpy()
        papr = 10 * np.log10(np.quantile(env2, 0.9999) / env2.mean())
        peak = 10 * np.log10(env2.max(axis=1).mean() / env2.mean())
        out[n_f] = (abs(g), sdr, papr, peak)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="w")
    ap.add_argument("--headroom", type=float, nargs="+", default=[1.0])
    ap.add_argument("--overshoot", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--write", help="measure every band at every --headroom and write this JSON table")
    a = ap.parse_args()
    if a.write:
        import json

        from data2g.config import BANDS

        table = {}
        for band in BANDS:
            for hr in a.headroom:
                r = measure(band, hr, a.overshoot, a.device)
                (g1, _, _, _), (g8, sdr, papr, peak) = r[1], r[8]
                table[f"{band}@{hr:g}"] = dict(gain_1f=round(g1, 4), gain=round(g8, 4), sdr_db=round(sdr, 2),
                                              papr_db=round(papr, 2), peak_db=round(peak, 2))
                print(band, hr, table[f"{band}@{hr:g}"], flush=True)
        with open(a.write, "w") as f:
            json.dump({"overshoot": list(a.overshoot), "entries": table}, f, indent=1)
        return
    for hr in a.headroom:
        r = measure(a.band, hr, a.overshoot, a.device)
        (g1, _, _, _), (g8, sdr, papr, peak) = r[1], r[8]
        print(f"{a.band} headroom {hr:4.1f} dB overshoot {tuple(a.overshoot)}: "
              f"gain {g1:.3f}/{g8:.3f}  SDR {sdr:5.2f} dB  PAPR(99.99%) {papr:5.2f} dB  "
              f"peak {peak:5.2f} dB   CLIP entry: ({{1: {g1:.3f}}}, {g8:.3f}, 10 ** (-{sdr:.1f} / 10))",
              flush=True)


if __name__ == "__main__":
    main()
