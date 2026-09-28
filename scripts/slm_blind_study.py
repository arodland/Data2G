"""Can the receiver find each symbol's SLM pattern without being told?

Sign patterns (slm_study.py) can't be found blind: every constellation
here is symmetric under negation. Random 8-PSK phase patterns can: a
carrier turned by an odd multiple of 45 degrees leaves the grid. Per data
OFDM symbol, the receiver picks the candidate whose derotated cells lie
nearest the constellation: sum over carriers of min |y - h p x|^2 / var,
from 8 distance tables per cell (one per phase), then a gather per
candidate.

Per mode near its 10% point, AWGN and MPD: the fraction of symbols whose
pattern is picked wrong, and the PEP gain of 8-PSK patterns against the
sign patterns (slm_study's measure).

    uv run --no-sync python scripts/slm_blind_study.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from data2g import constellation
from data2g.channel_torch import CHANNELS
from data2g.config import BANDS, SubmodeSpec
from data2g.waveform import ofdm

sys.path.insert(0, str(Path(__file__).parent))
from slm_study import N_F, AceChannel, measure  # noqa: E402

# (mode, band, constellation, headroom today)
CASES = (("w48-16qam-r2/3", "w48", "gray-qam16", 3), ("w48-64l-r2/3", "w48", "c64-w48-r23", 6),
         ("w48-256l-r5/8", "w48", "c256-w48-r58", 6), ("n10-16qam-r2/3", "n10", "gray-qam16", 2))


def patterns(c: int, nc: int, seed: int = 7) -> np.ndarray:
    """(c, nc) phase indices 0..7 (x 45 degrees); pattern 0 all zero."""
    p = np.random.default_rng(seed).integers(0, 8, size=(c, nc))
    p[0] = 0
    return p


def select(x, band, pats):
    """Lowest-peak pattern per symbol -> (rotated x, chosen index)."""
    mod = ofdm.band(band).mod
    rot = np.exp(1j * np.pi / 4 * pats)  # (c, nc)
    cand = x[..., None, :] * rot  # (B, n_f, 5, c, nc)
    k = np.abs(cand @ mod.T).max(axis=-1).argmin(axis=-1)
    return np.take_along_axis(cand, k[..., None, None], axis=-2)[..., 0, :], k


def detect(raw, h, var, pts, pats):
    """(B, n_f, 5, nc) -> the pattern picked per symbol."""
    rot = np.exp(1j * np.pi / 4 * np.arange(8))
    nc = pats.shape[1]
    out = []
    for i in range(len(raw)):  # a burst at a time: the tables are (n_f, 5, nc, 8, points)
        y = (raw[i] / h[i])[..., None, None]  # equalized
        d = (np.abs(y - rot[:, None] * pts) ** 2).min(axis=-1) * (np.abs(h[i]) ** 2 / var[i])[..., None]
        out.append(d[..., np.arange(nc)[None, :], pats].sum(axis=-1).argmin(axis=-1))  # (n_f, 5)
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", type=int, default=32)
    ap.add_argument("--ladder", default="runs/ladder_10pct.csv")
    a = ap.parse_args()
    torch.set_num_threads(8)
    p10 = {r["name"]: r for r in csv.DictReader(open(a.ladder))}
    print("mode channel snr C | symbols picked wrong", flush=True)
    for mode, band, const, hr in CASES:
        pts = constellation.load(const)
        m, nc = constellation.bits_per_symbol(pts), ofdm.band(band).nc
        x = constellation.modulate(np.random.default_rng(1).integers(0, 2, a.bursts * N_F * 5 * nc * m),
                                   pts).reshape(a.bursts, N_F, 5, nc)
        spec = SubmodeSpec(0, "b", "ldpc", const, 1, band=band)
        for c in (32, 128):
            pats = patterns(c, nc)
            xs, k = select(x, band, pats)
            ch = AceChannel(spec, N_F, dtype=torch.float64, clip_setting=(hr, BANDS[band].clip_overshoot))
            tx = ch.transmit(torch.tensor(xs))
            s0, s1 = measure(band, x, hr, 0, 2.0, 9.9), measure(band, xs, hr, 0, 2.0, 9.9)
            print(f"{mode} at {hr} dB headroom, 8-PSK x {c}: SDR {s0[0]:.2f} -> {s1[0]:.2f} dB, "
                  f"peak {s0[3]:.2f} -> {s1[3]:.2f} dB", flush=True)
            for chan in ("awgn", "mpd"):
                v = p10[mode][chan]
                if v in ("", "nan"):
                    continue
                g = torch.Generator().manual_seed(3)
                y = ch.channel(tx, CHANNELS[chan], float(v), g)
                raw, h, var = (t.numpy() for t in ch.receive(y, CHANNELS[chan]))
                # the clip's gain: data arrive at 0.7-1 of the pilots' channel
                h = h * (np.vdot(h * xs, raw) / np.vdot(h * xs, h * xs))
                wrong = float(np.mean(detect(raw, h, var, pts, pats) != k))
                print(f"  {mode} {chan} {float(v):.2f} {c} | {wrong:.4f}", flush=True)


if __name__ == "__main__":
    main()
