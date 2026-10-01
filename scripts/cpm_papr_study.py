"""CPM envelope peaks: TX variants against PEP-fair decode thresholds.

With the old filter, fsk32r62-r1/2's envelope peaked 2.8 dB over its
average (its real waveform 1.1 dB): a tight TX bandpass (bp 30 Hz past the
outer tones) rang at every tone jump, and SSB PEP is the envelope. This
chose the current one (bp 150/50, clip 0). Variants, TX only (the receiver
reads tone energies, not these); parts joined by "+" combine:
  base       as cpm.GRIDS ships
  old        bp 30 Hz, clip_db 0.5 (the filter before 2026-09-29: bp30+clip0.5)
  clip<d>    clip_db d
  bp<hz>     a wider TX bandpass margin
  glide<b>   the frequency trajectory smoothed over a fraction b of a
             symbol (raised cosine); the CPM prototype judged this at one
             AWGN SNR, genie sync, average power only
Per variant: envelope peak over average (per-burst max, median of bursts),
99% occupied bandwidth, and the 10% end-to-end failure point of a data
codeword (sync, header, codeword) on AWGN and MPP through
phy_session.ContinuousChannel with noise against each burst's envelope
peak (DATA2G_PEP_REF_DB=5): thresholds are PEP-fair as they stand. Seeded:
the same trials at every SNR and in every variant.

    uv run python scripts/cpm_papr_study.py --out runs/cpm_papr.csv
"""

import os

from data2g import threads  # noqa: E402

threads.limit(1)
os.environ.setdefault("DATA2G_PEP_REF_DB", "5")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import cpm, hfchannel  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.config import FS  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import outcome_data as O  # noqa: E402
import phy_session as PS  # noqa: E402

MODE = "fsk32r62-r1/2"
VARIANTS = ("base", "old", "clip0.25", "clip0", "bp60", "bp100", "bp150", "glide0.1", "glide0.2", "glide0.3")
SNRS = {"awgn": np.arange(-15.0, -5.0), "mpp": np.arange(-11.0, 0.0)}
_TONES, _BANDPASS = cpm.tones, cpm.bandpass


def apply(variant: str):
    """Patch the CPM transmitter for `variant` (per worker process); parts
    joined by "+" combine (bp150+glide0.3)."""
    cpm.tones, cpm.bandpass = _TONES, _BANDPASS
    fields = {}
    for part in variant.split("+"):
        if part.startswith("clip"):
            fields["clip_db"] = float(part[4:])
        elif part.startswith("bp"):
            fields["bp"] = float(part[2:])
        elif part == "old":
            fields.update(bp=30.0, clip_db=0.5)
        elif part.startswith("glide"):
            cpm.tones = glide_tones(float(part[5:]))
    if fields:
        cpm.bandpass = lambda g, x: _BANDPASS(replace(g, **fields), x)


def glide_tones(beta: float):
    def tones(g, sym):
        T = g.T
        a = np.repeat(sym.astype(float), T)
        n = max(1, int(beta * T))
        if n > 1:
            w = np.hanning(n + 2)[1:-1]
            a = np.convolve(np.pad(a, n, mode="edge"), w / w.sum(), mode="same")[n:-n]
        x = np.sqrt(2) * np.cos(2 * np.pi * np.cumsum(g.f0 + a * g.rate) / FS)
        n_ramp = int(cpm.RAMP_S * FS)
        ramp = (1 - np.cos(np.pi * (np.arange(n_ramp) + 0.5) / n_ramp)) / 2
        x[:n_ramp] *= ramp
        x[-n_ramp:] *= ramp[::-1]
        return x

    return tones


def shape(variant, n=40):
    """Envelope peak over average (median of per-burst values, dB) and the
    99% occupied bandwidth (Hz) of `n` 9-slot bursts."""
    apply(variant)
    rng = np.random.default_rng(1)
    papr, xs = [], []
    for _ in range(n):
        x = PHY.tx_audio(O.burst(MODE, 9, rng))
        z = hfchannel._analytic(x)
        papr.append(10 * np.log10(np.max(np.abs(z)) ** 2 / np.mean(np.abs(z) ** 2)))
        xs.append(x)
    f = np.fft.rfftfreq(FS, 1 / FS)
    p = np.zeros(len(f))
    for x in xs:
        for i in range(0, len(x) - FS, FS // 2):
            p += np.abs(np.fft.rfft(x[i:i + FS] * np.hanning(FS))) ** 2
    c = np.cumsum(p) / p.sum()
    return float(np.median(papr)), float(f[np.searchsorted(c, 0.995)] - f[np.searchsorted(c, 0.005)])


def trial(args):
    variant, chan, snr, seed = args
    apply(variant)
    rng = np.random.default_rng(seed)
    b = O.burst(MODE, 2, rng)
    ch = PS.ContinuousChannel(chan, snr, seed, 120.0)
    r = O.receive(ch.apply(PHY.tx_audio(b), float(rng.uniform(5, 100))), MODE)
    if r is None or r["spec"].name != MODE or r["n_cw"] != 2:
        return variant, chan, snr, seed, 0
    rx = PHY.ModemRx(r, {})
    return variant, chan, snr, seed, int(rx.decode(1, b.slots[1].mask_id, 0, None) == b.slots[1].payload)


def point(snrs, fail, target=0.1):
    """The SNR where the failure rate crosses `target` (linear interpolation)."""
    for i in range(len(snrs) - 1):
        if fail[i] >= target > fail[i + 1]:
            return snrs[i] + (fail[i] - target) / (fail[i] - fail[i + 1]) * (snrs[i + 1] - snrs[i])
    return float("nan")


def main():
    global MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/cpm_papr.csv")
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--mode", default=MODE)
    ap.add_argument("--shift", type=float, default=0.0, help="dB added to the SNR grids (a lower mode: negative)")
    a = ap.parse_args()
    if PS.PEP_REF_DB is None:
        ap.error("DATA2G_PEP_REF_DB must be set (PEP-fair thresholds)")
    variants = a.variants.split(",")
    MODE = a.mode
    for c in SNRS:
        SNRS[c] = SNRS[c] + a.shift
    jobs = [(v, c, float(s), 90000 + i) for v in variants for c in SNRS for s in SNRS[c] for i in range(a.trials)]
    with Pool(a.jobs) as pool:
        shapes = dict(zip(variants, pool.map(shape, variants)))
        res = pool.map(trial, jobs, chunksize=16)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "channel", "snr", "seed", "ok"])
        w.writerows(res)
    print(f"{MODE}: 10% end-to-end failure point, PEP-referenced (lower is better); envelope PAPR; 99% OBW")
    base = {}
    for v in variants:
        pts = {}
        for c in SNRS:
            fail = [1 - np.mean([r[4] for r in res if r[0] == v and r[1] == c and r[2] == s]) for s in SNRS[c]]
            pts[c] = point(list(SNRS[c]), fail)
        if v == "base":
            base = pts
        d = "  ".join(f"{c} {pts[c]:+6.2f} ({pts[c] - base.get(c, pts[c]):+.2f})" for c in SNRS)
        print(f"  {v:9s} {d}   PAPR {shapes[v][0]:.2f} dB   OBW {shapes[v][1]:.0f} Hz")


if __name__ == "__main__":
    main()
