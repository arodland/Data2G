"""Gear-shifter phase A: how well the receiver measures the link.

For bursts through the full numpy receiver (acquisition, header,
equalizer), compare against simulator truth:

  SNR      Burst.snr_db's estimate (pilot-based, 2500 Hz reference) vs the
           channel's SNR (average transmitted power)
  Doppler  est["spread_hz"] (2 sigma) vs the preset's Doppler spread
  delay    the delay support's width vs the preset's second-path delay
  MI       effective mutual information predicted from the equalizer's
           own h and noise (mean over channel uses of the AWGN BICM
           capacity at |h|^2 / var, the MIESM feature), vs the genie BMI
           of the burst's actual LLRs against the sent bits

    uv run python scripts/rx_audit.py --out runs/rx_audit.csv
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

from data2g import constellation, hfchannel, modem
from data2g.config import BANDS, FS, RS, SNR_REF_BW_HZ, SUBMODES

# QPSK data submodes, ~16 frames per burst
BURST = {"w": ("qpsk-r1/2", 2), "n10": ("n10-qpsk-r1/2", 2), "n4": ("n4-qpsk-r1/2", 1),
         "w48": ("w48-qpsk-r1/2", 4)}
CHANNELS = ["awgn", "mpg", "mpp", "mpd"]
SNRS = [-5.0, 0.0, 5.0, 10.0, 20.0]
GRID = np.arange(-20.0, 45.01, 0.25)


@lru_cache(maxsize=None)
def capacity_table(const: str) -> np.ndarray:
    """AWGN BICM mutual information (bits per coded bit) on GRID (dB).
    Shipped as codes_data/capacity_tables.npz (data2g.arq.predictor);
    this Monte Carlo is what made it."""
    pts = constellation.load(const)
    m = constellation.bits_per_symbol(pts)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 20000 * m)
    x = constellation.modulate(bits, pts)
    out = []
    for snr in GRID:
        var = np.full(x.shape, 10 ** (-snr / 10))
        y = x + np.sqrt(var / 2) * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape))
        l = constellation.llr(y, np.ones_like(x), var, pts)
        out.append(bmi(l, bits))
    return np.maximum.accumulate(np.array(out))


def bmi(l, bits) -> float:
    s = -(1.0 - 2.0 * bits) * np.clip(l, -50, 50)
    return float(1.0 - np.mean(np.logaddexp(0.0, s)) / np.log(2))


def effective_mi(h, var, const) -> float:
    snr = 10 * np.log10(np.maximum(np.abs(h) ** 2 / var, 1e-6))
    return float(np.mean(np.interp(snr, GRID, capacity_table(const))))


def trial(args):
    band, chan, snr, seed = args
    name, n_cw = BURST[band]
    spec = SUBMODES[name]
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_cw * spec.coded_bits)
    x = modem.modulate_bits(bits, spec)
    x = np.concatenate([np.zeros(int(rng.uniform(0.2, 0.6) * FS)), x, np.zeros(FS // 4)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=rng.uniform(-50, 50), ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    try:
        r = modem.receive(y)
    except modem.SyncError:
        return None
    if r["spec"] != spec or r["n_cw"] != n_cw:
        return None
    est = r["est"]
    h = est["h"]
    var = modem.noise_var(h, est) + est["mse"]
    l = constellation.llr(r["raw"][:, 1:], h, var, constellation.load(spec.constellation)).reshape(-1)
    nc = BANDS[band].nc
    snr_est = 10 * np.log10(est["p_sig"] / est["n0"] * nc * RS / SNR_REF_BW_HZ)
    d0, d1 = r["support"]
    out = dict(band=band, channel=chan, snr=snr, seed=seed, snr_est=snr_est, spread_est=est["spread_hz"],
               delay_est_ms=(d1 - d0) / FS * 1000, mi_genie=bmi(l, bits))
    for c in ("gray-qam4", "gray-qam16", "c64-snr18", "c256-snr26"):
        out[f"mi_{c}"] = effective_mi(h, var, c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(b, c, s, 1000 * i + j) for i, (b, c, s) in enumerate(
        (b, c, s) for b in BURST for c in CHANNELS for s in SNRS) for j in range(a.trials)]
    for c in ("gray-qam4", "gray-qam16", "c64-snr18", "c256-snr26"):
        capacity_table(c)  # before forking
    with Pool(a.jobs) as pool:
        rows = [r for r in pool.imap_unordered(trial, jobs, chunksize=4) if r is not None]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} of {len(jobs)} bursts received")


if __name__ == "__main__":
    main()
