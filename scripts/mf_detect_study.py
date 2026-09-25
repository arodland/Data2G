"""The matched-filter preamble detector (sync.detection_stat) on its own:
noise peaks per band (to calibrate config.PREAMBLE_THRESHOLDS) and the
statistic at the true start on signals (misses at a threshold).

    uv run python scripts/mf_detect_study.py noise --band n4 --seconds 600
    uv run python scripts/mf_detect_study.py signal --band n4 --chan awgn --snr -8 -6 --thr 40
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from multiprocessing import Pool

import numpy as np
from data2g import codes, hfchannel, modem
from data2g.config import FS, LEADIN_SAMPLES, SUBMODES
from data2g.waveform import ofdm, sync
from data2g.waveform.dsp import to_baseband

BURST = {"w": "ack-1f", "n10": "n10-ack-4f", "n4": "n4-ack-2f", "w48": "w48-qpsk-r1/2"}


def detect_stat(z, band, repeats=None):
    S, freqs = sync.detection_stat(z, ofdm.band(band), repeats=repeats)
    return S.max(axis=0), freqs[np.argmax(S, axis=0)]


def noise_peak(args):
    band, seed, secs, repeats = args
    z = to_baseband(np.random.default_rng(seed).normal(size=int(secs * FS)))
    return float(detect_stat(z, band, repeats)[0].max())


def sig_trial(args):
    band, chan, snr, seed = args
    spec = SUBMODES[BURST[band]]
    rng = np.random.default_rng(seed)
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec))], spec)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    f0 = rng.uniform(-50, 50)
    x = np.concatenate([np.zeros(lead), x, np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=f0, ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    d, f = detect_stat(to_baseband(y), band)
    start = lead + LEADIN_SAMPLES
    n = int(np.argmax(d))
    near = d[start - 40 : start + 41].max()
    return float(near), float(d.max()), abs(n - start) <= 40 and abs(f[n] - f0) <= sync.STEP_HZ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["noise", "signal"])
    ap.add_argument("--band", default="n4")
    ap.add_argument("--chan", default="awgn")
    ap.add_argument("--snr", type=float, nargs="+", default=[-8.0])
    ap.add_argument("--thr", type=float, default=40.0)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--repeats", type=int, help="noise mode: preamble repeats (default: config)")
    a = ap.parse_args()
    with Pool(a.jobs) as pool:
        if a.mode == "noise":
            chunk = 20
            peaks = pool.map(noise_peak, [(a.band, s, chunk, a.repeats) for s in range(int(a.seconds // chunk))])
            p = np.sort(peaks)
            print(f"{a.band}: noise peak over {a.seconds:g} s: {p[-1]:.1f}; per-{chunk}s-chunk peaks "
                  f"median {np.median(p):.1f}, 90% {np.quantile(p, 0.9):.1f}", flush=True)
            return
        for snr in a.snr:
            res = pool.map(sig_trial, [(a.band, a.chan, snr, 20_000 + i) for i in range(a.trials)], chunksize=4)
            near = np.array([r[0] for r in res])
            locked = np.array([r[0] >= a.thr and r[2] for r in res])
            print(f"{a.band} {a.chan} {snr:6.2f} dB: D at truth median {np.median(near):6.1f} 1% {np.quantile(near, 0.01):6.1f}"
                  f"  miss(thr {a.thr:g}) {np.mean(near < a.thr):.4f}  argmax lock+CFO fail {1 - locked.mean():.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
