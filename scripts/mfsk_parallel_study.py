"""Parallel multi-tone FSK (docs/hopping-fsk.md, "Parallel tones"): N copies
of a CPM grid side by side, each sending its own tone at once (N x the
rate in N x the band), through the TX clip-and-filter at a given headroom.
Genie sync, one r1/2 LDPC codeword per burst, noise against the burst's
envelope peak (PEP-fair); prints the median envelope peak over average and
the first SNR (of the sweep) at which at most 10% of codewords fail.

    uv run python scripts/mfsk_parallel_study.py c8r50 2 0 awgn -14 4 0.5    # grid N clip|none channel lo hi step

Channels: awgn or an hfchannel fading preset. The failure it recorded
(2026-10-07): with the best clip, X2 cost 5 dB and X4 8.5-9 dB of PEP-fair
SNR for 2x / 4x the rate, worse than anything already on the ladder.
"""

import sys
from multiprocessing import Pool

import numpy as np
from scipy import signal

from data2g import codes, cpm, hfchannel
from data2g.config import FS, SNR_REF_BW_HZ
from data2g.waveform.dsp import tx_condition

TRIALS = 40


def make(grid, N, clip, rng, spec):
    g = cpm.GRIDS[grid]
    payload = rng.bytes(codes.payload_bytes(spec))
    sym = cpm.to_tones(g, codes.encode(spec, payload))
    L = int(np.ceil(len(sym) / N))
    S = np.pad(sym, (0, L * N - len(sym))).reshape(L, N)  # time x copy: symbol i is (i // N, i % N)
    T = g.T
    f0 = round((g.center - (N * g.m - 1) * g.rate / 2) / g.rate) * g.rate
    x = np.zeros(L * T)
    for n in range(N):
        a = np.repeat(S[:, n].astype(float) + n * g.m, T)
        x += np.sqrt(2) * np.cos(rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.cumsum(f0 + a * g.rate) / FS)
    x /= np.sqrt(np.mean(x ** 2))
    pad = 400
    x = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    act = slice(pad, pad + L * T)
    bp = (f0 - g.bp, f0 + (N * g.m - 1) * g.rate + g.bp)
    if clip is None:  # filter only
        x = np.convolve(x, signal.firwin(201, bp, fs=FS, pass_zero=False), "same")
        x /= np.sqrt(np.mean(x[act] ** 2))
    else:
        x = tx_condition(x, clip, active=act, bandpass=bp)
    return x, act, f0, L, payload


def trial(args):
    grid, N, clip, ch, snr, seed = args
    rng = np.random.default_rng(seed)
    g = cpm.GRIDS[grid]
    spec = cpm.SPECS[f"fsk{g.m}r{int(g.rate)}-r1/2"]
    x, act, f0, L, payload = make(grid, N, clip, rng, spec)
    pep = (np.abs(signal.hilbert(x))[act] ** 2).max() / 2  # a unit-RMS sinusoid's = 1
    y = hfchannel.fading(x, ch, seed=seed) if ch != "awgn" else x
    y = y + rng.normal(scale=np.sqrt(pep * (FS / 2) / SNR_REF_BW_HZ / 10 ** (snr / 10)), size=len(y))
    T = g.T
    seg = y[act]
    Z = np.fft.fft((seg * np.exp(-2j * np.pi * f0 * np.arange(len(seg)) / FS)).reshape(L, T), axis=1)
    E = (np.abs(Z[:, :N * g.m]) ** 2).reshape(L * N, g.m)  # back to codeword order
    soft = cpm.llrs(g, E)[:spec.coded_bits]
    ok = any(p == payload and c for p, c in codes.decode_many(spec, soft[None, :], index=np.array([0])))
    return 10 * np.log10(pep / np.mean(x[act] ** 2)), ok


def main():
    grid, N, ch = sys.argv[1], int(sys.argv[2]), sys.argv[4]
    clip = None if sys.argv[3] == "none" else float(sys.argv[3])
    lo, hi, step = map(float, sys.argv[5:8])
    fails, peaks = [], []
    with Pool(4) as pool:
        for s in np.arange(lo, hi + 1e-9, step):
            r = pool.map(trial, [(grid, N, clip, ch, s, 1000 * i + 7) for i in range(TRIALS)])
            peaks.append(np.median([a for a, _ in r]))
            fails.append((s, 1 - np.mean([b for _, b in r])))
    thr = next((s for s, f in fails if f <= 0.1), None)
    print(f"{grid} X{N} clip={sys.argv[3]} {ch}: peak {np.median(peaks):.2f} dB, 10% PEP-fair {thr}  "
          + " ".join(f"{s:.1f}:{f:.2f}" for s, f in fails), flush=True)


if __name__ == "__main__":
    main()
