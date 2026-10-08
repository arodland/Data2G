"""Hopped fsk8r50 (docs/hopping-fsk.md): per draw, the SNR (dB in 2500 Hz,
noise against that burst's envelope peak: PEP-fair) at and above which its
codeword decodes, to 0.125 dB, for every TX variant on the same payload,
channel and noise shape (paired). Genie sync and CFO: the code and the
noncoherent demodulator (cpm.llrs) alone.

A variant is a hop_tx mode with TX parameters, e.g.
k4/bp=350/glide=0.1/passes=6. Channels: awgn | grid:<delay ms>:<Doppler Hz>
| ens (hop_tx.channel). Rows append to --out: variant, channel, seed,
threshold dB (inf: never decoded), envelope peak over average dB.

    uv run python scripts/hop_study.py --out runs/x.csv --variants k1,k4/bp=350 --channels awgn,ens --draws 200

runs/hop_round.sh runs the study the notes report.
"""

import os

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, cpm  # noqa: E402
from data2g.config import FS, SNR_REF_BW_HZ  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import hop_tx as H  # noqa: E402


def parse(variant: str):
    mode, *kv = variant.split("/")
    return mode, {k: float(v) for k, v in (s.split("=") for s in kv)}


def seed_of(i: int) -> int:
    return 7919 * i + 3


def run(args):
    variant, ch, seed = args
    mode, p = parse(variant)
    x, act, m = H.tx(mode, p, seed)
    pk_db, pep = H.papr(x, act)
    y = H.channel(x, ch, seed)
    L, T, f0, span, c = m["L"], H.G.T, m["f0"], m["span"], m["c"]
    n = np.random.default_rng(seed + 2).normal(size=len(y))
    mix = np.exp(-2j * np.pi * f0 * np.arange(L * T) / FS)
    Ys = np.fft.fft((y[act] * mix).reshape(L, T), axis=1)[:, :span]
    Ns = np.fft.fft((n[act] * mix).reshape(L, T), axis=1)[:, :span]
    cols = H.fidx(mode, np.arange(H.G.m)[None, :], c[:, None])  # each symbol's copy
    rows = np.arange(L)[:, None]

    def ok(snrs):
        soft = []
        for s in snrs:
            sig = np.sqrt(pep * (FS / 2) / SNR_REF_BW_HZ / 10 ** (s / 10))
            soft.append(cpm.llrs(H.G, (np.abs(Ys + sig * Ns) ** 2)[rows, cols]))
        res = codes.decode_many(H.SPEC, np.array(soft), index=np.zeros(len(snrs), int))
        return np.array([q == m["payload"] and k for q, k in res])

    # 1 dB steps, then 0.125 dB below the first SNR from which every step decodes
    coarse = np.arange(-15.0, 30.01, 1.0)
    bad = np.where(~ok(coarse))[0]
    t = coarse[0] if len(bad) == 0 else (coarse[bad[-1] + 1] if bad[-1] + 1 < len(coarse) else np.inf)
    if np.isfinite(t) and t > coarse[0]:
        fine = np.arange(t - 0.875, t + 0.001, 0.125)
        g = ok(fine)
        t = fine[0] if g.all() else fine[np.where(~g)[0][-1] + 1]
    return variant, ch, seed, float(t), round(pk_db, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", required=True, help="comma-separated")
    ap.add_argument("--channels", required=True, help="comma-separated")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--first", type=int, default=0, help="first draw index (seeds: 7919 i + 3)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    jobs = [(v, ch, seed_of(i)) for ch in a.channels.split(",") for i in range(a.first, a.first + a.draws)
            for v in a.variants.split(",")]
    with Pool(a.jobs) as pool, open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        for r in pool.imap_unordered(run, jobs, chunksize=4):
            w.writerow(r)
            f.flush()


if __name__ == "__main__":
    main()
