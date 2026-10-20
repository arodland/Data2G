"""BICM-ID: iterative demapping. When an LDPC decode fails, feed the
decoder's extrinsic LLRs back to the demapper as a priori bit
probabilities, re-demap the codeword's symbols, decode again. Receiver
only: no format change.

What it can gain is bounded by the gap between the constellation's BMI
(what a one-shot demapper delivers, codes.label_reliability) and its
coded-modulation MI (symbol-wise). A Gray label makes the two nearly
equal; QPSK's bits are independent, so it gains exactly nothing.

    uv run --no-sync python scripts/bicm_id_study.py exit
    uv run --no-sync python scripts/bicm_id_study.py per w48-64l-r1/2 [--outer 4 --inner 40 --n 200]
    uv run --no-sync python scripts/bicm_id_study.py per w48-64l-r1/2 --outer 0 --inner 200   # control

`exit`: the demapper's EXIT curve (extrinsic MI vs a priori MI, Gaussian
priors) on AWGN at the Es/N0 where BMI = rate * m, and the dB the
coded-modulation limit sits below that point. `per`: the submode's real
LDPC code and interleaver on AWGN (ideal CSI, no clipper), PER after
each outer iteration; the decoder restarts each pass. Runs with the same
seed share codewords and noise, so `--outer 0 --inner N*(outer+1)` is a
paired control: same decoder work, no feedback.

Found (2026-10-20). EXIT (I_E at I_A = 0 -> 1): gray-qam16 r1/2
0.499 -> 0.518; c64-w48-r12 0.496 -> 0.551; c64-w48-r712 0.580 -> 0.620;
c64-w48-r23 0.663 -> 0.691; c64-w48-r34 0.748 -> 0.763; c256-w48-r58
0.625 -> 0.671. The learned sets, trained for BMI through the clipper,
did not come out Gray. CM limit below the operating point: 0.52, 0.37,
0.24, 0.13, 0.58 dB for those learned sets; 0.16 dB for 16-QAM r1/2.

PER 10% point, 200 codewords per 0.25 dB step (about +-0.1 dB):

| submode | 40 iters (today) | 200 iters, no feedback | BICM-ID 5 x 40 | gain vs control |
|---|---:|---:|---:|---:|
| w48-16qam-r1/2 | 6.19 | 6.06 | 5.94 | 0.1 dB |
| w48-64l-r1/2 | 10.25 | 10.17 | 9.67 | 0.5 dB |
| w48-64l-r2/3 | 13.42 | 13.25 | 13.08 | 0.15 dB |
| w48-256l-r5/8 | 16.93 | 16.86 | 16.25 | 0.6 dB |

Most of the gain comes in the first one or two outer passes. The
prior-aware demapper costs m * 2^m per symbol and bit, so the 256l
experiment took about 2.5x the control's CPU.
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import time

import numpy as np

from data2g import codes, constellation, ldpc
from data2g.config import SUBMODES


def demap(y, h, var, points, la=None):
    """(N,) symbols -> (N, m) extrinsic LLRs (log P0/P1), given a priori
    LLRs `la` (N, m) of each symbol's label bits (None: constellation.llr)."""
    m = constellation.bits_per_symbol(points)
    if la is None:
        return constellation.llr(y, h, var, points).reshape(-1, m)
    lb = constellation.label_bits(m).astype(float)  # (M, m)
    d = -np.abs(y[:, None] - h[:, None] * points) ** 2 / var[:, None] - la @ lb.T  # + log P(label)
    out = np.empty((len(y), m))
    for i in range(m):
        t = d + lb[:, i] * la[:, i : i + 1]  # leave out bit i's own prior
        e = np.exp(t - t.max(axis=1, keepdims=True))
        out[:, i] = np.log(np.maximum(e @ (1 - lb[:, i]), 1e-300)) - np.log(np.maximum(e @ lb[:, i], 1e-300))
    return out


def mi(llr, bits):
    return 1 - np.mean(np.logaddexp(0, -(1 - 2 * bits) * np.clip(llr, -50, 50))) / np.log(2)


def _awgn(points, esn0_db, n, rng):
    m = constellation.bits_per_symbol(points)
    bits = rng.integers(0, 2, (n, m))
    x = constellation.modulate(bits.reshape(-1), points)
    var = np.full(n, 10 ** (-esn0_db / 10))
    return bits, x + np.sqrt(var / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n)), var


def _sigma(i_a, n=20000):
    """Gaussian prior LLR spread whose MI is i_a (the J^-1 function)."""
    z, lo, hi = np.random.default_rng(1).normal(size=n), 1e-3, 60.0
    for _ in range(40):
        s = (lo + hi) / 2
        lo, hi = (s, hi) if mi(s * s / 2 + s * z, np.zeros(n)) < i_a else (lo, s)
    return (lo + hi) / 2


def cm_mi(points, esn0_db, n=40000):
    """Coded-modulation MI (bits per symbol) on AWGN."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(points), n)
    var = 10 ** (-esn0_db / 10)
    y = points[idx] + np.sqrt(var / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
    d = -np.abs(y[:, None] - points) ** 2 / var
    return np.log2(len(points)) + np.mean(d[np.arange(n), idx] - np.logaddexp.reduce(d, axis=1)) / np.log(2)


def exit_study():
    cases = [("gray-qam16", 1 / 2), ("gray-qam16", 2 / 3), ("c64-w48-r12", 1 / 2), ("c64-w48-r712", 7 / 12),
             ("c64-w48-r23", 2 / 3), ("c64-w48-r34", 3 / 4), ("c256-w48-r58", 5 / 8)]
    ias = (0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    print("constellation  rate  Es/N0  I_E at I_A = " + " ".join(f"{a:g}" for a in ias) + "  CM limit below (dB)")
    for name, r in cases:
        p = constellation.load(name)
        m = constellation.bits_per_symbol(p)
        lo, hi = -10.0, 40.0  # the BMI operating point
        for _ in range(25):
            mid = (lo + hi) / 2
            bits, y, var = _awgn(p, mid, 20000, np.random.default_rng(0))
            lo, hi = (lo, mid) if mi(demap(y, np.ones(len(y)), var, p), bits) >= r else (mid, hi)
        es = hi
        rng = np.random.default_rng(0)
        bits, y, var = _awgn(p, es, 40000, rng)
        ie = []
        for a in ias:
            if a == 0:
                la = None
            elif a >= 1:
                la = (1 - 2 * bits) * 60.0
            else:
                s = _sigma(a)
                la = (1 - 2 * bits) * s * s / 2 + s * rng.normal(size=bits.shape)
            ie.append(mi(demap(y, np.ones(len(y)), var, p, la), bits))
        lo, hi = es - 6, es
        for _ in range(20):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if cm_mi(p, mid) >= r * m else (mid, hi)
        print(f"{name:14s} {r:.3f} {es:6.2f}  " + " ".join(f"{v:.3f}" for v in ie) + f"  {es - hi:.2f}", flush=True)


def per_study(name, snrs, n_cw, outer, inner):
    spec = SUBMODES[name]
    pts = constellation.load(spec.constellation)
    m = constellation.bits_per_symbol(pts)
    dec, perm = codes._decoder(spec), codes.interleaver(spec)
    rng = np.random.default_rng(0)
    for es in snrs:
        info = rng.integers(0, 2, (n_cw, spec.k)).astype(np.uint8)
        x = constellation.modulate(codes.encode_info(spec, info).reshape(-1), pts)
        var = np.full(x.shape, 10 ** (-es / 10))
        y = x + np.sqrt(var / 2) * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape))
        h = np.ones_like(x)
        soft, done, per, t0 = demap(y, h, var, pts).reshape(n_cw, -1), np.zeros(n_cw, bool), [], time.time()
        for o in range(outer + 1):
            deint = codes._deinterleave(spec, soft.astype(np.float32))
            out, ok, post = dec.decode(deint, iters=inner, posterior=True)
            done |= ok & (out == info).all(axis=1)
            per.append(1 - done.mean())
            if done.all():
                per += per[-1:] * (outer - o)
                break
            # the decoder's extrinsic: its posterior less the channel input it was given
            ext = np.clip(post - np.clip(deint, -ldpc.CH_CLAMP, ldpc.CH_CLAMP), -20, 20)[:, perm]
            soft = demap(y, h, var, pts, ext.reshape(-1, m)).reshape(n_cw, -1)
        print(f"{name:16s} Es/N0 {es:5.2f}  PER after pass " + " ".join(f"{v:.3f}" for v in per)
              + f"  ({time.time() - t0:.0f} s)", flush=True)


GRID = {"w48-16qam-r1/2": (5.0, 6.5), "w48-64l-r1/2": (9.0, 10.5), "w48-64l-r2/3": (12.25, 13.5),
        "w48-256l-r5/8": (15.5, 17.0)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("exit")
    p = sub.add_parser("per")
    p.add_argument("submode")
    p.add_argument("--snr", nargs=2, type=float, help="Es/N0 range (dB), 0.25 dB steps")
    p.add_argument("--n", type=int, default=200, help="codewords per point")
    p.add_argument("--outer", type=int, default=4, help="re-demap passes after the first decode")
    p.add_argument("--inner", type=int, default=codes.LDPC_ITERS, help="LDPC iterations per pass")
    a = ap.parse_args()
    if a.cmd == "exit":
        exit_study()
    else:
        lo, hi = a.snr or GRID[a.submode]
        per_study(a.submode, np.arange(lo, hi + 1e-9, 0.25), a.n, a.outer, a.inner)


if __name__ == "__main__":
    main()
