"""Gear-shifter phase A: how long the receiver takes to decode a burst,
which the IRS must do before it can ACK (part of every turnaround).

Per submode, a burst of --n-cw codewords (default: as many as MAX_CODEWORDS
allows), timed three ways: one codeword at a time as modem.demodulate
does today, batched on CPU torch, batched on CUDA. Two inputs: clean
LLRs (BP stops at the first iteration) and pure noise (runs all 40,
the worst case: a burst that fails).

    uv run python scripts/decode_latency.py --out runs/decode_latency.csv
"""

import argparse
import csv
import time

import numpy as np
import torch

from data2g import codes
from data2g.config import MAX_CODEWORDS, SUBMODES


def timed(fn, reps=3):
    fn()  # warm-up (lazy construction, CUDA kernels)
    t = []
    for _ in range(reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    return min(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cw", type=int, default=MAX_CODEWORDS)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(0)
    rows = []
    for spec in SUBMODES.values():
        bits = np.stack([codes.encode(spec, rng.bytes(codes.payload_bytes(spec))) for _ in range(a.n_cw)])
        inputs = {"clean": 8.0 * (1.0 - 2.0 * bits), "noise": rng.normal(scale=2.0, size=bits.shape)}
        for kind, llr in inputs.items():
            row = dict(submode=spec.name, code=spec.code, n=spec.coded_bits, n_cw=a.n_cw, input=kind)
            row["loop_cpu_s"] = timed(lambda: [codes.decode_llrs(spec, l[None]) for l in llr], reps=1)
            row["batch_cpu_s"] = timed(lambda: codes.decode_llrs(spec, llr))
            if torch.cuda.is_available():
                row["batch_cuda_s"] = timed(lambda: codes.decode_llrs(spec, llr, device="cuda"))
            rows.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
