"""Finite-length check of a searched protograph against NR's graph:
BPSK on AWGN, same (k, n), same decoder (BP, 50 iterations), same seeds.
PEXIT thresholds are asymptotic; this is what decides whether one ships.

    uv run python scripts/compare_ldpc.py --proto bg2:runs/proto/bg2_r0.5.npy \\
        --k 1024 --n 2048 --ebn0 1.0 1.2 1.4 1.6 1.8
"""

import argparse

import numpy as np
import torch

from data2g import ldpc


def bler(code, k, n, ebn0s, blocks, device, seed=1):
    dec = ldpc.MinSumDecoder(code, device=device)
    out = []
    for ebn0 in ebn0s:
        rng = np.random.default_rng(seed)
        fe = be = done = 0
        while done < blocks:
            bits = rng.integers(0, 2, (2000, k))
            x = 1 - 2.0 * code.encode(bits)
            sig = np.sqrt(1 / (2 * k / n * 10 ** (ebn0 / 10)))
            y = x + rng.normal(scale=sig, size=x.shape)
            est, _ = dec.decode(torch.tensor(2 * y / sig**2, dtype=torch.float32, device=device), iters=50)
            e = est.cpu().numpy() != bits
            fe += e.any(1).sum()
            be += e.sum()
            done += 2000
        out.append((fe / done, be / done / k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ebn0", type=float, nargs="+", required=True)
    ap.add_argument("--blocks", type=int, default=40000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    bg, path = a.proto.split(":", 1)
    bg = int(bg[2:])
    codes = {
        "NR": ldpc.nr_code(a.k, a.n, bg=bg, shifts="nr"),
        "searched": ldpc.protograph_code(np.load(path), bg, a.k, a.n),
    }
    print(f"k={a.k} n={a.n} z={codes['NR'].z}; BLER / BER at Eb/N0 {a.ebn0}")
    for name, code in codes.items():
        r = bler(code, a.k, a.n, a.ebn0, a.blocks, a.device)
        print(f"  {name:9s}", "  ".join(f"{b:.1e}/{e:.0e}" for b, e in r), flush=True)


if __name__ == "__main__":
    main()
