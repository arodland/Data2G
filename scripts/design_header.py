"""Pick a header code: an (N, 16) binary linear code (N = 192 for the
wide band; 2 * carriers * header symbols per band), the generator
with the largest minimum distance (then fewest minimum-weight words)
among random draws. Exact: k = 16 means all 65535 nonzero codewords are
enumerated. Writes data2g/codes_data/header_code.npy (the on-air format;
committed, never regenerated at import).

    uv run python scripts/design_header.py --draws 400
    uv run python scripts/design_header.py --n 200 --out data2g/codes_data/header_code_n10.npy
"""

import argparse

import numpy as np

K = 16


def weights(g):
    msgs = (np.arange(1, 2**K)[:, None] >> np.arange(K - 1, -1, -1)) & 1
    return ((msgs @ g) % 2).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--n", type=int, default=192, help="code length: 2 * carriers * header symbols")
    ap.add_argument("--out", default="data2g/codes_data/header_code.npy")
    a = ap.parse_args()
    best = None
    for seed in range(a.draws):
        g = np.random.default_rng(seed).integers(0, 2, (K, a.n))
        w = weights(g)
        key = (w.min(), -np.sum(w == w.min()))
        if best is None or key > best[0]:
            best = (key, seed, g)
            print(f"seed {seed}: d_min {key[0]}, {-key[1]} words at d_min", flush=True)
    np.save(a.out, best[2].astype(np.uint8))
    print(f"saved seed {best[1]} to {a.out}")


if __name__ == "__main__":
    main()
