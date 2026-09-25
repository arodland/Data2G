"""Low-PAPR pilot phases for an n-carrier band (50 Hz grid).

Minimizes the envelope peak-to-average power of one pilot symbol,
evaluated 16x oversampled over the whole period, from many random
starts (L-BFGS on a soft maximum), then quantizes each phase to a
rational turn NUM/1024 as SSTVAE's pilot is, so that C++ evaluates the
identical phasors. Prints the phases and the PAPR after quantization.

    uv run python scripts/design_pilot.py --carriers 4 10 24 --starts 200
"""

import argparse

import numpy as np
from scipy.optimize import minimize

DEN = 1024
OVERSAMPLE = 16


def papr_db(phases: np.ndarray) -> float:
    n = len(phases)
    t = np.arange(n * OVERSAMPLE) / (n * OVERSAMPLE)
    env = np.abs(np.exp(1j * (2 * np.pi * np.outer(t, np.arange(n)) + phases)).sum(axis=1)) ** 2
    return 10 * np.log10(env.max() / env.mean())


def design(n: int, starts: int, seed: int = 0) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    t = np.arange(n * OVERSAMPLE) / (n * OVERSAMPLE)
    e = np.exp(2j * np.pi * np.outer(t, np.arange(n)))

    def soft_peak(p, beta=40.0):
        env = np.abs(e @ np.exp(1j * p)) ** 2 / n
        return np.log(np.mean(np.exp(beta * (env - env.max())))) / beta + env.max()

    best = None
    for _ in range(starts):
        r = minimize(soft_peak, rng.uniform(0, 2 * np.pi, n), method="L-BFGS-B")
        num = np.round(r.x / (2 * np.pi) * DEN).astype(int) % DEN
        p = papr_db(2 * np.pi * num / DEN)
        if best is None or p < best[1]:
            best = (num, p)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carriers", type=int, nargs="+", default=[4, 10, 24])
    ap.add_argument("--starts", type=int, default=200)
    a = ap.parse_args()
    for n in a.carriers:
        num, p = design(n, a.starts)
        print(f"{n} carriers: PAPR {p:.2f} dB  NUM = {tuple(int(v) for v in num)}", flush=True)


if __name__ == "__main__":
    main()
