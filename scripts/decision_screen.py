"""Offline screen for outcome models: decisions, not likelihood.

scripts/outcome_data.py's samples send 4 candidate bursts at the same moment
through the same channel, so every candidate's outcome is known. Per sample,
a model picks the candidate with the best predicted goodput (P(usable) x
P(codeword | usable) x data codewords x payload over the burst's airtime
plus TURN_S of turnaround); its regret is the best candidate's realized
goodput less its pick's. Reported as mean regret over mean best goodput,
by channel and SNR band, so models can be ranked before a loss study.

A spec is a model file, or "router:<installed>,<gate>,<fast>" (either
region may be "-"): the gate where spread_est < GATE_SPREAD_HZ and snr_est
< GATE_SNR_DB, the fast region where spread_est >= FAST_SPREAD_HZ and
snr_est in [FAST_SNR_LO_DB, FAST_SNR_HI_DB), as GearShifter.gate (on the
row's own spread_est: rows hold no history). LOGIT_OFFSETS as shipped.

    python -m scripts.decision_screen runs/outcome_data_eval.csv shipped=router:a.npz,b.npz,- student=runs/s.npz
"""

from . import session_data as SD  # noqa: I001 (first: it sets the thread caps before numpy loads)

import argparse
from collections import defaultdict

import numpy as np

from data2g import codes
from data2g.arq import predictor as P

from . import train_outcome as T

TURN_S = 2.0
OFFSETS = {"w48-16qam-r1/2": -1.0, "n10-256l-r3/4": -0.5}


def logits(spec, x_e, x_n, spread, snr):
    """(rows, 2n) logits in T.MODES' order for a spec."""
    if not spec.startswith("router:"):
        m = P.outcome_model(spec)
        return m(x_e if m.energy else x_n)
    inst, gate, fast = spec[len("router:"):].split(",")
    region = np.zeros(len(snr), dtype=int)
    if gate != "-":
        region[(spread < P.GATE_SPREAD_HZ) & (snr < P.GATE_SNR_DB)] = 1
    if fast != "-":
        region[(spread >= P.FAST_SPREAD_HZ) & (snr >= P.FAST_SNR_LO_DB) & (snr < P.FAST_SNR_HI_DB)] = 2
    z = np.empty((len(snr), 2 * len(T.MODES)))
    for k, path in enumerate((inst, gate, fast)):
        s = region == k
        if s.any():
            m = P.outcome_model(path)
            z[s] = m(x_e[s] if m.energy else x_n[s])
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("specs", nargs="+", help="name=spec")
    a = ap.parse_args()
    x_e, mode, bok, dsent, dok, rows, _ = T.load(a.data, energy=True)
    x_n = T.load(a.data)[0]
    spread = np.array([float(r["spread_est"]) for r in rows])
    snr_est = np.array([float(r["snr_est"]) for r in rows])
    n = len(T.MODES)
    pay = np.array([codes.payload_bytes(SD.MODES[T.MODES[k]]) for k in mode], dtype=float)
    sec = np.array([float(r["cand_seconds"]) for r in rows]) + TURN_S
    real = bok * dok * pay / sec
    sample = np.array([int(r["seed"]) for r in rows])
    kind = np.array([r["kind"] for r in rows])
    snr = np.array([float(r["snr"]) for r in rows])
    band = np.where(snr < -2, "<-2", np.where(snr < 4, "-2..4", np.where(snr < 12, "4..12", ">=12")))
    _, first, inv = np.unique(sample, return_index=True, return_inverse=True)
    best = np.zeros(len(first))
    np.maximum.at(best, inv, real)
    cells = sorted({(kind[i], band[i]) for i in first})
    print(f"{len(first)} samples; mean regret / mean best goodput, by channel and SNR (PEP-referenced cell SNR)")
    print(f"{'':10s} {'all':>7s} " + " ".join(f"{k[:6]:>6s}{b:>6s}" for k, b in cells))
    for item in a.specs:
        name, spec = item.split("=", 1)
        z = logits(spec, x_e, x_n, spread, snr_est)
        for m, off in OFFSETS.items():
            if m in T.MODES:
                z[:, T.MODES.index(m)] += off
        pb = 1 / (1 + np.exp(-np.clip(z[np.arange(len(mode)), mode], -40, 40)))
        pc = 1 / (1 + np.exp(-np.clip(z[np.arange(len(mode)), mode + n], -40, 40)))
        pred = pb * pc * dsent * pay / sec
        pick = defaultdict(lambda: (-1.0, 0.0))
        for i in range(len(mode)):
            if pred[i] > pick[inv[i]][0]:
                pick[inv[i]] = (pred[i], real[i])
        got = np.array([pick[j][1] for j in range(len(first))])
        reg = best - got
        out = [f"{reg.sum() / best.sum():7.3f}"]
        for k, b in cells:
            s = (kind[first] == k) & (band[first] == b)
            out.append(f"{reg[s].sum() / max(best[s].sum(), 1e-9):12.3f}")
        print(f"{name:10s} " + " ".join(out), flush=True)


if __name__ == "__main__":
    main()
