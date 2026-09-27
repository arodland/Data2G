"""Re-decoding failed codewords with erasures, after modem73 (a CRC
failure retries with the worst 1/8, then 1/4, of its symbol rows erased,
then rows under 0.3 x the median quality). Our LLRs already weigh each
cell by its channel and noise estimate, so a retry only wins where that
model is wrong: a row hit by something the estimate doesn't see.

Row quality: each OFDM symbol's mean decision residual, min over the
constellation of |y - h x|^2 / var (about 1 where the model holds).
Erasing a row: its cells' variance to infinity (LLRs 0).

OFDM bursts of N_CW codewords around each mode's 10% point; for every
codeword the first decode fails, whether any retry decodes it right, and
any retry that passes its CRC with the wrong payload (a false accept).

    uv run --no-sync python scripts/erasure_retry_study.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import codes, constellation, hfchannel, modem
from data2g.arq import phy as PHY
from data2g.config import FS
from data2g.tnc import Blanker, receive_any

sys.path.insert(0, str(Path(__file__).parent))
import outcome_data as O  # noqa: E402

NAMES = ("qpsk-r1/2", "16qam-r1/2", "n10-qpsk-r1/2", "w48-16qam-r1/2")
# (channel, clicks per second): clicks at 20 dB over the signal, blanked
CHANNELS = (("mpg", 0), ("mpp", 0), ("mpd", 0), ("awgn", 5))
OFFSETS = (-2.0, -1.0, 0.0)  # dB from the smallest burst's 10% point (mostly a sync point: codewords fail below it)
N_CW = 8
RETRIES = ("worst 1/8", "worst 1/4", "residual > 3.3 x median")


def retries(r: dict):
    """-> [(label, soft (n_cw, coded_bits))] with rows erased."""
    spec, est = r["spec"], r["est"]
    raw, h = r["raw"][:, 1:], est["h"]
    var = modem.noise_var(h, est) + est["mse"]
    pts = constellation.load(spec.constellation)
    res = np.min(np.abs(raw[..., None] - h[..., None] * pts) ** 2, axis=-1) / var
    q = res.mean(axis=-1).reshape(-1)  # per OFDM symbol (frames x data symbols): high is bad
    order = np.argsort(-q)
    out = []
    for label, rows in (("worst 1/8", order[:max(1, len(q) // 8)]), ("worst 1/4", order[:max(1, len(q) // 4)]),
                        ("residual > 3.3 x median", np.flatnonzero(q > 3.3 * np.median(q)))):
        if not len(rows):
            out.append((label, None))
            continue
        v = var.copy()
        v.reshape(-1, v.shape[-1])[rows] = np.inf
        out.append((label, np.asarray(codes.despread(modem.soft_bits(r["raw"], h, v, spec), r["n_cw"],
                                                     spec.bits_per_cu))))
    return out


def trial(args):
    """-> None (burst not received), else (codewords, first-pass failures,
    per retry: recovered, false accepts; any retry recovered; any false)."""
    name, chan, rate, snr, seed = args
    rng = np.random.default_rng(seed)
    b = O.burst(name, N_CW, rng)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    x = np.concatenate([np.zeros(lead), PHY.tx_audio(b), np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=float(rng.uniform(-50, 50)), ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    if rate:
        y = Blanker()(hfchannel.clicks(y, rate, 20.0, seed=seed + 2, s_power=hfchannel.active_power(x)))
    try:
        r = receive_any(y, lead=FS)
    except Exception:  # noqa: BLE001
        return None
    if r is None or r["spec"].name != name or r["n_cw"] != N_CW:
        return None
    spec = r["spec"]
    masks = [PHY.mask_value(s.mask_id) for s in b.slots]
    first = codes.decode_many(spec, PHY.soft_bits(r), masks, index=0)
    failed = [i for i, (p, ok) in enumerate(first) if not (ok and p == b.slots[i].payload)]
    per = []
    got, bad = set(), set()
    if failed:
        for label, soft in retries(r):
            if soft is None:
                per.append((0, 0))
                continue
            dec = codes.decode_many(spec, soft[failed], [masks[i] for i in failed], index=0)
            rec = {i for i, (p, ok) in zip(failed, dec) if ok and p == b.slots[i].payload}
            wrong = {i for i, (p, ok) in zip(failed, dec) if ok and p != b.slots[i].payload}
            got |= rec
            bad |= wrong
            per.append((len(rec), len(wrong)))
    else:
        per = [(0, 0)] * len(RETRIES)
    return N_CW, len(failed), per, len(got), len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="runs/ladder_10pct.csv")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="runs/erasure_retry.csv")
    a = ap.parse_args()
    p10 = {r["name"]: r for r in csv.DictReader(open(a.ladder))}
    rows = []
    with Pool(a.jobs) as pool:
        for name in NAMES:
            for chan, rate in CHANNELS:
                for off in OFFSETS:
                    snr = float(p10[name][chan]) + off
                    res = [t for t in pool.map(trial, [(name, chan, rate, snr, 104729 * k + 3) for k in range(a.trials)])
                           if t is not None]
                    row = dict(name=name, channel=chan, clicks=rate, snr=round(snr, 3), bursts=len(res),
                               codewords=sum(t[0] for t in res), failed=sum(t[1] for t in res),
                               recovered=sum(t[3] for t in res), false_accepts=sum(t[4] for t in res))
                    for j, label in enumerate(RETRIES):
                        row[f"rec {label}"] = sum(t[2][j][0] for t in res)
                        row[f"false {label}"] = sum(t[2][j][1] for t in res)
                    rows.append(row)
                    print(row, flush=True)
                    with open(a.out, "w", newline="") as f:
                        w = csv.DictWriter(f, list(rows[0]))
                        w.writeheader()
                        w.writerows(rows)


if __name__ == "__main__":
    main()
