"""Outcome-trained link predictor: dataset (replaces predictor_data.py's
channel-MI targets).

Per sample, on one continuous fading process (scripts/phy_session.py's
ContinuousChannel, a slow SNR drift on top), everything through the real
modem and receiver:
- the receiver's history: a previous burst (sometimes none) and a current
  burst, in modes plausible at this SNR and sized as the shifter sends them,
  each measured (arq.phy.measure);
- then CANDIDATES next bursts, all sent at the same moment through the same
  fading (independent noise), in modes near their decision boundary for this
  channel, each with its outcome: burst_ok (synced, its header read right,
  its first codeword decoded: the burst was usable) and its other codewords
  decoded.

One row per candidate. scripts/train_outcome.py fits P(burst ok) and
P(codeword ok | burst ok) per submode from these.

    uv run python scripts/outcome_data.py --samples 20000 --out runs/outcome_data.csv
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

from data2g import codes, modem
from data2g.arq import phy as PHY
from data2g.arq import policy as G
from data2g.arq.link import Slot, TxBurst
from data2g.config import SUBMODES

sys.path.insert(0, str(Path(__file__).parent))
import ir_study as I  # noqa: E402
import phy_session as PS  # noqa: E402

CANDIDATES = 4
REPLY_MODES = {"ack-1f", "ack-4f", "n10-ack-4f", "n4-ack-2f", "n4-ack-8f", "qpsk-r1/5", "n10-qpsk-r1/3"}
KINDS = (("awgn", 0.15), ("mpg", 0.2), ("mpp", 0.2), ("mpd", 0.15), ("random", 0.3))
THRESHOLDS = I.thresholds()


def threshold(name: str, kind: str) -> float:
    """The mode's ladder threshold (1% codeword failures) on this channel family."""
    t = THRESHOLDS.get((I.ladder_name(SUBMODES[name]), kind))
    return t if t is not None else 99.0


def family(doppler: float) -> str:
    return "awgn" if doppler == 0 else "mpg" if doppler <= 0.3 else "mpp" if doppler <= 1.5 else "mpd"


def burst(name: str, n: int, rng) -> TxBurst:
    s = SUBMODES[name]
    return TxBurst(name, [Slot((5, 0, i), 0, bytes(rng.integers(0, 256, codes.payload_bytes(s), dtype=np.uint8)))
                          for i in range(n)], 0)


def n_for(name: str, seconds: float) -> int:
    s = SUBMODES[name]
    return max(1, sum(1 for k in range(1, 65) if modem.burst_seconds(s, k) <= seconds))


def sample(seed):
    rng = np.random.default_rng(seed)
    kind = str(rng.choice([k for k, _ in KINDS], p=[w for _, w in KINDS]))
    if kind == "random":
        doppler, delay = float(np.exp(rng.uniform(np.log(0.05), np.log(3.0)))), float(rng.uniform(0, 5))
    else:
        doppler, delay = PS.L.PRESETS[kind]
    fam = family(doppler)
    # ~1% outside the realistic range, so nothing is wholly out of
    # distribution (a lossless loopback measured 20-33 dB)
    u = rng.random()
    snr0 = float(rng.uniform(-14, -8) if u < 0.005 else rng.uniform(22, 40) if u < 0.01 else rng.uniform(-8, 22))
    drift = float(rng.normal(0, 1.5))  # dB per 30 s
    cap = 0 if rng.random() < 0.25 else 2
    allowed = [s.name for s in G.allowed(cap)]
    ch = PS.ContinuousChannel(fam, snr0, seed, 120.0, doppler=doppler, delay_ms=delay)

    def hear(b: TxBurst, t0: float):
        ch.snr_db = snr0 + drift * t0 / 30
        try:
            return modem.receive(ch.apply(PHY.tx_audio(b), t0))
        except modem.SyncError:
            return None

    def measured_burst(t0):
        # what the shifter sends: a reply (ACK and connect modes, short), or
        # a data mode in its working range (threshold 3-8 dB below the SNR)
        if rng.random() < 0.01:  # any mode at all, rarely (as above)
            name = str(rng.choice(allowed))
            n = n_for(name, float(rng.choice(G.SIZE_S)))
        elif rng.random() < 0.35:
            name = str(rng.choice([m for m in allowed if m in REPLY_MODES]))
            n = int(rng.integers(1, 3))
        else:
            working = [m for m in allowed if snr0 - 8 <= threshold(m, fam) <= snr0 - 3] or \
                [min(allowed, key=lambda m: abs(threshold(m, fam) - (snr0 - 3)))]
            name = str(rng.choice(working))
            n = n_for(name, float(rng.choice(G.SIZE_S)))
        b = burst(name, n, rng)
        r = hear(b, t0)
        ok = r is not None and r["spec"].name == name and r["n_cw"] == n
        return (PHY.measure(r) if ok else None), name, t0 + modem.burst_seconds(SUBMODES[name], n)

    t = 1.0
    prev, prev_name, t = measured_burst(t) if rng.random() < 0.85 else (None, None, t)
    t_prev_end = t
    t += float(rng.uniform(1.0, 15.0)) if prev_name else 0.0
    cur, cur_name, t_cur_end = measured_burst(t)
    if cur is None:
        return []
    gap = float(rng.uniform(1.5, 4.0))
    t_next = t_cur_end + gap
    near = [m for m in allowed if abs(threshold(m, fam) - snr0) <= 5]
    cands = list(rng.choice(near, size=min(len(near), CANDIDATES - 1), replace=False)) if near else []
    while len(cands) < CANDIDATES:
        cands.append(str(rng.choice(allowed)))
    base = dict(seed=seed, kind=kind, doppler=round(doppler, 3), delay_ms=round(delay, 2), snr=round(snr0, 2),
                snr_next=round(snr0 + drift * t_next / 30, 2), cap=cap, band=SUBMODES[cur_name].band, gap=round(gap, 2),
                **{k: v for k, v in cur.items()})
    if prev is not None:
        base.update(prev_band=SUBMODES[prev_name].band, prev_age=round(t_cur_end - t_prev_end, 2),
                    **{f"prev_{k}": v for k, v in prev.items()})
    rows = []
    for name in cands:
        secs = float(rng.choice(G.SIZE_S))
        n = max(2, n_for(name, secs))
        b = burst(name, n, rng)
        r = hear(b, t_next)
        right = r is not None and r["spec"].name == name and r["n_cw"] == n
        ok = []
        if right:
            rx = PHY.ModemRx(r, {})
            ok = [rx.decode(i, s.mask_id, 0, None) == s.payload for i, s in enumerate(b.slots)]
        rows.append(dict(base, cand=name, cand_n=n, cand_seconds=round(modem.burst_seconds(SUBMODES[name], n), 2),
                         burst_ok=int(bool(ok) and ok[0]), data_sent=n - 1, data_ok=int(sum(ok[1:])) if ok else 0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meas = ["snr_est", "spread_est", "delay_est_ms", "headroom", "frames"] + [f"mi_{c}" for c in PHY.P.CONSTS]
    fields = (["seed", "kind", "doppler", "delay_ms", "snr", "snr_next", "cap", "band", "gap"] + meas
              + ["prev_band", "prev_age"] + [f"prev_{k}" for k in meas]
              + ["cand", "cand_n", "cand_seconds", "burst_ok", "data_sent", "data_ok"])
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    start = 0 if new else len({r["seed"] for r in csv.DictReader(open(a.out))})
    done = 0
    with Pool(a.jobs) as pool, open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fields)
        if new:
            w.writeheader()
        for rows in pool.imap_unordered(sample, range(start * 11 + 1, (start + a.samples) * 11 + 1, 11), chunksize=2):
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
            done += 1
            if done % 200 == 0:
                f.flush()
                print(done, flush=True)


if __name__ == "__main__":
    main()
