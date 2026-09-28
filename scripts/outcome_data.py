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

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g import codes, cpm, modem
from data2g.arq import phy as PHY
from data2g.arq import policy as G
from data2g.arq.link import Slot, TxBurst, ctl_mask, data_mask
from data2g.arq.modes import MODES, burst_seconds, ctl_spec, is_cpm
from data2g.config import FS
from data2g.tnc import search_span

sys.path.insert(0, str(Path(__file__).parent))
import ir_study as I  # noqa: E402
import phy_session as PS  # noqa: E402

CANDIDATES = 4
REPLY_MODES = {"ack-1f", "ack-4f", "n10-ack-4f", "n4-ack-2f", "n4-ack-8f", "qpsk-r1/5", "n10-qpsk-r1/3"}
# --sustained: a supplement for sustained low SNR (a trial at MPP -6 dB PEP5
# found the model flat there: qpsk-r1/5 x9 predicted 0.71 per codeword sent,
# 0.26 actual). No drift, the measured burst often a reply (fragile: synced
# only in an up-fade, as a receiver in such a session measures), candidates
# at the long size classes the shifter sends at low SNR.
SUSTAINED = False
KINDS = (("awgn", 0.15), ("mpg", 0.2), ("mpp", 0.2), ("mpd", 0.15), ("random", 0.3))
THRESHOLDS = I.thresholds()
# CPM modes' 1% end-to-end points (the CPM prototype's study, 1% v2), for
# choosing modes near their working range only
CPM_THRESHOLDS = {"fsk16r25-r1/3": (-13.4, -5.0, -10.2, -10.8), "fsk16r25-r1/2": (-12.3, -2.8, -8.0, -8.2),
                  "fsk8r50-r1/3": (-11.2, -0.7, -6.9, -7.2), "fsk8r50-r1/2": (-9.7, 1.9, -5.4, -5.2),
                  "fsk32r62-r1/3": (-8.8, 1.8, -4.8, -5.2), "fsk32r62-r1/2": (-7.6, 2.9, -1.6, -2.2)}


def threshold(name: str, kind: str) -> float:
    """The mode's ladder threshold (1% codeword failures) on this channel family."""
    if name in CPM_THRESHOLDS:
        return CPM_THRESHOLDS[name][("awgn", "mpg", "mpp", "mpd").index(kind)]
    t = THRESHOLDS.get((I.ladder_name(MODES[name]), kind))
    return t if t is not None else 99.0


def family(doppler: float) -> str:
    return "awgn" if doppler == 0 else "mpg" if doppler <= 0.3 else "mpp" if doppler <= 1.5 else "mpd"


def burst(name: str, n: int, rng) -> TxBurst:
    """A control codeword, then n - 1 data codewords (random payloads)."""
    s = MODES[name]
    rand = lambda spec: bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8))  # noqa: E731
    return TxBurst(name, [Slot(ctl_mask(0, 0, 5), 0, rand(ctl_spec(s)))]
                   + [Slot(data_mask(0, i, 5), 0, rand(s)) for i in range(1, n)], 0)


def n_for(name: str, seconds: float) -> int:
    return G.slots_for(MODES[name], seconds)


def receive(y: np.ndarray, name: str) -> dict | None:
    """The receiver's result for a burst of `name` in y, or None (no sync).
    ponytail: listens for the sent family only (OFDM, or CPM on the sent
    grid); cross-family false locks are not in the data."""
    s = MODES[name]
    if not is_cpm(s):
        try:
            return modem.receive(y)
        except modem.SyncError:
            return None
    # the head only, as the streaming receiver searches (a whole-burst search
    # can lock on a later sync block)
    lock = cpm.find(cpm.GRIDS[s.grid], y[:int(PS.PAD_S * FS) + FS // 2 + search_span((), (s.grid,))])
    return cpm.receive(y, lock) if lock is not None else None


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
    if SUSTAINED:
        snr0, drift = float(rng.uniform(-10, 4)), 0.0
    cap = 0 if rng.random() < 0.25 else 2
    allowed = [s.name for s in G.allowed(cap)]
    ch = PS.ContinuousChannel(fam, snr0, seed, 120.0, doppler=doppler, delay_ms=delay)

    def hear(b: TxBurst, t0: float):
        ch.snr_db = snr0 + drift * t0 / 30
        try:
            return receive(ch.apply(PHY.tx_audio(b), t0), b.submode)
        except modem.SyncError:  # ponytail: an OFDM header read past the audio
            return None

    def measured_burst(t0):
        # what the shifter sends: a reply (ACK and connect modes, short), or
        # a data mode in its working range (threshold 3-8 dB below the SNR)
        if rng.random() < 0.01:  # any mode at all, rarely (as above)
            name = str(rng.choice(allowed))
            n = n_for(name, float(rng.choice(G.SIZE_S)))
        elif rng.random() < (0.6 if SUSTAINED else 0.35):
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
        return (PHY.measure(r) if ok else None), name, t0 + burst_seconds(MODES[name], n)

    t = 1.0
    prev, prev_name, t = measured_burst(t) if rng.random() < 0.85 else (None, None, t)
    t_prev_end = t
    t += float(rng.uniform(1.0, 15.0)) if prev_name else 0.0
    cur, cur_name, t_cur_end = measured_burst(t)
    if cur is None:
        return []
    gap = float(rng.uniform(1.5, 4.0))
    t_next = t_cur_end + gap
    # one reply mode (sent as replies are, 1-2 codewords: without these the
    # model barely knew the ACK modes' sync, and the shifter replied in
    # ack-4f at MPP -4 dB, 79% usable, over n10-ack-4f, 95%), two near their
    # decision boundary, one at random
    near = [m for m in allowed if abs(threshold(m, fam) - snr0) <= 5]
    cands = [str(rng.choice([m for m in allowed if m in REPLY_MODES]))]
    cands += list(rng.choice(near, size=min(len(near), CANDIDATES - 2), replace=False)) if near else []
    cpm_ok = [m for m in allowed if m in cpm.SPECS]
    if cpm_ok and rng.random() < 0.3:  # CPM candidates oversampled (a new family: few rows otherwise)
        cands.append(str(rng.choice(cpm_ok)))
    while len(cands) < CANDIDATES:
        cands.append(str(rng.choice(allowed)))
    base = dict(seed=seed, kind=kind, doppler=round(doppler, 3), delay_ms=round(delay, 2), snr=round(snr0, 2),
                snr_next=round(snr0 + drift * t_next / 30, 2), cap=cap, band=MODES[cur_name].band, gap=round(gap, 2),
                **{k: v for k, v in cur.items()})
    if prev is not None:
        base.update(prev_band=MODES[prev_name].band, prev_age=round(t_cur_end - t_prev_end, 2),
                    **{f"prev_{k}": v for k, v in prev.items()})
    rows = []
    for j, name in enumerate(cands):
        sizes = G.SIZE_S[2:] if SUSTAINED else G.SIZE_S
        n = int(rng.integers(1, 3)) if j == 0 else max(2, n_for(name, float(rng.choice(sizes))))
        b = burst(name, n, rng)
        r = hear(b, t_next)
        right = r is not None and r["spec"].name == name and r["n_cw"] == n
        ok = []
        if right:
            rx = PHY.ModemRx(r, {})
            ok = [rx.decode(i, s.mask_id, 0, None) == s.payload for i, s in enumerate(b.slots)]
        rows.append(dict(base, cand=name, cand_n=n, cand_seconds=round(burst_seconds(MODES[name], n), 2),
                         burst_ok=int(bool(ok) and ok[0]), data_sent=n - 1, data_ok=int(sum(ok[1:])) if ok else 0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, default=0, help="first sample number (another dataset's seeds: past its end)")
    ap.add_argument("--sustained", action="store_true", help="the sustained-low-SNR supplement (see SUSTAINED)")
    a = ap.parse_args()
    global SUSTAINED
    SUSTAINED = a.sustained
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
        for rows in pool.imap_unordered(sample, range((a.first + start) * 11 + 1, (a.first + start + a.samples) * 11 + 1, 11), chunksize=2):
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
            done += 1
            if done % 200 == 0:
                f.flush()
                print(done, flush=True)


if __name__ == "__main__":
    main()
