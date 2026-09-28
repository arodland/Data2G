"""Outcome-model training rows from real-modem ARQ sessions: what the
shifter actually sees (outcome model v4 was calibrated offline but lost
in sessions: its training SNR prior made a +3 dB reply reading mean "better
than 0 dB", and a session held at 0 dB is that prior's tail).

Per session (300 s, scripts/loss_study.py's real modem): a channel kind
(outcome_data.KINDS, random Doppler/delay included), SNR uniform -8..22 dB
(~1% at -14..-8 or 22..40) drifting slowly, cap 500 Hz a quarter of the
time. Every burst sent after its receiver has a measurement is a row: the
receiver's inputs at its last recommendation, and what the burst did
(burst_ok: control decoded; first-transmission data codewords sent/ok),
in scripts/outcome_data.py's columns. Exploration: EXPLORE of the data and
of the reply recommendations are a random allowed mode and size. Control is
never duplicated (the model's P(usable) is without it).

    uv run python scripts/session_data.py --sessions 800 --out runs/session_data.csv
"""

import os

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data2g.arq import policy as G
from data2g.arq import predictor as P
from data2g.arq.modes import MODES, burst_seconds

sys.path.insert(0, str(Path(__file__).parent))
import linksim as L  # noqa: E402
import loss_study as LS  # noqa: E402
import outcome_data as O  # noqa: E402
import phy_session as PS  # noqa: E402

EXPLORE = 0.2
MEAS = ["snr_est", "spread_est", "delay_est_ms", "headroom", "frames"] + [f"mi_{c}" for c in P.CONSTS]
FIELDS = (["seed", "kind", "doppler", "delay_ms", "snr", "snr_next", "cap", "band", "gap"] + MEAS
          + ["prev_band", "prev_age"] + [f"prev_{k}" for k in MEAS]
          + ["cand", "cand_n", "cand_seconds", "burst_ok", "data_sent", "data_ok", "explored"])


class Explorer(G.GearShifter):
    """The shifter, snapshotting its inputs at each recommendation and
    sometimes recommending at random."""

    def __init__(self, rng, **kw):
        super().__init__(use_cpm=True, **kw)
        self.rng, self.snap = rng, None

    def recommend(self, station):
        rec, hint, reply = super().recommend(station)
        self.want_dup = False
        if self.measured is None:
            return rec, hint, reply
        prev = None
        if self.prev is not None and self.measured_at - self.prev[2] <= G.PREV_MAX_S:
            prev = (self.prev[0], self.prev[1], self.measured_at - self.prev[2])
        ok = [s.name for s in G.allowed(station.cap) if not G.is_cpm(s) or P.outcome_knows(s.name)]
        explored = False
        if self.rng.random() < EXPLORE:
            rec, hint, explored = G.encode(self.rng.choice(ok)), self.rng.randrange(len(G.SIZE_S)), True
        if self.rng.random() < EXPLORE:
            reply = G.encode(self.rng.choice(ok))
        self.snap = dict(m=self.measured, band=self.measured_band, t=self.measured_at, prev=prev, explored=explored)
        return rec, hint, reply


class DataPhy(LS.AuditPhy):
    def __init__(self, ch, tag, pols, snr0, drift, out):
        super().__init__(ch, [], tag)
        self.pols, self.snr0, self.drift, self.out = pols, snr0, drift, out

    def send(self, burst, t0):
        self.ch.snr_db = self.snr0 + self.drift * t0 / 30
        rx = self.pols[1 - burst.slots[0].mask_id[1]]  # direction 0 is station a's: b receives
        snap, n = rx.snap, len(self.rows)
        res = super().send(burst, t0)
        if snap is not None and len(self.rows) > n:
            a = self.rows[-1]
            spec = MODES[burst.submode]
            dup = int(a.get("dup") or 0)
            row = dict(self.tag, snr_next=round(self.ch.snr_db, 2), band=snap["band"], gap=round(t0 - snap["t"], 2),
                       **{k: snap["m"].get(k, 0.0) for k in MEAS},
                       cand=spec.name, cand_n=len(burst.slots) - dup,
                       cand_seconds=round(burst_seconds(spec, len(burst.slots), bool(dup)), 2),
                       burst_ok=int(a["outcome"] == "ok"), data_sent=a["data_sent"], data_ok=a["data_ok"],
                       explored=int(snap["explored"]))
            if snap["prev"] is not None:
                pm, pb, age = snap["prev"]
                row.update(prev_band=pb, prev_age=round(age, 2), **{f"prev_{k}": pm.get(k, 0.0) for k in MEAS})
            self.out.append(row)
        return res


def session(seed):
    rng = np.random.default_rng(seed)
    kind = str(rng.choice([k for k, _ in O.KINDS], p=[w for _, w in O.KINDS]))
    if kind == "random":
        doppler, delay = float(np.exp(rng.uniform(np.log(0.05), np.log(3.0)))), float(rng.uniform(0, 5))
    else:
        doppler, delay = PS.L.PRESETS[kind]
    u = rng.random()
    snr0 = float(rng.uniform(-14, -8) if u < 0.005 else rng.uniform(22, 40) if u < 0.01 else rng.uniform(-8, 22))
    drift = float(rng.normal(0, 1.0))  # dB per 30 s
    cap = 0 if rng.random() < 0.25 else 2
    horizon = 300.0
    ch = PS.ContinuousChannel(O.family(doppler), snr0, seed, horizon + 60, doppler=doppler, delay_ms=delay)
    pols = [Explorer(random.Random(seed * 2 + i)) for i in range(2)]
    out = []
    tag = dict(seed=seed, kind=kind, doppler=round(doppler, 3), delay_ms=round(delay, 2), snr=round(snr0, 2), cap=cap)
    phy = DataPhy(ch, tag, pols, snr0, drift, out)
    try:
        L.run(pols[0], pols[1], None, L.WORKLOADS["bulk"](random.Random(seed + 7)), seed=seed, horizon=horizon,
              phy=phy, cap=cap)
    except AssertionError as e:  # a corrupt delivery would be a bug: keep the rows, say so
        print("session", seed, "assertion:", str(e)[:120], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=800)
    ap.add_argument("--first", type=int, default=300000)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    done = set() if new else {int(r["seed"]) for r in csv.DictReader(open(a.out))}
    todo = [s for s in range(a.first, a.first + a.sessions) if s not in done]
    n = 0
    with Pool(a.jobs) as pool, open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for rows in pool.imap_unordered(session, todo):
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})
            f.flush()
            n += 1
            if n % 20 == 0:
                print(n, "sessions", flush=True)


if __name__ == "__main__":
    main()
