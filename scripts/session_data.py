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

from data2g import interference as INTF
from data2g.arq import policy as G
from data2g.arq import predictor as P
from data2g.arq.modes import MODES, burst_seconds

sys.path.insert(0, str(Path(__file__).parent))
import linksim as L  # noqa: E402
import loss_study as LS  # noqa: E402
import outcome_data as O  # noqa: E402
import phy_session as PS  # noqa: E402

EXPLORE = 0.2
# --slow: sustained low SNR on slow fading, what the regular sessions (SNR
# -8..22 dB drifting ~1 dB per 30 s, 300 s) barely cover: the candidate
# model lost 13% at MPG -4 dB (600 s at a fixed SNR) where its own picks
# had decoded 0.99 of codewords in session data (2026-09-29). MPG 70% of
# sessions, else Doppler 0.05-0.3 Hz and 0-2 ms; SNR fixed, -8..0 dB; 600 s.
SLOW = False
# --high: high SNR on fading channels (+10..+30 dB, drifting; no AWGN). With
# v10's sessions gone, a retrain lost 13-22% at MPD +20, MPP +15 and MPG +15
# against v12 (runs/cpmc_round.sh, 2026-10-03).
HIGH = False
# --fastlow: sustained low SNR on fast fading, where the models smear the
# steep edges of the faster modes (MPD 0: w48-qpsk-r1/3 predicted 0.38-0.46
# at -2..0 dB, decoded 0; MPP -8: qpsk-r1/5 0.6 vs 0.21) because the steering
# model avoids them there and uniform exploration rarely lands on them. MPP
# and MPD presets, else Doppler 1-3 Hz and 0-5 ms; SNR fixed, -8..+6 dB;
# 600 s. Exploration FASTLOW_EXPLORE, of it FASTLOW_FASTER at a mode faster
# than the recommendation by up to FASTLOW_SPAN in asymptotic payload rate
# (the ladder is dense: the next two after fsk32r62-r1/2 are polar modes).
FASTLOW = False
FASTLOW_EXPLORE, FASTLOW_FASTER, FASTLOW_SPAN = 0.35, 0.7, 2.5
KINDS_ONLY = ()  # --kinds: only these channel kinds (outcome_data.KINDS names); with --high, AWGN allowed
MEAS = ["snr_est", "spread_est", "delay_est_ms", "headroom", "frames"] + [f"mi_{c}" for c in P.CONSTS]
FIELDS = (["seed", "kind", "doppler", "delay_ms", "snr", "snr_next", "cap", "band", "gap"] + MEAS
          + ["prev_band", "prev_age"] + [f"prev_{k}" for k in MEAS]
          + ["cand", "cand_n", "cand_seconds", "burst_ok", "data_sent", "data_ok", "explored"])
# the receiver's noise profile at the recommendation (tnc.NoiseProfile.snapshot()), and its interference
NOISE_COLS = ([f"noise_db{i}" for i in range(1, 6)] + [f"noise_tail{i}" for i in range(1, 6)]
              + ["impulses_per_min"])
# the link history at the recommendation (GearShifter.link_features; empty before any expected burst)
LINK_COLS = ["link_lost", "link_miss", "link_n"]
ENERGY_COLS = ["energy_db", "energy_n"]  # GearShifter.energy_features at the recommendation
FIELDS = FIELDS + NOISE_COLS + ["intf"] + LINK_COLS + ENERGY_COLS
# --interference [DRAWS]: each station's interference drawn from data2g.interference.draw()
INTERFERENCE = None
WANDER_DB = 0.0  # --wander: each station's floor wanders (phy_session.ContinuousChannel)


class Explorer(G.GearShifter):
    """The shifter, snapshotting its inputs at each recommendation and
    sometimes recommending at random."""

    def __init__(self, rng, **kw):
        super().__init__(use_cpm=True, **kw)
        self.rng, self.snap, self.noise = rng, None, None

    def observe(self, measured, submode, now):
        self.noise = measured.get("noise")  # kept here: a native shifter's measured drops it
        super().observe(measured, submode, now)

    def recommend(self, station):
        rec, hint, reply = super().recommend(station)
        self.want_dup = False
        if self.measured is None:
            return rec, hint, reply
        prev = None
        if self.prev is not None:  # as recommend() reads it: older history at PREV_MAX_S
            prev = (self.prev[0], self.prev[1], min(self.measured_at - self.prev[2], G.PREV_MAX_S))
        ok = [s.name for s in G.allowed(station.cap) if not G.is_cpm(s) or P.outcome_knows(s.name)]
        explored = False
        explore = FASTLOW_EXPLORE if FASTLOW else EXPLORE
        if self.rng.random() < explore:
            pick = self.rng.choice(ok)
            if FASTLOW and self.rng.random() < FASTLOW_FASTER:
                cur = G.decode(rec)
                faster = [m for m in ok if cur in MODES and rate(cur) < rate(m) <= FASTLOW_SPAN * rate(cur)]
                pick = self.rng.choice(faster) if faster else pick
            rec, hint, explored = G.encode(pick), self.rng.randrange(len(G.SIZE_S)), True
        if self.rng.random() < EXPLORE:
            reply = G.encode(self.rng.choice(ok))
        self.snap = dict(m=dict(self.measured, noise=self.noise), band=self.measured_band, t=self.measured_at,
                         prev=prev, explored=explored, link=self.link_now, energy=self.energy_features())
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
            noise = snap["m"].get("noise")
            if noise:
                row.update({f"noise_db{i + 1}": round(v, 2) for i, v in enumerate(noise["noise_db"])})
                row.update({f"noise_tail{i + 1}": round(v, 2) for i, v in enumerate(noise["noise_tail_db"])})
                row["impulses_per_min"] = round(noise["impulses_per_min"], 1)
            if snap["link"]:
                row.update(zip(LINK_COLS, (round(v, 4) for v in snap["link"][:3])))
            if snap["energy"]:
                row.update(zip(ENERGY_COLS, (round(v, 3) for v in snap["energy"][:2])))
            row["intf"] = INTF.describe(self.ch.intf[1 - (burst.slots[0].mask_id[1] & 1)].spec)
            self.out.append(row)
        return res


def rate(name: str) -> float:
    """A mode's asymptotic payload rate (bytes/s over a long burst), for --fastlow's 'faster'."""
    s = MODES[name]
    return G.codes.payload_bytes(s) / (burst_seconds(s, 9) - burst_seconds(s, 1)) * 8


def session(seed):
    rng = np.random.default_rng(seed)
    if FASTLOW:
        u = rng.random()
        kind = "mpp" if u < 0.35 else "mpd" if u < 0.7 else "random"
        doppler, delay = (PS.L.PRESETS[kind] if kind != "random"
                          else (float(np.exp(rng.uniform(np.log(1.0), np.log(3.0)))), float(rng.uniform(0, 5))))
        snr0, drift = float(rng.uniform(-8, 6)), 0.0
    elif SLOW:
        kind = "mpg" if rng.random() < 0.7 else "random"
        doppler, delay = (PS.L.PRESETS["mpg"] if kind == "mpg"
                          else (float(np.exp(rng.uniform(np.log(0.05), np.log(0.3)))), float(rng.uniform(0, 2))))
        snr0, drift = float(rng.uniform(-8, 0)), 0.0
    else:
        kinds = [(k, w) for k, w in O.KINDS if (k in KINDS_ONLY if KINDS_ONLY else not (HIGH and k == "awgn"))]
        ws = np.array([w for _, w in kinds])
        kind = str(rng.choice([k for k, _ in kinds], p=ws / ws.sum()))
        if kind == "random":
            doppler, delay = float(np.exp(rng.uniform(np.log(0.05), np.log(3.0)))), float(rng.uniform(0, 5))
        else:
            doppler, delay = PS.L.PRESETS[kind]
        u = rng.random()
        snr0 = float(rng.uniform(10, 30) if HIGH else rng.uniform(-14, -8) if u < 0.005
                     else rng.uniform(22, 40) if u < 0.01 else rng.uniform(-8, 22))
        drift = float(rng.normal(0, 1.0))  # dB per 30 s
    cap = 0 if rng.random() < 0.25 else 2
    horizon = 600.0 if SLOW or FASTLOW else 300.0
    intf = None
    if INTERFERENCE:
        irng = np.random.default_rng(np.random.SeedSequence([seed, 5]))
        intf = (INTF.draw(irng, INTERFERENCE), INTF.draw(irng, INTERFERENCE))
    ch = PS.ContinuousChannel(O.family(doppler), snr0, seed, horizon + 60, doppler=doppler, delay_ms=delay,
                              interference=intf, wander_db=WANDER_DB)
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
    ap.add_argument("--slow", action="store_true", help="sustained low SNR on slow fading (SLOW)")
    ap.add_argument("--fastlow", action="store_true", help="sustained low SNR on fast fading, edges explored (FASTLOW)")
    ap.add_argument("--high", action="store_true", help="high SNR on fading channels (HIGH)")
    ap.add_argument("--kinds", default="", help="comma-separated channel kinds only (KINDS_ONLY), e.g. awgn")
    ap.add_argument("--interference", nargs="?", const="v1", default=None, choices=sorted(INTF.DRAWS),
                    help="each station's interference drawn from interference.DRAWS[this] (default v1)")
    ap.add_argument("--wander", type=float, default=0.0, help="each station's floor wanders by this (dB)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--average-snr", action="store_true",
                    help="allow SNR against each burst's average power (without DATA2G_PEP_REF_DB)")
    a = ap.parse_args()
    if PS.PEP_REF_DB is None and not a.average_snr:
        ap.error("DATA2G_PEP_REF_DB is unset: set it (5: noise against each burst's peak, as data2g-host "
                 "transmits) or pass --average-snr")
    global SLOW, HIGH, INTERFERENCE, WANDER_DB, KINDS_ONLY, FASTLOW
    # set before the pool forks: the workers inherit them
    SLOW, HIGH, INTERFERENCE, WANDER_DB = a.slow, a.high, a.interference, a.wander
    KINDS_ONLY = tuple(filter(None, a.kinds.split(",")))
    FASTLOW = a.fastlow
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
