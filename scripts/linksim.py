"""Gear-shifter phase D: two ARQ sessions (the real data2g.arq code) on the
real mode ladder, with the modem replaced by a link-abstraction model:

- Channel: one continuous two-path Watterson process for the whole run
  (Gaussian Doppler, as hfchannel), an SNR (optionally a schedule: QSB, a
  band closing). Per band, per-carrier SNR over time: the transmitted
  power split over the band's carriers, noise in 50 Hz per carrier.
- Sync: a burst's preamble succeeds with a logistic in its effective SNR,
  placed so AWGN fails 1% at the band's measured sync floor
  (runs/sync_floor.csv) with the slope measured there.
- Codewords: codewords are spread over the burst, so each sees the burst's
  effective MI (mean capacity over its frames and carriers, clip noise
  included); P(decode) is the link abstraction's curve for the submode.
  A resend combined with stored soft bits decodes on combined_mi (IR on
  fresh parity, Chase on repeats; fitted to the real PHY).
- Measurements handed to the receiving policy: what the real receiver
  would report (predictor inputs): the truth plus a real receiver's error
  on a similar burst (runs/predictor_data_v5.csv).

    uv run python scripts/linksim.py validate              # abstraction PHY vs the ladder thresholds
    uv run python scripts/linksim.py run --policy fixed:qpsk-r1/2:8 --chan mpd --snr 5
"""

import argparse
import csv
import json
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np

from data2g import codes, modem
from data2g.arq import frames as F
from data2g.arq import predictor as P
from data2g.arq import session as S
from data2g.config import BANDS, FRAME_SAMPLES, FS, LEADIN_SAMPLES, SUBMODES, clip_consts
from data2g.waveform import ofdm

ROOT = Path(__file__).parent.parent
PRESETS = {"awgn": (0.0, 0.0), "mpg": (0.1, 0.5), "mpp": (1.0, 2.0), "mpd": (2.0, 4.0)}
SYNC_SLOPE_DB = 0.6  # logistic scale of sync failure in SNR (w awgn: 11% at -9, 1% at -7.5)
# The real acquisition loses more to fading than instantaneous SNR shows.
# Added to the AWGN floor, in Doppler (0, 0.1, 1, 2 Hz): measured floors
# (runs/sync_floor.csv) minus this model's emergent ones without it
# (2026-09-24; e.g. w mpd: -2.0 emergent vs 4.25 measured; refit to the
# 1% floors, not the 50% points, the same day; refit 2026-09-25 to the
# +-150 Hz search's floors, within 0.25 dB).
SYNC_PENALTY_DB = {"w": (0.0, -0.5, 1.25, 4.75), "n10": (0.0, 0.5, 0.25, 4.0), "w48": (0.0, 0.25, 0.25, 4.0)}
FRAME_S = FRAME_SAMPLES / FS
PTT_S, DECODE_S = 0.1, 0.25


def sync_floors() -> dict:
    out = {}
    for r in csv.DictReader(open(ROOT / "runs/sync_floor.csv")):
        if r["channel"] == "awgn":
            out[r["band"]] = float(r["sync_threshold_db"])
    return out


# --- channel ----------------------------------------------------------------------

class Channel:
    def __init__(self, doppler, delay_ms, snr_db, seed=0, duration=4000.0, schedule=None):
        self.doppler, self.delay = doppler, delay_ms * 1e-3
        self.snr_db, self.schedule = snr_db, schedule
        rng = np.random.default_rng(seed)
        if doppler:
            self.rate = max(64 * doppler, 8.0)
            n = int(duration * self.rate) + 2
            f = np.fft.fftfreq(n, 1 / self.rate)
            shape = np.exp(-(f ** 2) / (4 * (doppler / 2) ** 2))
            norm = np.sqrt(2 * np.mean(shape ** 2))
            self.g = [np.fft.ifft(np.fft.fft(rng.normal(size=n) + 1j * rng.normal(size=n)) * shape) / norm
                      for _ in range(2)]

    def snr_at(self, t):
        return self.snr_db + (self.schedule(t) if self.schedule else 0.0)

    def carrier_snr(self, band: str, t: np.ndarray) -> np.ndarray:
        """(len(t), nc) linear per-carrier SNR."""
        bb = ofdm.band(band).bb
        nc = len(bb)
        base = 10 ** (np.array([self.snr_at(x) for x in t]) / 10) * (2500 / 50) / nc
        if not self.doppler:
            return np.repeat(base[:, None], nc, axis=1)
        i = t * self.rate
        i0 = np.clip(i.astype(int), 0, len(self.g[0]) - 2)
        a = i - i0
        g1 = self.g[0][i0] * (1 - a) + self.g[0][i0 + 1] * a
        g2 = self.g[1][i0] * (1 - a) + self.g[1][i0 + 1] * a
        h = (g1[:, None] + g2[:, None] * np.exp(-2j * np.pi * bb[None, :] * self.delay)) / np.sqrt(2)
        return base[:, None] * np.abs(h) ** 2


# --- PHY model --------------------------------------------------------------------

def _ab():
    return P.abstraction()


def p_decode(submode: str, mi: float) -> float:
    c = _ab()[submode]
    return 1 / (1 + math.exp(-max(-40.0, min(40.0, c["slope"] * (mi - c["mi50"])))))


_OFFSETS = None


def mi_offset(band: str, doppler: float) -> float:
    """The receiver's estimation loss in MI, relative to the true channel's:
    the median per (band, preset) that makes this model fail 1% at the
    measured ladder thresholds (calibrate(): runs/linksim_offsets.csv),
    interpolated in Doppler between the presets (0, 0.1, 1, 2 Hz). Within a
    band and preset, submodes still spread by about +-0.05 MI (~1 dB): the
    model's accuracy limit, fine for comparing policies on it."""
    global _OFFSETS
    if _OFFSETS is None:
        from collections import defaultdict
        g = defaultdict(list)
        path = ROOT / "runs/linksim_offsets.csv"
        if path.exists():
            for r in csv.DictReader(open(path)):
                g[(r["band"], r["channel"])].append(float(r["offset"]))
        _OFFSETS = {k: float(np.median(v)) for k, v in g.items()}
    xs = [PRESETS[c][0] for c in ("awgn", "mpg", "mpp", "mpd")]
    ys = [_OFFSETS.get((band, c), 0.0) for c in ("awgn", "mpg", "mpp", "mpd")]
    return float(np.interp(doppler, xs, ys))


def burst_mi(ch: Channel, spec, t_data: float, n_frames: int, calibrated=True) -> float:
    """Effective MI (bits per coded bit) of a burst's data frames, as the
    receiver would estimate it (the true channel's, less mi_offset)."""
    t = t_data + (np.arange(n_frames) + 0.5) * FRAME_S
    snr = ch.carrier_snr(spec.band, t)
    ratio = clip_consts(spec.band, spec.headroom)[2]  # clip noise relative to signal
    snr = 1 / (1 / snr + ratio)
    mi = float(np.mean(P.capacity(10 * np.log10(snr), spec.constellation)))
    return max(0.0, mi - mi_offset(spec.band, ch.doppler)) if calibrated else mi


IR_LOSS = 0.06  # combining's loss against the model below, relative (scripts/ir_study.py, runs/ir_study.csv)


def combined_mi(submode: str, txs) -> float:
    """Effective MI of one codeword received several times, [(MI, rv)]:
    the transmissions' SNR spread over the mother-code buffer they cover
    (c codeword lengths, codes.rv_positions), c * C(sum SNR / c). One
    covered length is Chase (SNRs add); fresh parity each time is IR (MIs
    add). Fitted on the real PHY (scripts/ir_study.py): IR and Chase
    within ~0.05 in P on 16/64-QAM; low-rate QPSK and polar well below
    their threshold stay optimistic (as this sim's single-shot curve and
    sync are there)."""
    spec = SUBMODES[submode]
    grid, tables = P._capacity()
    tab = tables[P.const_family(spec.constellation)]
    snr = sum(10 ** (np.interp(np.clip(m, tab[0] + 1e-6, tab[-1] - 1e-6), tab, grid) / 10) for m, _ in txs)
    c = min(codes.buffer_len(spec), (max(rv for _, rv in txs) + 1) * spec.coded_bits) / spec.coded_bits
    return min(1.0, c * float(np.interp(10 * np.log10(snr / c), grid, tab))) * (1 - IR_LOSS)


class SimRx:
    """RxBurst for link.Station.handle: per-slot Bernoulli on P(decode)."""

    def __init__(self, burst, mi, rng, store, stats):
        self.burst, self.submode, self.n_cw = burst, burst.submode, len(burst.slots)
        self.mi, self.rng, self.store, self.stats = mi, rng, store, stats
        self.draw = {}

    def decode(self, i, mask_id, rv, key):
        s = self.burst.slots[i]
        if mask_id != s.mask_id or (key is not None and rv != s.rv):
            if key is not None and mask_id[0] == s.mask_id[0]:
                self.stats["mismatch"] += 1
            return None
        mi = self.mi
        if key is not None and self.store.get(key):
            mi = combined_mi(self.submode, self.store[key] + [(self.mi, rv)])
        if i not in self.draw:
            self.draw[i] = self.rng.random()
        if self.draw[i] < p_decode(self.submode, mi):
            return s.payload
        if key is not None:
            self.store.setdefault(key, []).append((self.mi, rv))
        return None

    def forget(self, key):
        self.store.pop(key, None)


class SimPhy:
    def __init__(self, ch: Channel, rng):
        self.ch, self.rng, self.floors = ch, rng, sync_floors()

    def send(self, burst, t0):
        """-> (on_air_end, (header time, submode, n_cw) or None, make_rx(store,
        stats, rng) -> RxBurst, measured dict)."""
        spec = SUBMODES[burst.submode]
        n = len(burst.slots)
        end = t0 + modem.burst_seconds(spec, n)
        sb = BANDS[spec.sync_band]
        t_pre = t0 + LEADIN_SAMPLES / FS + np.linspace(0, sb.preamble_samples / FS, 6)
        pre = self.ch.carrier_snr(spec.sync_band, t_pre)
        # effective preamble SNR (2500 Hz reference): invert the mean QPSK capacity
        cap = float(np.mean(P.capacity(10 * np.log10(pre), "gray-qam4")))
        grid, tables = P._capacity()
        eff_c = float(np.interp(cap, tables["gray-qam4"], grid))
        eff = eff_c + 10 * np.log10(sb.nc * 50 / 2500)
        floor = self.floors[spec.sync_band] + float(np.interp(self.ch.doppler, [0.0, 0.1, 1.0, 2.0],
                                                               SYNC_PENALTY_DB[spec.sync_band]))
        p_fail = 1 / (1 + math.exp((eff - (floor - 4.6 * SYNC_SLOPE_DB)) / SYNC_SLOPE_DB))
        if self.rng.random() < p_fail:
            return end, None, None, None
        t_header = t0 + (LEADIN_SAMPLES + sb.preamble_samples) / FS
        t_data = t_header + modem.header_samples(spec.sync_band) / FS
        n_frames = n * spec.frames_per_cw
        mi = burst_mi(self.ch, spec, t_data, n_frames)
        meas = self.measure(spec, t_data, n_frames)
        meas["frames"] = n_frames
        return (end, (t_header + 0.05, burst.submode, n), lambda store, stats, rng: SimRx(burst, mi, rng, store, stats),
                meas)

    def measure(self, spec, t_data, n_frames) -> dict:
        """What the receiver would report on this burst (predictor inputs):
        the truth plus a real receiver's error on a similar burst (same
        band, nearby Doppler and MI; residuals())."""
        t = t_data + (np.arange(n_frames) + 0.5) * FRAME_S
        snr = self.ch.carrier_snr(spec.band, t)
        nc = BANDS[spec.band].nc
        snr_db = 10 * np.log10(np.mean(snr) * nc * 50 / 2500)
        truth = [float(np.mean(P.capacity(10 * np.log10(snr), c))) for c in P.CONSTS]
        d, tr, res = residuals()[spec.band]
        score = np.abs(np.log((self.ch.doppler + 0.05) / (d + 0.05))) / 0.5 + np.abs(truth[0] - tr) / 0.1
        r = res[self.rng.choice(np.argsort(score)[:20])]
        wide = nc >= 24
        out = dict(snr_est=snr_db + float(np.clip(r[4], -10, 10)),
                   spread_est=max(0.03, self.ch.doppler + float(r[5])),
                   delay_est_ms=(self.ch.delay * 1e3 if (wide and snr_db > 2 and self.ch.delay >= 1e-3) else 0.0))
        # thermal-only effective MI (predictor inputs exclude the sender's clip noise)
        for c, v, e in zip(P.CONSTS, truth, r[:4]):
            out[f"mi_{c}"] = float(np.clip(v + e, 0, 1))
        out["headroom"] = spec.headroom
        return out


RESIDUALS = ROOT / "runs/predictor_data_v5.csv"  # scripts/predictor_data.py (truth1_* columns)


@lru_cache(maxsize=None)
def residuals() -> dict:
    """Per band: (doppler, true QPSK MI, [MI errors per constellation, SNR
    error dB, spread error Hz]) of the real receiver's measurements."""
    out = {}
    rows = list(csv.DictReader(open(RESIDUALS)))
    for b in BANDS:
        rs = [r for r in rows if r["band1"] == b]
        out[b] = (np.array([float(r["doppler"]) for r in rs]),
                  np.array([float(r["truth1_gray-qam4"]) for r in rs]),
                  np.array([[float(r[f"mi_{c}"]) - float(r[f"truth1_{c}"]) for c in P.CONSTS]
                            + [float(r["snr_est"]) - float(r["snr"]), float(r["spread_est"]) - float(r["doppler"])]
                            for r in rs]))
    return out


# --- policies -----------------------------------------------------------------------

def rv_cycle(submode):
    return codes.rv_cycle(SUBMODES[submode])


class FixedPolicy:
    """One data mode, fixed burst size (the baseline and the validation)."""

    def __init__(self, mode, n_cw, connect="qpsk-r1/5"):
        self.mode, self.n_cw, self.connect = mode, n_cw, connect

    def choose(self, station, escalation):
        return self.mode, self.n_cw

    def payload_bytes(self, m):
        return codes.payload_bytes(SUBMODES[m])

    def rv_cycle(self, m):
        return rv_cycle(m)

    def connect_mode(self, cap):
        return self.connect

    def airtime(self, m, n_cw):
        return modem.burst_seconds(SUBMODES[m], n_cw)

    def observe(self, measured, submode, now):
        pass


def _ladder(cap=2):
    """Allowed submodes, slowest to fastest (payload bits per second of a 6 s burst)."""
    from data2g.arq import policy as G

    def rate(s):
        n = max(1, sum(1 for k in range(1, 65) if modem.burst_seconds(s, k) <= 6.0))
        return (n - 1) * codes.payload_bytes(s) / modem.burst_seconds(s, n)
    return sorted((s.name for s in G.allowed(cap)), key=lambda m: rate(SUBMODES[m]))


class ArdopPolicy(FixedPolicy):
    """Sender-side counter, ARDOP style: up one step after 2 fully
    delivered bursts, down one on a burst with codewords still missing or
    on escalation (timeouts). Ignores the receiver's measurements."""

    def __init__(self, start=None, size_s=6.0):
        self.size_s = size_s
        # one bandwidth, as ARDOP negotiates one per session: w48's 15 modes
        # (a 42-rung ladder across all bands took the whole run to climb)
        self.ladder = [m for m in _ladder() if SUBMODES[m].band == "w48"]
        self.level = self.ladder.index(start) if start else len(self.ladder) // 2
        self.good = 0
        super().__init__(self.ladder[self.level], 0)

    def choose(self, station, escalation):
        if escalation:
            self.level, self.good = max(0, self.level - 1), 0
        elif station.tx.ack is not None:
            if station.tx.missing():
                self.level, self.good = max(0, self.level - 1), 0
            else:
                self.good += 1
                if self.good >= 2:
                    self.level, self.good = min(len(self.ladder) - 1, self.level + 1), 0
        spec = SUBMODES[self.ladder[self.level]]
        n = max(2, sum(1 for k in range(1, 65) if modem.burst_seconds(spec, k) <= self.size_s))
        return spec.name, n


class SnrPolicy(FixedPolicy):
    """Receiver-recommended by SNR alone, VARA style: the fastest mode whose
    AWGN ladder threshold + margin is under the measured SNR."""

    def __init__(self, margin=3.0, hint=2):
        from data2g.arq import policy as G
        self.G, self.margin, self.hint = G, margin, hint
        self.thr = {}
        names = {f"{'' if s.band == 'w' else s.band + '-'}{s.code}-{s.constellation}-f{s.frames_per_cw}-k{s.k}@h{s.headroom:g}": s.name
                 for s in SUBMODES.values()}
        for r in csv.DictReader(open(ROOT / "runs/ladder_final.csv")):
            if r["channel"] == "awgn" and r["name"] in names:
                self.thr[names[r["name"]]] = float(r["threshold_db"])
        self.ladder = _ladder()
        self.snr = None
        super().__init__(self.ladder[0], 0)

    def observe(self, measured, submode, now):
        self.snr = measured["snr_est"]

    def recommend(self, station):
        fb = self.G.encode(self.G.FALLBACK[station.cap])
        if self.snr is None:
            return fb, 1, fb
        ok = [m for m in self.ladder if self.thr.get(m, 99) + self.margin <= self.snr]
        return self.G.encode(ok[-1] if ok else self.ladder[0]), self.hint, fb

    def choose(self, station, escalation):
        return self.G.GearShifter.choose(self.G.GearShifter(), station, escalation)


def make_policy(spec: str):
    kind, *args = spec.split(":")
    if kind == "fixed":
        return FixedPolicy(args[0], int(args[1]))
    if kind == "ardop":
        return ArdopPolicy(size_s=float(args[0]) if args else 6.0)
    if kind == "snr":
        return SnrPolicy(float(args[0]) if args else 3.0, int(args[1]) if len(args) > 1 else 2)
    if kind in ("shift", "chat"):
        from data2g.arq.policy import GearShifter
        g = GearShifter()
        g.chat_on = kind == "chat"  # the harness sets CHAT ON on the sessions
        return g
    raise ValueError(spec)


def goodput(policy: str, chan: str, snr: float, seed: int, secs=300.0) -> float:
    """Bytes per second delivered one way (caller to callee) in `secs`."""
    return score("bulk", policy, chan, snr, seed, secs)


def score(workload: str, policy: str, chan: str, snr: float, seed: int, secs=None):
    """bulk: bytes/s over `secs` (300). winlink: seconds to finish the session
    (the horizon, 1800, if it does not). chat: mean delivery latency of the
    delivered messages, with each undelivered one counted at the horizon."""
    dop, dly = PRESETS[chan]
    horizon = secs or (300.0 if workload == "bulk" else 1800.0)
    ch = Channel(dop, dly, snr, seed=seed, duration=horizon + 60)
    steps = WORKLOADS[workload](random.Random(seed + 7))
    res = run(make_policy(policy), make_policy(policy), ch, steps, seed=seed, horizon=horizon)
    if workload == "bulk":
        return res["delivered"] / horizon
    if workload == "winlink":
        return res["t"] if res["complete"] else horizon
    lat = res["latency"] + [horizon] * (len(steps) - len(res["latency"]))
    return float(np.mean(lat))


# --- driver ---------------------------------------------------------------------------

# --- workloads ------------------------------------------------------------------------
#
# A workload is a list of application steps (writer "a" caller / "b" callee,
# bytes, the step it waits for or None, think time after that). A step is
# written when the step it waits for has been delivered (plus think time),
# and is delivered when its last byte reaches the other side.

APP_S = 0.2  # an application's reaction time to a delivered step


def bulk(rng, n=2_000_000):
    return [("a", n, None, 0.0)]


def winlink(rng):
    """B2F-like: the caller proposes its messages (~100 B), the callee
    answers (~20 B), the messages follow; then the callee proposes its own,
    the caller answers, those follow. Sizes log-normal, median 3 KB."""
    steps = []
    def msgs(k):
        return [int(min(40000, max(200, rng.lognormvariate(np.log(3000), 1.0)))) for _ in range(k)]
    out_a, out_b = msgs(rng.randint(1, 4)), msgs(rng.randint(0, 3))
    steps.append(("a", 60 + 40 * len(out_a), None, 0.0))  # proposals
    steps.append(("b", 20, 0, APP_S))  # accept
    last = 1
    for m in out_a:
        steps.append(("a", m, last, APP_S))
        last = len(steps) - 1
    steps.append(("b", 60 + 40 * len(out_b), last, APP_S))  # the callee's turn: its proposals
    steps.append(("a", 20, len(steps) - 1, APP_S))
    last = len(steps) - 1
    for m in out_b:
        steps.append(("b", m, last, APP_S))
        last = len(steps) - 1
    steps.append(("a", 20, last, APP_S))  # FF / FQ
    return steps


def chat(rng, n=10):
    """Short messages alternating direction, 5-30 s of thinking between."""
    steps = []
    for i in range(n):
        steps.append(("a" if i % 2 == 0 else "b", rng.randint(50, 200), i - 1 if i else None,
                      rng.uniform(5, 30) if i else 0.0))
    return steps


WORKLOADS = {"bulk": bulk, "winlink": winlink, "chat": chat}


def run(pol_a, pol_b, ch: Channel, steps, seed=0, horizon=1800.0, phy=None):
    """-> dict: per-step write and delivery times, delivered bytes, stats.
    `phy`: what carries bursts (default SimPhy on `ch`; scripts/phy_session.py
    has the real modem)."""
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    phy = phy or SimPhy(ch, nprng)
    a = S.Session("W1AW", pol_a, rng=random.Random(seed + 1))
    b = S.Session("K2XYZ", pol_b, rng=random.Random(seed + 2))
    who = {"a": a, "b": b}
    a.set_chat(getattr(pol_a, "chat_on", False))
    b.set_chat(getattr(pol_b, "chat_on", False))
    b.listen()
    a.connect("K2XYZ", 2, 0.0)
    stores = {id(a): {}, id(b): {}}
    stats = {"mismatch": 0, "bursts": 0, "modes": {}, "airtime": 0.0, "time": {}, "lost_sync": 0, "timeouts": 0}
    last_sent = {}
    ot = a._on_timeout

    def on_timeout(now, _ot=ot):
        stats["timeouts"] += 1
        return _ot(now)
    a._on_timeout = on_timeout
    written = {"a": 0, "b": 0}  # bytes written per direction
    delivered = {"a": 0, "b": 0}  # ... and delivered
    sent_data = {"a": bytearray(), "b": bytearray()}
    got = {"a": bytearray(), "b": bytearray()}
    step_end = []  # per step: cumulative bytes in its direction once written
    t_write = [None] * len(steps)
    t_done = [None] * len(steps)
    events, air = [], []
    t = 0.0
    while t < horizon:
        # the application: write every step whose prerequisite is delivered
        for i, (w, n, after, think) in enumerate(steps):
            if t_write[i] is None and (after is None or (t_done[after] is not None and t >= t_done[after] + think)):
                if who[w].state not in (S.CONNECTED, S.CONNECTING, S.LISTEN):
                    continue
                data = rng.randbytes(n)
                who[w].write(data)
                sent_data[w] += data
                written[w] += n
                t_write[i] = t
        ends = {"a": 0, "b": 0}
        for i, (w, n, _, _) in enumerate(steps):
            if t_write[i] is not None:
                ends[w] += n
                if t_done[i] is None and delivered[w] >= ends[w]:
                    t_done[i] = t
        if all(x is not None for x in t_done):
            break
        for me, other in ((a, b), (b, a)):
            burst = me.poll(t)
            if burst is None:
                continue
            start = t + PTT_S
            busy = max([e for s_, e, w_ in air[-4:] if w_ is other and e > start], default=None)
            if busy is not None:
                start = busy + 0.05
            end, hdr, make_rx, meas = phy.send(burst, start)
            air.append((start, end, me))
            stats["bursts"] += 1
            core = F.Core.unpack(burst.slots[0].payload)
            kind = ("repeat" if last_sent.get(id(me)) is burst else "poll" if core.ftype == F.PROBE
                    else "session" if core.ftype == F.SESSION else "data" if len(burst.slots) > core.n_ctl else "ctl")
            last_sent[id(me)] = burst
            key = ("a_" if me is a else "b_") + kind
            stats["time"][key] = stats["time"].get(key, 0.0) + (end - start)
            if hdr is None:
                stats["lost_sync"] += 1
            stats["airtime"] += end - start
            stats["modes"][burst.submode] = stats["modes"].get(burst.submode, 0) + 1
            events.append((end, "txend", me, burst, None))
            if hdr is not None:
                events.append((hdr[0], "header", other, burst, hdr))
                events.append((end + DECODE_S, "rx", other, burst, (make_rx, meas, hdr[1])))
        got["a"] += b.read()  # what the caller wrote, delivered at the callee
        got["b"] += a.read()
        for w in "ab":
            assert bytes(got[w]) == bytes(sent_data[w][:len(got[w])])
            delivered[w] = len(got[w])
        if a.state == S.CLOSED and b.state == S.CLOSED:
            break
        nxt = [e[0] for e in events] + [x for x in (a.next_event(), b.next_event()) if x is not None]
        pend = [t_done[after] + think for (_, _, after, think), tw in zip(steps, t_write)
                if tw is None and after is not None and t_done[after] is not None]
        nxt += [p for p in pend if p > t] + ([t + 0.05] if any(p <= t for p in pend) else [])
        if not nxt:
            break
        t = max(t, min(nxt))
        due = sorted([e for e in events if e[0] <= t], key=lambda e: e[0])
        events = [e for e in events if e[0] > t]
        for when, kind, whom, burst, extra in due:
            if kind == "txend":
                whom.on_tx_end(burst, when)
            elif kind == "header":
                whom.on_header(extra[1], extra[2], when)
            else:
                make_rx, meas, submode = extra
                whom.policy.observe(meas, submode, when)
                whom.on_rx(make_rx(stores[id(whom)], stats, rng), when)
    lat = [d - w for w, d in zip(t_write, t_done) if w is not None and d is not None]
    return dict(t=t, complete=all(x is not None for x in t_done), latency=lat,
                delivered=delivered["a"] + delivered["b"], a_state=a.state, b_state=b.state,
                reason=(a.close_reason, b.close_reason), **stats)


# --- validation ---------------------------------------------------------------------

def validate(bursts=300, frames=16):
    """Codeword failure of every submode at its ladder code thresholds under
    this PHY model: ~1% means the abstraction + true-channel MI reproduce
    the measured thresholds."""
    thr = {}
    for r in csv.DictReader(open(ROOT / "runs/ladder_final.csv")):
        thr[(r["name"], r["channel"])] = float(r["threshold_db"])
    rng = np.random.default_rng(0)
    for s in SUBMODES.values():
        name = f"{'' if s.band == 'w' else s.band + '-'}{s.code}-{s.constellation}-f{s.frames_per_cw}-k{s.k}@h{s.headroom:g}"
        row = []
        for chan, (dop, dly) in PRESETS.items():
            t = thr.get((name, chan))
            if t is None or not np.isfinite(t):
                row.append(f"{chan} -")
                continue
            ch = Channel(dop, dly, t, seed=int(rng.integers(1 << 30)), duration=bursts * 5.0 + 10)
            n_f = max(frames, s.frames_per_cw)
            fails = [1 - p_decode(s.name, burst_mi(ch, s, 1.0 + i * 5.0, n_f)) for i in range(bursts)]
            row.append(f"{chan} {100 * np.mean(fails):5.2f}%")
        print(f"{s.name:18s} " + "  ".join(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    s2 = sub.add_parser("sweep2")
    s2.add_argument("--out", default="runs/linksim_sweep2.csv")
    s2.add_argument("--workloads", nargs="+", default=["bulk", "winlink", "chat"])
    s2.add_argument("--seeds", type=int, default=4)
    sw = sub.add_parser("sweep")
    sw.add_argument("--out", default="runs/linksim_sweep.csv")
    sw.add_argument("--seeds", type=int, default=3)
    c = sub.add_parser("calibrate")
    c.add_argument("--out", default="runs/linksim_offsets.csv")
    r = sub.add_parser("run")
    r.add_argument("--policy", default="fixed:qpsk-r1/2:8")
    r.add_argument("--chan", default="awgn")
    r.add_argument("--snr", type=float, default=10.0)
    r.add_argument("--bytes", type=int, default=20000)
    r.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "validate":
        validate()
        return
    if a.cmd == "sweep2":
        sweep2(a.out, tuple(a.workloads), a.seeds)
        return
    if a.cmd == "sweep":
        sweep(a.out, a.seeds)
        return
    if a.cmd == "calibrate":
        calibrate(out=a.out)
        return
    dop, dly = PRESETS[a.chan]
    ch = Channel(dop, dly, a.snr, seed=a.seed)
    res = run(make_policy(a.policy), make_policy(a.policy), ch, a.bytes, 0, seed=a.seed)
    print(json.dumps({k: v for k, v in res.items()}, default=str, indent=1))



def calibrate(bursts=300, frames=16, out=None):
    """Per submode and channel: the MI offset d with which true-channel MI
    minus d reproduces the measured 1% threshold (the abstraction was fit
    on the receiver's estimated MI, which carries estimation loss)."""
    thr = {}
    for r in csv.DictReader(open(ROOT / "runs/ladder_final.csv")):
        thr[(r["name"], r["channel"])] = float(r["threshold_db"])
    rng = np.random.default_rng(0)
    rows = []
    for s in SUBMODES.values():
        name = f"{'' if s.band == 'w' else s.band + '-'}{s.code}-{s.constellation}-f{s.frames_per_cw}-k{s.k}@h{s.headroom:g}"
        for chan, (dop, dly) in PRESETS.items():
            t = thr.get((name, chan))
            if t is None or not np.isfinite(t):
                continue
            ch = Channel(dop, dly, t, seed=int(rng.integers(1 << 30)), duration=bursts * 5.0 + 10)
            n_f = max(frames, s.frames_per_cw)
            mis = np.array([burst_mi(ch, s, 1.0 + i * 5.0, n_f, calibrated=False) for i in range(bursts)])
            lo, hi = -0.3, 0.3
            for _ in range(30):
                d = (lo + hi) / 2
                fail = np.mean([1 - p_decode(s.name, m - d) for m in mis])
                lo, hi = (d, hi) if fail < 0.01 else (lo, d)
            rows.append((s.name, s.band, P.const_family(s.constellation), chan, (lo + hi) / 2))
            print(f"{s.name:18s} {chan}  d = {(lo + hi) / 2:+.3f}", flush=True)
    if out:
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["submode", "band", "const", "channel", "offset"])
            w.writerows(rows)



def _gp(args):
    pol, chan, snr, seed = args
    return pol, chan, snr, seed, goodput(pol, chan, snr, seed)


def sweep(out, seeds=3, jobs=8):
    """Goodput of each policy and of every fixed mode (the hindsight oracle
    takes each scenario's best) over channels and SNRs."""
    from multiprocessing import Pool
    fixed = []
    for m in _ladder():
        s = SUBMODES[m]
        n = max(2, sum(1 for k in range(1, 65) if modem.burst_seconds(s, k) <= 6.0))
        fixed.append(f"fixed:{m}:{n}")
    pols = ["shift", "snr", "ardop"] + fixed
    jobs_ = [(p, c, snr, seed) for c in PRESETS for snr in range(-8, 25, 4) for seed in range(seeds) for p in pols]
    with Pool(jobs) as pool, open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "channel", "snr", "seed", "goodput"])
        for r in pool.imap_unordered(_gp, jobs_, chunksize=8):
            w.writerow(r)
    summarize(out)


def summarize(path):
    from collections import defaultdict
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        g[(r["policy"], r["channel"], int(r["snr"]))].append(float(r["goodput"]))
    mean = {k: float(np.mean(v)) for k, v in g.items()}
    chans = sorted({k[1] for k in mean}, key=list(PRESETS).index)
    snrs = sorted({k[2] for k in mean})
    print("goodput, bytes/s (mean over seeds); oracle = best fixed mode per cell")
    for c in chans:
        print(f"\n{c:5s} " + " ".join(f"{s:>7d}" for s in snrs))
        oracle = [max(v for (p, cc, ss), v in mean.items() if cc == c and ss == s and p.startswith("fixed")) for s in snrs]
        print("oracle " + " ".join(f"{v:7.1f}" for v in oracle))
        for p in ("shift", "snr", "ardop"):
            print(f"{p:6s} " + " ".join(f"{mean.get((p, c, s), float('nan')):7.1f}" for s in snrs))


SIZES_S = (1.0, 3.0, 6.0, 12.0)  # as policy.SIZE_S


def _score(args):
    w, pol, chan, snr, seed = args
    return w, pol, chan, snr, seed, score(w, pol, chan, snr, seed)


def sweep2(out, workloads=("bulk", "winlink", "chat"), seeds=4, jobs=8):
    """Every policy on every workload; the oracle is the best fixed (mode,
    size) per cell, the baselines are reported at their best size."""
    from multiprocessing import Pool
    fixed = []
    for m in _ladder():
        sp = SUBMODES[m]
        for size in SIZES_S:
            n = max(2, sum(1 for k in range(1, 65) if modem.burst_seconds(sp, k) <= size))
            fixed.append(f"fixed:{m}:{n}")
    fixed = sorted(set(fixed))
    pols = (["shift", "chat"] + [f"snr:3:{h}" for h in (1, 2, 3)] + [f"ardop:{sz:g}" for sz in (3, 6, 12)] + fixed)
    jobs_ = [(w, p, c, snr, sd) for w in workloads for c in PRESETS for snr in range(-4, 21, 4)
             for sd in range(seeds) for p in pols]
    with Pool(jobs) as pool, open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["workload", "policy", "channel", "snr", "seed", "score"])
        for i, r in enumerate(pool.imap_unordered(_score, jobs_, chunksize=16)):
            wr.writerow(r)
            if i % 5000 == 0:
                f.flush()
                print(i, "of", len(jobs_), flush=True)
    summarize2(out)


def summarize2(path):
    from collections import defaultdict
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        g[(r["workload"], r["policy"], r["channel"], int(r["snr"]))].append(float(r["score"]))
    mean = {k: float(np.mean(v)) for k, v in g.items()}
    units = {"bulk": "bytes/s (higher is better)", "winlink": "s to finish the session (lower is better)",
             "chat": "s mean message latency (lower is better)"}
    for w in ("bulk", "winlink", "chat"):
        if not any(k[0] == w for k in mean):
            continue
        better = max if w == "bulk" else min
        print(f"\n=== {w}: {units[w]}; oracle = best fixed (mode, size) per cell, baselines at their best size")
        for c in PRESETS:
            snrs = sorted({k[3] for k in mean if k[0] == w and k[2] == c})
            print(f"{c:6s} " + " ".join(f"{x:>7d}" for x in snrs))
            rows = {"oracle": [better(v for (ww, p, cc, ss), v in mean.items()
                                      if ww == w and cc == c and ss == x and p.startswith("fixed")) for x in snrs],
                    "shift": [mean.get((w, "shift", c, x), float("nan")) for x in snrs],
                    "chat": [mean.get((w, "chat", c, x), float("nan")) for x in snrs],  # CHAT ON (shorter polls too)
                    "snr": [better(v for (ww, p, cc, ss), v in mean.items() if ww == w and cc == c and ss == x
                                   and p.startswith("snr")) for x in snrs],
                    "ardop": [better(v for (ww, p, cc, ss), v in mean.items() if ww == w and cc == c and ss == x
                                     and p.startswith("ardop")) for x in snrs]}
            for name, vals in rows.items():
                print(f"{name:6s} " + " ".join(f"{v:7.1f}" for v in vals))


if __name__ == "__main__":
    main()
