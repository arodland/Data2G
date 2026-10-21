"""The gear shifter (docs/arq.md §8): one per station.

As a receiver, from its measurements of the peer's last burst it predicts
P(a codeword decodes) for every submode inside the session's bandwidth cap
(data2g.arq.predictor), and recommends the mode and burst length that
maximize the expected goodput of the peer's next burst: delivered payload
over the whole turn's time. As a sender it follows the peer's
recommendation, except when escalating (its own timeouts, or the peer's
repeats and polls saying its replies are lost): then the robust mode.

The recommendation rides the core control word: 6 bits of submode (sync
band, index), 2 bits of burst length (an airtime class, SIZE_S).

GearShifter is also the link.Policy the engine runs, its plumbing methods
forwarding to modes and codes; native splits the two (GearShifter, which
reads a StationView, and GearPolicy). The env-driven study toggles are
grouped below the tuned constants; native has their defaults.
"""

import json
import os
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .. import codes, modem
from .. import cpm
from ..config import max_codewords
# the bandwidth and recommendation-code helpers are modes'; named here too,
# where the shifter and its callers (and tests/native_subs/gear.py) find them
from .modes import (CAP_HZ, MODES, burst_seconds, ctl_payload_bytes, decode, encode, is_cpm, max_ctl, min_cw,
                    width_hz)
from . import frames as F
from . import link as L
from . import predictor as P
from .session import LINK_LOST_S

SIZE_S = (1.0, 3.0, 6.0, 12.0)  # size-hint airtime classes (s): 0 shortest
TURN_S = 1.3  # a turnaround's dead time (decode + PTT + audio), for goodput
TIMEOUT_S = 1.0 + 1.0 + 1.5  # t_turn + reply start margin + a poll: what a lost turn costs before recovery
# the predictor's training range for the previous burst's age (scripts/predictor_data.py).
# Older history is used as this old, not dropped: with no history the model
# gave w48-64l-r1/2 0.34-0.55 of usable bursts at MPG -6 (0.07 with it),
# and escalation's long gaps sent 12 s 64-ary bursts that all failed
PREV_MAX_S = 30.0
BIAS_STEP = 1.0  # online correction: logit step per unit of surprise (its bound: BIAS_MAX, below)
DUP_BELOW = 0.9  # predicted P(burst usable) under which control is duplicated
CPM_SIZE_SCALE = 4.0  # SIZE_S for a CPM burst: 4-48 s (fsk8r50: 1-6 data codewords)
# ... but at most this long: a missed header loses the whole burst and no soft
# bits (MPP -8: two 30 s fsk32r62 bursts missed outright; a 48 s one then
# helped run out the 90 s link-lost clock)
CPM_MAX_S = 24.0
# a lost data burst spends the session's link-lost clock (LINK_LOST_S);
# recovering takes polls of about T_RECOVER_S each, and a session that runs
# out costs LOST_LINK_COST_S of expected turn time (reconnect, redo)
T_RECOVER_S = TIMEOUT_S + 2 * 5.0  # a timeout, a robust poll and its reply
LOST_LINK_COST_S = 300.0
REPLY_HOLD_MARGIN_S = 0.5  # the reply's start past my burst's end (0.4-0.6 s measured) and decode lag
# The data ladder: after LADDER_AFTER bursts in a row lost in the data mode I
# recommended, data may only go in modes LADDER_STEP_DB more robust than it on
# every channel (MODE_THRESHOLDS); each further loss steps down from the mode
# that failed, each usable data burst climbs a step. Climbed off it, the next
# data burst lost re-enters at once (on air, AG7EW 2026-10-06: the climbs out
# went straight to modes that failed). On air
# (recordings/20261002-232711) every fsk16r25 poll decoded and 0 of 12 data
# bursts did, predicted 0.74-0.93: the model had never seen CPM polls measured
# on a bad channel, and the per-mode bias hopped to unpenalised neighbours.
# ponytail: thresholds from one ladder study, not learned; the retrain on this
# branch's sessions should make the ladder rare
# The CPM floor (a stopgap): when the median snr_est of the peer's last
# CPM_FLOOR_HIST bursts (snr), or the energy inputs' mean (energy), is under
# X dB, data goes in CPM modes only. On air (KC2G-AG7EW, 2026-10-06, snr_est
# -2..-9 on fading) every OFDM data mode the model rated 0.5-0.8 decoded 0 of
# 24 while the CPM polls got through 75%+. On by default at snr:-4: on n10pe
# gated with n10qf (runs/cpm_floor_pe.sh, 13 cells x 12 seeds) MPP -10/-8/-6
# +12.5/+7.2/+3.7%, no cell worse beyond its SE (MPG 0 -2.8% on 2 seeds, MPD
# -8 -2.0%); snr:-3 cost MPG -4 7.5%. (kind, X, S): "snr" also wants the
# median spread_est of those bursts at least S Hz (fast fading: on slow
# fading a fade holds the median down, and OFDM recovers).
CPM_FLOOR_DEFAULT = ("snr", -4.0, 0.0)
CPM_FLOOR_HIST = 3
LADDER_AFTER = 2
LADDER_STEP_DB = 3.0
LINK_HIST = 8  # the link history's window: peer bursts I expected (predictor.N_LINK)
ENERGY_HIST = 4  # the energy inputs' window: the peer's last bursts (predictor.N_ENERGY)
# per mode, its 10% codeword failure SNR on awgn, mpg, mpp, mpd (codes_data/mode_thresholds.json)
MODE_THRESHOLDS = {m: tuple(v) for m, v in json.load(open(P.DATA / "mode_thresholds.json"))["modes"].items()}
# robust per cap. n10-ack-4f everywhere was tried (usable at -4 dB: MPP 95%
# vs ack-4f's 79%): the loss study got worse in every cell (MPP 0 dB 248 ->
# 187 bps), likely because polls are what the receiver measures, and a 500 Hz
# poll says little about the wide modes it recommends (2026-09-25)
FALLBACK = {0: "n10-ack-4f", 1: "ack-4f", 2: "ack-4f"}
CONNECT = {0: "n10-qpsk-r1/3", 1: "qpsk-r1/5", 2: "qpsk-r1/5"}  # >= 28 B payload, one control codeword
# escalation 2's mode: better than ack-4f on awgn and mpg, worse on mpp
# and mpd, so it alternates with the reply mode (1, 3)
ALT_POLL = "n4-ack-8f"
ROBUST_ESCALATION = 4  # from here on ROBUST_CONNECT (link caps escalation at 4)
ROBUST_CONNECT = "fsk16r25-r1/2"  # session-frame retries: 500 Hz, within every cap; CONNECT goes compact (20 B)

# --- study toggles (Python only; native has the defaults) --------------------
# Set any and tests/native_subs/gear.py's GEAR_STUDY_ENV keeps the shifter in Python.

# The online correction bounded at 6, not 3: without LOGIT_OFFSETS v10 sent
# w48-qpsk-r1/3 in 78% of MPP 0 dB data bursts, 13% of its codewords
# decoding, for a whole session; at a predicted 0.99 a -3 bound still leaves
# 0.83. (That it learns from data codewords alone is outcome()'s `usable`.)
# On since v12; DATA2G_BIAS_FIX=0 is the old bound.
BIAS_FIX = os.environ.get("DATA2G_BIAS_FIX", "1") == "1"
# A burst too short or weak for a Doppler estimate reports
# equalizer.DEFAULT_SPREAD_HZ (2.0, what MPD measures), so on AWGN 20-50% of
# w-band readings looked like fast fading and every outcome model blended the
# two (P(usable) low on AWGN, high on fading). Such a reading takes the peer's
# last measured spread instead (none yet: as is); the installed model is
# trained on it. DATA2G_SPREAD_CARRY=0 (studies): off.
SPREAD_CARRY = os.environ.get("DATA2G_SPREAD_CARRY", "1") == "1"
# DATA2G_COND_OFFSETS="spread_min:snr_max:mode:logit;..." (studies): P(usable)
# logit offsets that apply only while the peer's median spread_est is at least
# spread_min and snr_est under snr_max (a model's error confined to a regime,
# e.g. QPSK overrated at the CPM floor's edge on fast fading).
COND_OFFSETS = [(float(a), float(b), m, float(o)) for a, b, m, o in
                (e.split(":") for e in os.environ.get("DATA2G_COND_OFFSETS", "").split(";") if e)]
UNMEASURED_SPREAD_HZ = 2.0  # equalizer.DEFAULT_SPREAD_HZ (arq doesn't import the modem)
BIAS_MAX = 6.0 if BIAS_FIX else 3.0
# The noise rule (recommend(): candidates whose band is noisier in the
# receiver's profile than the measured one predicted at a lower SNR), its
# tail weight: on by default (runs/noise_rule_study.sh: carriers +54..+209%,
# edge +8..+37%, clean and wandering-floor cells unchanged).
# DATA2G_NOISE_RULE: a weight, or '' / 'off' for none.
_NR = os.environ.get("DATA2G_NOISE_RULE")
NOISE_RULE = 1.0 if _NR is None else (None if _NR in ("", "off") else float(_NR))


def _env_cpm_floor():
    """DATA2G_CPM_FLOOR: unset, CPM_FLOOR_DEFAULT; "snr:X[:S]" or
    "energy:X[:S]" (S default 0); '' turns the floor off (None)."""
    v = os.environ.get("DATA2G_CPM_FLOOR")
    if v is None:
        return CPM_FLOOR_DEFAULT
    if not v:
        return None
    f = v.split(":")
    return f[0], float(f[1]), float(f[2]) if len(f) > 2 else 0.0


CPM_FLOOR = _env_cpm_floor()
# DATA2G_DROP_MODES: comma-separated modes nobody picks (a pruned ladder):
# not a candidate, not taken from a peer, not trained on
# (scripts/train_outcome.py). They still decode when received.
DROP = frozenset(filter(None, os.environ.get("DATA2G_DROP_MODES", "").split(",")))
assert DROP <= set(MODES), f"DATA2G_DROP_MODES: unknown {sorted(DROP - set(MODES))}"


def allowed(cap: int) -> list:
    """The modes the shifter may pick within cap `cap` (modes.allowed less DROP)."""
    return [s for s in MODES.values() if width_hz(s) <= CAP_HZ[cap] and s.name not in DROP]


CTL_BYTES = 12  # a data burst's typical control: core 4, new 3, rv 3, a flag or two


def ctl_slots(spec) -> int:
    """Codewords a data burst's control takes in `spec`: one mostly, but 3 in
    a 4-byte reply mode (the shifter, counting one, recommended ack-4f for
    data at MPP -4 dB: 20+ turns of control-only bursts, no data)."""
    return min(max_ctl(spec), -(-CTL_BYTES // ctl_payload_bytes(spec)))


def slots_for(spec, seconds: float, data: bool = True, dup: bool = False) -> int:
    """Slots (control included) a burst of `spec` gets within `seconds`:
    at least min_cw; a CPM burst's duplicated control is on top; at most what
    its header can announce. CPM size classes are CPM_SIZE_SCALE times
    longer (a CPM data codeword is 3-10 s)."""
    if is_cpm(spec):
        seconds = min(seconds * CPM_SIZE_SCALE, CPM_MAX_S)
    n, lim = 1, 64 if is_cpm(spec) else max_codewords(spec.sync_band)
    while n < lim and burst_seconds(spec, n + 1) <= seconds:
        n += 1
    n = max(n, min_cw(spec, data), ctl_slots(spec) + data)
    if is_cpm(spec):
        n = min(n + (dup and data), 1 + dup + cpm.MAX_DATA)
    return n


class StationView(NamedTuple):
    """What the shifter reads of its link.Station (native: arq::StationView).
    KISS, with no station, passes one of its own."""
    cap: int = 2
    pending: bool = True  # tx.pending()
    peer_recommend: int | None = None
    peer_reply_recommend: int | None = None
    peer_size_hint: int = 1
    peer_wants_dup: bool = False
    chat: bool = False  # station.chat or station.peer_chat
    peer_queued: int = 0
    held: int = 0  # codewords held beyond the cumulative ACK: len(rx.buf)
    misses: int = 0
    esc_floor: int = 0
    timeouts: int = 0  # stats["timeouts"] (link_features: Python only)


def view(station) -> StationView:
    """A link.Station as the shifter reads it (a StationView as it is). An
    attribute it lacks takes the field's default, as in the binding's
    station(): tests hand the shifter stubs with only what one method reads."""
    if isinstance(station, StationView):
        return station
    try:  # a whole station (once per burst built: keep it cheap)
        return StationView(station.cap, station.tx.pending(), station.peer_recommend, station.peer_reply_recommend,
                           station.peer_size_hint, station.peer_wants_dup, bool(station.chat or station.peer_chat),
                           station.peer_queued, len(station.rx.buf), station.misses, station.esc_floor,
                           station.stats.get("timeouts", 0))
    except AttributeError:
        pass

    def g(name, default):
        return getattr(station, name, default)

    tx, rx = g("tx", None), g("rx", None)
    return StationView(cap=g("cap", 2), pending=tx.pending() if tx is not None else True,
                       peer_recommend=g("peer_recommend", None), peer_reply_recommend=g("peer_reply_recommend", None),
                       peer_size_hint=g("peer_size_hint", 1), peer_wants_dup=g("peer_wants_dup", False),
                       chat=bool(g("chat", False) or g("peer_chat", False)), peer_queued=g("peer_queued", 0),
                       held=len(rx.buf) if rx is not None else 0, misses=g("misses", 0), esc_floor=g("esc_floor", 0),
                       timeouts=g("stats", {}).get("timeouts", 0))


@dataclass
class GearShifter(L.Policy):
    gap_s: float = 2.5
    use_cpm: bool = True  # recommend CPM modes (only once the outcome model has learned them)
    # data modes need P(usable) x P(codeword) at least this: the KISS TNC
    # sets it (no resends there; a lost codeword is a lost AX.25 frame and a
    # retry), the ARQ doesn't (IR resends recover codewords)
    min_success: float = 0.0  # expected time from the end of the peer's burst to its next
    measured: dict | None = None  # the peer's last burst, as measured
    measured_band: str = "w"
    measured_at: float = 0.0
    heard: str | None = None  # the submode of the peer's last burst
    data_lost: int = 0  # bursts lost in a row in the data mode I recommended (LADDER_AFTER)
    ceiling: tuple | None = None  # the data ladder: per channel, the highest threshold data may have
    ladder_top: tuple | None = None  # the thresholds of the mode whose losses started it: climbed back there, it's off
    ladder_left: bool = False  # just climbed off the ladder: one data burst lost puts it back
    prev: tuple | None = None  # (measured, band, time) of the peer burst before the last
    bias: dict = field(default_factory=dict)  # online correction: logit shift per submode (codewords)
    bias_burst: dict = field(default_factory=dict)  # ... and of P(burst usable), outcome model only
    turns: list = field(default_factory=list)  # the last LINK_HIST expected peer bursts: "ok" | "lost" | "miss"
    timeouts_seen: int = 0  # station.stats["timeouts"] already counted as misses
    link_now: list | None = None  # link_features() at the last recommendation
    energies: list = field(default_factory=list)  # the peer's last ENERGY_HIST bursts' energy SNR, dB
    spreads: list = field(default_factory=list)  # the peer's last bursts' spread_est (predictor.GATE)
    last_spread: float | None = None  # the peer's last measured (not defaulted) spread_est (SPREAD_CARRY)
    snrs: list = field(default_factory=list)  # the peer's last CPM_FLOOR_HIST bursts' snr_est (CPM_FLOOR)
    snrs_long: list = field(default_factory=list)  # the same, the last GATE_HIST (the gate's steadiness)
    want_dup: bool = False  # ask the peer to duplicate its next data burst's control (link: T_DUPCTL)
    predicted: dict = field(default_factory=dict)  # submode -> the P I last predicted for it
    peer_had_data: bool = True
    log: list = field(default_factory=list)

    # -- link.Policy plumbing (native: GearPolicy): forwards to modes and codes -------

    def payload_bytes(self, m):
        return codes.payload_bytes(MODES[m])

    def ctl_payload_bytes(self, m):
        return ctl_payload_bytes(MODES[m])

    def max_ctl(self, m):
        return max_ctl(MODES[m])

    def rv_cycle(self, m):
        return codes.rv_cycle(MODES[m])

    def connect_mode(self, cap, tries: int = 0):
        """Session frames: the cap's connect mode first, then ROBUST_CONNECT
        (MPP -4 dB: qpsk-r1/5 bursts 38% usable, n4-qpsk-r1/3 91%; 9 of 12
        loss-study sessions there never connected in qpsk-r1/5)."""
        return CONNECT[cap] if tries == 0 else ROBUST_CONNECT

    def airtime(self, m, n_cw, dup=False):
        """`dup`: a CPM burst's control twice (the second copy is a short
        control codeword, not a data one: an x10 burst is 30.4 s, not 32.6)."""
        return burst_seconds(MODES[m], n_cw, dup)

    def mode_name(self, rec: int) -> str:
        return decode(rec) or f"?{rec}"

    # -- as a sender (native: GearShifter): my next burst, from the peer's advice -------

    def choose(self, station, escalation: int) -> tuple[str, int]:
        st = view(station)
        cap = st.cap
        # (strikes, a hold on modes that went unanswered, were removed: the
        # online bias routes around a mangled mode too, tests/test_engine.py,
        # and at -4 to 0 dB fading they held working modes: 12 seeds x 600 s,
        # 12-48% lower throughput with them)
        rec = st.peer_recommend if st.pending else st.peer_reply_recommend
        # escalation 1 and 3: the peer's reply mode; 2: ALT_POLL; 4 on, or
        # answering the peer's robust poll: control only in the connect
        # retry's mode (ack-4f polls went unanswered until the link was lost)
        # the floor there too (the last drop took it): my control-only bursts
        # as well, not only polls (MPP -8: the peer heard 4 of my 11
        # n4-ack-8f acks and 13 of 17 robust polls; its data waited on them)
        robust_floor = st.esc_floor >= ROBUST_ESCALATION and not st.pending
        if escalation >= ROBUST_ESCALATION or (escalation and self.heard == ROBUST_CONNECT) or robust_floor:
            return ROBUST_CONNECT, 1
        if escalation == 2:
            return ALT_POLL, 2
        if escalation:
            reply = decode(st.peer_reply_recommend) if st.peer_reply_recommend is not None else None
            return (reply if reply and MODES[reply] in allowed(cap) else FALLBACK[cap]), 2
        if rec is None:
            return FALLBACK[cap], 2
        mode = decode(rec)
        if mode is None or MODES[mode] not in allowed(cap):
            return FALLBACK[cap], 2
        return mode, slots_for(MODES[mode], SIZE_S[st.peer_size_hint], st.pending, st.peer_wants_dup)

    def next_capacity(self, station) -> int:
        """Payload bytes the next data burst will carry if it follows the
        peer's recommendation (no side effects: the host's BUFFER report)."""
        st = view(station)
        mode = decode(st.peer_recommend) if st.peer_recommend is not None else None
        if mode is None or MODES[mode] not in allowed(st.cap):
            mode = FALLBACK[st.cap]
        spec = MODES[mode]
        n = slots_for(spec, SIZE_S[st.peer_size_hint])
        return (n - ctl_slots(spec)) * codes.payload_bytes(spec)

    def reply_hold(self, station, burst) -> float:
        """Seconds past t_turn to wait for a reply to `burst` before timing
        out: the longest control-only reply the peer may send (the reply
        mode I asked for; its ladder's if `burst` was a poll; the robust
        mode if `burst` was in it or my floor is there). At MPP -8 the
        master missed reply headers and polled into the replies 2 s in
        (8 times in 717 s). A data reply (up to CPM_MAX_S) is not covered."""
        st = view(station)
        modes = [self.log[-1][2] if self.log else FALLBACK[st.cap]]
        if st.misses:
            modes += [FALLBACK[st.cap], ALT_POLL]
        if burst.submode == ROBUST_CONNECT or st.esc_floor >= ROBUST_ESCALATION:
            modes.append(ROBUST_CONNECT)
        return max(burst_seconds(MODES[m], ctl_slots(MODES[m])) for m in modes) + REPLY_HOLD_MARGIN_S

    # -- as a receiver: the peer's next burst, from my measurements of its last -------

    def observe(self, measured: dict, submode: str, now: float):
        """The receiver's measurements of a peer burst (predictor inputs)."""
        s = measured.get("spread_est")
        if s is not None and s != UNMEASURED_SPREAD_HZ:
            self.last_spread = float(s)
        elif SPREAD_CARRY and s is not None and self.last_spread is not None:
            measured = dict(measured, spread_est=self.last_spread)
        if self.measured is not None:
            self.prev = (self.measured, self.measured_band, self.measured_at)
        self.measured, self.measured_band, self.measured_at = measured, MODES[submode].band, now
        self.heard = submode
        self.spreads = (self.spreads + [float(measured.get("spread_est", 0.0))])[-P.GATE_HIST:]
        self.snrs = (self.snrs + [float(measured["snr_est"])])[-CPM_FLOOR_HIST:]
        self.snrs_long = (self.snrs_long + [float(measured["snr_est"])])[-P.GATE_HIST:]

    def cpm_floor(self) -> bool:
        """CPM_FLOOR holds: data in CPM modes only."""
        if CPM_FLOOR is None:
            return False
        kind, x, spread = CPM_FLOOR
        if spread and (not self.spreads or float(np.median(self.spreads[-CPM_FLOOR_HIST:])) < spread):
            return False
        if kind == "snr":
            return bool(self.snrs) and float(np.median(self.snrs)) < x
        e = self.energy_features()
        return e is not None and e[0] < x

    def gate(self):
        """The gate's model (predictor.GATE) or the fast region's
        (predictor.FAST) if its conditions hold, else None (the installed one)."""
        if not self.spreads or self.measured is None:
            return None
        med, snr = float(np.median(self.spreads)), self.measured["snr_est"]
        if P.GATE is not None and med < P.GATE[1] and snr < P.GATE[2] and (
                len(P.GATE) < 4 or (len(self.snrs_long) == P.GATE_HIST and float(np.std(self.snrs_long)) < P.GATE[3])):
            return P.outcome_model(P.GATE[0])
        if P.FAST is not None and med >= P.FAST[1] and P.FAST[2] <= snr < P.FAST[3]:
            return P.outcome_model(P.FAST[0])
        return None

    def outcome(self, submode: str, decoded: int, sent: int, usable: bool | None = None):
        """Codeword outcomes of a peer burst against what I predicted for its
        mode: one burst tells little (a fade takes a whole burst), so the
        bias moves a step per burst. It learns what one burst's features
        cannot tell (held-out: slow fading over-predicted by 0.1-0.2, a
        steady channel under-predicted as much). Kept per submode.
        `usable`: its control decoded, with decoded/sent its data codewords
        alone (counted with them, the control made a 0/7 burst score 1/8);
        None (KISS: no control): any codeword decoded."""
        if usable is not None:
            self.turns = (self.turns + ["ok" if usable else "lost"])[-LINK_HIST:]
        if usable and sent:
            self.data_lost, self.ladder_left = 0, False
            if self.ceiling is not None:  # climb a step
                self.ceiling = tuple(c + LADDER_STEP_DB for c in self.ceiling)
                if all(map(lambda c, t: c >= t, self.ceiling, self.ladder_top)):
                    self.ceiling = self.ladder_top = None
                    self.ladder_left = True
        elif usable is False and self.log and submode == self.log[-1][0]:
            self.data_lost += 1
            if self.data_lost >= LADDER_AFTER or self.ladder_left:  # step down from the mode that failed
                self.ladder_left = False
                down = tuple(t - LADDER_STEP_DB for t in MODE_THRESHOLDS[submode])
                if self.ceiling is None:
                    self.ceiling, self.ladder_top = down, MODE_THRESHOLDS[submode]
                else:
                    self.ceiling = tuple(map(min, down, self.ceiling))
        p = self.predicted.get(submode)
        if p is None or (sent == 0 and usable is None):
            return
        if usable is None:
            usable = decoded > 0
        pb, p = p
        self.bias_burst[submode] = float(np.clip(self.bias_burst.get(submode, 0.0) + BIAS_STEP * (usable - pb),
                                                 -BIAS_MAX, BIAS_MAX))
        if not usable or sent == 0:
            return
        # per mode: a family-wide bias let qpsk-r1/5's steady successes lift
        # qpsk-r1/3 over the eligibility floor, where it decoded 6%
        self.bias[submode] = float(np.clip(self.bias.get(submode, 0.0) + BIAS_STEP * (decoded / sent - p), -BIAS_MAX, BIAS_MAX))

    def observe_energy(self, snr_db: float, now: float):
        """A peer burst's in-band SNR from its power alone (heard or not)."""
        if snr_db is None:
            return
        self.energies = (self.energies + [snr_db])[-ENERGY_HIST:]

    def expected_reply(self) -> tuple[str, float] | None:
        """The burst I asked the peer for last, its mode and length on air (s):
        data in the mode and size I recommended if it had data, else a
        control-only reply (the engine's energy reading for a missed one)."""
        if not self.log:
            return None
        data, hint, reply = self.log[-1]
        if self.peer_had_data:
            spec = MODES[data]
            return data, burst_seconds(spec, slots_for(spec, SIZE_S[hint]))
        spec = MODES[reply]
        return reply, burst_seconds(spec, ctl_slots(spec))

    def energy_features(self) -> list | None:
        """The energy inputs (predictor.N_ENERGY); None before any."""
        if not self.energies:
            return None
        mean = 10 * np.log10(np.mean(10 ** (np.array(self.energies) / 10)))
        return [float(mean), len(self.energies) / ENERGY_HIST, 1.0]

    def hist_features(self) -> list | None:
        """The history inputs (predictor.N_HIST); None before any burst."""
        if not self.spreads:
            return None
        return [float(np.median(self.spreads)), float(np.median(self.snrs)), len(self.spreads) / P.GATE_HIST, 1.0]

    def link_features(self, station) -> list | None:
        """The link history as the outcome model reads it (predictor.N_LINK):
        my timeouts since the last call count as missed peer bursts. None
        before any expected burst."""
        t = view(station).timeouts
        if t > self.timeouts_seen:
            self.turns = (self.turns + ["miss"] * (t - self.timeouts_seen))[-LINK_HIST:]
        self.timeouts_seen = t
        if not self.turns:
            return None
        n = len(self.turns)
        return [self.turns.count("lost") / n, self.turns.count("miss") / n, n / LINK_HIST, 1.0]

    def recommend(self, station) -> tuple[int, int, int]:
        """-> (data mode, size hint, reply mode) for the peer's next burst:
        the mode it should use if it sends data, and if it sends none."""
        st = view(station)
        self.link_now = self.link_features(st)
        if self.measured is None:
            return encode(FALLBACK[st.cap]), 1, encode(FALLBACK[st.cap])
        # only modes the outcome model has learned (a mode added since is
        # left out until a model trained with it ships)
        cands = [s for s in allowed(st.cap) if P.outcome_knows(s.name) and (not is_cpm(s) or self.use_cpm)]
        m = dict(self.measured, link=self.link_now, energy=self.energy_features(), hist=self.hist_features())
        prev = None
        if self.prev is not None:
            prev = (self.prev[0], self.prev[1], min(self.measured_at - self.prev[2], PREV_MAX_S))

        # the outcome model (P from real decodes, scripts/train_outcome.py)
        # replaced the MI predictor's output corrections: at -4 dB AWGN on
        # the real modem, control losses 24% -> 3% of data bursts and 141 ->
        # 196 bps (scripts/loss_study.py). It ships with the package.
        gated = self.gate()
        cond = None
        if COND_OFFSETS and self.spreads:
            med = float(np.median(self.spreads))
            cond = {mo: o for a, b, mo, o in COND_OFFSETS if med >= a and self.measured["snr_est"] < b} or None
        if P.outcome_model() is None:
            raise RuntimeError(f"no outcome model: {P.DATA / 'outcome_predictor.npz'} (scripts/train_outcome.py)")
        omemo = {}
        # DATA2G_NOISE_RULE=w (studies): each candidate predicted as if its SNR
        # were lower by predictor.noise_shift_db (its band's noise in the
        # receiver's profile against the measured band's; w: the weight of
        # its often-loud moments). Unset: off.
        shift = {}
        if NOISE_RULE is not None:
            shift = {s.band: P.noise_shift_db(m.get("noise"), self.measured_band, s.band, NOISE_RULE) for s in cands}

        def predicted(s, n_cw):
            key = (round(burst_seconds(s, n_cw), 2), shift.get(s.band, 0.0))
            if key not in omemo:
                omemo[key] = P.predict_outcome(P.shifted(m, key[1]), self.measured_band, self.gap_s, key[0], cands, prev,
                                               gated, cond)
            return omemo[key][s.name]

        def logit_shift(q, b):
            q = float(np.clip(q, 1e-6, 1 - 1e-6))
            return float(1 / (1 + np.exp(-(np.log(q / (1 - q)) + b))))

        def q_burst(s, n_cw):
            """P(a burst of s, n_cw long, is usable: synced, header, control)."""
            return logit_shift(predicted(s, n_cw)[0], self.bias_burst.get(s.name, 0.0))

        def q_cw(s, n_cw):
            """P(one of its data codewords decodes | the burst is usable)."""
            return logit_shift(predicted(s, n_cw)[1], self.bias.get(s.name, 0.0))

        best, best_v = None, -1.0
        # my reply to the peer: the cheapest in expectation. A lost reply costs
        # a timeout and a poll round trip on top of itself (linksim: fragile
        # 0.6 s replies after long data bursts made timeouts the shifter's
        # biggest loss on fading).
        reply, reply_c, reply_p = None, float("inf"), 0.0
        for s in cands:
            t = burst_seconds(s, 1)
            ok = q_burst(s, 1)  # sync, then its control codeword (reciprocal channel)
            c = t + (1 - ok) * (TIMEOUT_S + t) / max(ok, 1e-3)  # geometric retries
            if c < reply_c:
                reply, reply_c, reply_p = s.name, c, ok
        # data: expected delivered payload per second of expected turn time,
        # losses priced as timeouts. Switching away from the mode the peer's
        # outstanding codewords use abandons them (link.py: a resend keeps its
        # submode), and what I hold beyond my cumulative ACK is sent again:
        # priced as payload not delivered this turn (linksim: mpg rate hopping
        # scored below every fixed mode it hopped between).
        cur = self.log[-1][0] if self.log else None
        held = st.held * codes.payload_bytes(MODES[cur]) if cur else 0
        # CHAT ON: the least expected time to deliver what the peer has queued
        # (its T_BUFFER; a chat line at least). A file is then sent in full
        # bursts, a line in the fewest codewords (a VARA client with CHAT ON
        # sent a file 242 bytes per burst when this planned for 200 B always)
        chat = st.chat
        queued = max(F.CHAT_LINE_BYTES, st.peer_queued)  # a chat line at least
        if chat:
            best_v = -float("inf")
        data_cands = cands
        if self.ceiling is not None:
            data_cands = [s for s in cands if all(map(lambda t, c: t <= c, MODE_THRESHOLDS[s.name], self.ceiling))]
            # none that robust: the most robust there is (the lowest worst-case threshold)
            data_cands = data_cands or [min(cands, key=lambda s: max(MODE_THRESHOLDS[s.name]))]
        if self.cpm_floor():
            data_cands = [s for s in data_cands if is_cpm(s)] or [s for s in cands if is_cpm(s)] or data_cands
        for s in data_cands:
            pb, c = codes.payload_bytes(s), ctl_slots(s)
            for hint, target in enumerate(SIZE_S):
                n = slots_for(s, target)
                if chat:
                    # CHAT ON: the least expected time to get a message across,
                    # every codeword of it (sizes are caps: take the smallest that fits)
                    k = -(-queued // pb)
                    if n < k + c and hint < len(SIZE_S) - 1:
                        continue
                    n = min(n, k + c)
                pn = q_cw(s, n)
                ok_ctl = q_burst(s, n)  # the burst survives sync and its control codeword
                if ok_ctl * pn < self.min_success:
                    continue
                # a CPM burst whose control is at risk goes with it twice (one
                # more slot, its short control codeword: choose(), slots_for)
                dup = is_cpm(s) and ok_ctl < DUP_BELOW
                tb = burst_seconds(s, n + dup, dup)
                t = tb + 2 * TURN_S + reply_c + (1 - ok_ctl) * TIMEOUT_S
                # lost, the burst leaves (LINK_LOST_S - tb) to recover in:
                # each try a poll and its reply, both at about my reply's P
                tries = max(0, int((LINK_LOST_S - tb - TIMEOUT_S) // T_RECOVER_S))
                t += (1 - ok_ctl) * (1 - reply_p ** 2) ** tries * LOST_LINK_COST_S
                if chat:
                    # every burst the message takes: one is retried until all
                    # of it arrives; past one, each codeword is resent (IR)
                    # until it does, so what a burst delivers is what counts
                    # (a client with CHAT ON sending a file: every codeword of
                    # an 18-codeword burst held 64-ary modes out at 20 dB,
                    # recordings/20261021-165549)
                    bursts = -(-k // max(n - c, 1))
                    ok = max(ok_ctl * (pn ** (n - c) if bursts == 1 else pn), 1e-3)
                    # ties (a line fits alike in several rates) go to the one
                    # carrying more, for whatever follows it
                    v = -t * bursts / ok + 1e-6 * (n - c) * pb
                else:
                    v = (ok_ctl * pn * (n - c) * pb - (held if s.name != cur else 0)) / t
                if v > best_v:
                    best, best_v = (s.name, hint), v
                if chat:
                    break
        if best is None:
            best = (FALLBACK[st.cap], 1)
        else:
            s = MODES[best[0]]
            nb = slots_for(s, SIZE_S[best[1]])
            self.predicted = {s.name: (q_burst(s, nb), q_cw(s, nb))}
            # its control at risk: ask for it twice (one codeword of the burst;
            # at MPP -4 dB 73% of data bursts lost their control while 172 of
            # their data codewords decoded, scripts/loss_study.py)
            self.want_dup = self.predicted[s.name][0] < DUP_BELOW
            if reply:
                self.predicted[reply] = (q_burst(MODES[reply], 1), q_cw(MODES[reply], 1))
        reply = reply or FALLBACK[st.cap]
        self.log.append((best[0], best[1], reply))
        return encode(best[0]), best[1], encode(reply)
