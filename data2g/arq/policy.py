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
"""

import os
from dataclasses import dataclass, field

import numpy as np

from .. import codes, modem
from .. import cpm
from .modes import MODES, burst_seconds, ctl_payload_bytes, is_cpm, max_ctl, min_cw
from . import frames as F
from . import predictor as P

SIZE_S = (1.0, 3.0, 6.0, 12.0)  # size-hint airtime classes (s): 0 shortest
TURN_S = 1.3  # a turnaround's dead time (decode + PTT + audio), for goodput
TIMEOUT_S = 1.0 + 1.0 + 1.5  # t_turn + reply start margin + a poll: what a lost turn costs before recovery
PREV_MAX_S = 30.0  # history older than the predictor's training range is dropped (scripts/predictor_data.py)
BIAS_STEP, BIAS_MAX = 1.0, 3.0  # online correction: logit step per unit of surprise, and its bound
# The online correction bounded at 6, not 3: without LOGIT_OFFSETS v10 sent
# w48-qpsk-r1/3 in 78% of MPP 0 dB data bursts, 13% of its codewords
# decoding, for a whole session; at a predicted 0.99 a -3 bound still leaves
# 0.83. (That it learns from data codewords alone is outcome()'s `usable`.)
# On since v12; DATA2G_BIAS_FIX=0 (studies) is the old bound.
BIAS_FIX = os.environ.get("DATA2G_BIAS_FIX", "1") == "1"
if BIAS_FIX:
    BIAS_MAX = 6.0
DUP_BELOW = 0.9  # predicted P(burst usable) under which control is duplicated
CHAT_BYTES = F.CHAT_LINE_BYTES  # a chat line, the least the latency objective plans for (more: T_BUFFER)
CPM_SIZE_SCALE = 4.0  # SIZE_S for a CPM burst: 4-48 s (fsk8r50: 1-6 data codewords)
CAP_HZ = {0: 500, 1: 1200, 2: 2400}
WIDTH_HZ = {"n4": 200, "n10": 500, "w": 1200, "w48": 2400}
BY_CODE = {code: band for band, code in F.BANDS_CODE.items()}
# robust per cap. n10-ack-4f everywhere was tried (usable at -4 dB: MPP 95%
# vs ack-4f's 79%): the loss study got worse in every cell (MPP 0 dB 248 ->
# 187 bps), likely because polls are what the receiver measures, and a 500 Hz
# poll says little about the wide modes it recommends (2026-09-25)
FALLBACK = {0: "n10-ack-4f", 1: "ack-4f", 2: "ack-4f"}
CONNECT = {0: "n10-qpsk-r1/3", 1: "qpsk-r1/5", 2: "qpsk-r1/5"}  # >= 28 B payload, one control codeword
ROBUST_CONNECT = "fsk16r25-r1/2"  # session-frame retries: 500 Hz, within every cap; CONNECT goes compact (20 B)


CPM_CODE = 3  # the recommendation's band code for CPM modes (index: data2g.cpm.SPECS' order)


def encode(submode: str) -> int:
    s = MODES[submode]
    if is_cpm(s):
        return CPM_CODE << 4 | list(cpm.SPECS).index(submode)
    return F.BANDS_CODE[s.sync_band] << 4 | s.index


def decode(rec: int) -> str | None:
    from ..modem import BY_INDEX

    if rec >> 4 == CPM_CODE:
        names = list(cpm.SPECS)
        return names[rec & 15] if (rec & 15) < len(names) else None
    band = BY_CODE.get(rec >> 4)
    s = BY_INDEX.get((band, rec & 15)) if band else None
    return s.name if s else None


# DATA2G_DROP_MODES (studies): comma-separated modes nobody picks (a pruned
# ladder): not a candidate, not taken from a peer, not trained on
# (scripts/train_outcome.py). They still decode when received.
DROP = frozenset(filter(None, os.environ.get("DATA2G_DROP_MODES", "").split(",")))
assert DROP <= set(MODES), f"DATA2G_DROP_MODES: unknown {sorted(DROP - set(MODES))}"


def allowed(cap: int) -> list:
    return [s for s in MODES.values() if width_hz(s) <= CAP_HZ[cap] and s.name not in DROP]


def width_hz(s) -> float:
    return cpm.GRIDS[s.band].bandwidth if is_cpm(s) else WIDTH_HZ[s.band]


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
        seconds *= CPM_SIZE_SCALE
    n = 1
    while n < 64 and burst_seconds(spec, n + 1) <= seconds:
        n += 1
    n = max(n, min_cw(spec, data), ctl_slots(spec) + data)
    if is_cpm(spec):
        n = min(n + (dup and data), 1 + dup + cpm.MAX_DATA)
    return n


@dataclass
class GearShifter:
    gap_s: float = 2.5
    use_cpm: bool = True  # recommend CPM modes (only once the outcome model has learned them)
    # data modes need P(usable) x P(codeword) at least this: the KISS TNC
    # sets it (no resends there; a lost codeword is a lost AX.25 frame and a
    # retry), the ARQ doesn't (IR resends recover codewords)
    min_success: float = 0.0  # expected time from the end of the peer's burst to its next
    measured: dict | None = None  # the peer's last burst, as measured
    measured_band: str = "w"
    measured_at: float = 0.0
    prev: tuple | None = None  # (measured, band, time) of the peer burst before the last
    bias: dict = field(default_factory=dict)  # online correction: logit shift per submode (codewords)
    bias_burst: dict = field(default_factory=dict)  # ... and of P(burst usable), outcome model only
    want_dup: bool = False  # ask the peer to duplicate its next data burst's control (link: T_DUPCTL)
    predicted: dict = field(default_factory=dict)  # submode -> the P I last predicted for it
    peer_had_data: bool = True
    log: list = field(default_factory=list)

    # -- link.Policy ------------------------------------------------------------------

    def choose(self, station, escalation: int) -> tuple[str, int]:
        cap = station.cap
        # (strikes, a hold on modes that went unanswered, were removed: the
        # online bias routes around a mangled mode too, tests/test_engine.py,
        # and at -4 to 0 dB fading they held working modes: 12 seeds x 600 s,
        # 12-48% lower throughput with them)
        rec = station.peer_recommend if station.tx.pending() else station.peer_reply_recommend
        if escalation or rec is None:
            return FALLBACK[cap], 2
        mode = decode(rec)
        if mode is None or MODES[mode] not in allowed(cap):
            return FALLBACK[cap], 2
        return mode, slots_for(MODES[mode], SIZE_S[station.peer_size_hint], station.tx.pending(),
                               getattr(station, "peer_wants_dup", False))

    def next_capacity(self, station) -> int:
        """Payload bytes the next data burst will carry if it follows the
        peer's recommendation (no side effects: the host's BUFFER report)."""
        mode = decode(station.peer_recommend) if station.peer_recommend is not None else None
        if mode is None or MODES[mode] not in allowed(station.cap):
            mode = FALLBACK[station.cap]
        spec = MODES[mode]
        n = slots_for(spec, SIZE_S[station.peer_size_hint])
        return (n - ctl_slots(spec)) * codes.payload_bytes(spec)

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

    # -- receiver side ------------------------------------------------------------------

    def observe(self, measured: dict, submode: str, now: float):
        """The receiver's measurements of a peer burst (predictor inputs)."""
        if self.measured is not None:
            self.prev = (self.measured, self.measured_band, self.measured_at)
        self.measured, self.measured_band, self.measured_at = measured, MODES[submode].band, now

    def outcome(self, submode: str, decoded: int, sent: int, usable: bool | None = None):
        """Codeword outcomes of a peer burst against what I predicted for its
        mode: one burst tells little (a fade takes a whole burst), so the
        bias moves a step per burst. It learns what one burst's features
        cannot tell (held-out: slow fading over-predicted by 0.1-0.2, a
        steady channel under-predicted as much). Kept per submode.
        `usable`: its control decoded, with decoded/sent its data codewords
        alone (counted with them, the control made a 0/7 burst score 1/8);
        None (KISS: no control): any codeword decoded."""
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

    def recommend(self, station) -> tuple[int, int, int]:
        """-> (data mode, size hint, reply mode) for the peer's next burst:
        the mode it should use if it sends data, and if it sends none."""
        if self.measured is None:
            return encode(FALLBACK[station.cap]), 1, encode(FALLBACK[station.cap])
        cands = [s for s in allowed(station.cap) if (not is_cpm(s) or (self.use_cpm and P.outcome_knows(s.name)))]
        m = self.measured
        prev = None
        if self.prev is not None and self.measured_at - self.prev[2] <= PREV_MAX_S:
            prev = (self.prev[0], self.prev[1], self.measured_at - self.prev[2])

        # the outcome model (P from real decodes, scripts/train_outcome.py)
        # replaced the MI predictor's output corrections: at -4 dB AWGN on
        # the real modem, control losses 24% -> 3% of data bursts and 141 ->
        # 196 bps (scripts/loss_study.py). It ships with the package.
        if P.outcome_model() is None:
            raise RuntimeError(f"no outcome model: {P.DATA / 'outcome_predictor.npz'} (scripts/train_outcome.py)")
        omemo = {}

        def logit_shift(q, b):
            q = float(np.clip(q, 1e-6, 1 - 1e-6))
            return float(1 / (1 + np.exp(-(np.log(q / (1 - q)) + b))))

        def q_burst(s, n_cw):
            """P(a burst of s, n_cw long, is usable: synced, header, control)."""
            sec = round(burst_seconds(s, n_cw), 2)
            if sec not in omemo:
                omemo[sec] = P.predict_outcome(m, self.measured_band, self.gap_s, sec, cands, prev)
            return logit_shift(omemo[sec][s.name][0], self.bias_burst.get(s.name, 0.0))

        def q_cw(s, n_cw):
            """P(one of its data codewords decodes | the burst is usable)."""
            q_burst(s, n_cw)
            return logit_shift(omemo[round(burst_seconds(s, n_cw), 2)][s.name][1], self.bias.get(s.name, 0.0))

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
        held = len(station.rx.buf) * codes.payload_bytes(MODES[cur]) if cur else 0
        # CHAT ON: the least expected time to deliver what the peer has queued
        # (its T_BUFFER; a chat line at least). A file is then sent in full
        # bursts, a line in the fewest codewords (a VARA client with CHAT ON
        # sent a file 242 bytes per burst when this planned for 200 B always)
        chat = station.chat or station.peer_chat
        queued = max(CHAT_BYTES, getattr(station, "peer_queued", 0))
        if chat:
            best_v = -float("inf")
        for s in cands:
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
                if chat:
                    # every burst the message takes, each retried until it all arrives
                    ok_all = max(ok_ctl * pn ** (n - c), 1e-3)
                    # ties (a line fits alike in several rates) go to the one
                    # carrying more, for whatever follows it
                    v = -t * -(-k // max(n - c, 1)) / ok_all + 1e-6 * (n - c) * pb
                else:
                    v = (ok_ctl * pn * (n - c) * pb - (held if s.name != cur else 0)) / t
                if v > best_v:
                    best, best_v = (s.name, hint), v
                if chat:
                    break
        if best is None:
            best = (FALLBACK[station.cap], 1)
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
        reply = reply or FALLBACK[station.cap]
        self.log.append((best[0], best[1], reply))
        return encode(best[0]), best[1], encode(reply)
