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

import json
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .. import codes, modem
from ..config import BANDS, SUBMODES
from . import frames as F
from . import predictor as P

SIZE_S = (1.0, 3.0, 6.0, 12.0)  # size-hint airtime classes (s): 0 shortest
TURN_S = 1.3  # a turnaround's dead time (decode + PTT + audio), for goodput
TIMEOUT_S = 1.0 + 1.0 + 1.5  # t_turn + reply start margin + a poll: what a lost turn costs before recovery
PREV_MAX_S = 30.0  # history older than the predictor's training range is dropped (scripts/predictor_data.py)
BIAS_STEP, BIAS_MAX = 1.0, 3.0  # online correction: logit step per unit of surprise, and its bound
CHAT_BYTES = F.CHAT_LINE_BYTES  # a chat line, the least the latency objective plans for (more: T_BUFFER)
CAP_HZ = {0: 500, 1: 1200, 2: 2400}
WIDTH_HZ = {"n4": 200, "n10": 500, "w": 1200, "w48": 2400}
BY_CODE = {code: band for band, code in F.BANDS_CODE.items()}
FALLBACK = {0: "n10-ack-4f", 1: "ack-4f", 2: "ack-4f"}  # robust per cap
CONNECT = {0: "n10-qpsk-r1/3", 1: "qpsk-r1/5", 2: "qpsk-r1/5"}  # >= 28 B payload, one control codeword


def encode(submode: str) -> int:
    s = SUBMODES[submode]
    return F.BANDS_CODE[s.sync_band] << 4 | s.index


def decode(rec: int) -> str | None:
    from ..modem import BY_INDEX

    band = BY_CODE.get(rec >> 4)
    s = BY_INDEX.get((band, rec & 15)) if band else None
    return s.name if s else None


@lru_cache(maxsize=None)
def _sync():
    return json.load(open(P.DATA / "sync_floors.json"))


def p_sync(band: str, snr_db: float, doppler: float) -> float:
    """P(a burst's preamble and header get through) in its sync band, from
    the measured 1% floors (codes_data/sync_floors.json) interpolated in
    Doppler, as a logistic in SNR: steep on AWGN (0.6 dB), shallow on
    fading (3 dB, outage-like), where it is 1% at the floor."""
    d = _sync()
    names = ["awgn", "mpg", "mpp", "mpd"]
    xs = [d["doppler"][c] for c in names]
    floor = float(np.interp(doppler, xs, [d["floors_db"][band][c] for c in names]))
    scale = 0.6 if doppler < 0.05 else 3.0
    z = (snr_db - floor) / scale + 4.6  # logit(0.99) at the floor
    return float(1 / (1 + np.exp(-np.clip(z, -40, 40))))


def _family(s) -> tuple:
    """The online correction's key: what the predictor errs on together
    (a bias learned on QPSK would otherwise inflate 64-QAM's P)."""
    return s.band, P.const_family(s.constellation)


STRIKE_HOLD = 4  # decisions a struck mode or family sits out, doubling per consecutive strike
STRIKE_HOLD_MAX = 64
STRIKES_TO_HOLD = {"rx": 2, "tx": 1}  # the receiver's single failure can be a fade; the sender's is two


def _strike(table: dict, key, now: int):
    """One more failure for `key`: after enough in a row it sits out a hold."""
    c = table.get(key, [0, 0])
    c[0] += 1
    need = STRIKES_TO_HOLD["tx" if isinstance(key, str) else "rx"]
    if c[0] >= need:
        c[1] = now + min(STRIKE_HOLD_MAX, STRIKE_HOLD * 2 ** (c[0] - need))
    table[key] = c


def _held(table: dict, key, now: int) -> bool:
    c = table.get(key)
    return c is not None and now < c[1]


def allowed(cap: int) -> list:
    return [s for s in SUBMODES.values() if WIDTH_HZ[s.band] <= CAP_HZ[cap]]


@dataclass
class GearShifter:
    gap_s: float = 2.5  # expected time from the end of the peer's burst to its next
    measured: dict | None = None  # the peer's last burst, as measured
    measured_band: str = "w"
    measured_at: float = 0.0
    prev: tuple | None = None  # (measured, band, time) of the peer burst before the last
    bias: dict = field(default_factory=dict)  # online correction: logit shift per submode (codewords)
    bias_burst: dict = field(default_factory=dict)  # ... and of P(burst usable), outcome model only
    predicted: dict = field(default_factory=dict)  # submode -> the P I last predicted for it
    rx_strikes: dict = field(default_factory=dict)  # family -> [count, hold until recommendation #]
    tx_strikes: dict = field(default_factory=dict)  # submode -> [count, hold until choice #]
    _n_rec: int = 0
    _n_choose: int = 0
    _last: tuple | None = None  # (data mode followed, my tx base then)
    peer_had_data: bool = True
    log: list = field(default_factory=list)

    # -- link.Policy ------------------------------------------------------------------

    def choose(self, station, escalation: int) -> tuple[str, int]:
        cap = station.cap
        # sender-side strikes: a burst in the recommended mode, and its
        # repeat, both unanswered (escalation now) while polls get through
        # is a mode the peer can't hear, whatever it predicts; without this
        # the two stations jab forever (the audio loopback: 64-QAM clipped
        # to death, polls answered, 64-QAM again)
        self._n_choose += 1
        last, self._last = self._last, None
        if last is not None:
            m, base = last
            if escalation:
                _strike(self.tx_strikes, m, self._n_choose)
            elif station.tx.base > base:
                self.tx_strikes.pop(m, None)
        rec = station.peer_recommend if station.tx.pending() else station.peer_reply_recommend
        if escalation or rec is None:
            return FALLBACK[cap], 2
        mode = decode(rec)
        if mode is not None and _held(self.tx_strikes, mode, self._n_choose):
            # the robust data mode, not the peer's reply mode (chosen for
            # control-only bursts: the loopback sent 64 six-byte codewords)
            mode = CONNECT[cap] if not _held(self.tx_strikes, CONNECT[cap], self._n_choose) else None
        if mode is None or SUBMODES[mode] not in allowed(cap):
            return FALLBACK[cap], 2
        if station.tx.pending():
            self._last = (mode, station.tx.base)
        spec = SUBMODES[mode]
        target = SIZE_S[station.peer_size_hint]
        n = 1
        while n < 64 and modem.burst_seconds(spec, n + 1) <= target:
            n += 1
        return mode, n

    def next_capacity(self, station) -> int:
        """Payload bytes the next data burst will carry if it follows the
        peer's recommendation (no side effects: the host's BUFFER report)."""
        mode = decode(station.peer_recommend) if station.peer_recommend is not None else None
        if mode is None or SUBMODES[mode] not in allowed(station.cap):
            mode = FALLBACK[station.cap]
        spec = SUBMODES[mode]
        n = 1
        while n < 64 and modem.burst_seconds(spec, n + 1) <= SIZE_S[station.peer_size_hint]:
            n += 1
        return (n - 1) * codes.payload_bytes(spec)  # one codeword or more is control

    def payload_bytes(self, m):
        return codes.payload_bytes(SUBMODES[m])

    def rv_cycle(self, m):
        return codes.rv_cycle(SUBMODES[m])

    def connect_mode(self, cap):
        return CONNECT[cap]

    def airtime(self, m, n_cw):
        return modem.burst_seconds(SUBMODES[m], n_cw)

    # -- receiver side ------------------------------------------------------------------

    def observe(self, measured: dict, submode: str, now: float):
        """The receiver's measurements of a peer burst (predictor inputs)."""
        if self.measured is not None:
            self.prev = (self.measured, self.measured_band, self.measured_at)
        self.measured, self.measured_band, self.measured_at = measured, SUBMODES[submode].band, now

    def outcome(self, submode: str, decoded: int, sent: int):
        """Codeword outcomes of a peer burst against what I predicted for its
        mode: one burst tells little (a fade takes a whole burst), so the
        bias moves a step per burst. It learns what one burst's features
        cannot tell (held-out: slow fading over-predicted by 0.1-0.2, a
        steady channel under-predicted as much). Kept per submode."""
        k = _family(SUBMODES[submode])
        # receiver-side strikes: headers heard in a family, controls failed
        if decoded == 0:
            _strike(self.rx_strikes, k, self._n_rec)
        else:
            self.rx_strikes.pop(k, None)
        p = self.predicted.get(submode)
        if p is None or sent == 0:
            return
        pb, p = p
        # the burst usable (its control decoded) or not; then its codewords
        self.bias_burst[submode] = float(np.clip(self.bias_burst.get(submode, 0.0) + BIAS_STEP * ((decoded > 0) - pb),
                                                 -BIAS_MAX, BIAS_MAX))
        if decoded == 0:
            return
        # per mode: a family-wide bias let qpsk-r1/5's steady successes lift
        # qpsk-r1/3 over the eligibility floor, where it decoded 6%
        self.bias[submode] = float(np.clip(self.bias.get(submode, 0.0) + BIAS_STEP * (decoded / sent - p), -BIAS_MAX, BIAS_MAX))

    def recommend(self, station) -> tuple[int, int, int]:
        """-> (data mode, size hint, reply mode) for the peer's next burst:
        the mode it should use if it sends data, and if it sends none."""
        if self.measured is None:
            return encode(FALLBACK[station.cap]), 1, encode(FALLBACK[station.cap])
        self._n_rec += 1
        cands = [s for s in allowed(station.cap) if not _held(self.rx_strikes, _family(s), self._n_rec)]
        m = self.measured
        memo = {}
        prev = None
        if self.prev is not None and self.measured_at - self.prev[2] <= PREV_MAX_S:
            prev = (self.prev[0], self.prev[1], self.measured_at - self.prev[2])

        def p(s, n_cw):
            """P(codeword decodes) over an n_cw burst of s from the MI
            predictor (the fallback without the outcome model): the window
            snapped to a power of two (one forward pass each)."""
            w = 2 ** int(np.clip(np.round(np.log2(n_cw * s.frames_per_cw)), 1, 6))
            if w not in memo:
                memo[w] = P.predict(m, self.measured_band, self.gap_s, cands, window=w, prev=prev)
            q = float(np.clip(memo[w][s.name], 1e-6, 1 - 1e-6))
            return float(1 / (1 + np.exp(-(np.log(q / (1 - q)) + self.bias.get(s.name, 0.0)))))

        # the outcome model (P from real decodes, scripts/train_outcome.py)
        # replaced the MI predictor's output corrections: at -4 dB AWGN on
        # the real modem, control losses 24% -> 3% of data bursts and 141 ->
        # 196 bps (scripts/loss_study.py); the MI predictor stays as fallback
        outcome = P.outcome_model() is not None
        omemo = {}

        def logit_shift(q, b):
            q = float(np.clip(q, 1e-6, 1 - 1e-6))
            return float(1 / (1 + np.exp(-(np.log(q / (1 - q)) + b))))

        def q_burst(s, n_cw):
            """P(a burst of s, n_cw long, is usable: synced, header, control)."""
            if not outcome:
                return ps[s.name] * p(s, n_cw)
            sec = round(modem.burst_seconds(s, n_cw), 2)
            if sec not in omemo:
                omemo[sec] = P.predict_outcome(m, self.measured_band, self.gap_s, sec, cands, prev)
            return logit_shift(omemo[sec][s.name][0], self.bias_burst.get(s.name, 0.0))

        def q_cw(s, n_cw):
            """P(one of its data codewords decodes | the burst is usable)."""
            if not outcome:
                return p(s, n_cw)
            q_burst(s, n_cw)
            return logit_shift(omemo[round(modem.burst_seconds(s, n_cw), 2)][s.name][1], self.bias.get(s.name, 0.0))

        # the band's SNR from the measured one: same power over another width
        # is the same SNR (2500 Hz reference); the outcome model has sync inside
        ps = {} if outcome else {s.name: p_sync(s.sync_band, m["snr_est"], m["spread_est"]) for s in cands}
        best, best_v = None, -1.0
        # my reply to the peer: the cheapest in expectation. A lost reply costs
        # a timeout and a poll round trip on top of itself (linksim: fragile
        # 0.6 s replies after long data bursts made timeouts the shifter's
        # biggest loss on fading).
        reply, reply_c, reply_p = None, float("inf"), 0.0
        for s in cands:
            t = modem.burst_seconds(s, 1)
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
        held = len(station.rx.buf) * codes.payload_bytes(SUBMODES[cur]) if cur else 0
        # CHAT ON: the least expected time to deliver what the peer has queued
        # (its T_BUFFER; a chat line at least). A file is then sent in full
        # bursts, a line in the fewest codewords (a VARA client with CHAT ON
        # sent a file 242 bytes per burst when this planned for 200 B always)
        chat = station.chat or station.peer_chat
        queued = max(CHAT_BYTES, getattr(station, "peer_queued", 0))
        if chat:
            best_v = -float("inf")
        for s in cands:
            pb = codes.payload_bytes(s)
            for hint, target in enumerate(SIZE_S):
                n = max(1, sum(1 for k in range(1, 65) if modem.burst_seconds(s, k) <= target))
                if chat:
                    # CHAT ON: the least expected time to get a message across,
                    # every codeword of it (sizes are caps: take the smallest that fits)
                    k = -(-queued // pb)
                    if n < k + 1 and hint < len(SIZE_S) - 1:
                        continue
                    n = min(n, k + 1)
                pn = q_cw(s, n)
                ok_ctl = q_burst(s, n)  # the burst survives sync and its control codeword
                tb = modem.burst_seconds(s, n)
                t = tb + 2 * TURN_S + reply_c + (1 - ok_ctl) * TIMEOUT_S
                if chat:
                    # every burst the message takes, each retried until it all arrives
                    ok_all = max(ok_ctl * pn ** (n - 1), 1e-3)
                    # ties (a line fits alike in several rates) go to the one
                    # carrying more, for whatever follows it
                    v = -t * -(-k // max(n - 1, 1)) / ok_all + 1e-6 * (n - 1) * pb
                else:
                    v = (ok_ctl * pn * (n - 1) * pb - (held if s.name != cur else 0)) / t
                if v > best_v:
                    best, best_v = (s.name, hint), v
                if chat:
                    break
        if best is None:
            best = (FALLBACK[station.cap], 1)
        else:
            s = SUBMODES[best[0]]
            nb = max(1, sum(1 for k in range(1, 65) if modem.burst_seconds(s, k) <= SIZE_S[best[1]]))
            self.predicted = {s.name: (q_burst(s, nb), q_cw(s, nb))}
            if reply:
                self.predicted[reply] = (q_burst(SUBMODES[reply], 1), q_cw(SUBMODES[reply], 1))
        reply = reply or FALLBACK[station.cap]
        self.log.append((best[0], best[1], reply))
        return encode(best[0]), best[1], encode(reply)
