"""ARQ session (docs/arq.md §6, §7): connect, disconnect and timers
around link.Station. Driven by its caller's clock:

    s.on_header(submode, n_cw, now)   a burst header was decoded (its end is known)
    s.on_rx(rx, now)                  a whole burst arrived (link.RxBurst)
    s.poll(now) -> TxBurst | None     anything to send now?
    s.on_tx_end(burst, now)           that burst finished going out

Timers are absolute times; nothing here sleeps. The same object runs in
the simulator (scripts/linksim.py) and on the air.

Idle rule (docs/arq.md §6a): the caller keeps the link alive with a
poll after a random 15-30 s (KEEPALIVE_S). The callee may break idle with its new
data (a wake burst, WAKE_*), but owns no retry timer: any burst from the
caller cancels its wake, and the caller stays the only station that
retries on a timeout.

Native: arq/session.hpp.
"""

import logging
import random
from dataclasses import dataclass, field

from . import burst as B
from . import frames as F
from . import link as L

T_SESS = F.T_SESS  # the extension carrying a session frame
VERSION = 4  # 2: T_COMP (a v1 peer would deliver compressed codewords raw); 3: deflate primed with frames.ZDICT;
# 4: T_INLINE (a v3 peer ignores it: each inline codeword waits for its resend in a slot)
CONNECT_TRIES = 5
CONNECT_CTL_BYTES = 4 + 2 + F.Connect.BYTES  # a CONNECT's Control: core, T_SESS header, body
DISC_TRIES = 3
# past t_turn: a reply's header must have been heard by then: the peer's
# decode and PTT (~0.7 s), audio latency both ways, the reply's lead-in,
# preamble and header (0.4-0.6 s) and the receiver's search step (0.25 s)
# (the audio loopback timed out on replies already on air at 1.0). On air
# (recordings/20261002-224419) a reply's whole head was in at 1.9 s and its
# header reported at 2.2 s: 1.5 left no margin for a slower peer or radio
REPLY_START_S = 2.5
IDLE_CLOSE_S = 300.0
KEEPALIVE_S = (15.0, 30.0)  # idle poll: a random wait in this range after each exchange
KEEPALIVE_DOUBLING = False  # v1 (scripts/idle_study.py): the low end, doubling to the high end
LINK_LOST_S = 90.0  # 60 s of retries past the longest keepalive gap
# the callee's wake: after t_turn + REPLY_START_S + WAKE_GUARD_S of silence
# since its last burst (the caller's answer or timeout retry starts within
# t_turn + REPLY_START_S of it; the guard covers PTT, audio latency and
# header detection), plus a random backoff, 0-WAKE_JITTER_S doubling with
# each unanswered send (CSMA-like). A lost wake is repeated identically, at
# most WAKE_TRIES sends per idle period (CHAT_WAKE_TRIES in chat mode).
WAKE_GUARD_S = 1.0
WAKE_JITTER_S = 1.0
WAKE_TRIES = 2
CHAT_WAKE_TRIES = 6
CHAT_KEEPALIVE_S = KEEPALIVE_S  # CHAT ON: the callee's wakes carry its lines (v1 polled every 2-4 s)
REPEAT_MAX_S = 3.0  # repeat a timed-out burst identically only if it is this short
COALESCE_S = 0.1  # CHAT OFF: a short write to an idle link waits this long for more before it goes
# the callee, with nothing to send, holds its reply this long to a burst
# that brought its host data and ended the caller's (T_DRAINED): an
# interactive host's answer (a BBS prompt) then rides it instead of
# waiting ~5 s for a wake. A write ends the hold. Either station holds its
# next burst as long when CLEANTXBUFFER's limbo has just settled: the host's
# client, told SETTLED, writes what goes next (after the rest of a record
# begun on air, if any).
REPLY_HOLD_S = 0.25

log = logging.getLogger("data2g.session")

IDLE, LISTEN, CONNECTING, CONNECTED, DISCONNECTING, CLOSED = (
    "idle", "listen", "connecting", "connected", "disconnecting", "closed")
# CONNECT_NAK reasons: the caller stops dialing on any of them
NAK_VERSION, NAK_NOT_LISTENING, NAK_BUSY, NAK_REFUSED = 1, 2, 3, 4
# close reasons (DISCONNECTED's, docs/arq.md §9) are one of these words, then detail if any
NAK_NAMES = {NAK_VERSION: "VERSION", NAK_NOT_LISTENING: "NOT_LISTENING", NAK_BUSY: "BUSY", NAK_REFUSED: "REFUSED"}


def session_key(caller: str, callee: str, nonce: int) -> int:
    """Nonzero 16-bit key; 0 means no session (connect frames)."""
    h = 0x811C9DC5
    for b in f"{caller}|{callee}|{nonce}".encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return (h & 0xFFFF) or 1


@dataclass
class Session:
    call: str
    policy: L.Policy  # connect_mode and airtime too
    t_turn: float = 1.0
    rng: random.Random = field(default_factory=random.Random)
    state: str = IDLE
    peer: str = ""
    cap: int = 2
    station: L.Station | None = None
    events: list = field(default_factory=list)  # host notifications, oldest first
    close_reason: str = ""
    _nonce: int = 0
    _out: B.TxBurst | None = None
    _due: float | None = None  # when _out may go
    _deadline: float | None = None  # master: reply must be heard by then
    _answer_mode: str | None = None  # callee: the mode the CONNECT came in (its ACK goes in it)
    _tries: int = 0
    _last_heard: float = 0.0
    _last_data: float = 0.0
    _idle_wait: float = 0.0
    _build_at: float | None = None  # the caller's idle poll, or the callee's held reply: built when it goes
    _quiet_from: float = 0.0  # callee: the end of its last burst, or the last burst heard since
    _wake_wait: float = 0.0  # callee: silence after _quiet_from before a wake
    _wakes: int = 0  # callee: wake bursts sent since the caller was last heard
    _want_disc: bool = False
    _disc_why: str = "CLIENT"  # who wanted the DISC: CLIENT or IDLE
    _sent_disc_ack: bool = False
    _pending_write: bytearray = field(default_factory=bytearray)
    _nak: B.TxBurst | None = None  # a refusal in _out: not the session's burst (no reply timer, no wake)
    chat: bool = False  # CHAT ON: set with set_chat()
    idle_close_s: float | None = None  # host TIMEOUT; None: IDLE_CLOSE_S
    aliases: tuple = ()  # more calls this station answers to (VARA's MYCALL takes several)
    accept_connect: object = None  # (caller, callee) -> bool, before a CONNECT is answered: False refuses it (host ACCEPT)
    stats_interval_s: float = 60.0  # throughput line in the log this often while connected (0: off)
    _now: float = 0.0  # the latest time the caller gave
    _stats_since: float = 0.0
    _stats_prev: dict = field(default_factory=dict)
    _connected_at: float = 0.0
    _coalesce_until: float | None = None  # write(): an idle poll or wake held for more data

    # -- host side -------------------------------------------------------------------

    def set_chat(self, on: bool):
        """VARA's CHAT ON / OFF: latency over throughput for this session."""
        self.chat = on
        if self.station is not None:
            self.station.chat = on

    def listen(self):
        self.state = LISTEN

    def connect(self, peer: str, cap: int, now: float):
        self.peer, self.cap, self.state = peer.upper(), cap, CONNECTING
        self._nonce = self.rng.randrange(1 << 16)
        self._tries = 0
        self._queue(self._connect_burst(), now)

    def disconnect(self):
        if self.state in (IDLE, LISTEN, CLOSED):
            return  # nothing to close; a flag left set would end the next session at its first burst
        self._want_disc, self._disc_why = True, "CLIENT"
        if self._build_at is not None:
            self._build_at = float("-inf")  # the caller's idle poll: a DISC now instead

    def write(self, data: bytes):
        st = self.station
        if st is None:
            self._pending_write += data
            return
        # CHAT OFF: a write to an empty buffer, less than a full burst, holds
        # the idle poll or wake it brings forward COALESCE_S for more (clients
        # write a message in pieces). A full burst's worth ends the hold.
        holding = self._coalesce_until is not None and self._now < self._coalesce_until
        fresh = holding or not st.tx.pending()
        st.write(data)
        if fresh and not self.chat and (cap := self.policy.next_capacity(st)) is not None and st.new_available() < cap:
            if not holding:
                self._coalesce_until = self._now + COALESCE_S
        else:
            self._coalesce_until = None
        if self._build_at is not None:  # an idle poll due: send it now (or at the hold's end), with the data
            self._build_at = self._coalesce_until if self._coalesce_until is not None else float("-inf")
            self._idle_wait = 0.0

    def read(self) -> bytes:
        return self.station.read() if self.station else b""

    def queued_bytes(self, vara: bool = False) -> tuple[int, int]:
        """(unsent, unacked): host bytes written that are in no codeword
        yet, and that the peer has not acked, sent or not. Writes waiting
        for the session count as both. `vara`: without those written before
        the last CLEANTXBUFFER (VARA drops them all, on air or not)."""
        n = len(self._pending_write)
        if self.station is None:
            return n, n
        u, a = self.station.tx.queued(vara)
        return n + u, n + a

    # -- clock side ------------------------------------------------------------------

    def poll(self, now: float) -> B.TxBurst | None:
        self._now = now
        if (self.stats_interval_s and self.state in (CONNECTED, DISCONNECTING)
                and now >= self._stats_since + self.stats_interval_s):
            self._log_stats("stats", now, self._stats_since, self._stats_prev)
            self._stats_since, self._stats_prev = now, self._stats()
        if self.state == CLOSED and self._out is None:
            return None
        if self.state in (CONNECTED, DISCONNECTING):
            # the very expressions next_event() reports: (t + d) - t can come
            # out a hair under d, and an event loop then spins at that
            # instant forever (found by scripts/arq_stress.py --sessions)
            if now >= self._last_heard + LINK_LOST_S:
                return self._close("LINK_LOST" if self.state == CONNECTED else f"{self._disc_why} unconfirmed")
            if self.state == CONNECTED and now >= self._last_data + (IDLE_CLOSE_S if self.idle_close_s is None else self.idle_close_s) and not self._want_disc:
                self._want_disc, self._disc_why = True, "IDLE"
        if self._deadline is not None and now >= self._deadline:
            self._deadline = None
            self._on_timeout(now)
        if self._build_at is not None and now >= self._build_at and self.state == CONNECTED:
            self._build_at = None
            if not self._disc(now):
                self._queue(self.station.build(), now)
                self.station.answered()
        if (w := self._wake_time()) is not None and now >= w:
            self._wakes += 1
            self._wake_wait = float("inf")  # until on_tx_end arms the next
            st = self.station
            if self._disc(now):
                log.info("wake: breaking idle with DISC")
            elif self._wakes == 1:
                log.info("wake: breaking idle with %d unacked, %d B queued", st.tx.next - st.tx.base,
                         st.new_available())
                self._queue(st.build(), now)
            else:
                log.info("wake %d: no answer, repeating", self._wakes)
                self._queue(st.last_sent, now)
        if self._out is not None and self._due is not None and now >= self._due:
            out, self._out, self._due = self._out, None, None
            return out
        return None

    def claims(self, rx: B.RxBurst) -> bool:
        """`rx` is my peer's: connected (or disconnecting) and its first
        control codeword decodes under the session key (the station's own
        decode of it is then a memo hit, phy.ModemRx)."""
        st = self.station
        return (st is not None and self.state in (CONNECTED, DISCONNECTING)
                and rx.decode(0, B.ctl_mask(st.peer, 0, st.key), 0, None) is not None)

    def idle(self) -> bool:
        """Connected with nothing in flight or queued, either way: the air is
        free for a broadcast burst (the engine's KISS link)."""
        return (self.state == CONNECTED and self.station is not None and self._out is None
                and self._deadline is None and not self._want_disc and not self._pending_write
                and not self.station.tx.pending())

    def next_event(self) -> float | None:
        """The earliest time poll() may do something (for an event loop)."""
        ts = [t for t in (self._due, self._deadline, self._build_at, self._wake_time()) if t is not None]
        if self.state in (CONNECTED, DISCONNECTING):
            ts.append(self._last_heard + LINK_LOST_S)
        return min(ts) if ts else None

    def on_tx_end(self, burst: B.TxBurst, now: float):
        """Arm the reply timer (only the caller retries, §6, and a callee's
        DISC), or the callee's wake."""
        if burst is self._nak:
            self._nak = None
            return
        if (self.master and self.state in (CONNECTING, CONNECTED)) or self.state == DISCONNECTING:
            wait = REPLY_START_S
            if (self.state == CONNECTED and self.station is not None
                    and (hold := self.policy.reply_hold(self.station, burst)) is not None):
                # a reply whose header I miss is still on air: don't poll over it
                wait = max(wait, hold)
            self._deadline = now + self.t_turn + wait
        elif not self.master:
            self._quiet_from = now
            self._wake_wait = (self.t_turn + REPLY_START_S + WAKE_GUARD_S
                               + self.rng.uniform(0, WAKE_JITTER_S * 2 ** self._wakes))

    def on_header(self, submode: str, n_cw: int, now: float):
        """A burst started arriving: wait for its end instead of timing out
        (caller), or hold a wake past it (callee)."""
        end = now + self.policy.airtime(submode, n_cw)
        if self._deadline is not None:
            # a peer's ID burst (one codeword in the connect mode) runs straight
            # into its reply, whose header must still be found (engine.ID_INTERVAL_S)
            is_id = n_cw == 1 and submode == self.policy.connect_mode(self.cap)
            self._deadline = max(self._deadline, end + self.t_turn + (REPLY_START_S if is_id else 0.0))
        if not self.master:
            self._quiet_from = max(self._quiet_from, end)

    def on_rx(self, rx: B.RxBurst, now: float):
        ctl = self._session_frame(rx)
        if ctl is not None:
            self._on_session_frame(ctl, now)
            return
        st = self.station
        if st is None or self.state not in (CONNECTED, DISCONNECTING):
            return
        undrained, limbo = len(st.rx.out), st.tx.limbo is not None
        if not st.handle(rx):
            return  # not ours, or unreadable: stay silent (§3)
        self._heard(now)
        if st.state == L.FAILED:
            self._close(f"LINK_FAILED {st.fail_reason}")
            return
        st.settle_limbo()
        settled = limbo and st.tx.limbo is None
        if st.rx.out or st.last_rx_data or st.tx.pending():
            self._last_data = now  # data moving either way; a sender that hears no data is not idle
        if self._disc(now):
            return
        if not self.master:
            if settled or (len(st.rx.out) > undrained and st.peer_drained and not st.tx.pending()
                           and not self._pending_write):
                self._build_at = now + REPLY_HOLD_S  # poll() builds it; write() brings it forward
            else:
                self._build_at = None
                self._queue(st.build(), now)
                st.answered()
            return
        # the caller: go on now, or after an idle backoff when neither side
        # has data. The idle poll is built when it goes, so data the host
        # writes meanwhile rides it (write() also brings it forward).
        # a burst while the idle poll waits is the callee's wake: answer it
        # at once even without data in it (its policy may have sent control only)
        woke, self._build_at = self._build_at is not None, None
        # the peer's reply was lost: it may hold data I never heard of. An
        # abandon re-sliced data the peer holds, maybe in a control-only
        # (escalated) burst: idling would leave it to wake me ~5 s later
        busy = (st.tx.pending() or st.last_rx_data or bool(self._pending_write) or woke or st.peer_reply_lost
                or st.peer_abandon)
        if busy and settled:
            self._idle_wait = 0.0
            self._build_at = now + REPLY_HOLD_S  # poll() builds it; write() brings it forward
        elif busy:
            self._idle_wait = 0.0
            self._queue(st.build(), now)
            st.answered()
        else:
            lo, hi = CHAT_KEEPALIVE_S if st.chat or st.peer_chat else KEEPALIVE_S
            if KEEPALIVE_DOUBLING:
                self._idle_wait = lo if not self._idle_wait else min(hi, 2 * self._idle_wait)
            else:
                self._idle_wait = self.rng.uniform(lo, hi)
            self._build_at = now + self._idle_wait

    @property
    def master(self) -> bool:
        """The caller: the station that retries (its link.Station's, once there is one)."""
        return self.station.master if self.station else self.state == CONNECTING

    # -- internals ---------------------------------------------------------------------

    def _queue(self, burst: B.TxBurst, when: float):
        self._out, self._due = burst, when

    def _heard(self, now: float):
        self._last_heard = now
        self._deadline = None
        self._quiet_from, self._wakes = now, 0

    def _disc(self, now: float) -> bool:
        """Queue a DISC if the host asked for one and all our data is acked."""
        if not self._want_disc or self.station.tx.pending():
            return False
        self._tries = 0
        self._queue(self._disc_burst(), now)
        self.state = DISCONNECTING
        return True

    def _wake_time(self) -> float | None:
        """Callee: when to break idle with queued data or a DISC, if it may.
        Only from idle: its last burst carried no data, so the caller's
        receive state is what this station last heard it ack and a fresh
        build is exact (link.Station.build). Not while a burst is queued."""
        st = self.station
        if (st is None or self.master or self.state != CONNECTED or self._out is not None
                or self._wakes >= (CHAT_WAKE_TRIES if st.chat or st.peer_chat else WAKE_TRIES)
                or st.last_sent is None):
            return None
        disc = self._want_disc and not st.tx.pending()
        if not self._wakes and not disc and (st.data_in_flight() or not st.tx.pending()):
            return None
        return max(self._quiet_from + self._wake_wait, self._coalesce_until or float("-inf"))

    def _close(self, why: str, final: B.TxBurst | None = None, now: float = 0.0):
        """-> None. `final`: one last burst to send (a DISC_ACK)."""
        self._build_at = None
        if self.state != CLOSED:
            self.events.append(f"DISCONNECTED {why}")
            if self.station is not None:
                self._log_stats("session", max(now, self._now), self._connected_at, {})
        self.state, self.close_reason = CLOSED, why
        self._out, self._due = (final, now) if final is not None else (None, None)
        self._deadline = None
        return None

    def _connected(self, now: float):
        self.state = CONNECTED
        self._heard(now)
        self._last_data = self._connected_at = self._stats_since = now
        self._stats_prev = {}

    def _stats(self) -> dict:
        st = self.station
        return dict(st.stats, tx_bytes=st.tx.acked, rx_bytes=st.rx.reader.delivered,
                    tx_plain=st.tx.acked_plain, tx_wire=st.tx.acked_wire, rx_plain=st.rx.plain, rx_wire=st.rx.wire)

    def _log_stats(self, label: str, now: float, since: float, prev: dict):
        d = {k: v - prev.get(k, 0) for k, v in self._stats().items()}
        dt = max(now - since, 1e-9)
        cw = d.get("cw_new", 0) + d.get("cw_resend", 0)

        def ratio(k):  # uncompressed / compressed payload bytes
            return "%.2f" % (d[k + "_plain"] / d[k + "_wire"]) if d[k + "_wire"] else "-"
        log.info("%s %.0f s: tx %d B acked (%.0f bps, compression %s), rx %d B (%.0f bps, compression %s)"
                 " | data cw sent %d, %.0f%% resends | bursts heard %d, %d control lost | timeouts %d", label, dt,
                 d["tx_bytes"], 8 * d["tx_bytes"] / dt, ratio("tx"), d["rx_bytes"], 8 * d["rx_bytes"] / dt,
                 ratio("rx"), cw, 100 * d.get("cw_resend", 0) / max(cw, 1),
                 d.get("rx_ok", 0) + d.get("rx_lost", 0), d.get("rx_lost", 0), d.get("timeouts", 0))

    def _on_timeout(self, now: float):
        if self.state == CONNECTING:
            self._tries += 1
            log.info("no answer to CONNECT %s, try %d of %d", self.peer, self._tries, CONNECT_TRIES)
            if self._tries >= CONNECT_TRIES:
                self._close("NO_ANSWER")
            else:
                self._queue(self._connect_burst(), now + self.rng.uniform(3.0, 5.0))
        elif self.state == DISCONNECTING:
            self._tries += 1
            log.info("no answer to DISC, try %d of %d", self._tries, DISC_TRIES)
            if self._tries >= DISC_TRIES:
                self._close(f"{self._disc_why} unconfirmed")
            else:
                self._queue(self._disc_burst(), now)
        elif self.state == CONNECTED:
            # an identical repeat only after a short burst: after a long one,
            # the lost reply is likelier and a poll recovers it for ~1 s
            # instead of the whole burst again (linksim: repeats of 12 s
            # bursts took 30% of the time on mpg)
            last = self.station.last_sent
            short = last is None or self.policy.airtime(last.submode, len(last.slots), B.dup_ctl(last)) <= REPEAT_MAX_S
            b = self.station.on_timeout(allow_repeat=short)
            if b is None:
                self._close(f"LINK_FAILED {self.station.fail_reason}")
            else:
                self._queue(b, now)

    # session frames: control-only bursts with a T_SESS extension. Connect
    # frames use key 0 (no session yet); DISC / DISC_ACK the session key.

    def _session_burst(self, direction: int, key: int, body: bytes, mode: str | None = None) -> B.TxBurst:
        """`mode`: an answer goes in the mode its frame came in (a caller that
        needed the robust connect mode hears its answer in it); otherwise
        the policy's, more robust on retries."""
        mode = mode or self.policy.connect_mode(self.cap, self._tries)
        cpb = self.policy.ctl_payload_bytes(mode)
        if body[0] == F.CONNECT and self._compact(mode):
            slots = [B.Slot(B.COMPACT_CONNECT, 0, F.pack_connect(body) + bytes(cpb - F.COMPACT_BYTES))]
        else:
            ctl = F.Control(F.Core(ftype=F.SESSION), {F.T_SESS: body}).pack(cpb)
            slots = [B.Slot(B.ctl_mask(direction, i, key), 0, p) for i, p in enumerate(ctl)]
        retry = f" try {self._tries + 1}" if body[0] in (F.CONNECT, F.DISC) else ""
        log.info("TX %s %s x%d%s", _frame_desc(body), mode, len(slots), retry)
        return B.TxBurst(mode, slots, 0, self.cap)

    def _compact(self, mode: str) -> bool:
        """A CONNECT in `mode` goes compact (frames.pack_connect): in its
        Control envelope it needs more codewords than the mode's control
        may have (CPM: one of 20 B)."""
        return -(-CONNECT_CTL_BYTES // self.policy.ctl_payload_bytes(mode)) > self.policy.max_ctl(mode)

    def _connect_burst(self) -> B.TxBurst:
        body = F.Connect(VERSION, self.call, self.peer, self._nonce, self.cap, round(self.t_turn * 10)).pack()
        return self._session_burst(0, 0, body)

    def _disc_burst(self) -> B.TxBurst:
        st = self.station
        return self._session_burst(st.direction, st.key, bytes([F.DISC]))

    def _session_frame(self, rx) -> dict | None:
        """Decode rx as a session frame addressed to this station, if it is one.
        What each state accepts:
          station exists   DISC / DISC_ACK from the peer under the session key
          CONNECTING       CONNECT_ACK / CONNECT_NAK, key 0
          any              CONNECT, key 0: answered, or refused with a CONNECT_NAK
        A decode is memoised per slot (ModemRx), so each extra mask is a CRC check."""
        tries = []
        if self.station is not None:
            tries.append((self.station.peer, self.station.key))
        elif self.state == CONNECTING:
            tries.append((1, 0))
        tries.append((0, 0))
        if (0, 0) in tries and self._compact(rx.submode):
            p = rx.decode(0, B.COMPACT_CONNECT, 0, None)
            if p is not None:
                body = F.unpack_connect(p[:F.COMPACT_BYTES])
                if why := _check_frame(body):
                    log.warning("RX malformed %s: dropped", why)
                    return None
                return {"key": 0, "body": body, "mode": rx.submode}
        for direction, key in tries:
            first = rx.decode(0, B.ctl_mask(direction, 0, key), 0, None)
            if first is None:
                continue
            core = F.Core.unpack(first)
            if core.ftype != F.SESSION:
                return None
            if core.n_ctl > rx.n_cw:
                log.warning("RX malformed session frame (%d control codewords in a burst of %d): dropped",
                            core.n_ctl, rx.n_cw)
                return None
            payloads = [first]
            for i in range(1, core.n_ctl):
                p = rx.decode(i, B.ctl_mask(direction, i, key), 0, None)
                if p is None:
                    return None
                payloads.append(p)
            try:
                body = F.Control.unpack(payloads).ext.get(F.T_SESS, b"")
            except ValueError as e:
                log.warning("RX malformed session frame (%s): dropped", e)
                return None
            if body and (why := _check_frame(body)):
                log.warning("RX malformed %s: dropped", why)
                return None
            return {"key": key, "body": body, "mode": rx.submode} if body else None
        return None

    def _on_session_frame(self, f: dict, now: float):
        body, key = f["body"], f["key"]
        sub = body[0]
        log.info("RX %s %s", _frame_desc(body), f["mode"])
        if sub == F.CONNECT and len(body) >= F.Connect.BYTES and key == 0:
            c = F.Connect.unpack(body)
            caller, callee, nonce = c.caller, c.callee, c.nonce
            if callee != self.call.upper() and callee not in self.aliases:
                return
            if self.station is not None and not self.master and nonce == self._nonce:
                self._answer_mode = f["mode"]
                self._queue(self._accept_burst(), now)  # our ACK was lost: say it again
                return
            if (self.station is not None and not self.master and caller == self.peer
                    and self.state in (CONNECTED, DISCONNECTING)):
                # my caller dialed again: it gave up on our session (lost my
                # CONNECT_ACK and ran out of tries), so take the new one. Held
                # on air, the old one ignored it until link lost
                # (recordings/20261002-232711)
                self._close("PEER_RECONNECTED", now=now)
                self.state, self._want_disc, self._sent_disc_ack, self._tries = LISTEN, False, False, 0
            if self.state != LISTEN:
                busy = self.state in (CONNECTING, CONNECTED, DISCONNECTING)
                return self._refuse(c, f["mode"], NAK_BUSY if busy else NAK_NOT_LISTENING, now)
            if c.version != VERSION:
                return self._refuse(c, f["mode"], NAK_VERSION, now)
            if self.accept_connect is not None and not self.accept_connect(caller, callee):
                return self._refuse(c, f["mode"], NAK_REFUSED, now)
            self.call = callee  # the call it dialed (the session key derives from it)
            self.peer, self._nonce, self.cap = caller, nonce, min(c.cap, self.cap)
            self.station = L.Station(1, self.policy, master=False, key=session_key(caller, self.call, nonce),
                                     cap=self.cap, max_misses=None, chat=self.chat)
            self._flush_writes()
            self._connected(now)
            self.events.append(f"CONNECTED {caller}")
            self._answer_mode = f["mode"]
            self._queue(self._accept_burst(), now)
        elif sub == F.CONNECT_ACK and self.state == CONNECTING and key == 0:
            a = F.ConnectAck.unpack(body)
            if a.nonce != self._nonce:
                return
            self.cap = min(a.cap, self.cap)
            self.station = L.Station(0, self.policy, master=True, key=session_key(self.call, self.peer, self._nonce),
                                     cap=self.cap, max_misses=None, chat=self.chat)
            self._flush_writes()
            self._connected(now)
            self.events.append(f"CONNECTED {self.peer}")
            self._queue(self.station.build(), now)
        elif sub == F.CONNECT_NAK and self.state == CONNECTING and key == 0:
            n = F.ConnectNak.unpack(body)
            if n.nonce == self._nonce:
                self._close(NAK_NAMES.get(n.reason, f"REFUSED reason {n.reason}"))
        elif sub == F.DISC and self.station is not None and key == self.station.key:
            # also when already closed: our DISC_ACK may have been lost
            self._heard(now)
            self._close("PEER",
                        final=self._session_burst(self.station.direction, key, bytes([F.DISC_ACK]), f["mode"]), now=now)
        elif sub == F.DISC_ACK and self.state == DISCONNECTING and key == self.station.key:
            self._close(self._disc_why)

    def _refuse(self, c: F.Connect, mode: str, reason: int, now: float):
        """A CONNECT_NAK for this CONNECT, unless a burst of the session's own
        is queued (that goes; the caller's next try is refused instead)."""
        log.info("refusing CONNECT from %s: %s", c.caller, NAK_NAMES[reason])
        if self._out is None:
            self._nak = self._session_burst(1, 0, F.ConnectNak(c.nonce, reason).pack(), mode)
            self._queue(self._nak, now)

    def _accept_burst(self) -> B.TxBurst:
        body = F.ConnectAck(self._nonce, self.cap, round(self.t_turn * 10)).pack()
        return self._session_burst(1, 0, body, self._answer_mode)

    def _flush_writes(self):
        if self._pending_write:
            self.station.write(bytes(self._pending_write))
            self._pending_write.clear()


FRAME_NAMES = {F.CONNECT: "CONNECT", F.CONNECT_ACK: "CONNECT_ACK", F.CONNECT_NAK: "CONNECT_NAK",
               F.DISC: "DISC", F.DISC_ACK: "DISC_ACK"}
FRAME_BYTES = {F.CONNECT: F.Connect.BYTES, F.CONNECT_ACK: F.ConnectAck.BYTES, F.CONNECT_NAK: F.ConnectNak.BYTES}


def _check_frame(body: bytes) -> str | None:
    """What is wrong with a CRC-valid session frame body, or None. A bad
    one is dropped as if it had not decoded (the peer retries)."""
    name = FRAME_NAMES.get(body[0], f"session frame {body[0]}")
    if len(body) < FRAME_BYTES.get(body[0], 1):
        return f"{name} of {len(body)} B"
    if body[0] == F.CONNECT:
        try:
            F.Connect.unpack(body)
        except ValueError as e:
            return f"{name}: {e}"
    if body[0] == F.CONNECT_ACK and (cap := F.ConnectAck.unpack(body).cap) > 2:
        return f"{name}: cap code {cap}"
    return None


def _frame_desc(body: bytes) -> str:
    out = FRAME_NAMES.get(body[0], f"session frame {body[0]}")
    if body[0] == F.CONNECT and len(body) >= 18:
        out += " {}>{}".format(*F.Connect.calls(body))
    return out
