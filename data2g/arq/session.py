"""ARQ session (docs/arq.md §6, §7): connect, disconnect and timers
around link.Station. Driven by its caller's clock:

    s.on_header(submode, n_cw, now)   a burst header was decoded (its end is known)
    s.on_rx(rx, now)                  a whole burst arrived (link.RxBurst)
    s.poll(now) -> TxBurst | None     anything to send now?
    s.on_tx_end(burst, now)           that burst finished going out

Timers are absolute times; nothing here sleeps. The same object runs in
the simulator (scripts/linksim.py) and on the air.

Idle rule (docs/arq.md §6a): the caller keeps the link alive with a
jittered backoff (KEEPALIVE_S). The callee may break idle with its new
data (a wake burst, WAKE_*), but owns no retry timer: any burst from the
caller cancels its wake, and the caller stays the only station that
retries on a timeout.
"""

import logging
import random
from dataclasses import dataclass, field

from . import frames as F
from . import link as L

T_SESS = 10  # extension type carrying a session frame
VERSION = 2  # 2: T_COMP (a v1 peer would deliver compressed codewords raw)
CONNECT_TRIES = 5
DISC_TRIES = 3
# past t_turn: a reply's header must have been heard by then: the peer's
# decode and PTT (~0.7 s), audio latency both ways, the reply's lead-in,
# preamble and header (0.4-0.6 s) and the receiver's search step (0.25 s)
# (the audio loopback timed out on replies already on air at 1.0)
REPLY_START_S = 1.5
IDLE_CLOSE_S = 300.0
KEEPALIVE_S = (15.0, 60.0)  # first and largest idle poll interval
# after the longest keepalive gap, v1's retry budget (90 s link lost
# against 16 s polls): a lost keepalive must not close the link
LINK_LOST_S = KEEPALIVE_S[1] + 75.0
KEEPALIVE_JITTER = 0.3  # each interval stretched by up to this fraction (no lockstep with the callee's wakes)
# the callee's wake: after t_turn + REPLY_START_S + WAKE_GUARD_S of silence
# since its last burst (the caller's answer or timeout retry starts within
# t_turn + REPLY_START_S of it; the guard covers PTT, audio latency and
# header detection), plus a random 0-WAKE_JITTER_S. A lost wake is repeated
# identically after WAKE_RETRY_S, at most WAKE_TRIES sends per idle period.
WAKE_GUARD_S = 1.0
WAKE_JITTER_S = 1.0
WAKE_RETRY_S = (6.0, 10.0)
WAKE_TRIES = 2
CHAT_KEEPALIVE_S = (2.0, 4.0)  # the same while either side has CHAT ON: the callee's message waits for a poll
REPEAT_MAX_S = 3.0  # repeat a timed-out burst identically only if it is this short

log = logging.getLogger("data2g.session")

IDLE, LISTEN, CONNECTING, CONNECTED, DISCONNECTING, CLOSED = (
    "idle", "listen", "connecting", "connected", "disconnecting", "closed")


def session_key(caller: str, callee: str, nonce: int) -> int:
    """Nonzero 16-bit key; 0 means no session (connect frames)."""
    h = 0x811C9DC5
    for b in f"{caller}|{callee}|{nonce}".encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return (h & 0xFFFF) or 1


@dataclass
class Session:
    call: str
    policy: object  # link.Policy plus connect_mode(cap_code) -> submode, airtime(submode, n_cw) -> s
    t_turn: float = 1.0
    rng: random.Random = field(default_factory=random.Random)
    state: str = IDLE
    peer: str = ""
    cap: int = 2
    station: L.Station | None = None
    events: list = field(default_factory=list)  # host notifications, oldest first
    close_reason: str = ""
    _nonce: int = 0
    _out: L.TxBurst | None = None
    _due: float | None = None  # when _out may go
    _deadline: float | None = None  # master: reply must be heard by then
    _answer_mode: str | None = None  # callee: the mode the CONNECT came in (its ACK goes in it)
    _tries: int = 0
    _last_heard: float = 0.0
    _last_data: float = 0.0
    _idle_wait: float = 0.0
    _build_at: float | None = None  # the caller's idle poll: built when it goes
    _quiet_from: float = 0.0  # callee: the end of its last burst, or the last burst heard since
    _wake_wait: float = 0.0  # callee: silence after _quiet_from before a wake
    _wakes: int = 0  # callee: wake bursts sent since the caller was last heard
    _want_disc: bool = False
    _sent_disc_ack: bool = False
    _pending_write: bytearray = field(default_factory=bytearray)
    chat: bool = False  # CHAT ON: set with set_chat()
    aliases: tuple = ()  # more calls this station answers to (VARA's MYCALL takes several)
    stats_interval_s: float = 60.0  # throughput line in the log this often while connected (0: off)
    _now: float = 0.0  # the latest time the caller gave
    _stats_since: float = 0.0
    _stats_prev: dict = field(default_factory=dict)
    _connected_at: float = 0.0

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
        self._want_disc = True

    def write(self, data: bytes):
        if self.station is not None:
            self.station.write(data)
            if self._build_at is not None:
                self._build_at = float("-inf")  # an idle poll due: send it now, with the data
                self._idle_wait = 0.0
        else:
            self._pending_write += data

    def read(self) -> bytes:
        return self.station.read() if self.station else b""

    # -- clock side ------------------------------------------------------------------

    def poll(self, now: float) -> L.TxBurst | None:
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
                return self._close("link lost" if self.state == CONNECTED else "disconnected (unconfirmed)")
            if self.state == CONNECTED and now >= self._last_data + IDLE_CLOSE_S:
                self._want_disc = True
        if self._deadline is not None and now >= self._deadline:
            self._deadline = None
            self._on_timeout(now)
        if self._build_at is not None and now >= self._build_at and self.state == CONNECTED:
            self._build_at = None
            self._queue(self.station.build(), now)
            self.station.answered()
        if (w := self._wake_time()) is not None and now >= w:
            self._wakes += 1
            self._wake_wait = float("inf")  # until on_tx_end arms the next
            st = self.station
            if self._wakes == 1:
                log.info("wake: breaking idle with %d B queued", st._new_available())
                self._queue(st.build(), now)
            else:
                log.info("wake %d: no answer, repeating", self._wakes)
                self._queue(st.last_sent, now)
        if self._out is not None and self._due is not None and now >= self._due:
            out, self._out, self._due = self._out, None, None
            return out
        return None

    def next_event(self) -> float | None:
        """The earliest time poll() may do something (for an event loop)."""
        ts = [t for t in (self._due, self._deadline, self._build_at, self._wake_time()) if t is not None]
        if self.state in (CONNECTED, DISCONNECTING):
            ts.append(self._last_heard + LINK_LOST_S)
        return min(ts) if ts else None

    def on_tx_end(self, burst: L.TxBurst, now: float):
        """Arm the reply timer (only the caller retries, §6), or the callee's wake."""
        if self._master and self.state in (CONNECTING, CONNECTED, DISCONNECTING):
            self._deadline = now + self.t_turn + REPLY_START_S
        elif not self._master:
            self._quiet_from = now
            self._wake_wait = (self.t_turn + REPLY_START_S + WAKE_GUARD_S + self.rng.uniform(0, WAKE_JITTER_S) if not self._wakes
                               else self.rng.uniform(*WAKE_RETRY_S))

    def on_header(self, submode: str, n_cw: int, now: float):
        """A burst started arriving: wait for its end instead of timing out
        (caller), or hold a wake past it (callee)."""
        end = now + self.policy.airtime(submode, n_cw)
        if self._deadline is not None:
            self._deadline = max(self._deadline, end + self.t_turn)
        if not self._master:
            self._quiet_from = max(self._quiet_from, end)

    def on_rx(self, rx: L.RxBurst, now: float):
        ctl = self._session_frame(rx)
        if ctl is not None:
            self._on_session_frame(ctl, now)
            return
        st = self.station
        if st is None or self.state not in (CONNECTED, DISCONNECTING):
            return
        if not st.handle(rx):
            return  # not ours, or unreadable: stay silent (§3)
        self._heard(now)
        if st.state == L.FAILED:
            self._close(f"link failed: {st.fail_reason}")
            return
        if st.rx.out or st.last_rx_data or st.tx.pending():
            self._last_data = now  # data moving either way; a sender that hears no data is not idle
        if self._want_disc and not st.tx.pending():
            self._tries = 0
            self._queue(self._disc_burst(), now)
            self.state = DISCONNECTING
            return
        if not self._master:
            self._queue(st.build(), now)
            st.answered()
            return
        # the caller: go on now, or after an idle backoff when neither side
        # has data. The idle poll is built when it goes, so data the host
        # writes meanwhile rides it (write() also brings it forward).
        # a burst while the idle poll waits is the callee's wake: answer it
        # at once even without data in it (its policy may have sent control only)
        woke, self._build_at = self._build_at is not None, None
        busy = st.tx.pending() or st.last_rx_data or bool(self._pending_write) or woke
        if busy:
            self._idle_wait = 0.0
            self._queue(st.build(), now)
            st.answered()
        else:
            lo, hi = CHAT_KEEPALIVE_S if st.chat or st.peer_chat else KEEPALIVE_S
            self._idle_wait = lo if not self._idle_wait else min(hi, 2 * self._idle_wait)
            self._build_at = now + min(hi, self._idle_wait * (1 + self.rng.uniform(0, KEEPALIVE_JITTER)))

    # -- internals ---------------------------------------------------------------------

    @property
    def _master(self) -> bool:
        return self.station.master if self.station else self.state == CONNECTING

    def _queue(self, burst: L.TxBurst, when: float):
        self._out, self._due = burst, when

    def _heard(self, now: float):
        self._last_heard = now
        self._deadline = None
        self._quiet_from, self._wakes = now, 0

    def _wake_time(self) -> float | None:
        """Callee: when to break idle with queued data, if it may. Only from
        idle: its last burst carried no data, so the caller's receive state
        is what this station last heard it ack and a fresh build is exact
        (link.Station.build). Not while a burst is queued or in chat mode,
        where the caller's fast polls carry it."""
        st = self.station
        if (st is None or self._master or self.state != CONNECTED or self._out is not None or self._want_disc
                or self._wakes >= WAKE_TRIES or st.chat or st.peer_chat or st.last_sent is None):
            return None
        if not self._wakes and (st._sent_seqs.get(st._latest) or not st.tx.pending()):
            return None
        return self._quiet_from + self._wake_wait

    def _close(self, why: str, final: L.TxBurst | None = None, now: float = 0.0):
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
        return dict(st.stats, tx_bytes=st.tx.acked, rx_bytes=st.rx.reader.delivered)

    def _log_stats(self, label: str, now: float, since: float, prev: dict):
        d = {k: v - prev.get(k, 0) for k, v in self._stats().items()}
        dt = max(now - since, 1e-9)
        cw = d.get("cw_new", 0) + d.get("cw_resend", 0)
        log.info("%s %.0f s: tx %d B acked (%.0f bps), rx %d B (%.0f bps) | data cw sent %d, %.0f%% resends,"
                 " %d new compressed | bursts heard %d, %d control lost | timeouts %d", label, dt, d["tx_bytes"],
                 8 * d["tx_bytes"] / dt, d["rx_bytes"], 8 * d["rx_bytes"] / dt, cw,
                 100 * d.get("cw_resend", 0) / max(cw, 1), d.get("cw_comp", 0),
                 d.get("rx_ok", 0) + d.get("rx_lost", 0), d.get("rx_lost", 0), d.get("timeouts", 0))

    def _on_timeout(self, now: float):
        if self.state == CONNECTING:
            self._tries += 1
            log.info("no answer to CONNECT %s, try %d of %d", self.peer, self._tries, CONNECT_TRIES)
            if self._tries >= CONNECT_TRIES:
                self._close("no answer")
            else:
                self._queue(self._connect_burst(), now + self.rng.uniform(3.0, 5.0))
        elif self.state == DISCONNECTING:
            self._tries += 1
            log.info("no answer to DISC, try %d of %d", self._tries, DISC_TRIES)
            if self._tries >= DISC_TRIES:
                self._close("disconnected (unconfirmed)")
            else:
                self._queue(self._disc_burst(), now)
        elif self.state == CONNECTED:
            # an identical repeat only after a short burst: after a long one,
            # the lost reply is likelier and a poll recovers it for ~1 s
            # instead of the whole burst again (linksim: repeats of 12 s
            # bursts took 30% of the time on mpg)
            last = self.station.last_sent
            short = last is None or self.policy.airtime(last.submode, len(last.slots), L.dup_ctl(last)) <= REPEAT_MAX_S
            b = self.station.on_timeout(allow_repeat=short)
            if b is None:
                self._close(f"link failed: {self.station.fail_reason}")
            else:
                self._queue(b, now)

    # session frames: control-only bursts with a T_SESS extension. Connect
    # frames use key 0 (no session yet); DISC / DISC_ACK the session key.

    def _session_burst(self, direction: int, key: int, body: bytes, mode: str | None = None) -> L.TxBurst:
        """`mode`: an answer goes in the mode its frame came in (a caller that
        needed the robust connect mode hears its answer in it); otherwise
        the policy's, more robust on retries."""
        mode = mode or self.policy.connect_mode(self.cap, self._tries)
        ctl = F.Control(F.Core(ftype=F.SESSION), {T_SESS: body}).pack(self.policy.payload_bytes(mode))
        slots = [L.Slot(L.ctl_mask(direction, i, key), 0, p) for i, p in enumerate(ctl)]
        retry = f" try {self._tries + 1}" if body[0] in (F.CONNECT, F.DISC) else ""
        log.info("TX %s %s x%d%s", _frame_desc(body), mode, len(slots), retry)
        return L.TxBurst(mode, slots, 0)

    def _connect_burst(self) -> L.TxBurst:
        body = (bytes([F.CONNECT, VERSION]) + F.pack_call(self.call) + F.pack_call(self.peer)
                + self._nonce.to_bytes(2, "big") + bytes([self.cap, round(self.t_turn * 10)]))
        return self._session_burst(0, 0, body)

    def _disc_burst(self) -> L.TxBurst:
        st = self.station
        return self._session_burst(st.direction, st.key, bytes([F.DISC]))

    def _session_frame(self, rx) -> dict | None:
        """Decode rx as a session frame addressed to this station, if it is one.
        What each state accepts:
          station exists   DISC / DISC_ACK from the peer under the session key
          callee           a repeated CONNECT (its CONNECT_ACK was lost), key 0
          LISTEN           CONNECT, key 0
          CONNECTING       CONNECT_ACK / CONNECT_NAK, key 0"""
        tries = []
        if self.station is not None:
            tries.append((self.station.peer, self.station.key))
            if not self.station.master:
                tries.append((0, 0))
        elif self.state == LISTEN:
            tries.append((0, 0))
        elif self.state == CONNECTING:
            tries.append((1, 0))
        for direction, key in tries:
            first = rx.decode(0, L.ctl_mask(direction, 0, key), 0, None)
            if first is None:
                continue
            core = F.Core.unpack(first)
            if core.ftype != F.SESSION:
                return None
            payloads = [first]
            for i in range(1, core.n_ctl):
                p = rx.decode(i, L.ctl_mask(direction, i, key), 0, None)
                if p is None:
                    return None
                payloads.append(p)
            try:
                body = F.Control.unpack(payloads).ext.get(T_SESS, b"")
            except ValueError:
                return None
            return {"key": key, "body": body, "mode": rx.submode} if body else None
        return None

    def _on_session_frame(self, f: dict, now: float):
        body, key = f["body"], f["key"]
        sub = body[0]
        log.info("RX %s %s", _frame_desc(body), f["mode"])
        if sub == F.CONNECT and len(body) >= 22 and key == 0:
            caller, callee = F.unpack_call(body[2:10]), F.unpack_call(body[10:18])
            nonce = int.from_bytes(body[18:20], "big")
            if callee != self.call.upper() and callee not in self.aliases:
                return
            if self.station is not None and nonce == self._nonce:
                self._answer_mode = f["mode"]
                self._queue(self._accept_burst(), now)  # our ACK was lost: say it again
                return
            if self.state != LISTEN:
                return
            if body[1] != VERSION:
                self._queue(self._session_burst(1, 0, bytes([F.CONNECT_NAK]) + body[18:20] + b"\x01", f["mode"]), now)
                return
            self.call = callee  # the call it dialed (the session key derives from it)
            self.peer, self._nonce, self.cap = caller, nonce, min(body[20], self.cap)
            self.station = L.Station(1, self.policy, master=False, key=session_key(caller, self.call, nonce),
                                     cap=self.cap, max_misses=None, chat=self.chat)
            self._flush_writes()
            self._connected(now)
            self.events.append(f"CONNECTED {caller}")
            self._answer_mode = f["mode"]
            self._queue(self._accept_burst(), now)
        elif sub == F.CONNECT_ACK and self.state == CONNECTING and key == 0:
            if int.from_bytes(body[1:3], "big") != self._nonce:
                return
            self.cap = body[3]
            self.station = L.Station(0, self.policy, master=True, key=session_key(self.call, self.peer, self._nonce),
                                     cap=self.cap, max_misses=None, chat=self.chat)
            self._flush_writes()
            self._connected(now)
            self.events.append(f"CONNECTED {self.peer}")
            self._queue(self.station.build(), now)
        elif sub == F.CONNECT_NAK and self.state == CONNECTING and key == 0:
            if int.from_bytes(body[1:3], "big") == self._nonce:
                self._close("refused")
        elif sub == F.DISC and self.station is not None and key == self.station.key:
            # also when already closed: our DISC_ACK may have been lost
            self._heard(now)
            self._close("disconnected by peer",
                        final=self._session_burst(self.station.direction, key, bytes([F.DISC_ACK]), f["mode"]), now=now)
        elif sub == F.DISC_ACK and self.state == DISCONNECTING and key == self.station.key:
            self._close("disconnected")

    def _accept_burst(self) -> L.TxBurst:
        body = bytes([F.CONNECT_ACK]) + self._nonce.to_bytes(2, "big") + bytes([self.cap, round(self.t_turn * 10)])
        return self._session_burst(1, 0, body, self._answer_mode)

    def _flush_writes(self):
        if self._pending_write:
            self.station.write(bytes(self._pending_write))
            self._pending_write.clear()


FRAME_NAMES = {F.CONNECT: "CONNECT", F.CONNECT_ACK: "CONNECT_ACK", F.CONNECT_NAK: "CONNECT_NAK",
               F.DISC: "DISC", F.DISC_ACK: "DISC_ACK"}


def _frame_desc(body: bytes) -> str:
    out = FRAME_NAMES.get(body[0], f"session frame {body[0]}")
    if body[0] == F.CONNECT and len(body) >= 18:
        out += f" {F.unpack_call(body[2:10])}>{F.unpack_call(body[10:18])}"
    return out
