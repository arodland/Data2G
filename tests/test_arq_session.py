"""ARQ sessions on a simulated clock: connect, transfer both ways,
disconnect, with loss and dead links. Invariants: delivered bytes are
always an exact prefix, the two stations never key over each other, and
every run ends (docs/arq.md §6, §7, §10)."""

import random

import pytest

from data2g.arq import frames as F
from data2g.arq import link as L
from data2g.arq import session as S

from test_arq import MODES, FakeRx

PTT_S, HEADER_S, DECODE_S = 0.1, 0.25, 0.2


class Policy(L.Policy):
    def __init__(self, rng, change=0.1):
        self.rng, self.change, self.mode = rng, change, "m22"

    def choose(self, station, escalation):
        if escalation:
            return "m46", 2
        if self.rng.random() < self.change:
            self.mode = self.rng.choice(list(MODES))
        return self.mode, self.rng.randint(1, 12)

    def payload_bytes(self, m):
        return MODES[m][0]

    def rv_cycle(self, m):
        return MODES[m][1]

    def connect_mode(self, cap, tries=0):
        return "m46"  # a connect frame (28 B) in one control codeword

    def airtime(self, m, n_cw, dup=False):
        return 0.4 + 0.12 * n_cw


def run(seed, p_burst=0.0, p_cw=0.0, n_a=2000, n_b=800, die_at=None, horizon=3000.0, ack_loss_first=0,
        b_write_at=None, chat=False, disc_at=None, disc_by="a"):
    """`b_write_at`: the callee writes its data then (idle by then) instead
    of up front; `ack_loss_first` then loses its first bursts from then on.
    `disc_by` ("a" or "b") disconnects once all data is in, not before `disc_at`."""
    rng = random.Random(seed)
    a = S.Session("W1AW", Policy(random.Random(seed + 1)), rng=random.Random(seed + 2))
    b = S.Session("K2XYZ-7", Policy(random.Random(seed + 3)), rng=random.Random(seed + 4))
    data_a = bytes(rng.randrange(256) for _ in range(n_a))
    data_b = bytes(rng.randrange(256) for _ in range(n_b))
    a.set_chat(chat)
    b.set_chat(chat)
    b.listen()
    a.write(data_a)
    if b_write_at is None:
        b.write(data_b)
    a.connect("K2XYZ-7", 2, 0.0)
    stores = {id(a): {}, id(b): {}}
    stats = {"mismatch": 0, "collisions": 0, "bursts": 0}
    air = []  # (start, end, sender) of every transmission
    events = []  # (time, kind, target, payload)
    # as data2g.arq.engine: no poll (nothing built) while the other's burst is
    # on air or pending decode; a burst built earlier and deferred past it
    # could act on a stale ACK
    held = {id(a): set(), id(b): set()}
    got_a, got_b = bytearray(), bytearray()
    disconnect_asked = False
    t = 0.0
    lost_first = ack_loss_first
    while t < horizon:
        if b_write_at is not None and t >= b_write_at and stats.get("b_written") is None:
            b.write(data_b)
            stats["b_written"] = t
        for me, other in ((a, b), (b, a)):
            if held[id(me)]:
                continue
            burst = me.poll(t)
            if burst is None:
                continue
            start = t + PTT_S
            busy_until = max([e for s, e, who in air[-4:] if who is other and e > start], default=None)
            if busy_until is not None:  # listen before talk: defer past their carrier
                start = busy_until + 0.05
            end = start + me.policy.airtime(burst.submode, len(burst.slots))
            if any(s < end and start < e for s, e, _ in air[-4:]):
                stats["collisions"] += 1
            air.append((start, end, me))
            held[id(other)].add(id(burst))
            stats["bursts"] += 1
            events.append((end, "txend", me, burst))
            lost = (die_at is not None and start >= die_at) or rng.random() < p_burst
            if me is b and lost_first > 0 and start >= (b_write_at or 0.0):
                lost, lost_first = True, lost_first - 1
            if not lost:
                events.append((start + HEADER_S, "header", other, burst))
                events.append((end + DECODE_S, "rx", other, burst))
        got_a += a.read()
        got_b += b.read()
        if got_a == data_b and "b_done" not in stats and stats.get("b_written") is not None:
            stats["b_done"] = t
        assert bytes(got_b) == data_a[:len(got_b)]
        assert bytes(got_a) == data_b[:len(got_a)]
        if got_a == data_b and got_b == data_a and not disconnect_asked and t >= (disc_at or 0.0):
            (a if disc_by == "a" else b).disconnect()
            disconnect_asked = True
        if a.state == S.CLOSED and b.state == S.CLOSED and not events and a._out is None and b._out is None:
            break
        nxt = [e[0] for e in events] + [x for x in (s.next_event() for s in (a, b) if not held[id(s)]) if x is not None]
        if b_write_at is not None and stats.get("b_written") is None:
            nxt.append(b_write_at)
        if disc_at is not None and not disconnect_asked:
            nxt.append(disc_at)
        if not nxt:
            break
        t = max(t, min(nxt))
        due = sorted([e for e in events if e[0] <= t], key=lambda e: e[0])
        events = [e for e in events if e[0] > t]
        for when, kind, who, burst in due:
            if kind == "txend":
                who.on_tx_end(burst, when)
                if not any(e[1] == "rx" and e[3] is burst for e in events):
                    held[id(a)].discard(id(burst))  # lost: nothing to wait for
                    held[id(b)].discard(id(burst))
            elif kind == "header":
                who.on_header(burst.submode, len(burst.slots), when)
            else:
                who.on_rx(FakeRx(burst, rng, p_cw, stores[id(who)], stats), when)
                held[id(who)].discard(id(burst))
    return dict(a=a, b=b, got_a=bytes(got_a), got_b=bytes(got_b), data_a=data_a, data_b=data_b, t=t, **stats)


@pytest.mark.parametrize("seed", range(20))
def test_connect_transfer_disconnect(seed):
    p_burst = [0.0, 0.05, 0.15, 0.25][seed % 4]
    p_cw = [0.0, 0.1, 0.2][seed % 3]
    r = run(seed, p_burst, p_cw)
    assert r["collisions"] == 0 and r["mismatch"] == 0
    assert r["got_b"] == r["data_a"] and r["got_a"] == r["data_b"], (r["a"].close_reason, r["b"].close_reason)
    assert r["a"].state == S.CLOSED
    assert r["a"].events[0] == "CONNECTED K2XYZ-7" and r["b"].events[0] == "CONNECTED W1AW"


def test_lost_connect_ack_is_repeated():
    r = run(7, ack_loss_first=2)
    assert r["got_b"] == r["data_a"] and r["got_a"] == r["data_b"]


def test_redial_replaces_a_dead_callee_session():
    """The caller lost my CONNECT_ACK, gave up and dialed again (a new
    nonce): I drop the session it abandoned and answer the new one, instead
    of ignoring it until link lost (on air, recordings/20261002-232711)."""
    def heard(burst):
        return FakeRx(burst, random.Random(0), 0.0, {}, {"mismatch": 0})

    b = S.Session("K2XYZ", Policy(random.Random(3)), rng=random.Random(4))
    b.listen()
    a1 = S.Session("W1AW", Policy(random.Random(1)), rng=random.Random(2))
    a1.connect("K2XYZ", 2, 0.0)
    b.on_rx(heard(a1.poll(0.0)), 1.0)
    assert b.state == S.CONNECTED and b.poll(1.0) is not None  # its CONNECT_ACK, lost
    a2 = S.Session("W1AW", Policy(random.Random(5)), rng=random.Random(6))
    a2.connect("K2XYZ", 2, 40.0)
    assert a2._nonce != a1._nonce
    b.on_rx(heard(a2.poll(40.0)), 41.0)
    ack = b.poll(41.0)
    assert ack is not None and b.events[-2:] == ["DISCONNECTED PEER_RECONNECTED", "CONNECTED W1AW"]
    a2.on_rx(heard(ack), 42.0)
    assert a2.state == S.CONNECTED and a2.station.key == b.station.key


@pytest.mark.parametrize("seed", range(5))
def test_dead_link_closes_both_within_bound(seed):
    r = run(50 + seed, n_a=50000, die_at=20.0)
    assert r["a"].state == S.CLOSED and r["b"].state == S.CLOSED
    assert "LINK" in r["a"].close_reason and "LINK" in r["b"].close_reason
    assert r["t"] <= 20.0 + S.LINK_LOST_S + 5
    assert r["collisions"] == 0


def test_connect_retries_in_the_robust_mode_and_is_answered_in_it():
    """A lost CONNECT is retried in ROBUST_CONNECT; the callee answers in
    the mode the CONNECT came in (MPP -4 dB: 9 of 12 loss-study sessions
    never connected when every try went in qpsk-r1/5)."""
    from data2g.arq import policy as G

    class Good(G.GearShifter):  # recommends a real data mode, not the fallback's 4 B
        def recommend(self, station):
            return G.encode("qpsk-r1/2"), 1, G.encode("qpsk-r1/2")

    a = S.Session("W1AW", Good(), rng=random.Random(1))
    b = S.Session("K2XYZ", Good(), rng=random.Random(2))
    b.listen()
    a.connect("K2XYZ", 2, 0.0)
    first = a.poll(0.0)
    assert first.submode == G.CONNECT[2]
    a.on_tx_end(first, 2.0)  # lost: nobody hears it
    t, retry = 2.0, None
    while retry is None and t < 60:
        t = max(t + 0.1, a.next_event() or t)
        retry = a.poll(t)
    assert retry is not None and retry.submode == G.ROBUST_CONNECT
    a.on_tx_end(retry, t + 4.2)
    b.on_rx(FakeRx(retry, random.Random(3), 0.0, {}, {"mismatch": 0}), t + 4.5)
    ack = b.poll(t + 4.5)
    assert b.state == S.CONNECTED and ack.submode == G.ROBUST_CONNECT


def test_a_refusal_sent_leaves_the_reply_timer_alone():
    """A NAK BUSY sent while dialing is not the session's burst: its end
    arms nothing. on_tx_end knows it by identity, so the burst handed back
    must be the one poll gave (the native binding once rebuilt it)."""
    from data2g.arq import policy as G

    a = S.Session("W1AW", G.GearShifter(), rng=random.Random(1))
    c = S.Session("N0CALL", G.GearShifter(), rng=random.Random(2))
    a.connect("K2XYZ", 2, 0.0)
    a.on_tx_end(a.poll(0.0), 2.0)
    deadline = a.next_event()
    c.connect("W1AW", 2, 2.1)
    a.on_rx(FakeRx(c.poll(2.1), random.Random(3), 0.0, {}, {"mismatch": 0}), 2.5)
    nak = a._out
    assert nak is not None and a.poll(2.5) is nak
    a.on_tx_end(nak, 3.0)
    assert a.next_event() == deadline


def test_callee_breaks_idle():
    """Data the callee writes on an idle link goes in its own wake burst, not
    at the caller's next keepalive (KEEPALIVE_S[0] or more away)."""
    r = run(3, n_a=200, n_b=40, b_write_at=120.0)
    assert r["got_a"] == r["data_b"] and r["collisions"] == 0
    assert r["b_done"] - r["b_written"] < S.KEEPALIVE_S[0] - 5


@pytest.mark.parametrize("by", ["a", "b"])
def test_disconnect_breaks_idle(by):
    """Either side's DISCONNECT on an idle link goes out at once (the
    callee's as a wake), not at the caller's next keepalive (up to ~32 s
    late at these times before)."""
    for at in range(100, 160, 4):
        r = run(3, n_a=200, n_b=40, disc_at=at, disc_by=by)
        assert r["a"].state == S.CLOSED and r["b"].state == S.CLOSED and r["collisions"] == 0
        assert r["a"].close_reason.startswith(("CLIENT", "PEER")) and r["b"].close_reason.startswith(("CLIENT", "PEER"))
        assert r["t"] - at < 8, at


@pytest.mark.parametrize("lost", [1, 2, 3])
def test_lost_wake_is_recovered(lost):
    """Lost wakes (the repeat too, with 2 or more): the caller's keepalive
    still collects the data, and the callee never keys over the caller."""
    r = run(3, n_a=200, n_b=40, b_write_at=120.0, ack_loss_first=lost)
    assert r["got_a"] == r["data_b"] and r["got_b"] == r["data_a"]
    assert r["collisions"] == 0
    assert r["b_done"] - r["b_written"] < S.KEEPALIVE_S[1] + 10


def test_chat_wakes_back_off():
    """CHAT ON: the callee keeps waking (CSMA-like backoff) through 4 lost
    wakes, and its line arrives within that retry budget."""
    r = run(3, n_a=200, n_b=40, b_write_at=120.0, ack_loss_first=4, chat=True)
    assert r["got_a"] == r["data_b"] and r["collisions"] == 0
    guard = 1.0 + S.REPLY_START_S + S.WAKE_GUARD_S
    budget = sum(guard + S.WAKE_JITTER_S * 2 ** k + 1.0 for k in range(5))  # + airtime
    assert r["b_done"] - r["b_written"] < budget + 5


@pytest.mark.parametrize("chat", [False, True])
def test_short_write_coalesces(chat):
    """CHAT OFF: short writes to an idle caller hold its idle poll
    COALESCE_S for more; a full burst's worth, or CHAT ON, sends at once."""
    from data2g.arq import policy as G

    class Good(G.GearShifter):  # recommends a real data mode, not the fallback's 4 B
        def recommend(self, station):
            return G.encode("qpsk-r1/2"), 1, G.encode("qpsk-r1/2")

    a = S.Session("W1AW", G.GearShifter(), rng=random.Random(1))
    b = S.Session("K2XYZ", Good(), rng=random.Random(2))
    a.set_chat(chat)
    b.listen()
    a.connect("K2XYZ", 2, 0.0)
    t, src, dst = 0.0, a, b
    while a._build_at is None:  # CONNECT, accept, then empty bursts until the caller idles
        assert t < 30
        burst = src.poll(t)
        if burst is None:
            t = max(t + 0.1, src.next_event() or t)
            continue
        src.on_tx_end(burst, t + 1.0)
        t += 1.5
        dst.on_rx(FakeRx(burst, random.Random(3), 0.0, {}, {"mismatch": 0}), t)
        src, dst = dst, src
    a.poll(t)  # the session's clock
    cap = a.policy.next_capacity(a.station)
    assert cap > 100
    a.write(b"x" * 30)
    a.write(b"x" * 30)  # inside the hold: it keeps its end
    assert a._build_at == (float("-inf") if chat else t + S.COALESCE_S)
    a.write(b"x" * cap)  # a full burst
    assert a._build_at == float("-inf")


def _idle_link():
    """A connected pair, the caller idling: -> (caller, callee, now, step). step(src, dst, t)
    sends src's next burst (1 s on air, heard 0.5 s after) and returns the time dst heard it."""
    from data2g.arq import policy as G

    class Good(G.GearShifter):  # recommends a real data mode, not the fallback's 4 B
        def recommend(self, station):
            return G.encode("qpsk-r1/2"), 1, G.encode("qpsk-r1/2")

    a = S.Session("W1AW", Good(), rng=random.Random(1))
    b = S.Session("K2XYZ", Good(), rng=random.Random(2))
    b.listen()
    a.connect("K2XYZ", 2, 0.0)

    def step(src, dst, t, drop=False):
        burst = src.poll(t)
        while burst is None:
            assert t < 60
            t = max(t + 0.01, src.next_event() or t)
            burst = src.poll(t)
        src.on_tx_end(burst, t + 1.0)
        t += 1.5
        if not drop:
            dst.on_rx(FakeRx(burst, random.Random(3), 0.0, {}, {"mismatch": 0}), t)
        step.last = burst
        return t

    t, src, dst = 0.0, a, b
    while a._build_at is None:  # CONNECT, accept, then empty bursts until the caller idles
        assert t < 30
        t = step(src, dst, t)
        src, dst = dst, src
    return a, b, t, step


@pytest.mark.parametrize("who", ["callee", "caller"])
@pytest.mark.parametrize("answer", [None, 0.0, 0.24])
def test_reply_is_held_for_the_hosts_answer(who, answer):
    """A burst that brings a station's host data and ends the peer's, with
    nothing to send, holds the reply REPLY_HOLD_S: the host's answer (a BBS
    prompt) rides it instead of waiting for a wake (the callee) or another
    turn (the caller, woken from idle here). No answer: a plain ACK at the
    hold's end. `answer`: when in the hold the host writes; the reply never
    goes later than the hold's end. Meanwhile the air is not free for a
    broadcast burst."""
    a, b, t, step = _idle_link()
    asks, holds = (a, b) if who == "callee" else (b, a)
    asks.write(b"LIST")
    t = step(asks, holds, t)
    assert holds._out is None and holds._build_at == t + S.REPLY_HOLD_S and not holds.idle()
    assert holds.read() == b"LIST"
    wait = S.REPLY_HOLD_S
    if answer is not None:
        assert holds.poll(t + answer) is None  # the session's clock, as its host's write finds it
        holds.write(b"prompt>")
        # a short write to a quiet link waits for more: the hold's end is brought forward to that, never put off
        wait = min(answer + S.COALESCE_S, S.REPLY_HOLD_S)
    t2 = step(holds, asks, t + (answer or 0.0))
    assert t + wait - 0.001 <= t2 - 1.5 <= t + wait + 0.011  # step polls every 10 ms
    assert asks.read() == (b"prompt>" if answer is not None else b"")
    if answer is not None and who == "callee":
        # mid-exchange (no idle poll was waiting): the caller holds for its host's answer to the prompt
        assert a._out is None and a._build_at == t2 + S.REPLY_HOLD_S and not a.idle()


def test_caller_in_a_recovery_answers_at_once():
    """The hold is not for a recovery: a burst that says the callee's reply
    was lost is answered at once though it brings data and ends the callee's."""
    a, b, t, step = _idle_link()
    t = step(a, b, t)  # the caller's idle poll
    t = step(b, a, t, drop=True)  # its answer, lost
    b.write(b"LIST")
    t = step(a, b, t)  # the caller's repeat
    t = step(b, a, t)
    assert a.station.peer_reply_lost and a.read() == b"LIST"
    assert a._out is not None and a._build_at is None and not a.idle()


def test_a_second_burst_leaves_the_callers_hold_standing():
    """A burst heard again while the caller holds its reply (a duplicate
    decode, a late nod) neither answers it at once nor drops it for an idle
    poll: the held reply goes when it was due."""
    a, b, t, step = _idle_link()
    b.write(b"LIST")
    t = step(b, a, t)
    due = a._build_at
    assert due == t + S.REPLY_HOLD_S and a.read() == b"LIST"
    heard = a.station.stats["rx_ok"]
    a.on_rx(FakeRx(step.last, random.Random(3), 0.0, {}, {"mismatch": 0}), t + 0.1)
    assert a.station.stats["rx_ok"] == heard + 1
    assert a._out is None and a._build_at == due and not a.idle()
    assert step(a, b, t + 0.1) - 1.5 == pytest.approx(due, abs=0.011)


def _frame(body, direction):
    """A CRC-valid session frame (key 0) carrying `body`."""
    ctl = F.Control(F.Core(ftype=F.SESSION), {S.T_SESS: body}).pack(46)
    burst = L.TxBurst("m46", [L.Slot(L.ctl_mask(direction, i, 0), 0, p) for i, p in enumerate(ctl)], 0)
    return FakeRx(burst, random.Random(0), 0.0, {}, {"mismatch": 0})


@pytest.mark.parametrize("bad", ["short ack", "ack cap", "callsign"])
def test_malformed_session_frame_is_dropped(bad):
    """A CRC-valid but malformed session frame is dropped as if it had not
    decoded (the peer retries), never raised."""
    a = S.Session("W1AW", Policy(random.Random(1)), rng=random.Random(2))
    b = S.Session("K2XYZ", Policy(random.Random(3)), rng=random.Random(4))
    a.connect("K2XYZ", 2, 0.0)
    b.listen()
    nonce = a._nonce.to_bytes(2, "big")
    if bad == "callsign":
        body = bytes([F.CONNECT, S.VERSION]) + b"\xff" * 8 + F.pack_call("K2XYZ") + nonce + b"\x02\x0a"
        b.on_rx(_frame(body, 0), 1.0)
        assert b.state == S.LISTEN and b.poll(1.0) is None
    else:
        body = bytes([F.CONNECT_ACK]) + nonce + (b"" if bad == "short ack" else b"\x07\x0a")
        a.on_rx(_frame(body, 1), 1.0)
        assert a.state == S.CONNECTING and a.station is None


def test_compact_connect_over_the_cpm_modem():
    """ROBUST_CONNECT is CPM: its one 20 B control codeword can't hold a
    CONNECT's Control (28 B), so the retry goes compact under its own mask,
    and the callee, hearing it through the real modem, connects and answers
    in it."""
    import numpy as np

    from data2g import cpm
    from data2g.arq import frames as F
    from data2g.arq import phy as PHY
    from data2g.arq import link as L
    from data2g.arq import policy as G
    from data2g.arq.modes import MODES as REAL

    a = S.Session("VE3/W1AW-1", G.GearShifter(), rng=random.Random(1), t_turn=2.5)
    b = S.Session("K2XYZ", G.GearShifter(), rng=random.Random(2))
    b.listen()
    a.connect("K2XYZ", 0, 0.0)
    a.on_tx_end(a.poll(0.0), 2.0)  # lost
    t, retry = 2.0, None
    while retry is None and t < 60:
        t = max(t + 0.1, a.next_event() or t)
        retry = a.poll(t)
    assert retry.submode == G.ROBUST_CONNECT and [s.mask_id for s in retry.slots] == [L.COMPACT_CONNECT]
    body = F.unpack_connect(retry.slots[0].payload)
    assert F.unpack_call(body[2:10]) == "VE3/W1AW-1" and body[20:] == bytes([0, 25])

    spec = REAL[retry.submode]
    y = np.concatenate([np.zeros(2000), PHY.tx_audio(retry), np.zeros(2000)])
    y += np.random.default_rng(1).normal(0, 0.05, len(y))
    rx = PHY.ModemRx(cpm.receive(y, cpm.find(cpm.GRIDS[spec.grid], y)), {})
    assert rx.decode(0, L.ctl_mask(0, 0), 0, None) is None  # the plain connect mask misses it
    b.on_rx(rx, t + 6.0)
    ack = b.poll(t + 6.0)
    assert b.state == S.CONNECTED and b.peer == "VE3/W1AW-1" and b.cap == 0
    assert ack.submode == G.ROBUST_CONNECT and len(ack.slots) == 1


def test_compact_connect_round_trip():
    from data2g.arq import frames as F

    body = (bytes([F.CONNECT, S.VERSION]) + F.pack_call("W1AW") + F.pack_call("K2XYZ/P")
            + (0xBEEF).to_bytes(2, "big") + bytes([2, 63]))
    p = F.pack_connect(body)
    assert len(p) == F.COMPACT_BYTES and F.unpack_connect(p) == body
    with pytest.raises(ValueError):
        F.pack_connect(body[:-1] + bytes([64]))  # t_turn over 6.3 s


def test_reply_lost_is_answered_at_once():
    """The callee's data burst is lost and its control-only answer to the
    caller's repeat says so (reply_lost): the caller must answer at once.
    That control-only burst would otherwise look idle and wait out a
    keepalive, though the callee holds data the caller never heard of."""
    a = S.Session("W1AW", Policy(random.Random(1), change=0.0), rng=random.Random(2))
    b = S.Session("K2XYZ-7", Policy(random.Random(3), change=0.0), rng=random.Random(4))
    b.listen()
    a.connect("K2XYZ-7", 2, 0.0)

    def send(me, other, t, drop=False, before_rx=None):
        burst = None
        while burst is None:
            t = me.next_event()
            burst = me.poll(t)
        end = t + me.policy.airtime(burst.submode, len(burst.slots))
        me.on_tx_end(burst, end)
        if before_rx:
            before_rx()
        if not drop:
            other.on_rx(FakeRx(burst, random.Random(0), 0.0, {}, {"mismatch": 0}), end + DECODE_S)
        return burst, end + DECODE_S

    t = 0.0
    for _ in range(3):  # CONNECT, its ACK, and the caller's first ARQ burst
        _, t = send(a, b, t)
        _, t = send(b, a, t)
    assert a.next_event() > t + S.KEEPALIVE_S[0] - 1  # idle: the keepalive poll is armed
    _, t = send(a, b, t, before_rx=lambda: b.write(bytes(500)))
    data, t = send(b, a, t, drop=True)
    assert data.slots and b.station.tx.pending()
    _, t = send(a, b, t)  # the caller's timeout repeat
    reply, t = send(b, a, t)
    assert a.station.peer_reply_lost and a.next_event() < t + 1  # answered now, not at a keepalive


def test_abandon_is_answered_at_once(monkeypatch):
    """The callee's data burst is lost; the caller polls (not a repeat), and
    the callee's escalated answer abandons (a submode change) in a
    control-only burst. The caller must answer at once: the callee holds
    re-sliced data, and idling left it to a wake ~5 s later (on air,
    2026-10-15: b3 abandon, then the b4 wake, both control only)."""
    class OneCwEscalation(Policy):
        def choose(self, station, escalation):
            return ("m46", 1) if escalation else ("m22", 6)

    monkeypatch.setattr(S, "REPEAT_MAX_S", 0.0)  # every timeout polls
    a = S.Session("W1AW", OneCwEscalation(random.Random(1), change=0.0), rng=random.Random(2))
    b = S.Session("K2XYZ-7", OneCwEscalation(random.Random(3), change=0.0), rng=random.Random(4))
    b.listen()
    a.connect("K2XYZ-7", 2, 0.0)

    def send(me, other, t, drop=False, before_rx=None):
        burst = None
        while burst is None:
            t = me.next_event()
            burst = me.poll(t)
        end = t + me.policy.airtime(burst.submode, len(burst.slots))
        me.on_tx_end(burst, end)
        if before_rx:
            before_rx()
        if not drop:
            other.on_rx(FakeRx(burst, random.Random(0), 0.0, {}, {"mismatch": 0}), end + DECODE_S)
        return burst, end + DECODE_S

    t = 0.0
    for _ in range(3):  # CONNECT, its ACK, and the caller's first ARQ burst
        _, t = send(a, b, t)
        _, t = send(b, a, t)
    _, t = send(a, b, t, before_rx=lambda: b.write(bytes(500)))
    data, t = send(b, a, t, drop=True)
    assert data.slots and b.station.tx.pending()
    poll, t = send(a, b, t)  # the caller's timeout poll
    assert len(poll.slots) == 1
    reply, t = send(b, a, t)
    assert len(reply.slots) == 1 and a.station.peer_abandon and not a.station.peer_reply_lost
    assert a.next_event() < t + 1  # answered now, not at a keepalive


def test_a_recovery_is_answered_at_once(monkeypatch):
    """The caller's reply to the callee's data is lost; the caller polls,
    and the callee answers in control only (its reply to a poll, escalated),
    holding its data back. The caller must answer at once, not idle into the
    callee's wake (on air 2026-11-15: a lost nod, then 16-32 s each time)."""
    class OneCwEscalation(Policy):
        def choose(self, station, escalation):
            return ("m46", 1) if escalation else ("m22", 6)

    monkeypatch.setattr(S, "REPEAT_MAX_S", 0.0)  # every timeout polls
    a = S.Session("W1AW", OneCwEscalation(random.Random(1), change=0.0), rng=random.Random(2))
    b = S.Session("K2XYZ-7", OneCwEscalation(random.Random(3), change=0.0), rng=random.Random(4))
    b.listen()
    a.connect("K2XYZ-7", 2, 0.0)

    def send(me, other, t, drop=False, before_rx=None):
        burst = None
        while burst is None:
            t = me.next_event()
            burst = me.poll(t)
        end = t + me.policy.airtime(burst.submode, len(burst.slots))
        me.on_tx_end(burst, end)
        if before_rx:
            before_rx()
        if not drop:
            other.on_rx(FakeRx(burst, random.Random(0), 0.0, {}, {"mismatch": 0}), end + DECODE_S)
        return burst, end + DECODE_S

    t = 0.0
    for _ in range(3):  # CONNECT, its ACK, and the caller's first ARQ burst
        _, t = send(a, b, t)
        _, t = send(b, a, t)
    _, t = send(a, b, t, before_rx=lambda: b.write(bytes(3000)))
    data, t = send(b, a, t)  # heard
    assert data.slots and b.station.tx.pending()
    _, t = send(a, b, t, drop=True)  # the caller's reply, lost
    poll, t = send(a, b, t)  # the caller's timeout poll
    assert len(poll.slots) == 1
    reply, t = send(b, a, t)
    assert len(reply.slots) == 1 and b.station.tx.pending()  # control only, data held back
    assert not a.station.peer_abandon and not a.station.peer_reply_lost
    assert a.next_event() < t + 1  # answered now, not at a keepalive


def test_another_stations_burst_is_foreign_but_my_peers_and_a_connect_are_not():
    """A burst under another session's key, heard while I am connected (its
    first codeword converged: the engine asks only then), is foreign: not
    my peer's loss to score. My peer's burst and a third station's CONNECT
    (to be refused) are not."""
    def heard(burst):
        return FakeRx(burst, random.Random(0), 0.0, {}, {"mismatch": 0})

    def connected(caller, callee, seed):
        a = S.Session(caller, Policy(random.Random(seed), change=0), rng=random.Random(seed + 1))
        b = S.Session(callee, Policy(random.Random(seed + 2), change=0), rng=random.Random(seed + 3))
        b.listen()
        a.connect(callee, 2, 0.0)
        b.on_rx(heard(a.poll(0.0)), 1.0)
        a.on_rx(heard(b.poll(1.0)), 2.0)
        assert a.state == S.CONNECTED == b.state
        a.write(b"data")
        return a, b, a.poll(30.0)  # the caller's first burst

    _, b, mine = connected("W1AW", "K2XYZ", 1)
    _, _, other = connected("N0CALL", "K2ABC", 11)
    assert b.foreign(heard(other))
    assert not b.foreign(heard(mine))
    e = S.Session("N1NEW", Policy(random.Random(21)), rng=random.Random(22))
    e.connect("K2XYZ", 2, 0.0)
    assert not b.foreign(heard(e.poll(0.0)))
    assert not S.Session("K2XYZ", Policy(random.Random(31)), rng=random.Random(32)).foreign(heard(other))
