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


class Policy:
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
        b_write_at=None, chat=False):
    """`b_write_at`: the callee writes its data then (idle by then) instead
    of up front; `ack_loss_first` then loses its first bursts from then on."""
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
        if got_a == data_b and got_b == data_a and not disconnect_asked:
            a.disconnect()
            disconnect_asked = True
        if a.state == S.CLOSED and b.state == S.CLOSED and not events and a._out is None and b._out is None:
            break
        nxt = [e[0] for e in events] + [x for x in (s.next_event() for s in (a, b) if not held[id(s)]) if x is not None]
        if b_write_at is not None and stats.get("b_written") is None:
            nxt.append(b_write_at)
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
    assert ack is not None and b.events[-2:] == ["DISCONNECTED peer reconnected", "CONNECTED W1AW"]
    a2.on_rx(heard(ack), 42.0)
    assert a2.state == S.CONNECTED and a2.station.key == b.station.key


@pytest.mark.parametrize("seed", range(5))
def test_dead_link_closes_both_within_bound(seed):
    r = run(50 + seed, n_a=50000, die_at=20.0)
    assert r["a"].state == S.CLOSED and r["b"].state == S.CLOSED
    assert "link" in r["a"].close_reason and "link" in r["b"].close_reason
    assert r["t"] <= 20.0 + S.LINK_LOST_S + 5
    assert r["collisions"] == 0


def test_connect_retries_in_the_robust_mode_and_is_answered_in_it():
    """A lost CONNECT is retried in ROBUST_CONNECT; the callee answers in
    the mode the CONNECT came in (MPP -4 dB: 9 of 12 loss-study sessions
    never connected when every try went in qpsk-r1/5)."""
    from data2g.arq import policy as G

    a = S.Session("W1AW", G.GearShifter(), rng=random.Random(1))
    b = S.Session("K2XYZ", G.GearShifter(), rng=random.Random(2))
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


def test_callee_breaks_idle():
    """Data the callee writes on an idle link goes in its own wake burst, not
    at the caller's next keepalive (KEEPALIVE_S[0] or more away)."""
    r = run(3, n_a=200, n_b=40, b_write_at=120.0)
    assert r["got_a"] == r["data_b"] and r["collisions"] == 0
    assert r["b_done"] - r["b_written"] < S.KEEPALIVE_S[0] - 5


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
