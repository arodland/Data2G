"""ARQ sessions on a simulated clock: connect, transfer both ways,
disconnect, with loss and dead links. Invariants: delivered bytes are
always an exact prefix, the two stations never key over each other, and
every run ends (docs/arq.md §6, §7, §10)."""

import random

import pytest

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

    def airtime(self, m, n_cw):
        return 0.4 + 0.12 * n_cw


def run(seed, p_burst=0.0, p_cw=0.0, n_a=2000, n_b=800, die_at=None, horizon=3000.0, ack_loss_first=0):
    rng = random.Random(seed)
    a = S.Session("W1AW", Policy(random.Random(seed + 1)), rng=random.Random(seed + 2))
    b = S.Session("K2XYZ-7", Policy(random.Random(seed + 3)), rng=random.Random(seed + 4))
    data_a = bytes(rng.randrange(256) for _ in range(n_a))
    data_b = bytes(rng.randrange(256) for _ in range(n_b))
    b.listen()
    a.write(data_a)
    b.write(data_b)
    a.connect("K2XYZ-7", 2, 0.0)
    stores = {id(a): {}, id(b): {}}
    stats = {"mismatch": 0, "collisions": 0, "bursts": 0}
    air = []  # (start, end, sender) of every transmission
    events = []  # (time, kind, target, payload)
    got_a, got_b = bytearray(), bytearray()
    disconnect_asked = False
    t = 0.0
    lost_first = ack_loss_first
    while t < horizon:
        for me, other in ((a, b), (b, a)):
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
            stats["bursts"] += 1
            events.append((end, "txend", me, burst))
            lost = (die_at is not None and start >= die_at) or rng.random() < p_burst
            if me is b and lost_first > 0:
                lost, lost_first = True, lost_first - 1
            if not lost:
                events.append((start + HEADER_S, "header", other, burst))
                events.append((end + DECODE_S, "rx", other, burst))
        got_a += a.read()
        got_b += b.read()
        assert bytes(got_b) == data_a[:len(got_b)]
        assert bytes(got_a) == data_b[:len(got_a)]
        if got_a == data_b and got_b == data_a and not disconnect_asked:
            a.disconnect()
            disconnect_asked = True
        if a.state == S.CLOSED and b.state == S.CLOSED and not events and a._out is None and b._out is None:
            break
        nxt = [e[0] for e in events] + [x for x in (a.next_event(), b.next_event()) if x is not None]
        if not nxt:
            break
        t = max(t, min(nxt))
        due = sorted([e for e in events if e[0] <= t], key=lambda e: e[0])
        events = [e for e in events if e[0] > t]
        for when, kind, who, burst in due:
            if kind == "txend":
                who.on_tx_end(burst, when)
            elif kind == "header":
                who.on_header(burst.submode, len(burst.slots), when)
            else:
                who.on_rx(FakeRx(burst, rng, p_cw, stores[id(who)], stats), when)
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
