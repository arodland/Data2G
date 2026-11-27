"""Unconditional T_REPLY retention (docs/arq.md §8a, wire 1.0): T_REPLY is sent
only when it changes (empty T_REPLY = none); omitted, the peer keeps its last
reply recommendation across lost bursts and nods."""

import random

import pytest

from data2g.arq import frames as F
from data2g.arq import link as L

from test_arq import CTL_BYTES, payload
from test_nod import AskingPolicy, FullRx, _ctl


def run(seed, nods=False, max_turns=3000):
    """Lockstep, 15% of bursts lost: -> (a, b, controls read, of them without T_REPLY)."""
    rng = random.Random(seed)
    data_a, data_b = payload(rng, 1500, "mixed"), payload(rng, 100, "mixed")
    a = L.Station(0, AskingPolicy(random.Random(seed + 1)), master=True, nod=nods, nod_peer=nods)
    b = L.Station(1, AskingPolicy(random.Random(seed + 2)), nod=nods, nod_peer=nods)
    a.write(data_a)
    b.write(data_b)
    got_a, got_b, n, bare = bytearray(), bytearray(), 0, 0
    burst, sender = a.build(nod=True), a
    for _ in range(max_turns):
        receiver = b if sender is a else a
        ok = False
        if rng.random() >= 0.15:
            said = sender._said(sender._latest)
            if burst.nod is not None:
                before = receiver.peer_reply_recommend
                ok = receiver.handle(receiver.nod_rx(burst.nod[0]))
                assert ok and receiver.peer_reply_recommend == before
            else:
                ext = _ctl(burst).ext
                ok = receiver.handle(FullRx(burst))
                assert ok
                n += 1
                bare += F.T_REPLY not in ext
            if said != L._UNKNOWN:
                assert receiver.peer_reply_recommend == said, (said, receiver.peer_reply_recommend)
        got_a += a.read()
        got_b += b.read()
        assert a.state != L.FAILED and b.state != L.FAILED, (a.fail_reason, b.fail_reason)
        if got_a == data_b and got_b == data_a and not a.tx.pending() and not b.tx.pending():
            return a, b, n, bare
        if ok:
            burst, sender = receiver.build(nod=True), receiver
            receiver.answered()
        else:
            burst, sender = a.on_timeout(), a
            assert burst is not None, a.fail_reason
    raise AssertionError("stuck")


@pytest.mark.parametrize("nods", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_reply_is_kept_through_losses(seed, nods, monkeypatch):
    monkeypatch.setattr(L, "NOD", True)
    a, b, n, bare = run(seed, nods=nods)
    assert bare > n // 4, (bare, n)  # most recommendations repeat the last


def test_t_reply_only_on_change_from_burst_zero():
    """With unconditional retention, T_REPLY goes on burst 0 when non-None, is
    omitted as soon as the peer acts on it, and changing to None sends empty
    T_REPLY once before returning to bare controls."""
    modes = ("fsk16r25-r1/2",)
    a = L.Station(0, AskingPolicy(random.Random(1), 0.0, modes, p_kept=0.0), master=True)
    b = L.Station(1, AskingPolicy(random.Random(2), 0.0, modes, p_kept=0.0))
    a.policy.reply, b.policy.reply = 3, 7
    a.write(payload(random.Random(3), 4000, "mixed"))
    x = a.build()
    assert _ctl(x).ext[F.T_REPLY] == b"\x03"
    a.on_timeout()  # lost: still unacknowledged
    x = a.on_timeout(allow_repeat=False)
    assert _ctl(x).ext[F.T_REPLY] == b"\x03"
    assert b.handle(FullRx(x)) and b.peer_reply_recommend == 3
    r = b.build()
    b.answered()
    assert _ctl(r).ext[F.T_REPLY] == b"\x07"
    assert a.handle(FullRx(r)) and a.peer_reply_recommend == 7
    # b acted on a's burst carrying 3, so a's next burst omits T_REPLY
    x = a.build()
    a.answered()
    assert F.T_REPLY not in _ctl(x).ext
    assert b.handle(FullRx(x)) and b.peer_reply_recommend == 3
    r = b.build()
    b.answered()
    assert F.T_REPLY not in _ctl(r).ext and a.handle(FullRx(r)) and a.peer_reply_recommend == 7
    # a change to None sends empty T_REPLY once, then omits it
    a.policy.recommend = lambda st: (0, 1, None)
    x = a.build()
    assert _ctl(x).ext[F.T_REPLY] == b""
    assert b.handle(FullRx(x)) and b.peer_reply_recommend is None
    r = b.build()
    b.answered()
    assert a.handle(FullRx(r))
    x = a.build()
    assert F.T_REPLY not in _ctl(x).ext
    assert b.handle(FullRx(x)) and b.peer_reply_recommend is None


def test_ack_control_shrinks_to_one_codeword_once_reply_held():
    """An OFDM ACK mode's 4 B codeword holds Core alone; after each side's first
    T_REPLY is acted on, subsequent ACKs are 1 codeword instead of 2."""
    class Tight(AskingPolicy):
        want_dup = False

        def payload_bytes(self, m):
            return 4

        def ctl_payload_bytes(self, m):
            return 4

        def max_ctl(self, m):
            return 4

        def rv_cycle(self, m):
            return 4

    a = L.Station(0, Tight(random.Random(1), 0.0, ("ack-4f",), p_kept=0.0), master=True)
    b = L.Station(1, Tight(random.Random(2), 0.0, ("ack-4f",), p_kept=0.0))
    a.policy.reply = b.policy.reply = 3
    seen, burst, sender, receiver = [], a.build(), a, b
    for _ in range(8):
        ext = F.Control.unpack([x.payload for x in burst.slots]).ext
        seen.append((len(burst.slots), F.T_REPLY in ext))
        assert receiver.handle(FullRx(burst))
        burst, sender, receiver = receiver.build(), receiver, sender
        sender.answered()
    assert seen[:2] == [(2, True), (2, True)]
    assert a.peer_reply_recommend == 3 == b.peer_reply_recommend
    assert seen[2:] == [(1, False)] * 6, seen


CPM, ACK = "fsk16r25-r1/2", "ack-4f"


class Mixed(AskingPolicy):
    want_dup = False
    draw = False

    def choose(self, station, escalation):
        if self.draw and self.rng.random() < 0.5:
            self.mode = self.rng.choice((CPM, ACK))
        return self.mode, 3

    def payload_bytes(self, m):
        return 40 if m == CPM else 4

    def ctl_payload_bytes(self, m):
        return CTL_BYTES if m == CPM else 4

    def max_ctl(self, m):
        return 1 if m == CPM else 4

    def rv_cycle(self, m):
        return 4


def _whole(burst):
    n = F.Core.unpack(burst.slots[0].payload).n_ctl
    return F.Control.unpack([x.payload for x in burst.slots[:n]])


def _pair(mode_a=ACK, mode_b=ACK):
    a = L.Station(0, Mixed(random.Random(1), 0.0, (mode_a,), p_kept=0.0), master=True)
    b = L.Station(1, Mixed(random.Random(2), 0.0, (mode_b,), p_kept=0.0))
    return a, b


def run_mixed(seed, turns=400):
    rng = random.Random(seed)
    a = L.Station(0, Mixed(random.Random(seed + 1), 0.0, (CPM, ACK), p_kept=0.3), master=True, max_misses=None,
                  nod=True, nod_peer=True)
    b = L.Station(1, Mixed(random.Random(seed + 2), 0.0, (CPM, ACK), p_kept=0.3), max_misses=None,
                  nod=True, nod_peer=True)
    a.policy.draw = b.policy.draw = True
    a.write(payload(rng, 3000, "mixed"))
    b.write(payload(rng, 600, "mixed"))
    n = bare = 0
    burst, sender = a.build(nod=True), a
    for _ in range(turns):
        receiver = b if sender is a else a
        ok = False
        if rng.random() >= 0.2:
            said = sender._said(sender._latest)
            if burst.nod is not None:
                ok = receiver.handle(receiver.nod_rx(burst.nod[0]))
            else:
                ext = _whole(burst).ext
                ok = receiver.handle(FullRx(burst))
                n += 1
                bare += F.T_REPLY not in ext
            if ok and said != L._UNKNOWN:
                assert receiver.peer_reply_recommend == said, (said, receiver.peer_reply_recommend, burst.nod)
        a.read()
        b.read()
        assert a.state != L.FAILED and b.state != L.FAILED, (a.fail_reason, b.fail_reason)
        if ok:
            burst, sender = receiver.build(nod=True), receiver
            receiver.answered()
        else:
            burst, sender = a.on_timeout(), a
            assert burst is not None, a.fail_reason
    return a, b, n, bare


@pytest.mark.parametrize("first", range(0, 600, 25))
def test_reply_agrees_through_losses_and_nods(first, monkeypatch):
    monkeypatch.setattr(L, "NOD", True)
    for seed in range(first, first + 25):
        a, b, n, bare = run_mixed(seed)
        assert bare > n // 4, (seed, bare, n)


def _kept_pair():
    """Two CPM stations with reply recommendations established, a's turn next: -> (a, b)."""
    a, b = _pair(CPM, CPM)
    a.policy.reply, b.policy.reply = 3, 7
    burst, sender, receiver = a.build(), a, b
    for _ in range(4):
        assert receiver.handle(FullRx(burst))
        burst, sender, receiver = receiver.build(), receiver, sender
        sender.answered()
    assert sender is a and b.peer_reply_recommend == 3
    return a, b


@pytest.mark.parametrize("size,left", [(7, b"\x05"), (8, b""), (9, None)])
def test_a_shed_t_reply_says_none_where_that_fits(size, left):
    """A changed T_REPLY that an inline codeword leaves no room for is cut to
    none (empty T_REPLY, a byte less) before it is dropped."""
    a, b = _kept_pair()
    a.policy.reply = 5
    a.write(bytes(range(1, size + 1)))
    x = a.build()
    ext = _whole(x).ext
    assert F.T_INLINE in ext and ext.get(F.T_REPLY) == left
    assert b.handle(FullRx(x)) and b.peer_reply_recommend == {b"\x05": 5, b"": None, None: 3}[left]
    assert a._said(a._latest) == {b"\x05": 5, b"": None, None: L._UNKNOWN}[left]
    y = b.build()
    b.answered()
    assert a.handle(FullRx(y))
    assert _whole(a.build()).ext.get(F.T_REPLY) == (None if left == b"\x05" else b"\x05")


def test_primed_hypotheses_leave_t_reply_out_first_once_kept():
    receiver, _ = _kept_pair()

    class Rx:
        submode, n_cw, n_ctl_slots = CPM, 1, 1

    hs = receiver.expect(Rx())
    exts = [F.Control.unpack([t]).ext for t, *_ in hs]
    assert F.T_REPLY not in exts[0] and any(F.T_REPLY in e for e in exts[1:3]), exts
    assert len(hs) <= L.PRIME_TRIES


def test_expected_modes_leave_out_a_shed_reply_mode():
    a, _ = _kept_pair()
    assert "3" in a.expected_modes()  # kept: as said last
    a.policy.reply = 5
    a.write(bytes(range(1, 10)))
    x = a.build()
    assert F.T_INLINE in _whole(x).ext and F.T_REPLY not in _whole(x).ext
    assert "3" in a.expected_modes() and "5" not in a.expected_modes()


def _reply_form(ext):
    return ext.get(F.T_REPLY)


def test_a_held_none_costs_nothing_and_only_a_change_to_none_is_said():
    """Initial None reply costs zero bytes from burst 0; changing from a mode to
    None sends empty T_REPLY once, then omits it."""
    a, b = _pair(CPM, CPM)
    a.policy.reply = b.policy.reply = None
    burst, sender, receiver = a.build(), a, b
    for _ in range(6):
        assert F.T_REPLY not in _whole(burst).ext
        assert receiver.handle(FullRx(burst)) and receiver.peer_reply_recommend is None
        burst, sender, receiver = receiver.build(), receiver, sender
        sender.answered()
    a, b = _kept_pair()
    a.policy.reply = None
    forms = []
    burst, sender, receiver = a.build(), a, b
    for _ in range(6):
        if sender is a:
            forms.append(_reply_form(_whole(burst).ext))
        assert receiver.handle(FullRx(burst))
        if receiver is b:
            assert b.peer_reply_recommend is None
        burst, sender, receiver = receiver.build(), receiver, sender
        sender.answered()
    assert forms == [b"", None, None], forms


def test_primed_hypotheses_follow_the_peers_t_reply():
    class Rx:
        submode, n_cw, n_ctl_slots = CPM, 1, 1

    a, b = _pair(CPM, CPM)
    a.policy.reply, b.policy.reply = 3, 7
    burst, sender, receiver = a.build(), a, b
    missed = [0, 0]
    for turn in range(12):
        if turn == 8:
            sender.policy.reply = None  # its next control: an empty T_REPLY
        if turn >= 2:
            actual = _reply_form(_whole(burst).ext)
            forms = [_reply_form(F.Control.unpack([t]).ext) for t, *_ in receiver.expect(Rx())]
            assert actual in forms, (turn, actual, forms)
            missed[receiver is b] += forms[0] != actual
        assert receiver.handle(FullRx(burst))
        burst, sender, receiver = receiver.build(), receiver, sender
        sender.answered()
    assert a._peer_omits and b._peer_omits
    assert max(missed) <= 2, missed
