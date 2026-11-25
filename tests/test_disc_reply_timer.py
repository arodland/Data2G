"""DISC reply timing with real mode lengths and no oracle carrier deferral."""
import random

import pytest

from data2g.arq import policy as G
from data2g.arq import session as S
from test_arq import FakeRx


class FixedMode(G.GearShifter):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def connect_mode(self, cap, tries=0, last=False):
        return self.mode

    def choose(self, station, escalation):
        return self.mode, 2


@pytest.fixture(params=["python", "native"])
def session_type(request, pure):
    return S.Session if request.param == "python" else request.getfixturevalue("native").arq.Session


def heard(burst):
    return FakeRx(burst, random.Random(7), 0.0, {}, {"mismatch": 0})


def ready(session, now):
    for _ in range(50):
        burst = session.poll(now)
        if burst is not None:
            return now, burst
        event = session.next_event()
        assert event is not None
        now = max(now + .001, event)
    pytest.fail("no transmission became ready")


def pair(session_type, mode):
    policies = [FixedMode("qpsk-r1/5"), FixedMode("qpsk-r1/5")]
    a, b = [session_type(call, p, rng=random.Random(i + 1))
            for i, (call, p) in enumerate(zip(("W1AW", "K2XYZ"), policies))]
    b.listen()
    a.connect("K2XYZ", 2, 0.0)
    now = 0.0
    for _ in range(50):
        if a.state == b.state == S.CONNECTED and all(
                s._out is None and s._deadline is None and s.queued_bytes() == (0, 0) for s in (a, b)):
            for policy in policies:
                policy.mode = mode
            return a, b, now
        for src, dst in ((a, b), (b, a)):
            burst = src.poll(now)
            if burst is None:
                continue
            end = now + src.policy.airtime(burst.submode, len(burst.slots))
            src.on_tx_end(burst, end)
            dst.on_header(burst.submode, len(burst.slots), now)
            dst.on_rx(heard(burst), end)
            now = end + max(src.t_turn, dst.t_turn)
            break
        else:
            events = [s.next_event() for s in (a, b) if s.next_event() is not None]
            assert events
            now = max(now + .001, min(events))
    pytest.fail("setup did not reach idle")


MODES = ("qpsk-r1/5", "n10-qpsk-r1/3", G.ROBUST_CONNECT, G.ROBUST_FAST, "ack-4f")


def lost_ack(session_type, mode, by, ack_delay=0.0):
    """A missing reply header must not let either side retry over the ACK.

    Unlike the general session simulator, do not defer a transmission using
    knowledge of the other station's on-air burst: this ACK is unheard.
    """
    sessions = pair(session_type, mode)
    src, dst, now = sessions[by], sessions[1 - by], sessions[2]
    src.disconnect()
    start, disc = ready(src, now)
    end = start + src.policy.airtime(disc.submode, len(disc.slots))
    src.on_tx_end(disc, end)
    dst.on_rx(heard(disc), end)
    ack_start, ack = ready(dst, end + src.t_turn + ack_delay)
    ack_end = ack_start + dst.policy.airtime(ack.submode, len(ack.slots))
    dst.on_tx_end(ack, ack_end)
    # Erase this ACK completely at src: neither on_header nor on_rx runs.
    assert src.poll(ack_end - .001) is None
    assert src.state == S.DISCONNECTING
    retry_start, retry = ready(src, ack_end)
    assert ack_end <= retry_start <= end + src.t_turn + S.REPLY_START_S + src.policy.airtime(disc.submode, len(disc.slots)) + .001
    retry_end = retry_start + src.policy.airtime(retry.submode, len(retry.slots))
    src.on_tx_end(retry, retry_end)
    dst.on_rx(heard(retry), retry_end)  # the closed peer re-ACKs the same key
    ack_start, ack = ready(dst, retry_end + src.t_turn)
    ack_end = ack_start + dst.policy.airtime(ack.submode, len(ack.slots))
    dst.on_tx_end(ack, ack_end)
    src.on_header(ack.submode, len(ack.slots), ack_start)
    src.on_rx(heard(ack), ack_end)
    assert src.state == dst.state == S.CLOSED
    assert src.close_reason == "CLIENT" and dst.close_reason == "PEER"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("by", (0, 1))
def test_lost_disc_ack_finishes_before_retry(session_type, mode, by):
    lost_ack(session_type, mode, by)


@pytest.mark.parametrize("by", (0, 1))
def test_late_ofdm_ack_finishes_before_retry(session_type, by):
    # The OFDM header could still arrive in the existing start window, but
    # the full reply extends past it. No header is reported in this case.
    lost_ack(session_type, "qpsk-r1/5", by, ack_delay=1.0)


def absent_replies(src, now):
    """No reply to any DISC try. -> (first try keyed, close time, tries sent,
    the last try's start and end)."""
    src.disconnect()
    start, burst = ready(src, now)
    first, tries = start, 0
    while burst is not None:
        tries += 1
        end = start + src.policy.airtime(burst.submode, len(burst.slots))
        src.on_tx_end(burst, end)
        deadline = src.next_event()
        hold = src.policy.airtime(burst.submode, len(burst.slots))
        assert end < deadline <= end + src.t_turn + S.REPLY_START_S + hold + .001
        burst = src.poll(deadline)
        if burst is not None:
            assert src.state == S.DISCONNECTING
            start = deadline
    assert src.state == S.CLOSED and src.close_reason == "CLIENT unconfirmed"
    return first, deadline, tries, start, end


@pytest.mark.parametrize("mode", (G.ROBUST_CONNECT, G.ROBUST_FAST, "qpsk-r1/5"))
@pytest.mark.parametrize("by", (0, 1))
def test_absent_disc_replies_close_within_the_budget(session_type, mode, by):
    """Unanswered, the DISC tries end within DISC_BUDGET_S of the first
    (BPQ32 sends ABORT 30 s after its DISCONNECT, #215), and a try is given
    up only when it would not fit: after the last reply window, the hold for
    the ACK, the retry's airtime and its own reply window overrun the budget.
    The counts depend on the profile's mode lengths (HF: below)."""
    sessions = pair(session_type, mode)
    src = sessions[by]
    first, closed, tries, last_start, last_end = absent_replies(src, sessions[2])
    assert closed - first <= S.DISC_BUDGET_S
    if tries < S.DISC_TRIES:
        window = closed - last_end
        air = last_end - last_start
        assert closed + air + air + window > first + S.DISC_BUDGET_S


@pytest.mark.hf_only("HF's mode lengths: ROBUST_CONNECT 4.98 s, ROBUST_FAST 3.62 s")
@pytest.mark.parametrize("mode", (G.ROBUST_CONNECT, G.ROBUST_FAST, "qpsk-r1/5"))
@pytest.mark.parametrize("by", (0, 1))
def test_absent_disc_tries_per_mode(session_type, mode, by):
    """At t_turn 1 s, a DISC whose every try is in ROBUST_CONNECT (4.98 s)
    gets two (a third would end at 35.4 s); in ROBUST_FAST (3.62 s) or
    qpsk-r1/5 all three."""
    sessions = pair(session_type, mode)
    tries = absent_replies(sessions[by], sessions[2])[2]
    assert tries == (2 if mode == G.ROBUST_CONNECT else S.DISC_TRIES)


@pytest.mark.parametrize("by", (0, 1))
def test_the_default_disc_ladder_keeps_three_tries_within_the_budget(session_type, by):
    """GearShifter's ladder (the cap's connect mode, then ROBUST_CONNECT):
    all DISC_TRIES tries, the ACK waited out before each retry, inside the
    budget (28.7 s at t_turn 1 s)."""
    sessions = pair(session_type, "qpsk-r1/5")
    src = sessions[by]
    src.policy.connect_mode = lambda cap, tries=0, last=False: G.GearShifter.connect_mode(src.policy, cap, tries, last)
    first, closed, tries, *_ = absent_replies(src, sessions[2])
    assert tries == S.DISC_TRIES and closed - first <= S.DISC_BUDGET_S


@pytest.mark.parametrize("mode", MODES)
def test_heard_disc_ack_closes_at_once(session_type, mode):
    a, b, now = pair(session_type, mode)
    a.disconnect()
    start, disc = ready(a, now)
    end = start + a.policy.airtime(disc.submode, len(disc.slots))
    a.on_tx_end(disc, end)
    b.on_rx(heard(disc), end)
    start, ack = ready(b, end + a.t_turn)
    ack_end = start + b.policy.airtime(ack.submode, len(ack.slots))
    a.on_header(ack.submode, len(ack.slots), start)
    assert a.poll(ack_end - .001) is None
    a.on_rx(heard(ack), ack_end)
    assert a.close_reason == "CLIENT" and a.state == b.state == S.CLOSED
    assert a.next_event() is None
