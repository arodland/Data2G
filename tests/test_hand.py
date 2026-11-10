"""Hand burst (data2g/hand.py) and the host primitives over it (Engine,
Host): a decoded round trip, a full round between two live stacks, the
command replies and their strict arguments, a round's window (its hold over
my own bursts, the gap its session bursts leave, a head it could not capture,
a slot the radio was busy past), HAND CANCEL, the ref window, which client the
hand lines go to on the command port, and the KISS port's REFS ON clients
(docs/arq.md §9)."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from data2g import hand, hfchannel
from data2g.arq.engine import Engine
from data2g.config import FS, SNR_REF_BW_HZ
from data2g.host import Host
from data2g.kisslink import KissLink

BLOCK = FS // 10
ACK = 0x1234  # the KISS ACKMODE tag of the frame that anchors a window (HAND LISTEN's ack)

sys.path.insert(0, str(Path(__file__).parent))
from test_kiss import frame  # noqa: E402  # an AX.25 UI frame, as the KISS port takes it


def link(a: Engine, b: Engine, snr_db: float, seconds: float, until, seed=0):
    """As tests/test_engine.py: step both engines, each hearing the other's
    last block plus noise, until `until()` or `seconds` of engine time."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt((FS / 2) / SNR_REF_BW_HZ / 10 ** (snr_db / 10))
    a_out = b_out = np.zeros(BLOCK)
    g = 2.2
    for _ in range(int(seconds * FS / BLOCK)):
        a_out, _ = a.step(g * b_out + rng.normal(0, sigma, BLOCK))
        b_out, _ = b.step(g * a_out + rng.normal(0, sigma, BLOCK))
        if until():
            return True
    return False


def key(e: Engine, audio: np.ndarray, acks=()):
    """My burst on air, carrying KISS frames acked with `acks` (engine._test_acks:
    a burst's acks are what names a window's anchor), and nobody's session."""
    e.tx = [hand.HandBurst(), audio, 0]
    e._tx_kiss = True  # nobody's session: no reply timer
    e._test_acks = [(0, a) for a in acks]


def hand_round(e: Engine, d_s: float, tag: int, channels: list[int], hands=(), gaps=(),
               at_open=None, arm=True, decoys=()):
    """One listen round on a bare engine (docs/arq.md §9): arm the window (unless
    the caller armed it already), key a 0.3 s burst carrying ACK to anchor it,
    then run the window to its end. `decoys`: the acks of 0.3 s bursts keyed
    first (None: a burst with no frame acked), which must not open it. `hands` is (channel, payload, seconds after the anchor) of
    Hand audio on the input; `gaps` is (start s, length s) of my own keying inside
    the window; `at_open` runs once the window has opened.

    -> (lines, keyed, anchor): the round's REF/HAND lines, the block-start samples
    at which the radio was keyed after the window opened, and the anchor burst's
    end sample (the window's zero point)."""
    if arm:
        e.hand_listen(d_s, tag, ACK, channels)
    for acks in [() if a is None else (a,) for a in decoys] + [(ACK,)]:  # the window opens at the last's end
        key(e, np.zeros(3 * BLOCK), acks)
        for _ in range(3):
            e.step(np.zeros(BLOCK))
    anchor = e.n  # its end sample: an exact whole number of blocks
    hi = anchor + int(round(d_s * FS)) + hand.SEARCH_SAMPLES + hand.LEN_SAMPLES
    y = np.zeros(hi - anchor + 3 * BLOCK)
    for c, payload, start in hands:
        x = hand.modulate(tag, hand.payload_bits(payload), c)
        i = int(round(start * FS))
        y[i:i + len(x)] += x
    if at_open:
        at_open()
    keyed = []
    while e.n < hi:
        for g0, gl in gaps:  # a session burst keys inside the window
            if e.tx is None and e.n == anchor + int(round(g0 * FS)):
                key(e, np.zeros(int(round(gl * FS))))
        i = e.n - anchor
        _, ptt = e.step(y[i:i + BLOCK])
        if ptt:
            keyed.append(i)
    lines, end = [], time.monotonic() + 30  # the decode runs on its own thread
    while not any(l == "HAND END" for l in lines):
        lines += e.take_hand_events()
        assert time.monotonic() < end, lines
        time.sleep(0.02)
    return lines, keyed, anchor


def ref_at(e: Engine, seconds_ahead: float) -> Engine:
    """A heard-burst reference, as _hear leaves one for every burst decoded: end
    `seconds_ahead` engine seconds from now, no offset."""
    e.hand_refs[1] = (e.now + seconds_ahead, 0.0, 5.0)
    return e


def test_hand_decode_roundtrip():
    """One channel, 2 Hz off and at 6 dB: the payload comes back, with its
    landing time reported to within a search step."""
    rng = np.random.default_rng(7)
    tag, c, payload = 0x5A, 20, int(rng.integers(0, 2 ** hand.PAYLOAD))
    jitter = int(rng.integers(0, hand.SEARCH_SAMPLES))  # lands this late
    y = np.concatenate([np.zeros(hand.SEARCH_SAMPLES + jitter),
                        hand.modulate(tag, hand.payload_bits(payload), c),
                        np.zeros(hand.SEARCH_SAMPLES)])
    y = hfchannel.awgn(hfchannel.freq_shift(y, 2.0), 6.0, seed=3)
    got, = hand.decode_window(y, tag, [c], hand.SEARCH_SAMPLES)
    assert got["payload"] == payload
    assert got["snr"] > 0
    assert abs(got["dt_ms"] - 1000.0 * jitter / FS) < 30


def test_hand_decode_with_the_window_head_lost():
    """The capture starts 100 ms late (my own burst ran into the window), so
    nominal is under SEARCH_SAMPLES: a Hand landing on its slot still decodes."""
    rng = np.random.default_rng(8)
    tag, c, payload = 0x5A, 20, int(rng.integers(0, 2 ** hand.PAYLOAD))
    lost = FS // 10
    y = np.concatenate([np.zeros(hand.SEARCH_SAMPLES - lost),
                        hand.modulate(tag, hand.payload_bits(payload), c),
                        np.zeros(hand.SEARCH_SAMPLES)])
    y = hfchannel.awgn(y, 6.0, seed=4)
    got, = hand.decode_window(y, tag, [c], hand.SEARCH_SAMPLES - lost)
    assert got["payload"] == payload


@pytest.mark.parametrize("hand_round", ["full", "cheap"])
def test_hand_decode_at_the_sensitivity_floor(hand_round):
    """The shipped threshold: -18 dB, 3 Hz off, landing in the search's
    middle. Both grids bring the payload back, so what --hand-decoder cheap
    costs (~0.2-0.3 dB, docs/hand-burst.md) stays inside the margin."""
    rng = np.random.default_rng(4)
    tag, c, payload = 0x9C, 12, 0x1234ABCDEF012
    jitter = int(rng.integers(hand.SEARCH_SAMPLES // 3, 2 * hand.SEARCH_SAMPLES // 3))
    y = np.concatenate([np.zeros(hand.SEARCH_SAMPLES + jitter),
                        hand.modulate(tag, hand.payload_bits(payload), c),
                        np.zeros(hand.SEARCH_SAMPLES)])
    y = hfchannel.awgn(hfchannel.freq_shift(y, -3.0), -18.0, seed=104)
    got, = hand.decode_window(y, tag, [c], hand.SEARCH_SAMPLES, hand.ROUNDS[hand_round])
    assert got["payload"] == payload


@pytest.mark.parametrize("hand_round", ["full", "cheap"])
def test_hand_round_between_two_stations(hand_round):
    """A (net control) arms a listen named by ACK, sends a CQ and then the
    KISS frame (ANY TRAFFIC's stand-in) acked with ACK, and gets B's Hand back
    on its channel: the window hangs on the frame's burst, not the CQ that went
    first. B's own CQ, queued behind the Hand, waits for it (the Hand isn't
    dropped), and the payload's 64th bit is ignored. Both --hand-decoder
    settings on net control: the cheap grids find the Hand as well as the
    shipped ones."""
    a, b = Engine("W1AW", seed=1, kiss=KissLink()), Engine("K2XYZ", seed=2)
    a.hand_round = hand_round
    tag, payload, d = 0x5A, 0x1234567890ABC, 1.0
    a.hand_listen(d, tag, ACK, [20])
    a.send_cq("W1AW", 2)
    a.kiss.enqueue(frame("QST", "W1AW", 0x03, b"ANY TRAFFIC"), 0, 7 << 16 | ACK)  # an id, its tag low
    b_ev = []

    def b_heard():
        b_ev.extend(b.take_hand_events())
        return sum(ev.startswith("REF") for ev in b_ev) >= 2

    assert link(a, b, 12, 40, b_heard)  # B hears the CQ, then the call: two REF lines
    ref = int([ev for ev in b_ev if ev.startswith("REF")][1].split()[1])  # the call's
    b.hand_send(ref, tag, d, 20, payload | 1 << 63)
    b.send_cq("K2XYZ", 2)
    a_ev = []

    def a_heard():
        a_ev.extend(a.take_hand_events())
        b_ev.extend(b.take_hand_events())
        return any(x.startswith("HAND END") for x in a_ev)

    link(a, b, 12, 40, a_heard)  # the round runs on the engine clock
    end = time.monotonic() + 20  # the decode happens on its own thread, in real time
    while not a_heard() and time.monotonic() < end:
        time.sleep(0.05)
    hs = [x for x in a_ev if x.split()[:2] == ["HAND", "20"]]
    assert len(hs) == 1, a_ev
    _, _, hex16, snr, dt = hs[0].split()
    assert int(hex16, 16) == payload
    assert float(snr) > 0
    # both engines share one clock, so B's Hand lands on its slot and what is left is
    # this harness's own delay: one block (100 ms) of link delay (each station hears
    # the other's previous block) plus B keying up at the first block boundary past
    # its slot, which is always less than one block. Under 200 ms, and 180 here: it
    # is the round's latency, nothing like the +-hand.SEARCH_S (600 ms) the search
    # covers, which no failure short of a decode in the wrong place could exceed.
    assert abs(float(dt)) < 200, hs
    assert any(ev.startswith("HAND SENT 20") for ev in b_ev), b_ev


def test_hand_window_hangs_on_its_named_burst():
    """Bursts that don't carry the listen's ack (none at all, or another tag,
    even one sharing its low byte) don't open the window: the Hand, placed for
    the named burst's end, decodes on time (docs/arq.md §9)."""
    e = Engine("W1AW", seed=1)
    lines, _, _ = hand_round(e, 1.0, 0x5A, [20], [(20, 0x4321, 1.0)], decoys=[None, 0x9934, 1 << 16 | 0x4312])
    got = [l for l in lines if l.startswith("HAND 20 ")]
    assert len(got) == 1 and int(got[0].split()[2], 16) == 0x4321, lines
    assert abs(float(got[0].split()[4])) < 30, lines


def test_hand_crc_binds_the_channel():
    """The CRC is seeded with the channel too (hand.crc_init): a Hand that
    lands a channel over (here 60 Hz low: channel 5 at 1650 Hz lands on 3's
    1590) fails there instead of being credited to channel 3, while one sent
    for channel 3 decodes."""
    tag, pad = 0x5A, np.zeros(hand.SEARCH_SAMPLES)
    for c, shift, want in ((5, -60.0, []), (3, 0.0, [0x777])):
        x = hfchannel.freq_shift(hand.modulate(tag, hand.payload_bits(0x777), c), shift)
        y = hfchannel.awgn(np.concatenate([pad, x, pad]), 10.0, seed=5)
        assert [r["payload"] for r in hand.decode_window(y, tag, [3], hand.SEARCH_SAMPLES)] == want, c


def test_hand_channels_fit_the_width():
    """Channels count from the centre out (1470, 1530, 1410 Hz, ...), and the
    host's width keeps a prefix: 8 at 500 Hz, 20 at 1200, all 40 at 2400,
    every tone inside +-width/2 of 1500 Hz. HAND SEND and HAND LISTEN refuse a
    channel past it, the narrower of the BW command's and --kiss-bw."""
    assert [hand.channel_freq(c) for c in range(4)] == [1470.0, 1530.0, 1410.0, 1590.0]
    for cap, n in ((500, 8), (1200, 20), (2400, 40)):
        assert hand.channels_within(cap) == n
        assert all(abs(hand.channel_freq(c) - 1500) + hand.M * hand.RATE / 2 <= cap / 2 for c in range(n))
    for bw, kiss_cap, ok, wrong in (("BW500", None, 7, 8), ("BW1200", None, 19, 20), ("BW2300", 0, 7, 8)):
        e = Engine("W1AW", seed=1, kiss=KissLink(cap=kiss_cap) if kiss_cap is not None else None)
        h = Host(ref_at(e, 5.0))
        h.command(bw)
        for c, want in ((wrong, "WRONG"), (ok, "OK")):
            h.out_cmd.clear()
            h.command(f"HAND LISTEN 3.0 90 1234 {c}")
            assert h.out_cmd == [want], (bw, kiss_cap, c)
        h.out_cmd.clear()
        h.command(f"HAND SEND 1 90 3.0 {wrong} 1")
        h.command(f"HAND SEND 1 90 3.0 {ok} 1")
        assert h.out_cmd == ["WRONG", "OK"], (bw, kiss_cap)
        e.hand_cancel()


def test_hand_commands_reply():
    """The §9 commands' replies, each case guarding the check it names: a bad
    channel on a Host with nothing armed, and bad hex once a ref exists - otherwise
    the already-armed window or the missing ref answers WRONG first and the case
    proves nothing."""
    h = Host(Engine("W1AW", seed=1))
    h.command("HAND LISTEN 3.0 90 1234 20 21")
    assert h.out_cmd == ["OK"]
    h.out_cmd.clear()
    for line in ("HAND LISTEN 3.0 91 1234 22", "HAND LISTEN 3.0 90 1234", "HAND SEND 1 90 3.0 20 1"):
        h.command(line)  # the window above is still armed; no burst heard: no refs
        assert h.out_cmd == ["WRONG"], line
        h.out_cmd.clear()

    for line in ("HAND LISTEN 3.0 90 1234 20 40",  # channel 40: there are 0-39
                 "HAND LISTEN 3.0 90 123 20", "HAND LISTEN 3.0 90 12345 20", "HAND LISTEN 3.0 90 zzzz 20"):  # ack: 4 hex
        fresh = Host(Engine("W1AW", seed=1))  # nothing armed: the argument is what refuses
        fresh.command(line)
        assert fresh.out_cmd == ["WRONG"], line

    ref_at(h.engine, 5.0)  # a ref, so the payload is what the reply is about
    h.command("HAND SEND 1 90 3.0 20 zz")
    assert h.out_cmd == ["WRONG"]  # hex16, not "zz"


def test_hand_send_rejections():
    """Each HAND SEND limit refuses on its own (docs/arq.md §9: WRONG for a gone
    ref, bad numbers, or a Hand already waiting), with a ref in place so the number
    is what's refused. A fresh Host per case: while one Hand waits, every later
    command is refused for that reason first."""
    for line, why in (("HAND SEND 1 90 3.0 40 1", "channel 40: there are 0-39"),
                      ("HAND SEND 1 256 3.0 20 1", "tag 256: 0-255"),
                      ("HAND SEND 1 90 0 20 1", "d = 0"),
                      ("HAND SEND 1 90 30 20 1", "d = the 30 s arm expiry: a Hand waits no longer"),
                      ("HAND SEND 9 90 3.0 20 1", "no ref 9")):
        h = Host(ref_at(Engine("W1AW", seed=1), 5.0))
        h.command(line)
        assert h.out_cmd == ["WRONG"], (line, why)

    h = Host(ref_at(Engine("W1AW", seed=1), -4.0))  # the ref's end is 4 s behind us
    h.command("HAND SEND 1 90 3.0 20 1")
    assert h.out_cmd == ["WRONG"]  # its slot, 3 s after that, has passed

    h = Host(ref_at(Engine("W1AW", seed=1), 5.0))
    h.command("HAND SEND 1 90 3.0 20 1")
    assert h.out_cmd == ["OK"]
    h.command("HAND SEND 1 90 4.0 21 2")
    assert h.out_cmd == ["OK", "WRONG"]  # a Hand is already waiting


def test_hand_args_are_strict():
    """The §9 arguments are exact character classes: int() and float() quietly take
    underscores and a leading minus, so the Host checks the text itself (with a ref
    in place, so the argument is what the reply is about)."""
    h = Host(ref_at(Engine("W1AW", seed=1), 5.0))
    for line in ("HAND LISTEN 1_0 90 1234 20", "HAND SEND 1 90 3.0 20 -0",
                 "HAND SEND 1 90 3.0 20 zz", "HAND SEND 1 90 3.0 20 f_f"):
        h.command(line)
        assert h.out_cmd == ["WRONG"], line
        h.out_cmd.clear()
    h.command("HAND SEND 1 90 3.0 20 FFFFffffFFFFff00")  # the class it does take
    assert h.out_cmd == ["OK"]


def test_hand_listen_expiry_and_cancel():
    """An armed HAND LISTEN that never gets its burst gives up after
    HAND_ARM_EXPIRE_S of engine time with one lone HAND END, and a listen can be
    armed again afterwards; HAND CANCEL ends the same wait early, always answers
    OK on the port, and never raises when there is nothing to cancel (docs/arq.md
    §9)."""
    e = Engine("W1AW", seed=1)
    h = Host(e)
    h.command("HAND CANCEL")  # nothing pending at all
    assert h.out_cmd == ["OK"]
    e.hand_cancel()  # and at the Engine level: idempotent, never raises
    e.hand_cancel()

    e.hand_listen(3.0, 0x5A, ACK, [20])
    keyed = []
    for i in range(int(31 * FS / BLOCK)):
        keyed.append(e.step(np.zeros(BLOCK))[1])
        if i == int(29 * FS / BLOCK):
            assert e.take_hand_events() == []  # 29 s in: still waiting for its burst
    assert not any(keyed)  # and it keys nothing while it waits
    assert e.take_hand_events() == ["HAND END"]  # one line, once
    assert e.take_hand_events() == []  # and nothing after it: the listen is gone
    e.hand_listen(3.0, 0x5A, ACK, [20])  # armed again, and the reply is about that one
    h.out_cmd.clear()
    h.command("HAND LISTEN 3.0 90 1234 20")
    assert h.out_cmd == ["WRONG"]  # a window is still armed
    h.command("HAND CANCEL")
    assert h.out_cmd == ["WRONG", "OK"] and e.take_hand_events() == ["HAND END"]
    h.command("HAND CANCEL")  # twice: OK again, no second HAND END
    assert h.out_cmd == ["WRONG", "OK", "OK"] and e.take_hand_events() == []
    h.out_cmd.clear()
    h.command("HAND LISTEN 3.0 90 1234 20")  # and a fresh listen is accepted
    assert h.out_cmd == ["OK"]


def test_hand_cancel_clears_a_waiting_hand_and_an_open_window():
    """A cancelled Hand never keys: the radio stays silent through its slot, with
    no HAND SENT and no HAND DROPPED either. And a window already capturing is
    cancelled too: one HAND END for it, and no round afterwards - the decode never
    runs, so no HAND lines follow (docs/arq.md §9)."""
    e = Engine("W1AW", seed=1)
    h = Host(e)
    ref_at(e, 1.0)
    h.command("HAND SEND 1 90 1.0 20 1")  # its slot: 2 s from now
    assert h.out_cmd == ["OK"]
    h.command("HAND CANCEL")
    assert h.out_cmd == ["OK", "OK"]
    for _ in range(40):  # well past the slot
        _, ptt = e.step(np.zeros(BLOCK))
        assert not ptt
    assert e.take_hand_events() == []  # neither sent nor dropped

    e = Engine("K2XYZ", seed=2)
    e.hand_listen(1.0, 0x5A, ACK, [20])
    key(e, np.zeros(3 * BLOCK), [ACK])  # the burst that anchors it
    for _ in range(3 + 5):  # and 0.5 s into the window it opened
        e.step(np.zeros(BLOCK))
    e.hand_cancel()
    assert e.take_hand_events() == ["HAND END"]
    for _ in range(int(12 * FS // BLOCK)):  # the rest of the window passes unused
        e.step(np.zeros(BLOCK))
    assert e.take_hand_events() == []  # no HAND SHORT, no decodes, no second end


def test_hand_dropped_past_the_slot():
    """A Hand whose slot arrives with the radio keyed is dropped (docs/arq.md §9:
    HAND DROPPED c if the radio was busy more than 0.5 s past the slot), and the
    lateness is measured from its key-up sample - the slot minus the host's own
    PTT delay - not from where its audio would start."""

    def keyed_until(late_s: float):
        """One station whose radio keys until `late_s` past its Hand's key-up
        sample: -> (engine, the block's PTT, the lines that decision made)."""
        e = Engine("W1AW", seed=1)
        ref_at(e, 2.05)
        e.hand_send(1, 0x5A, 1.0, 20, 0x123)
        up = e._hand_tx[0]  # the key-up sample the engine scheduled
        key(e, np.zeros(up + int(round(late_s * FS)) - e.n))
        while e.tx is not None:
            e.step(np.zeros(BLOCK))
        assert e.n - up == int(round(late_s * FS)) and e.take_hand_events() == []
        _, ptt = e.step(np.zeros(BLOCK))  # the first idle block decides
        return e, ptt, e.take_hand_events()

    e, ptt, ev = keyed_until(0.45)  # still inside LATE_S of its slot: it goes
    assert ptt and ev == ["HAND SENT 20"], ev
    e, ptt, ev = keyed_until(0.55)  # LATE_S plus 0.05 s ...
    assert not ptt and ev == ["HAND DROPPED 20"], ev
    # ... which, measured from where the audio would start (the key-up sample plus
    # the host's PTT delay), would only be 0.45 s late: late by the key-up reading
    # and on time by the audio one, so only the first can be what the engine used.
    for _ in range(20):  # and the Hand is gone: nothing keys at the slot later
        _, ptt = e.step(np.zeros(BLOCK))
        assert not ptt
    assert e.take_hand_events() == []


def test_hand_listen_holds_its_head():
    """A HAND LISTEN that would open its window less than 1.0 s after my burst is
    refused (its head cannot be captured), and a round whose head this station was
    busy for still reports itself: HAND SHORT ms (the head it lost) before HAND END,
    with no decodes at all when the slot itself is missing from the buffer
    (docs/arq.md §9)."""
    e = Engine("W1AW", seed=1)
    with pytest.raises(ValueError):
        e.hand_listen(0.5, 0x5A, ACK, [20])  # below the 1.0 s floor
    h = Host(e)
    h.command("HAND LISTEN 0.5 90 1234 20")
    assert h.out_cmd == ["WRONG"]

    # my own burst keys from the window's opening until 1.2 s in: the first 0.8 s
    # of it is gone, and with it the slot (0.4 s past the anchor).
    lines, _, _ = hand_round(e, 1.0, 0x5A, [20], [(20, 0x1234567, 1.0)], gaps=[(0.0, 1.2)])
    assert lines == ["HAND SHORT 800", "HAND END"], lines


def test_hand_listen_decodes_a_channel_once():
    """The channel list may repeat: HAND LISTEN 1.0 90 1234 20 20 21 decodes channel 20
    once, and reports it once (docs/arq.md §9)."""
    e = Engine("W1AW", seed=1)
    h = Host(e)
    h.command("HAND LISTEN 1.0 90 1234 20 20 21")
    assert h.out_cmd == ["OK"]
    lines, _, _ = hand_round(e, 1.0, 90, [20, 20, 21], [(20, 0x111, 0.9), (21, 0x222, 1.0)], arm=False)
    for c, payload in ((20, 0x111), (21, 0x222)):
        got = [l for l in lines if l.split()[:2] == ["HAND", str(c)]]
        assert len(got) == 1, lines
        assert int(got[0].split()[2], 16) == payload, lines


def test_hand_holds_my_own_bursts():
    """While a Hand waits and while its window is open, this host starts no CQ,
    broadcast or ID of its own (docs/arq.md §9); a session burst inside the window
    still keys, and the window counts its time as silence - which is what keeps the
    reported dt true."""
    e = Engine("W1AW", seed=1, kiss=KissLink())
    ref_at(e, 1.0)
    e.send_cq("W1AW", 2)
    e.kiss.enqueue(frame("APRS", "W1AW", 0x03, b"!beacon"))
    e.hand_send(1, 0x5A, 1.0, 20, 0x1)  # slot 2 s from now
    up = e._hand_tx[0]
    for _ in range(19):  # up to the slot: the radio never keys
        _, ptt = e.step(np.zeros(BLOCK))
        assert not ptt
    assert e._extra and e.kiss.queue  # the CQ frame and the beacon are both still queued
    while e.tx is None:
        e.step(np.zeros(BLOCK))
    assert e.n > up and e.take_hand_events() == ["HAND SENT 20"]  # it keyed the Hand, not mine
    assert e._extra and e.kiss.queue

    # and with the window open: a session burst keys (a Hand 1.3 s into a window
    # opening 1.0 s after the anchor), for 0.6 s, between the head and that Hand.
    e2 = Engine("K2XYZ", seed=2, kiss=KissLink())

    def queue_mine():  # queued while the window is open
        e2.send_cq("K2XYZ", 2)
        e2.kiss.enqueue(frame("APRS", "K2XYZ", 0x03, b"!beacon"))

    lines, keyed, anchor = hand_round(e2, 1.0, 0x5A, [20], [(20, 0x1234567890ABC, 1.3)],
                                      gaps=[(0.5, 0.6)], at_open=queue_mine)
    g0, g1 = anchor + int(0.5 * FS), anchor + int(1.1 * FS)
    hi = anchor + int(1.0 * FS) + hand.SEARCH_SAMPLES + hand.LEN_SAMPLES
    # only the session burst, up to the block that straddles the window's end: its
    # input completes the capture, and what it plays goes out after
    assert [s for s in keyed if not g0 <= anchor + s < g1 and anchor + s + BLOCK <= hi] == [], keyed
    assert e2.kiss.queue  # the CQ goes in the block that completes the capture, the beacon after
    got = [l for l in lines if l.startswith("HAND 20 ")]
    assert len(got) == 1, lines
    assert int(got[0].split()[2], 16) == 0x1234567890ABC, lines
    # the Hand landed 300 ms late; had the keyed gap not been kept as silence in the
    # buffer, it would read 600 ms earlier (a -300 ms dt).
    assert abs(float(got[0].split()[4]) - 300) < 30, lines


def test_hand_refs_count_up_and_lose_the_oldest():
    """Every burst decoded is a reference, its id counting up, and only the last 64
    stay usable: after 65 the first is gone (WRONG) while the newest answers
    (docs/arq.md §9)."""
    a, b = Engine("W1AW", seed=1), Engine("K2XYZ", seed=2)
    h = Host(b)
    seen = []

    def heard():
        seen.extend(x for x in b.take_hand_events() if x.startswith("REF "))
        if len(seen) < 65 and a.tx is None and not a._extra:
            a.send_cq("W1AW", 2)  # keep bursting until B has heard 65
        return len(seen) >= 65

    assert link(a, b, 12, 300, heard)
    assert [int(x.split()[1]) for x in seen] == list(range(1, 66))  # ids count up
    # only the last 64 stay usable: the burst heard first is gone, heard 2 .. 65 stay
    assert sorted(b.hand_refs) == list(range(2, 66))
    h.out_cmd.clear()
    h.command("HAND SEND 1 90 3.0 20 1")  # 3 s after that burst's end
    assert h.out_cmd == ["WRONG"]  # ... which is what the port answers for it
    h.out_cmd.clear()
    h.command("HAND SEND 65 90 3.0 20 1")  # the newest still answers
    assert h.out_cmd == ["OK"]
    h.engine.hand_cancel()  # don't leave the scheduled Hand keying


def test_client_gone_cancels_the_hand():
    """The client that owned a Hand goes: its waiting Hand and its armed window are
    cancelled with it - nothing keys at the slot, and the window's HAND END is not
    left behind for the client that comes next (docs/arq.md §9)."""
    e = Engine("W1AW", seed=1)
    h = Host(e)
    ref_at(e, 1.0)
    h.command("HAND SEND 1 90 1.0 20 1")  # slot 2 s from now
    h.command("HAND LISTEN 3.0 90 1234 20")
    assert h.out_cmd == ["OK", "OK"]
    h.out_cmd.clear()
    h.client_gone()
    for _ in range(40):  # past the slot
        _, ptt = e.step(np.zeros(BLOCK))
        assert not ptt, "a gone client's Hand keyed the radio"
    h.after_step(False)  # the audio loop keeps running, ungated now
    assert not [l for l in h.out_cmd if l.startswith("HAND")], h.out_cmd
    h.out_cmd.clear()
    h.command("HAND LISTEN 3.0 90 1234 20")  # the next client arms its own
    assert h.out_cmd == ["OK"]
    h.after_step(False)
    assert not [l for l in h.out_cmd if l == "HAND END"], h.out_cmd  # no stray end
    h.engine.hand_cancel()


def test_hand_lines_are_gated_to_the_client_that_asked():
    """REF and HAND lines go only to a client that sent DATA2G ON or a HAND
    command; a plain VARA client hears neither, and its leaving turns the gate off
    again (docs/arq.md §9)."""

    def gated(*cmds, arm_listen=True):
        """A station that hears one burst (a REF line) and runs one armed listen
        that never gets its burst (a HAND END): -> what its client is shown for
        each, after sending `cmds`. `arm_listen=False` leaves the arming to the
        client's own HAND command, or to the Engine."""
        e = Engine("W1AW", seed=1)
        h = Host(e)
        for c in cmds:
            h.command(c)
        h.out_cmd.clear()
        other = Engine("K2XYZ", seed=2)
        other.send_cq("K2XYZ", 2)
        link(other, e, 12, 4, lambda: h.after_step(e.tx is not None) or False)
        refs = [l for l in h.out_cmd if l.startswith("REF ")]
        h.out_cmd.clear()
        if arm_listen:
            e.hand_listen(3.0, 90, ACK, [20])  # the Engine's own round, not the client's
        h.out_cmd.clear()
        for _ in range(int(31 * FS // BLOCK)):  # its burst never goes out
            _, ptt = e.step(np.zeros(BLOCK))
            h.after_step(ptt)
        return refs, [l for l in h.out_cmd if l.startswith("HAND")]

    assert gated() == ([], [])  # a plain client: no REF, no HAND END of a round it never asked for
    refs, hands = gated("DATA2G ON")
    assert refs and refs[0].startswith("REF 1 "), refs  # same station, known client
    assert hands == ["HAND END"], hands
    refs, hands = gated("HAND LISTEN 3.0 91 1234 21", arm_listen=False)  # the command alone gates
    assert refs and refs[0].startswith("REF 1 "), refs
    assert hands == ["HAND END"], hands

    e = Engine("W1AW", seed=1)
    h = Host(e)
    h.command("DATA2G ON")
    h.out_cmd.clear()
    other = Engine("K2XYZ", seed=2)
    other.send_cq("K2XYZ", 2)
    link(other, e, 12, 4, lambda: h.after_step(e.tx is not None) or False)
    assert [l for l in h.out_cmd if l.startswith("REF ")]  # known, so far
    h.client_gone()  # ... and its leaving resets that
    h.out_cmd.clear()
    other = Engine("K3ABC", seed=3)
    other.send_cq("K3ABC", 2)
    link(other, e, 12, 4, lambda: h.after_step(e.tx is not None) or False)
    assert not [l for l in h.out_cmd if l.startswith(("REF ", "HAND"))], h.out_cmd


def test_kiss_refs():
    """REF lines also go to the KISS port, but only to a client that sent the
    documented SETHARDWARE "REFS ON", as literal SETHARDWARE REF frames (subcommand
    0x11) ahead of the burst's frames; REFS OFF and the connection dropping both
    stop them, and the host keeps them for it in out_kiss_refs."""
    import socket
    import threading

    from data2g import tnc

    for p, want in ((b"REFS ON\r", True), (b"refs off\0", False), (b"MODE qpsk-r1/2", None), (b"REFS ON X", None)):
        assert tnc.refs_command(p) is want, p
    srv = tnc.KissServer(("127.0.0.1", 0), lambda *a: None)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def readn(s, fend):
        """fend FEND-delimited bytes of frames, as they arrived: pinned as bytes,
        so the subcommand and its order are the server's, not a decoder's."""
        buf = b""
        end = time.monotonic() + 5
        while buf.count(0xC0) < fend or not buf.endswith(b"\xc0"):
            assert time.monotonic() < end, buf
            buf += s.recv(4096)
        return buf

    def wait_until(f):
        end = time.monotonic() + 5
        while not f():
            assert time.monotonic() < end
            time.sleep(0.01)

    try:
        a, b = (socket.create_connection(srv.server_address, timeout=5) for _ in range(2))
        a.sendall(b"\xc0\x06REFS ON\xc0")  # the documented command, byte for byte
        wait_until(lambda: len(srv.refs) == 1 and len(srv.clients) == 2)
        srv.send_ref("REF 1 3.62 -0.0 13.8")
        srv.broadcast(b"frame")
        assert readn(a, 4) == b"\xc0\x06\x11REF 1 3.62 -0.0 13.8\xc0" + b"\xc0\x00frame\xc0"
        assert readn(b, 2) == b"\xc0\x00frame\xc0"  # no REF for the client that never asked
        b.sendall(b"\xc0\x06REFS ON\xc0")  # per connection: now both have them
        wait_until(lambda: len(srv.refs) == 2)
        a.sendall(b"\xc0\x06REFS OFF\xc0")  # ... and off again, on the same connection
        wait_until(lambda: len(srv.refs) == 1)
        srv.send_ref("REF 2 4.10 -1.5 11.2")
        srv.broadcast(b"after")
        assert readn(a, 2) == b"\xc0\x00after\xc0"
        assert readn(b, 4) == b"\xc0\x06\x11REF 2 4.10 -1.5 11.2\xc0" + b"\xc0\x00after\xc0"
        b.close()  # leaving drops the socket from refs: send_ref has no dead socket to write
        wait_until(lambda: len(srv.clients) == 1 and not srv.refs)
        srv.send_ref("REF 3 5.00 2.0 10.0")
        srv.broadcast(b"still")
        assert readn(a, 2) == b"\xc0\x00still\xc0"
    finally:
        srv.shutdown()
        srv.server_close()

    e = Engine("W1AW", seed=1)
    h, other = Host(e), Engine("K2XYZ", seed=2)
    other.send_cq("K2XYZ", 2)
    link(other, e, 12, 20, lambda: (h.after_step(False), bool(h.out_kiss_refs))[1])
    assert h.out_kiss_refs[0].startswith("REF 1 ")
    assert not any(x.startswith("REF") for x in h.out_cmd)  # a plain VARA client: no REF on the command port
