"""Broadcast over KISS (data2g.kisslink, docs/broadcast.md): AX.25 parsing,
groups on ports, the self-checking control, statuses and acks, and two TNCs
over the real modem shifting modes from each other's reports."""

import numpy as np

from data2g import hfchannel
from data2g.arq import phy as PHY
from data2g.arq import policy as G
from data2g.arq.modes import MODES
from data2g.config import FS
from data2g.kisslink import BROADCAST, REPORT_MAX_S, KissLink, group_key, parse_ax25
from data2g.tnc import receive_any


def addr(call: str, last: bool = False, h: bool = False) -> bytes:
    c, _, ssid = call.partition("-")
    b = bytes(ord(x) << 1 for x in c.ljust(6))
    return b + bytes([0x60 | (int(ssid or 0) << 1) | (0x80 if h else 0) | (1 if last else 0)])


def frame(dst: str, src: str, ctrl: int, info: bytes = b"", digis=()) -> bytes:
    a = addr(dst) + addr(src, last=not digis)
    for i, (d, h) in enumerate(digis):
        a += addr(d, last=i == len(digis) - 1, h=h)
    return a + bytes([ctrl]) + (bytes([0xF0]) + info if ctrl & 1 == 0 or ctrl & 0xEF == 0x03 else b"")


def test_parse_ax25():
    i = parse_ax25(frame("KC2G", "W1AW-7", 0x00, b"hello"))
    assert (i.dst, i.src, i.next_hop, i.sender, i.connected) == ("KC2G", "W1AW-7", "KC2G", "W1AW-7", True)
    assert parse_ax25(frame("KC2G", "W1AW", 0x21)).connected  # RR (S frame)
    assert parse_ax25(frame("KC2G", "W1AW", 0x3F)).connected  # SABM (a U frame, not UI)
    assert not parse_ax25(frame("APRS", "W1AW", 0x03, b"!pos")).connected  # UI
    assert not parse_ax25(frame("APRS", "W1AW", 0x13, b"!pos")).connected  # UI with P/F
    d = parse_ax25(frame("KC2G", "W1AW", 0x00, b"x", digis=[("RELAY", True), ("WIDE2-1", False)]))
    assert (d.next_hop, d.sender) == ("WIDE2-1", "RELAY")
    assert parse_ax25(b"not an ax.25 frame at all") is None
    assert parse_ax25(b"\x01\x02") is None


def over_air(tx: KissLink, rx: KissLink, snr: float = 20.0, seed: int = 0):
    """tx's next burst through AWGN into rx: (the burst sent, (port, frame)s rx got)."""
    return heard(tx.next_burst(), rx, snr, seed)


def heard(burst, rx: KissLink, snr: float = 20.0, seed: int = 0):
    x = PHY.tx_audio(burst)
    y = np.concatenate([np.zeros(FS // 2), x, np.zeros(FS // 2)])
    y = hfchannel.awgn(y, snr, seed=seed, s_power=hfchannel.active_power(x))
    r = receive_any(y, lead=FS)
    return burst, (rx.on_burst(r) if r is not None else [])


def shifting(**kw):
    """A link with rate shifting on port 0 (BCAST MODE 0 AUTO), the cap's
    robust mode as the fallback."""
    k = KissLink(**kw)
    k.set_mode(0, k.broadcast, auto=True)
    return k


def test_modes_shift_from_reports_and_broadcasts_stay_robust():
    t = [0.0]
    a, b = shifting(clock=lambda: t[0]), shifting(clock=lambda: t[0])
    i1 = frame("KC2G", "W1AW", 0x00, b"connected hello")
    a.enqueue(i1)
    burst, got = over_air(a, b, seed=1)
    assert burst.submode == BROADCAST[2] and got == [(0, i1)]  # no report yet: robust
    t[0] += 3
    b.enqueue(frame("W1AW", "KC2G", 0x21))  # RR back: carries B's report on A
    burst, got = over_air(b, a, seed=2)
    assert len(got) == 1
    t[0] += 3
    i2 = frame("KC2G", "W1AW", 0x22, b"x" * 300)
    a.enqueue(i2)
    burst, got = over_air(a, b, seed=3)
    assert burst.submode != BROADCAST[2] and MODES[burst.submode] in G.allowed(2), burst.submode
    assert got == [(0, i2)]
    # UI and non-AX.25 stay robust whoever they're for
    for f in (frame("KC2G", "W1AW", 0x03, b"UI to KC2G"), b"plain bytes, not AX.25"):
        a.enqueue(f)
        burst, got = over_air(a, b, seed=4)
        assert burst.submode == BROADCAST[2] and got == [(0, f)]
    # a stale report is not followed
    t[0] += REPORT_MAX_S + 1
    a.enqueue(frame("KC2G", "W1AW", 0x24, b"late"))
    assert a.next_burst().submode == BROADCAST[2]


def test_500_hz_cap_stays_inside_500_hz():
    t = [0.0]
    a, b = shifting(cap=0, clock=lambda: t[0]), shifting(cap=0, clock=lambda: t[0])
    a.enqueue(frame("KC2G", "W1AW", 0x00, b"hi"))
    burst, _ = over_air(a, b, seed=5)
    assert G.width_hz(MODES[burst.submode]) <= 500
    b.enqueue(frame("W1AW", "KC2G", 0x21))
    over_air(b, a, seed=6)
    a.enqueue(frame("KC2G", "W1AW", 0x22, b"y" * 200))
    burst, got = over_air(a, b, seed=7)
    assert G.width_hz(MODES[burst.submode]) <= 500 and got


def test_first_transmission_success_is_the_target():
    """No resends under KISS: at 0 dB AWGN the shifted mode must still
    deliver every frame (plain goodput-first shifting picked w48-qpsk-r1/3
    there and lost half of them)."""
    t = [0.0]
    a, b = shifting(clock=lambda: t[0]), shifting(clock=lambda: t[0])
    got_n = 0
    for k in range(4):
        f = frame("KC2G", "W1AW", 0x00, b"x" * 256)
        a.enqueue(f)
        _, got = over_air(a, b, 0.0, seed=10 + k)
        got_n += got == [(0, f)]
        t[0] += 3
        b.enqueue(frame("W1AW", "KC2G", 0x21))
        over_air(b, a, 0.0, seed=20 + k)
        t[0] += 3
    assert got_n == 4


def test_broadcast_mode_option():
    import pytest

    a, b = KissLink(broadcast="n4-qpsk-r1/3"), KissLink()
    f = frame("APRS", "W1AW", 0x03, b"!beacon")
    a.enqueue(f)
    burst, got = over_air(a, b, seed=8)
    assert burst.submode == "n4-qpsk-r1/3" and got == [(0, f)]
    with pytest.raises(ValueError):
        KissLink(cap=0, broadcast="w48-qpsk-r1/5")  # 2400 Hz under a 500 Hz cap
    with pytest.raises(ValueError):
        KissLink(broadcast="no-such-mode")


def test_kiss_commands_set_channel_access():
    k = KissLink()
    k.command(2, bytes([127]))
    k.command(3, bytes([10]))  # a client's 100 ms: shorter than our carrier sense
    k.command(1, bytes([50]))  # TXDELAY: ours is --ptt-on-delay-ms
    assert (k.persist, k.slot_s) == (127, 1.0)
    k.command(3, bytes([200]))
    assert k.slot_s == 2.0


def test_groups_go_to_their_ports_only():
    a, b, c = KissLink(), KissLink(), KissLink()
    na, nb = a.open("APRS"), b.open("aprs")
    f = b"!beacon, not AX.25"
    a.enqueue(f, na)
    _, got = over_air(a, b, seed=30)
    assert got == [(nb, f)] and b.events == [f"BCAST {nb} HEARD"]
    a.enqueue(f, na)  # a station without the group: a broadcast burst, nothing for it
    _, got = over_air(a, c, seed=31)
    assert got == [] and c.events == []
    a.enqueue(b"plain", 0)  # port 0 ("KISS 0") to port 0 only
    _, got = over_air(a, b, seed=32)
    assert got == [(0, b"plain")]


def test_group_from_names_the_sender():
    a, b = KissLink(), KissLink()
    na, nb = a.open("CHAT", from_call="w1aw"), b.open("CHAT")
    a.enqueue(b"hello", na)
    _, got = over_air(a, b, seed=33)
    assert got == [(nb, b"hello")] and b.events == [f"BCAST {nb} HEARD W1AW"]


def control_lost(burst):
    """`burst` with its control codeword's CRC broken (masked with another key)."""
    from data2g.arq.link import Slot, TxBurst

    s0 = burst.slots[0]
    return TxBurst(burst.submode, [Slot((s0.mask_id[0] ^ 1, *s0.mask_id[1:]), 0, s0.payload)] + burst.slots[1:],
                   burst.burst_seq)


def test_control_lost_data_finds_its_port_but_never_a_colliding_one():
    a, b = KissLink(), KissLink()
    na, nb = a.open("APRS"), b.open("APRS")
    a.enqueue(b"x" * 40, na)
    _, got = heard(control_lost(a.next_burst()), b, seed=34)
    assert got == [(nb, b"x" * 40)] and b.events == [f"BCAST {nb} HEARD"]
    assert group_key("G102291") == group_key("APRS")  # another group, the same key: open it too
    nc = b.open("G102291")
    b.take_events()
    a.enqueue(b"y" * 40, na)
    _, got = heard(control_lost(a.next_burst()), b, seed=35)
    assert got == [] and sorted(b.take_events()) == sorted([f"BCAST {nb} LOST 1", f"BCAST {nc} LOST 1"])
    a.enqueue(b"z", na)  # with its control, the name settles it
    _, got = over_air(a, b, seed=36)
    assert got == [(nb, b"z")] and b.events == [f"BCAST {nb} HEARD"]


def test_acks_when_sent_and_drops_reported():
    a = KissLink()
    n = a.open("APRS")
    a.enqueue(b"one", n, ack=b"\x00\x01")
    a.enqueue(b"two", n, ack=b"\x00\x02")
    a.enqueue(b"no ack", n)
    burst = a.next_burst()
    assert a.acks == []  # not until it has gone
    a.on_sent(burst)
    assert a.acks == [(n, b"\x00\x01"), (n, b"\x00\x02")]
    a.enqueue(b"queued", n, ack=b"\x00\x03")
    a.close(n)
    assert a.events[-1] == f"BCAST {n} DROPPED 1" and not a.queue
    a.enqueue(b"to a closed port", n)
    assert a.events[-1] == f"BCAST {n} DROPPED 1"
    a.enqueue(bytes(60000))  # more than any burst in the port's mode carries
    assert a.next_burst() is None and a.events[-1] == "BCAST 0 DROPPED 1" and a.acks == [(n, b"\x00\x01"),
                                                                                         (n, b"\x00\x02")]


def test_rate_shifting_off_by_default_and_when_turned_off():
    from data2g.kisslink import T_REPORTS, parse_tlvs

    a = KissLink()
    a.enqueue(frame("KC2G", "W1AW", 0x00, b"x"))
    assert T_REPORTS not in parse_tlvs(a.next_burst().slots[0].payload[1:])  # off: no reports sent
    t = [0.0]
    a, b = shifting(clock=lambda: t[0]), shifting(clock=lambda: t[0])
    a.enqueue(frame("KC2G", "W1AW", 0x00, b"hello"))
    over_air(a, b, seed=40)
    t[0] += 3
    b.enqueue(frame("W1AW", "KC2G", 0x21))
    over_air(b, a, seed=41)
    t[0] += 3
    a.enqueue(frame("KC2G", "W1AW", 0x22, b"x" * 300))
    assert a.next_burst().submode != BROADCAST[2]  # on: shifted from b's report
    a.set_mode(0, BROADCAST[2])  # off again
    report = next(iter(a.peers.values())).report
    t[0] += 3
    b.enqueue(frame("W1AW", "KC2G", 0x21))
    over_air(b, a, seed=42)
    assert next(iter(a.peers.values())).report == report  # reports heard ignored
    a.enqueue(frame("KC2G", "W1AW", 0x24, b"y" * 300))
    burst = a.next_burst()
    assert burst.submode == BROADCAST[2] and T_REPORTS not in parse_tlvs(burst.slots[0].payload[1:])


def test_every_mode_fits_its_control_or_is_refused():
    """A port's control (group+from, and room for a report) fits the mode's
    control codewords, or BCAST MODE refuses the mode."""
    import pytest

    for cap in (0, 2):
        for spec in G.allowed(cap):
            a = KissLink(cap=cap)
            n = a.open("CHAT", from_call="VK2ABC-15")
            try:
                a.set_mode(n, spec.name, auto=True)
            except ValueError:
                continue
            a.enqueue(b"x", n)
            burst = a.next_burst()
            assert burst.submode == spec.name and sum(s.mask_id[2] >= 128 for s in burst.slots) <= 4
    with pytest.raises(ValueError):
        KissLink().set_mode(0, "no-such-mode")
    with pytest.raises(ValueError):
        KissLink(cap=0).set_mode(0, "w48-qpsk-r1/5")  # 2400 Hz under a 500 Hz cap
