"""The KISS TNC's link layer (data2g.kisslink): AX.25 parsing, and two
TNCs over the real modem shifting modes from each other's reports."""

import numpy as np

from data2g import hfchannel
from data2g.arq import phy as PHY
from data2g.arq import policy as G
from data2g.arq.modes import MODES
from data2g.config import FS
from data2g.kisslink import BROADCAST, REPORT_MAX_S, KissLink, parse_ax25
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
    """tx's next burst through AWGN into rx: (the burst sent, frames rx got)."""
    burst = tx.next_burst()
    x = PHY.tx_audio(burst)
    y = np.concatenate([np.zeros(FS // 2), x, np.zeros(FS // 2)])
    y = hfchannel.awgn(y, snr, seed=seed, s_power=hfchannel.active_power(x))
    r = receive_any(y, lead=FS)
    return burst, (rx.on_burst(r) if r is not None else [])


def test_modes_shift_from_reports_and_broadcasts_stay_robust():
    t = [0.0]
    a, b = KissLink(clock=lambda: t[0]), KissLink(clock=lambda: t[0])
    i1 = frame("KC2G", "W1AW", 0x00, b"connected hello")
    a.enqueue(i1)
    burst, got = over_air(a, b, seed=1)
    assert burst.submode == BROADCAST[2] and got == [i1]  # no report yet: robust
    t[0] += 3
    b.enqueue(frame("W1AW", "KC2G", 0x21))  # RR back: carries B's report on A
    burst, got = over_air(b, a, seed=2)
    assert len(got) == 1
    t[0] += 3
    i2 = frame("KC2G", "W1AW", 0x22, b"x" * 300)
    a.enqueue(i2)
    burst, got = over_air(a, b, seed=3)
    assert burst.submode != BROADCAST[2] and MODES[burst.submode] in G.allowed(2), burst.submode
    assert got == [i2]
    # UI and non-AX.25 stay robust whoever they're for
    for f in (frame("KC2G", "W1AW", 0x03, b"UI to KC2G"), b"plain bytes, not AX.25"):
        a.enqueue(f)
        burst, got = over_air(a, b, seed=4)
        assert burst.submode == BROADCAST[2] and got == [f]
    # a stale report is not followed
    t[0] += REPORT_MAX_S + 1
    a.enqueue(frame("KC2G", "W1AW", 0x24, b"late"))
    assert a.next_burst().submode == BROADCAST[2]


def test_500_hz_cap_stays_inside_500_hz():
    t = [0.0]
    a, b = KissLink(cap=0, clock=lambda: t[0]), KissLink(cap=0, clock=lambda: t[0])
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
    a, b = KissLink(clock=lambda: t[0]), KissLink(clock=lambda: t[0])
    got_n = 0
    for k in range(4):
        f = frame("KC2G", "W1AW", 0x00, b"x" * 256)
        a.enqueue(f)
        _, got = over_air(a, b, 0.0, seed=10 + k)
        got_n += got == [f]
        t[0] += 3
        b.enqueue(frame("W1AW", "KC2G", 0x21))
        over_air(b, a, 0.0, seed=20 + k)
        t[0] += 3
    assert got_n == 4
