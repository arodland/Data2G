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
from data2g.receiver import receive_any


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


def heard_snr(events: list, n: int, call: str = "") -> float:
    """The SNR of the one status, BCAST n HEARD [call] snr."""
    (ev,) = events
    head, snr = ev.rsplit(" ", 1)
    assert head == f"BCAST {n} HEARD" + (f" {call}" if call else ""), ev
    return float(snr)


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
    assert got == [(nb, f)] and abs(heard_snr(b.events, nb) - 20) < 3  # over_air's 20 dB
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
    assert got == [(nb, b"hello")] and heard_snr(b.events, nb, "W1AW")


def test_control_lost_data_finds_its_port_but_never_a_colliding_one():
    a, b = KissLink(), KissLink()
    na, nb = a.open("APRS"), b.open("APRS")
    a.enqueue(b"x" * 40, na)
    r, rx = soft_rx(a.next_burst(), {0: None})  # the control lost
    assert b.on_burst(r, rx) == [(nb, b"x" * 40)] and heard_snr(b.take_events(), nb)
    assert group_key("G102291") == group_key("APRS")  # another group, the same key: open it too
    nc = b.open("G102291")
    a.enqueue(b"y" * 40, na)
    r, rx = soft_rx(a.next_burst(), {0: None})
    assert b.on_burst(r, rx) == [] and sorted(b.take_events()) == sorted([f"BCAST {nb} LOST 1", f"BCAST {nc} LOST 1"])
    a.enqueue(b"z", na)  # with its control, the name settles it
    _, got = over_air(a, b, seed=36)
    assert got == [(nb, b"z")] and heard_snr(b.events, nb)


def test_a_decoded_slot_0_that_is_no_control_is_another_burst():
    """OFDM: a slot 0 that decodes but isn't a broadcast control (here one
    under another key: an ARQ burst's, say) means another kind of burst,
    not a broadcast whose control was lost: its data is never looked at."""
    from data2g.arq.burst import Slot, TxBurst

    a, b = KissLink(), KissLink()
    na, nb = a.open("APRS"), b.open("APRS")
    a.enqueue(b"x" * 40, na)
    burst = a.next_burst()
    s0 = burst.slots[0]
    other = TxBurst(burst.submode, [Slot((s0.mask_id[0] ^ 1, *s0.mask_id[1:]), 0, s0.payload)] + burst.slots[1:],
                    burst.burst_seq)
    _, got = heard(other, b, seed=34)
    assert got is None and b.events == []


def test_lost_control_search_stops_at_another_bursts_codeword(reference):
    """Control lost: the search stops at the first slot that decoded as a
    codeword but is neither an open port's data nor the rest of its
    control, and goes on past ones that are the rest of its control."""
    from data2g import kisslink
    from data2g.arq.burst import ctl_mask

    find = reference(kisslink, "KissLink").find_data
    key = group_key("APRS")

    class Rx:
        n_ctl_slots = 0

        def __init__(self, n_cw, good):
            self.n_cw, self.good, self.asked = n_cw, good, []

        def decode(self, slot, mask_id, rv, k):
            self.asked.append(slot)
            return b"ok" if self.good.get(slot) == mask_id else None

        def converged(self, slot, pair=False):
            return not pair and (slot in self.good or slot in self.foreign)

    rx = Rx(6, {})
    rx.foreign = {1}  # slot 1 decoded, but under no open key: another burst
    assert find(rx, [key]) is None and max(rx.asked) == 1
    # a 3-codeword control: slots 1-2 its rest, data from slot 3 (confirmed by slot 4)
    good = {1: ctl_mask(0, 1, key), 2: ctl_mask(0, 2, key), 3: kisslink.data_start_mask(3, 3, key),
            4: kisslink.data_start_mask(4, 3, key)}
    rx = Rx(6, good)
    rx.foreign = set()
    assert find(rx, [key]) == (key, 3)
    # the same under DUP: control codeword 1 at RV 0 in slot 2, data from slot 6
    good = {2: ctl_mask(0, 1, key), 6: kisslink.data_start_mask(6, 6, key), 7: kisslink.data_start_mask(7, 6, key)}
    rx = Rx(8, good)
    rx.foreign = set()
    assert find(rx, [key]) == (key, 6)


def test_a_frame_starting_a_codeword_survives_the_codeword_before():
    """A frame that would start a codeword gets a zero byte before it, so
    where it starts is known from its own codeword: frames that fill whole
    codewords (3 bytes short: that zero, COBS's byte, the ending zero) lose
    only the one in a lost codeword."""
    from data2g.kisslink import pack_frames, unpack_frames

    frames = [bytes([i + 1]) * 73 for i in range(4)]
    payloads = pack_frames(frames, 76)
    assert len(payloads) == 4 and all(p[0] == 0 for p in payloads)
    for lost in range(4):
        ok = [i != lost for i in range(4)]
        assert unpack_frames(payloads, ok) == ([f for i, f in enumerate(frames) if i != lost], 1)


def soft_rx(burst, sigma: dict, seed: int = 0):
    """`burst` received (the modem's dict, 20 dB), its soft bits then
    replaced (BPSK LLRs): slot i at noise sigma[i] (default 0.3), or lost
    (noise alone) where sigma[i] is None. No DD, which would remake them from
    the signal -> (the received dict, a phy.ModemRx of it)."""
    from data2g import codes
    from data2g.arq.modes import ctl_spec, is_cpm

    r = receive_any(np.concatenate([np.zeros(FS // 2), PHY.tx_audio(burst), np.zeros(FS // 2)]), lead=FS)
    spec, rng = MODES[burst.submode], np.random.default_rng(seed)
    n_ctl = sum(s.mask_id[2] >= 128 for s in burst.slots) if is_cpm(spec) else 0
    soft = []
    for i, s in enumerate(burst.slots):
        sp = ctl_spec(spec) if i < n_ctl else spec
        x = 1.0 - 2.0 * codes.encode(sp, s.payload, s.rv, PHY.mask_value(s.mask_id), i)
        if sp.code == "gf":  # GF(M): a symbol's likelihoods, the sent value's at +1
            p = sp.bits_per_cu
            sym = ((1 - x.reshape(-1, p)) / 2) @ (1 << np.arange(p)[::-1])
            x = (np.arange(1 << p)[None] == sym[:, None]).reshape(-1).astype(float)
        sg = sigma.get(i, 0.3)
        # lost: noise alone (all-zero LLRs would "decode" to the all-zero codeword)
        soft.append(2 * rng.standard_normal(len(x)) if sg is None else 2 * (x + sg * rng.standard_normal(len(x))) / sg**2)
    r.pop("hp", None)  # DD off
    r["_soft"] = soft if n_ctl else np.array(soft)
    return r, PHY.ModemRx(r, {})


def statuses(events: list) -> list:
    """BCAST n HEARD/LOST statuses without HEARD's trailing SNR."""
    return [e.rsplit(" ", 1)[0] if " HEARD" in e else e for e in events]


def test_dup_control_decodes_from_its_pair():
    """BCAST MODE n DUP: each control codeword twice, RV 0 then RV 1; a
    control too weak alone decodes from the two combined (LDPC IR, polar
    Chase), and the burst's frames follow."""
    import data2g.kisslink as KL
    from data2g.arq.burst import ctl_mask

    for mode in ("qpsk-r1/5", "fsk8r50-r1/2"):
        a, b = KissLink(), KissLink()
        na, nb = a.open("APRS"), b.open("APRS")
        a.set_mode(na, mode, dup=True)
        a.enqueue(b"dup control", na)
        burst = a.next_burst()
        assert [(s.mask_id[2], s.rv) for s in burst.slots[:2]] == [(128, 0), (128, 1)], mode
        assert burst.slots[0].payload[0] == KL.VERSION << 4 | KL.DUP
        found = False
        for sg in np.arange(1.0, 8.0, 0.25):
            noisy = {0: sg, 1: sg}
            r, rx = soft_rx(burst, noisy, seed=1)
            if rx.decode(0, ctl_mask(0, 0, group_key("APRS")), 0, None) is not None:
                continue  # decodes alone: not the case in question
            r, rx = soft_rx(burst, noisy, seed=1)
            if b.on_burst(r, rx) == [(nb, b"dup control")]:
                found = True
                break
        assert found, mode


def test_control_lost_finds_where_the_data_starts():
    """Control lost, its burst's data still found: each data codeword's
    mask names the slot its data starts at, so a burst with a longer
    control (DUP, several codewords) is found too, and from any of its
    first data slots (a frame starts after the next zero byte)."""
    a, b = KissLink(), KissLink()
    na, nb = a.open("APRS"), b.open("APRS")
    # 4-byte codewords: a 3-codeword control, sent twice: data starts at slot 6
    a.set_mode(na, "n4-ack-2f", dup=True)
    a.enqueue(b"h", na)
    burst = a.next_burst()
    assert len(burst.slots) == 7  # "h" framed: 4 bytes (a zero, COBS's 2, the ending zero), one codeword
    r, rx = soft_rx(burst, {i: None for i in range(6)})
    assert b.on_burst(r, rx) == [(nb, b"h")] and statuses(b.take_events()) == [f"BCAST {nb} HEARD"]
    # the first data codeword lost too: the frame it cut is lost, the next
    # found (slot 2 starts it, slot 3 agrees: one slot alone could be another
    # burst's codeword passing a CRC by chance)
    a.set_mode(na, "qpsk-r1/5")  # 46 B codewords
    for f in (b"x" * 50, b"y" * 10, b"z" * 40):
        a.enqueue(f, na)
    burst = a.next_burst()
    assert len(burst.slots) == 4
    r, rx = soft_rx(burst, {0: None, 1: None})
    assert b.on_burst(r, rx) == [(nb, b"y" * 10), (nb, b"z" * 40)]
    assert statuses(b.take_events()) == [f"BCAST {nb} HEARD", f"BCAST {nb} LOST 1"]
    r, rx = soft_rx(burst, {0: None, 1: None, 3: None})  # slot 2 alone: not believed
    assert b.on_burst(r, rx) is None
    a.set_mode(na, "qpsk-r1/5", dup=True)  # and with DUP: both copies lost
    a.enqueue(b"z" * 30, na)
    burst = a.next_burst()
    r, rx = soft_rx(burst, {0: None, 1: None})
    assert b.on_burst(r, rx) == [(nb, b"z" * 30)]


def test_dup_port_over_air():
    for mode in ("qpsk-r1/5", "fsk8r50-r1/2"):
        a, b = KissLink(), KissLink()
        na, nb = a.open("CHAT", from_call="W1AW"), b.open("CHAT")
        a.set_mode(na, mode, dup=True)
        a.enqueue(b"twice", na)
        _, got = over_air(a, b, seed=50)
        assert got == [(nb, b"twice")] and heard_snr(b.events, nb, "W1AW"), mode


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


def test_a_256_byte_frame_burst_is_one_receivers_accept():
    """A KISS burst grows to carry its first frame: with a full control,
    sent twice or not, a 256-byte frame's burst still has a header
    receivers take (engine.receive_accept; CPM: at most cpm.MAX_DATA data
    codewords), on every mode but the ACK modes."""
    from data2g import cpm
    from data2g.arq.engine import receive_accept
    from data2g.arq.modes import is_cpm

    cap = dict(receive_accept().max_cw)
    for spec in G.allowed(len(G.CAP_HZ) - 1):
        if "ack" in spec.name:
            continue
        for dup in (False, True):
            a = KissLink()
            n = a.open("CHAT", from_call="VK2ABC-15")
            try:
                a.set_mode(n, spec.name, auto=True, dup=dup)
            except ValueError:  # CPM: group+from and a report don't fit its 20 B
                n = a.open("CHAT")
                a.set_mode(n, spec.name, auto=True, dup=dup)
            a.enqueue(frame("CQ", "VK2ABC", 0x03, bytes(range(1, 241))), n)  # no zero: COBS's most
            burst = a.next_burst()
            assert burst is not None and burst.submode == spec.name and not a.events, (spec.name, dup)
            n_ctl = sum(s.mask_id[2] >= 128 for s in burst.slots)
            if is_cpm(spec):
                assert n_ctl == 1 + dup and len(burst.slots) - n_ctl <= cpm.MAX_DATA, (spec.name, dup)
            else:
                assert len(burst.slots) <= cap[spec.name], (spec.name, dup)


def test_framing_loses_only_what_a_bad_codeword_touches():
    from data2g.kisslink import pack_frames, unpack_frames

    packets = [bytes([i]) * n for i, n in enumerate((10, 60, 20, 5))]
    payloads = pack_frames(packets, 46)  # qpsk-r1/5's payload
    assert len(payloads) == 3 and all(len(p) == 46 for p in payloads)
    assert unpack_frames(payloads, [True] * len(payloads)) == (packets, 0)
    # codeword 1 (bytes 46-91) holds the end of packet 1 and all but the
    # zero ending packet 2: both lost (counted once: a lost codeword hides how
    # many it held), and packet 3 found again after the zero at byte 95
    ok = [True] * len(payloads)
    ok[1] = False
    assert unpack_frames(payloads, ok) == ([packets[0], packets[3]], 1)
    ok = [False, True, True]  # the first lost: parsing picks up at the next good zero
    assert unpack_frames(payloads, ok) == (packets[2:], 1)
    ok = [True, True, False]  # the last, holding packet 3's zero
    assert unpack_frames(payloads, ok) == (packets[:2], 1)


def test_cobs_framing_costs_what_framed_len_says():
    from data2g.kisslink import cobs, framed_len, pack_frames, uncobs, unpack_frames

    rng = np.random.default_rng(0)
    frames = [b"", b"\0", b"\0" * 300, bytes(range(1, 256)) * 3, b"\1" * 253, b"\1" * 254, b"\1" * 255, b"\1" * 508]
    frames += [bytes(rng.integers(0, 256, n, dtype=np.uint8)) for n in (1, 17, 253, 254, 256, 600)]
    for f in frames:
        c = cobs(f)
        assert 0 not in c and uncobs(c) == f, len(f)
        assert len(c) + 1 <= framed_len(len(f))
        if 0 not in f:
            assert len(c) + 1 == framed_len(len(f))  # the bound is met
    assert framed_len(253) == 255 and framed_len(256) == 259
    payloads = pack_frames([f for f in frames if f], 46)
    assert unpack_frames(payloads, [True] * len(payloads)) == ([f for f in frames if f], 0)
    assert uncobs(b"\5ab") is None and uncobs(b"\0") is None
