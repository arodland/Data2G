"""Two live stacks (data2g.arq.engine) back to back through a noisy channel,
on the sample clock: connect, data both ways, disconnect."""

import numpy as np

from data2g.arq import session as S
from data2g.arq.engine import Engine
from data2g.config import FS, SNR_REF_BW_HZ

BLOCK = FS // 10


def link(a: Engine, b: Engine, snr_db: float, seconds: float, until, seed=0):
    """Step both engines, each hearing the other's last block plus noise
    (SNR against a unit-power burst), until `until()` or `seconds` pass."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt((FS / 2) / SNR_REF_BW_HZ / 10 ** (snr_db / 10))
    a_out = b_out = np.zeros(BLOCK)
    g = 2.2  # an Engine plays bursts at peak 1.0: ~unit RMS again (peak/RMS ~2.2 after the clipper)
    for _ in range(int(seconds * FS / BLOCK)):
        a_out, _ = a.step(g * b_out + rng.normal(0, sigma, BLOCK))
        b_out, _ = b.step(g * a_out + rng.normal(0, sigma, BLOCK))
        if until():
            return True
    return False


def test_connect_exchange_disconnect(tmp_path):
    a, b = Engine("W1AW", seed=1, record_dir=tmp_path / "a"), Engine("K2XYZ", seed=2)
    b.listen()
    a.connect("K2XYZ", 2)
    assert link(a, b, 12, 60, lambda: a.session.state == S.CONNECTED and b.session.state == S.CONNECTED)
    rng = np.random.default_rng(3)
    # raw and deflated (T_COMP) codewords both, over the real PHY
    up = rng.bytes(1500) + (b"CQ CQ de W1AW QTH FN31 RST 599 GM OM 73 " * 60)[:2000]
    down = rng.bytes(400)
    a.session.write(up)
    b.session.write(down)
    got_a, got_b = bytearray(), bytearray()

    def done():
        got_a.extend(a.session.read())
        got_b.extend(b.session.read())
        return len(got_b) >= len(up) and len(got_a) >= len(down)
    assert link(a, b, 12, 240, done, seed=1)
    assert bytes(got_b) == up and bytes(got_a) == down
    assert 0 < a.session.station.stats["cw_comp"] < a.session.station.stats["cw_new"]
    a.session.disconnect()
    assert link(a, b, 12, 60, lambda: a.session.state == S.CLOSED and b.session.state == S.CLOSED, seed=2)
    ev_a, ev_b = a.events(), b.events()
    assert ev_a[0] == "CONNECTED K2XYZ" and ev_b[0] == "CONNECTED W1AW"
    assert ev_a[-1].startswith("DISCONNECTED") and ev_b[-1].startswith("DISCONNECTED")
    log = (tmp_path / "a" / "events.jsonl").read_text()
    assert '"kind": "tx"' in log and '"kind": "rx"' in log and list((tmp_path / "a").glob("rx_*.npz"))


def test_vara_commands_drive_a_session(tmp_path):
    import json

    from data2g.config import SUBMODES
    from data2g.host import Host

    a = Host(Engine("NOCALL", seed=4, record_dir=tmp_path / "a"))
    b = Host(Engine("NOCALL", seed=5, record_dir=tmp_path / "b"))
    for line in ("MYCALL K2XYZ", "LISTEN ON", "BW500"):
        b.command(line)
    a.command("BW500")
    a.command("CONNECT W1AW K2XYZ")
    assert a.out_cmd == ["OK", "OK"] and b.out_cmd == ["OK"] * 3

    def pump():
        a.after_step(bool(a.engine.tx))
        b.after_step(bool(b.engine.tx))
        return "CONNECTED W1AW K2XYZ 500" in a.out_cmd and "CONNECTED W1AW K2XYZ 500" in b.out_cmd
    assert link(a.engine, b.engine, 12, 60, pump, seed=3)
    assert a.engine.session.cap == 0  # BW500: narrow modes only
    a.data_in(b"hello from W1AW")

    def got():
        pump()
        return bytes(b.out_data) == b"hello from W1AW"
    assert link(a.engine, b.engine, 12, 120, got, seed=4)
    a.command("DISCONNECT")

    def closed():
        pump()
        return "DISCONNECTED" in a.out_cmd and "DISCONNECTED" in b.out_cmd
    assert link(a.engine, b.engine, 12, 60, closed, seed=5)
    assert b.engine.session.state == S.LISTEN  # listening again
    a.command("NONSENSE")
    assert a.out_cmd[-1] == "WRONG"
    # BW500 on air: every burst either station sent was 500 Hz or narrower
    sent = [json.loads(line)["submode"] for side in "ab" for line in open(tmp_path / side / "events.jsonl")
            if '"kind": "tx"' in line]
    assert sent and all(SUBMODES[m].band in ("n10", "n4") for m in sent), sent


def test_a_mode_the_link_mangles_does_not_stall_the_session(monkeypatch):
    """The audio loopback's livelock: 64/256-QAM bursts clipped to death on
    the way out (their headers still heard), everything else fine, a clean
    30 dB link the shifter would put 64-QAM on. The online bias must route around
    it and the data must still arrive."""
    from data2g.arq import phy as PHY
    from data2g.config import SUBMODES

    tx = PHY.tx_audio

    def mangled(burst):
        x = tx(burst)
        if SUBMODES[burst.submode].constellation[:3] in ("c64", "c25"):
            x = np.clip(x, -0.3, 0.3) / 0.3
        return x
    monkeypatch.setattr(PHY, "tx_audio", mangled)
    a, b = Engine("W1AW", seed=6), Engine("K2XYZ", seed=7)
    b.listen()
    a.connect("K2XYZ", 2)
    assert link(a, b, 30, 60, lambda: a.session.state == S.CONNECTED)
    up = np.random.default_rng(8).bytes(4000)
    a.session.write(up)
    got = bytearray()
    assert link(a, b, 30, 300, lambda: got.extend(b.session.read()) or len(got) >= len(up), seed=9)
    assert bytes(got) == up


def test_data_from_the_client_is_always_answered_with_buffer():
    """Pat's VARA driver adds each write to a local count that only a BUFFER
    line resets, and blocks while it exceeds 7x the next write: every data
    write must get a BUFFER line back, even when the value is unchanged."""
    from data2g.host import Host

    h = Host(Engine("W1AW", seed=1))
    h.engine.session.write = lambda data: None  # taken whole by the next burst: BUFFER stays 0
    h.after_step(False)
    h.out_cmd.clear()
    for _ in range(3):
        h.data_in(b"F> BC\r")
        h.after_step(False)
        assert any(line.startswith("BUFFER") for line in h.out_cmd)
        h.out_cmd.clear()


def test_buffer_credit_caps_or_disables_the_next_burst_allowance():
    from types import SimpleNamespace

    from data2g.host import Host

    # 400 of the 1000 bytes sent and not yet acked: VARA counts them. With a
    # credit, only unsent bytes past it count, and 1 stands for the rest
    # until they're acked (Pat's Flush waits for 0; its writes never block)
    for credit, expect in ((None, 1), (100, 500), (0, 1000)):
        h = Host(Engine("W1AW", seed=1), credit)
        st = SimpleNamespace(tx=SimpleNamespace(buf_off=0, buf=bytearray(1000), stream_end=400), read=lambda: b"")
        h.engine.session.station = st
        h.engine.session.policy.next_capacity = lambda station: 5000
        h.after_step(False)
        assert f"BUFFER {expect}" in h.out_cmd
        h.out_cmd.clear()
        h.engine.n += int(31 * FS)  # unchanged, but repeated: Pat times out after a minute without one
        h.after_step(False)
        assert f"BUFFER {expect}" in h.out_cmd


def test_cqframe_is_heard_without_a_session():
    """VARA's CQFRAME: sent outside any session, heard by a listening and by
    an idle station alike, notified as CQFRAME call bw, no connection made;
    at 500 Hz it goes out in a narrow mode."""
    from data2g.config import SUBMODES
    from data2g.host import Host

    a, b = Host(Engine("NOCALL", seed=11)), Host(Engine("NOCALL", seed=12))
    b.command("LISTEN ON")
    a.command("CQFRAME W1AW 500")
    assert a.out_cmd == ["OK"] and SUBMODES[a.engine._extra[0].submode].band in ("n10", "n4")

    def heard():
        a.after_step(bool(a.engine.tx))
        b.after_step(bool(b.engine.tx))
        return "CQFRAME W1AW 500" in b.out_cmd
    assert link(a.engine, b.engine, 12, 30, heard, seed=13)
    assert b.engine.session.state == S.LISTEN and a.engine.session.state == S.IDLE
    a.command("CQFRAME W1AW 9999")
    assert a.out_cmd[-1] == "WRONG"


def test_chat_on_sends_a_file_in_full_fast_bursts_and_a_line_short(tmp_path):
    """CHAT ON planned every burst for a 200-byte chat line, so a file went
    out 2 codewords at a time in the slowest mode that fit a line (a VARA
    client at BW500). With the sender's queue reported (T_BUFFER), a file
    goes in full bursts at the best rate, and a line stays short."""
    import json

    from data2g.config import SUBMODES

    def run(nbytes, d):
        a, b = Engine("W1AW", seed=21, record_dir=d / "a"), Engine("K2XYZ", seed=22)
        for e in (a, b):
            e.set_chat(True)
        b.listen()
        a.connect("K2XYZ", 0)
        assert link(a, b, 30, 60, lambda: a.session.state == S.CONNECTED)
        a.session.write(np.random.default_rng(23).bytes(nbytes))
        got = bytearray()
        assert link(a, b, 30, 400, lambda: got.extend(b.session.read()) or len(got) >= nbytes, seed=24)
        return [(e["submode"], len(e["slots"])) for e in map(json.loads, open(d / "a" / "events.jsonl"))
                if e["kind"] == "tx" and any(s["mask"][2] < 128 for s in e["slots"])]

    file_bursts = run(3000, tmp_path / "file")
    assert max(n for _, n in file_bursts) > 3, file_bursts
    assert any(SUBMODES[m].k / SUBMODES[m].coded_bits > 0.5 for m, _ in file_bursts), file_bursts
    line_bursts = run(150, tmp_path / "line")
    assert all(n <= 3 for _, n in line_bursts), line_bursts


def test_mycall_with_several_calls_answers_each():
    """VARA clients send MYCALL with several calls (VarAC: 'MYCALL KC2G
    KC2G-T'); a connect to any of them is answered, as the call dialed."""
    from data2g.host import Host

    a, b = Host(Engine("NOCALL", seed=31)), Host(Engine("NOCALL", seed=32))
    for line in ("MYCALL KC2G KC2G-T", "LISTEN CQ", "LISTEN ON", "IGNOREKISSDCD ON"):
        b.command(line)
    assert b.out_cmd == ["OK"] * 4
    a.command("CONNECT W1AW KC2G-T")

    def up():
        a.after_step(bool(a.engine.tx))
        b.after_step(bool(b.engine.tx))
        return "CONNECTED W1AW KC2G-T 2300" in b.out_cmd and "CONNECTED W1AW KC2G-T 2300" in a.out_cmd
    assert link(a.engine, b.engine, 20, 60, up, seed=33)


def test_iamalive_every_60_seconds():
    from data2g.host import ALIVE_S, Host

    h = Host(Engine("W1AW", seed=41))
    for _ in range(int(2.5 * ALIVE_S * FS / BLOCK)):
        h.engine.step(np.zeros(BLOCK))
        h.after_step(False)
    assert h.out_cmd.count("IAMALIVE") == 2


def test_losing_the_command_client_ends_its_session():
    """The command client owns the session: when its TCP connection drops,
    the session is disconnected gracefully (the peer hears DISC) and the
    station stops listening."""
    from data2g.host import Host

    a, b = Host(Engine("W1AW", seed=51)), Host(Engine("K2XYZ", seed=52))
    b.command("LISTEN ON")
    a.command("CONNECT W1AW K2XYZ")

    def pump(cond):
        def f():
            a.after_step(bool(a.engine.tx))
            b.after_step(bool(b.engine.tx))
            return cond()
        return f
    assert link(a.engine, b.engine, 20, 60, pump(lambda: b.engine.session.state == S.CONNECTED))
    b.client_gone()  # the callee's client crashed
    assert link(a.engine, b.engine, 20, 60, pump(lambda: "DISCONNECTED" in a.out_cmd
                                                 and b.engine.session.state in (S.IDLE, S.CLOSED)), seed=53)
    assert not b.listening and b.engine.session.state != S.LISTEN


def test_a_session_in_cpm_modes():
    """Both directions forced onto a CPM mode (control in its short
    codeword, data up to 8 codewords, duplicated control when asked): the
    Receiver's Costas path, tx/rx dispatch and the link's one-control-
    codeword fit, end to end on audio."""
    from data2g.arq import policy as G

    class CpmOnly(G.GearShifter):
        def choose(self, station, escalation):
            if escalation:
                return super().choose(station, escalation)
            m = "fsk32r62-r1/2"
            return m, G.slots_for(G.MODES[m], 12.0, station.tx.pending(), station.peer_wants_dup)

    a, b = Engine("W1AW", policy=CpmOnly, seed=11), Engine("K2XYZ", policy=CpmOnly, seed=12)
    b.listen()
    a.connect("K2XYZ", 2)
    assert link(a, b, 10, 60, lambda: a.session.state == S.CONNECTED and b.session.state == S.CONNECTED)
    up = np.random.default_rng(13).bytes(500)
    a.session.write(up)
    got = bytearray()
    sent = []
    tx = a.session.station.build

    def build(*args, **kw):
        burst = tx(*args, **kw)
        sent.append(burst.submode)
        return burst
    a.session.station.build = build
    assert link(a, b, 10, 240, lambda: got.extend(b.session.read()) or len(got) >= len(up), seed=14)
    assert bytes(got) == up
    assert sent.count("fsk32r62-r1/2") >= 2, sent


def test_kiss_and_vara_personalities_share_one_engine():
    """One engine serves both: a KISS frame crosses; an ARQ session still
    connects and delivers; a KISS frame queued during it waits for the
    session to end (ARQ first; KISS only between sessions)."""
    import sys
    from pathlib import Path

    from data2g.kisslink import KissLink

    sys.path.insert(0, str(Path(__file__).parent))
    from test_kiss import frame

    a = Engine("W1AW", seed=21, kiss=KissLink())
    b = Engine("K2XYZ", seed=22, kiss=KissLink())
    ui = frame("APRS", "W1AW", 0x03, b"!beacon")
    a.kiss.enqueue(ui)
    assert link(a, b, 12, 30, lambda: ui in b.kiss_rx)
    b.listen()
    a.connect("K2XYZ", 2)
    assert link(a, b, 12, 60, lambda: a.session.state == S.CONNECTED and b.session.state == S.CONNECTED)
    late = frame("K2XYZ", "W1AW", 0x03, b"queued during the session")
    a.kiss.enqueue(late)
    up = np.random.default_rng(23).bytes(300)
    a.session.write(up)
    got = bytearray()
    assert link(a, b, 12, 120, lambda: got.extend(b.session.read()) or len(got) >= len(up), seed=1)
    assert bytes(got) == up and late not in b.kiss_rx  # held while the session runs
    a.session.disconnect()
    assert link(a, b, 12, 90, lambda: late in b.kiss_rx, seed=2)
    assert a.session.state == S.CLOSED


class _Busy:
    """A receiver whose BUSY is set by hand."""

    def __init__(self, until: float):
        self.until, self.t = until, 0.0

    @property
    def busy(self):
        return self.t < self.until

    channel_busy = busy

    def feed(self, x):
        self.t += len(x) / FS
        return []

    def reset(self):
        pass


def _first_tx(e: Engine, seconds: float) -> float | None:
    for i in range(int(seconds * FS / BLOCK)):
        if e.step(np.zeros(BLOCK))[1]:
            return (i + 1) * BLOCK / FS
    return None


def test_kiss_waits_out_busy_but_not_a_stuck_one():
    from data2g.kisslink import KissLink

    e = Engine("W1AW", seed=1, kiss=KissLink(busy_limit_s=5.0))
    e.receiver = _Busy(until=1e9)
    e.kiss.enqueue(b"frame")
    assert 5.0 <= _first_tx(e, 8) <= 5.2


def test_kiss_stations_queued_under_one_burst_do_not_all_collide():
    """Both queue under the same burst: p-persistence spreads them over
    1 s slots (longer than our 0.44-0.79 s to sense a burst). A reply on a
    free channel goes at once."""
    from data2g.kisslink import KissLink

    def starts(seed, persist):
        out = []
        for s in (seed, seed + 1000):
            e = Engine("W1AW", seed=s, kiss=KissLink(persist=persist))
            e.receiver = _Busy(until=2.0)
            e.kiss.enqueue(b"frame")
            out.append(_first_tx(e, 60))
        return out

    hit = [abs(a - b) < 0.8 for a, b in (starts(k, 63) for k in range(40))]
    assert sum(hit) < 0.3 * len(hit)  # (P + 1) / 256 = 1/4: 1/7 expected
    assert all(abs(a - b) < 0.8 for a, b in (starts(k, 255) for k in range(5)))
    e = Engine("W1AW", seed=1, kiss=KissLink())
    e.receiver = _Busy(until=0.0)
    e.kiss.enqueue(b"reply")
    assert _first_tx(e, 1) == BLOCK / FS
