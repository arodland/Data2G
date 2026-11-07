"""Hand burst (data2g/hand.py) and the host primitives over it (Engine,
Host): a decoded round trip, a full round between two live stacks, and the
command replies."""

import time

import numpy as np

from data2g import hand, hfchannel
from data2g.arq.engine import Engine
from data2g.config import FS, SNR_REF_BW_HZ
from data2g.host import Host

BLOCK = FS // 10


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


def test_hand_decode_roundtrip():
    """One channel, 2 Hz off and at 6 dB: the payload comes back, with its
    landing time reported to within a search step."""
    rng = np.random.default_rng(7)
    tag, c, payload = 0x5A, 20, int(rng.integers(0, 2 ** hand.PAYLOAD))
    jitter = int(rng.integers(0, hand.SEARCH_SAMPLES))  # lands this late
    y = np.concatenate([np.zeros(hand.SEARCH_SAMPLES + jitter),
                        hand.modulate(tag, hand.payload_bits(payload), hand.channel_freq(c)),
                        np.zeros(hand.SEARCH_SAMPLES)])
    y = hfchannel.awgn(hfchannel.freq_shift(y, 2.0), 6.0, seed=3)
    got, = hand.decode_window(y, tag, [c], hand.SEARCH_SAMPLES)
    assert got["payload"] == payload
    assert got["snr"] > 0
    assert abs(got["dt_ms"] - 1000.0 * jitter / FS) < 30


def test_hand_round_between_two_stations():
    """A (net control) arms a listen, sends the anchor burst (CQ as ANY
    TRAFFIC's stand-in), and gets B's Hand back on its channel."""
    a, b = Engine("W1AW", seed=1), Engine("K2XYZ", seed=2)
    tag, payload, d = 0x5A, 0x1234567890ABC, 1.0
    a.hand_listen(d, tag, [20])
    a.send_cq("W1AW", 2)
    b_ev = []

    def b_heard():
        b_ev.extend(b.take_hand_events())
        return any(ev.startswith("REF") for ev in b_ev)

    assert link(a, b, 12, 20, b_heard)  # B hears the CQ: a REF line
    ref = int(next(ev for ev in b_ev if ev.startswith("REF")).split()[1])
    b.hand_send(ref, tag, d, 20, payload)
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
    assert abs(float(dt)) < 600  # shared clock: block quantization and measurement error only
    assert any(ev.startswith("HAND SENT 20") for ev in b_ev)


def test_hand_commands_reply():
    h = Host(Engine("W1AW", seed=1))
    h.command("HAND LISTEN 3.0 90 20 21")
    assert h.out_cmd == ["OK"]
    h.out_cmd.clear()
    for line in ("HAND LISTEN 3.0 90", "HAND LISTEN 3.0 90 20 40",
                 "HAND SEND 1 90 3.0 20 1", "HAND SEND 1 90 3.0 20 zz"):
        h.command(line)  # no burst heard yet: no refs, bad channel, bad hex
        assert h.out_cmd == ["WRONG"], line
        h.out_cmd.clear()
