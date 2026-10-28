"""Primed reception (docs/arq.md §3a): a peer control that fails blind is
decoded again with what the waiting station expects of it.

Link level: test_arq's lockstep stations through a fake PHY whose control
slots fail blind and pass a primed retry only when the hypothesis' known
bits agree with what was sent, or (adversarial) pass forged controls that
agree with the hypothesis everywhere it pinned. Python and C++ must send
identical bursts. PHY level: ModemRx.decode_known on real codewords.
"""

import logging
import random

import numpy as np
import pytest

from data2g import codes, cpm
from data2g.arq import frames as F
from data2g.arq import link as L
from data2g.arq import phy as PHY
from data2g.arq.burst import Slot, TxBurst, ctl_mask

from test_arq import FakeRx, RandomPolicy, payload
from test_native_arq_link import _burst, logged


class PrimeRx(FakeRx):
    """FakeRx whose control slots fail blind with probability `p_ctl`, and
    whose primed retries pass when the hypothesis agrees with what was sent
    (`p_known`), or, with probability `forge`, return a forged control: the
    hypothesis' known bits, random where it knew nothing (a false accept
    that passed the CRC)."""

    def __init__(self, burst, rng, p_cw, store, stats, p_ctl=0.0, p_known=0.8, forge=0.0):
        super().__init__(burst, rng, p_cw, store, stats)
        for i, s in enumerate(burst.slots):
            if s.mask_id[2] >= F.SEQ_MOD and rng.random() < p_ctl:
                self.good[i] = False
        self.p_known, self.forge = p_known, forge

    def decode_known(self, slot, mask_id, template, tmask, pair=False):
        self.stats["primed"] += 1
        if slot >= self.n_cw:
            return None
        s = self.burst.slots[slot]
        if mask_id == s.mask_id and len(template) == len(s.payload) and \
                all((a ^ b) & m == 0 for a, b, m in zip(s.payload, template, tmask)):
            if self.rng.random() < self.p_known:
                self.stats["primed_ok"] += 1
                return s.payload
        if self.rng.random() < self.forge:
            self.stats["forged"] += 1
            return bytes(t & m | self.rng.randrange(256) & ~m & 255 for t, m in zip(template, tmask))
        return None


def lockstep(seed, cls_a, cls_b, p_burst, p_ctl, n_a, n_b, forge=0.0, modes=("m4", "m22", "m46"), max_cw=20,
             max_turns=4000):
    """Two stations, a's data to b and b's to a, through PrimeRx: -> (trace, outcome, stats)."""
    rng = random.Random(seed)
    data_a, data_b = payload(rng, n_a, "mixed"), payload(rng, n_b, "mixed")
    a = cls_a(0, RandomPolicy(random.Random(seed + 1), 0.2, modes, max_cw), master=True)
    b = cls_b(1, RandomPolicy(random.Random(seed + 2), 0.2, modes, max_cw))
    a.write(data_a)
    b.write(data_b)
    stores, stats, trace = {0: {}, 1: {}}, {"mismatch": 0, "primed": 0, "primed_ok": 0, "forged": 0}, []
    got_a, got_b = bytearray(), bytearray()
    burst, sender = a.build(), a
    for _ in range(max_turns):
        trace.append((sender.direction, _burst(burst)))
        receiver = b if sender is a else a
        ok = False
        if rng.random() >= p_burst:
            ok = receiver.handle(PrimeRx(burst, rng, 0.1, stores[receiver.direction], stats, p_ctl, forge=forge))
        got_a += a.read()
        got_b += b.read()
        # the stream is never corrupted, whatever was accepted
        assert bytes(got_b) == data_a[:len(got_b)] and bytes(got_a) == data_b[:len(got_a)]
        if a.state == L.FAILED or b.state == L.FAILED:
            return trace, ("failed", a.fail_reason or b.fail_reason), stats
        if got_a == data_b and got_b == data_a and not a.tx.pending() and not b.tx.pending():
            return trace, ("done", dict(a.stats), dict(b.stats)), stats
        if ok:
            burst, sender = receiver.build(), receiver
            receiver.answered()
        else:
            burst, sender = a.on_timeout(), a
            if burst is None:
                return trace, ("failed", a.fail_reason), stats
    return trace, ("stuck",), stats


@pytest.mark.parametrize("seed", range(8))
def test_primed_rescues_controls(seed):
    """Control codewords lost blind 30% of the time: primed retries rescue many, and
    every run delivers both streams exactly."""
    modes = [("m4", "m22", "m46"), ("m4", "m46", "c60", "c40")][seed % 2]
    _, out, stats = lockstep(700 + seed, L.Station, L.Station, 0.05, 0.3, 3000, 1500, modes=modes)
    assert out[0] == "done", out
    # a primed codeword counts once its whole control is taken (another may still fail)
    assert 0 < out[1].get("rx_primed", 0) + out[2].get("rx_primed", 0) <= stats["primed_ok"]


def test_prime_off(monkeypatch):
    monkeypatch.setattr(L, "PRIME", False)
    _, out, stats = lockstep(700, L.Station, L.Station, 0.05, 0.3, 3000, 1500)
    assert out[0] == "done" and stats["primed"] == 0


@pytest.mark.parametrize("seed", range(16))
def test_forged_primed_controls_never_corrupt(seed):
    """Adversarial: a third of primed retries return a forged control (a
    false accept). The stream stays an exact prefix (asserted every turn),
    and every run ends in delivery or a bounded disconnect (§10)."""
    modes = [("m4", "m22", "m46"), ("m4", "m46", "c60", "c40")][seed % 2]
    _, out, stats = lockstep(800 + seed, L.Station, L.Station, 0.1, 0.5, 2000, 2000, forge=0.33, modes=modes)
    assert out[0] in ("done", "failed"), out
    assert stats["forged"] > 0


def test_expect_hits_a_clean_reply():
    """No loss: the peer's reply is the first hypothesis for every burst
    after the first exchange (a fixed policy, data one way) until the write's
    tail (its T_INLINE is not predicted)."""
    class Fixed(RandomPolicy):
        def choose(self, station, escalation):
            return "m22", 6

        @property
        def want_dup(self):
            return False

    rng = random.Random(3)
    a, b = L.Station(0, Fixed(rng), master=True), L.Station(1, Fixed(rng))
    a.write(payload(rng, 600, "random"))
    burst, sender, hits = a.build(), a, []
    for turn in range(12):
        receiver = b if sender is a else a
        rx = FakeRx(burst, rng, 0.0, {}, {"mismatch": 0})
        rx.decode_known = lambda *args: None  # primed reception on: expect() is asked by the test only
        true = b"".join(s.payload for s in burst.slots if s.mask_id[2] >= F.SEQ_MOD and s.rv == 0)
        hs = receiver.expect(rx)
        if 2 <= turn < 10:
            t, m, _ = hs[0]
            hits.append(len(t) == len(true) and all((x ^ y) & z == 0 for x, y, z in zip(t, true, m)))
            if not hits[-1]:
                print(turn, receiver.master, true.hex(), [(t.hex(), m.hex()) for t, m, _ in hs])
        assert receiver.handle(rx)
        burst, sender = receiver.build(), receiver
        receiver.answered()
    assert all(hits) and len(hits) == 8


# --- C++ parity -------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_cpp_stations_prime_identically(native, pure, caplog, seed):
    """Hypotheses, primed retries and drops: the same calls in the same order,
    so the same bursts and log lines, Python, C++ and mixed."""
    caplog.set_level(logging.INFO)
    modes = [("m4", "m22", "m46"), ("m4", "m46", "c60", "c40")][seed % 2]
    py, cc = L.Station, native.arq.Station
    args = (0.1, 0.5, 2500, 1200, [0.0, 0.2][seed % 2], modes)
    want, want_log = logged(caplog, lockstep, 600 + seed, py, py, *args)
    assert want[1][0] in ("done", "failed") and want[2]["primed"] > 0
    got, got_log = logged(caplog, lockstep, 600 + seed, cc, cc, *args)
    assert got == want and got_log == want_log
    for pair in ((py, cc), (cc, py)):
        assert lockstep(600 + seed, *pair, *args) == want


# --- PHY: ModemRx.decode_known -------------------------------------------------------------

def _cpm_rx(ctl: bytes, soft: np.ndarray, grid="c16r25"):
    """A received CPM control-only burst whose control slot reads `soft`."""
    spec = cpm.SPECS[next(n for n, s in cpm.SPECS.items() if s.grid == grid)]
    r = dict(family="cpm", spec=spec, band=grid, n_cw=1, n_ctl_slots=1, dup=False, soft=[soft], E=None)
    return PHY.ModemRx(r, {})


def _erased(spec, payload: bytes, mask: int, keep: float, seed: int, flip: float = 0.0) -> np.ndarray:
    """The codeword's soft bits (index 0) as +-4, a deterministic share `keep`
    of them kept and the rest erased (0), a share `flip` of the kept wrong."""
    c = codes.encode(spec, payload, 0, mask, 0)
    x = 4.0 * (1 - 2.0 * c)
    r = np.random.default_rng(seed)
    x[r.random(len(x)) >= keep] = 0.0
    x[r.random(len(x)) < flip] *= -1
    return x


def test_decode_known_rescues_a_cpm_control():
    """An ACK control (8 B) that neither the plain decode nor the padding
    cuts recover: the whole control known but the cumulative ACK (7 bits)
    does, and only with the right template. Erasures (not noise) make the
    premise hold on every platform: 30% of the soft bits kept, seed 5."""
    spec = cpm.CTL["c16r25"]
    mid = ctl_mask(1, 0, 0x5A5A)
    m = PHY.mask_value(mid)
    ctl = F.Control(F.Core(burst_seq=3, acted_on=2, cum=77, recommend=50, size_hint=2),
                    {F.T_REPLY: bytes([50]), F.T_DUPCTL: b""}).pack(codes.payload_bytes(spec))[0]
    rx = _cpm_rx(ctl, _erased(spec, ctl, m, 0.3, 5))
    assert rx.decode(0, mid, 0, None) is None
    known = bytearray(b"\xff" * len(ctl))
    known[2] &= 0x80  # cum: core bits 21..15
    known[1] &= 0xC0
    assert rx.decode_known(0, mid, ctl, bytes(known)) == ctl
    wrong = bytearray(ctl)
    wrong[6] ^= 1  # another T_REPLY value, pinned
    assert _cpm_rx(ctl, _erased(spec, ctl, m, 0.3, 5)).decode_known(0, mid, bytes(wrong), bytes(known)) is None
    other = ctl_mask(1, 0, 0x1234)  # another session's mask: its CRC fails
    assert _cpm_rx(ctl, _erased(spec, ctl, m, 0.3, 5)).decode_known(0, other, ctl, bytes(known)) is None


def test_decode_known_on_noise_and_erasures():
    """Nothing there (all erased, or noise): no primed control, even with
    every payload bit known (the CRC is never known; the score bounds noise)."""
    spec = cpm.CTL["c8r50"]
    mid = ctl_mask(0, 0, 0x5A5A)
    ctl = F.Control(F.Core(cum=5)).pack(codes.payload_bytes(spec))[0]
    full = b"\xff" * len(ctl)
    assert _cpm_rx(ctl, np.zeros(spec.coded_bits), "c8r50").decode_known(0, mid, ctl, full) is None
    rng = np.random.default_rng(11)
    for _ in range(20):
        noise = rng.normal(0, 2.0, spec.coded_bits)
        assert _cpm_rx(ctl, noise, "c8r50").decode_known(0, mid, ctl, full) is None


def test_known_bits_and_score():
    spec = cpm.CTL["c32r62"]
    tmpl, mask = bytes([0xA5, 0x0F]), bytes([0xFF, 0x0F])
    k = codes.known_bits(spec, tmpl, mask)
    pb = codes.payload_bytes(spec)
    assert list(k[:8]) == [1, 0, 1, 0, 0, 1, 0, 1] and list(k[8:16]) == [-1] * 4 + [1] * 4
    assert (k[16:8 * pb + codes.crc_bits(spec)] == -1).all() and (k[8 * pb + codes.crc_bits(spec):] == 0).all()
    p = bytes(range(pb))
    x = 30.0 * (1 - 2.0 * codes.encode(spec, p, 0, 7, 0))
    assert codes.score(spec, x, p, 7, 0) == pytest.approx(spec.coded_bits * np.log(2), rel=1e-6)
    assert codes.score(spec, np.zeros(spec.coded_bits), p, 7, 0) == 0.0


def test_cpp_decode_known_matches(native, pure):
    """ModemRx.decode_known, known_bits and score: C++ against Python on the same soft bits."""
    spec = cpm.CTL["c16r25"]
    mid = ctl_mask(1, 0, 0x5A5A)
    m = PHY.mask_value(mid)
    rng = random.Random(4)
    nphy = native.phy
    for trial in range(30):
        ctl = F.Control(F.Core(burst_seq=rng.randrange(8), cum=rng.randrange(128)),
                        {F.T_REPLY: bytes([rng.randrange(64)])}).pack(codes.payload_bytes(spec))[0]
        soft = _erased(spec, ctl, m, rng.uniform(0.2, 0.5), trial, rng.uniform(0.0, 0.1))
        known = bytes(rng.choice([0, 0xFF, 0xF0]) for _ in ctl)
        r = dict(family="cpm", spec=cpm.SPECS["fsk16r25-r1/3"], band="c16r25", n_cw=1, n_ctl_slots=1, dup=False,
                 soft=[soft], E=None)
        want = PHY.ModemRx(dict(r), {}).decode_known(0, mid, ctl, known)
        got = nphy.ModemRx(dict(r), {}).decode_known(0, mid, ctl, known)
        assert got == want
        assert list(native.codes.known_bits(spec.name, ctl, known)) == list(codes.known_bits(spec, ctl, known))
        assert native.codes.score(spec.name, soft, ctl, m, 0) == pytest.approx(codes.score(spec, soft, ctl, m, 0))


# --- connect frames ---------------------------------------------------------------------

class CpmConnect(L.Policy):
    """Connects in a CPM-like mode: one 20 B control codeword (a compact CONNECT)."""

    def choose(self, station, escalation):
        return "c60", 2

    def payload_bytes(self, m):
        return 60

    def ctl_payload_bytes(self, m):
        return 20

    def max_ctl(self, m):
        return 1

    def rv_cycle(self, m):
        return 4

    def connect_mode(self, cap, tries=0):
        return "c60"

    def airtime(self, m, n_cw, dup=False):
        return 3.0


class BlindFail:
    """Every blind decode fails; a primed one returns what was sent when the
    hypothesis agrees with it, else (`forge`) the hypothesis with random bits
    where it knew nothing."""

    def __init__(self, burst, forge=None):
        self.burst, self.submode, self.n_cw, self.forge = burst, burst.submode, len(burst.slots), forge
        self.asked = []

    def decode(self, slot, mask_id, rv, key):
        return None

    def forget(self, key):
        pass

    def decode_known(self, slot, mask_id, template, tmask, pair=False):
        s = self.burst.slots[slot]
        self.asked.append(tuple(mask_id))
        if self.forge is not None:
            return bytes(t & m | self.forge.randrange(256) & ~m & 255 for t, m in zip(template, tmask))
        if mask_id == s.mask_id and all((a ^ b) & m == 0 for a, b, m in zip(s.payload, template, tmask)):
            return s.payload
        return None


def handshake(cls, forge=None, t_turn_b=1.0):
    a = cls("W1AW", CpmConnect(), rng=random.Random(1))
    b = cls("K2XYZ", CpmConnect(), rng=random.Random(2))
    b.t_turn = t_turn_b
    b.listen()
    a.connect("K2XYZ", 2, 0.0)
    connect = a.poll(0.0)
    a.on_tx_end(connect, 1.0)
    b.on_rx(BlindFail(connect, forge), 1.5)
    ack = b.poll(10.0)
    if ack is not None:
        b.on_tx_end(ack, 11.0)
        a.on_rx(BlindFail(ack, forge), 11.5)
    return a.state, b.state, list(a.events), list(b.events)


def test_primed_connect_and_connect_ack():
    """Neither connect frame decodes blind: the callee primes the CONNECT with
    its own call, the caller its CONNECT_ACK with its nonce, and they connect.
    A callee with another t_turn still gets through (its second hypothesis)."""
    from data2g.arq import session as S
    assert handshake(S.Session)[:2] == (S.CONNECTED, S.CONNECTED)
    assert handshake(S.Session, t_turn_b=2.5)[:2] == (S.CONNECTED, S.CONNECTED)


def test_forged_connect_frames_rejected():
    """Forged primed frames (random where unknown): a CONNECT's caller rarely
    unpacks (10 characters in a 40 of 64 alphabet), a CONNECT_ACK's cap over
    2 is dropped; whatever is accepted is a well-formed frame."""
    from data2g.arq import session as S
    rng = random.Random(9)
    states = [handshake(S.Session, forge=rng)[1] for _ in range(200)]
    assert states.count(S.CONNECTED) < 10  # (5/8)^10 of them, about 1%


def test_cpp_sessions_prime_connect_identically(native, pure):
    from data2g.arq import session as S
    for kw in (dict(), dict(t_turn_b=2.5)):
        assert handshake(native.arq.Session, **kw) == handshake(S.Session, **kw)
    for seed in range(20):
        assert handshake(native.arq.Session, forge=random.Random(seed)) == handshake(S.Session, forge=random.Random(seed))


# --- the expected slot (cpm.prime_lock, receiver.ReplySlot) ----------------------------------

def _faded_cpm(seed):
    """A control-only fsk16r25 burst through MPP at -17 dB (phy_session's
    channel, numpy's generator: the same on every platform)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import phy_session as PS

    from data2g.receiver import receive_any
    rng = np.random.default_rng(seed)
    b = TxBurst("fsk16r25-r1/3", [Slot(ctl_mask(1, 0, 7), 0, bytes(rng.integers(0, 256, 20, dtype=np.uint8)))], 0, 2)
    y = PS.ContinuousChannel("mpp", -17.0, seed, 60.0).apply(PHY.tx_audio(b), 10.0)
    return y, receive_any(y, lead=int(PS.PAD_S * PS.FS) + PS.FS // 2), int(PS.PAD_S * PS.FS), PS.CFO_HZ


def test_prime_lock_finds_what_blind_acquisition_lost():
    """Seed 0: no blind path locks (the front faded); in a +-0.15 s window at
    the known CFO the expected slot finds it, start and word right."""
    y, r, p0, cfo = _faded_cpm(0)
    assert r is None
    lock = cpm.prime_lock(cpm.GRIDS["c16r25"], y, p0 - 1200, p0 + 1200, cfo)
    assert lock is not None and lock["start"] == p0 and lock["spec"].name == "fsk16r25-r1/3" and lock["n_data"] == 0
    assert lock["score"] >= cpm.PRIME_LLR


def test_prime_lock_on_noise():
    rng = np.random.default_rng(3)
    for g in cpm.GRIDS.values():
        for _ in range(5):
            assert cpm.prime_lock(g, rng.normal(0, 1.0, 6 * 8000), 1200, 3600, 4.5) is None


def test_reply_slot():
    from data2g.receiver import ReplySlot
    s = ReplySlot()
    assert s.window(10.0) is None
    s.heard(0.35, None)
    assert s.window(10.0) is None  # no CFO yet
    for g in (0.3, 0.4, 5.0, None):  # 5 s: not a reply
        s.heard(g, 4.5)
    lo, hi = s.window(10.0)
    assert (lo, hi) == pytest.approx((10.35 - s.TOL_S, 10.35 + s.TOL_S)) and s.cfo == 4.5


def test_cpp_prime_lock_matches(native, pure):
    for seed in range(6):
        y, _, p0, cfo = _faded_cpm(seed)
        for g in cpm.GRIDS.values():
            want = cpm.prime_lock(g, y, p0 - 1200, p0 + 1200, cfo)
            got = native.cpm.prime_lock(g.name, y, p0 - 1200, p0 + 1200, cfo)
            assert (got is None) == (want is None)
            if want is not None:
                assert got["start"] == want["start"] and got["spec"] == want["spec"].name
                assert got["n_data"] == want["n_data"] and got["cfo"] == pytest.approx(want["cfo"])
                assert got["score"] == pytest.approx(want["score"], rel=1e-9)


def _stream(rcv, y, expect=None):
    if expect is not None:
        rcv.expect(*expect)
    out = []
    for i in range(0, len(y), 800):
        out += rcv.feed(y[i:i + 800])
    return out


def test_receiver_expected_slot():
    """The streaming receiver: seed 0's burst, which no blind path locks,
    is found in the expected slot (and decodes), and not without it."""
    from data2g.arq.engine import receive_accept
    from data2g.receiver import Receiver
    y, _, p0, cfo = _faded_cpm(0)
    y = np.concatenate([y, np.random.default_rng(1).normal(0, np.std(y[:p0]), 2 * 8000)])
    assert not _stream(Receiver(receive_accept(), cpm_grids=tuple(cpm.GRIDS)), y)
    ev = _stream(Receiver(receive_accept(), cpm_grids=tuple(cpm.GRIDS)), y, (["c16r25"], p0 - 1200, p0 + 1200, cfo))
    assert [k for k, _ in ev] == ["header", "burst"]
    assert ev[0][1]["start"] == p0 and ev[1][1]["rx"] is not None and ev[1][1]["rx"]["spec"].name == "fsk16r25-r1/3"
