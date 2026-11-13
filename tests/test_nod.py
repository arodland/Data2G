"""Nods (docs/arq.md §3b): keyed c16r25 sequences standing for a control-only
reply. Detection on audio, and at link level: T_NODS until acted on, nods
only to a peer that sent it, and every nod a listener takes expands to the
control it stands for, both streams arriving whole through lost bursts and
lost nods."""

import random

import numpy as np
import pytest

from data2g.arq import frames as F
from data2g.arq import link as L
from data2g.arq import nod as N

from test_arq import CTL_BYTES, payload


def test_codebook_and_family():
    assert len(set(N.CODEBOOK)) == N.N_MEANINGS <= N.M
    a, b = N.family(1, 0, 2, 3), N.family(1, 1, 2, 3)
    assert a.shape == (N.M, N.L) and (a == N.family(1, 0, 2, 3)).all() and not (a == b).all()
    # splitmix64, as native nod::family (pinned: a change is a wire change)
    assert N.family(0x1234ABCD, 1, 5, 2)[0, :8].tolist() == FAMILY_PIN
    # the threshold holds 1e-6 false accepts per slot over a wider window: THR at +-0.15 s, more beyond
    assert N.thr(1) == N.thr(N.THR_POS) == N.THR and N.thr(31) == pytest.approx(39.82, abs=0.01)


FAMILY_PIN = [11, 12, 1, 8, 2, 11, 7, 3]


def test_detect():
    rng = np.random.default_rng(5)
    fam = N.family(77, 1, 4, 6)
    pad = np.zeros(3000)
    for m in (0, N.N_MEANINGS - 1):
        x = np.concatenate([pad, N.waveform(fam[m]), pad])
        x = x + rng.normal(0, 0.3, len(x))
        got = N.detect(x, 2700, 3300, 0.0, fam)
        assert got is not None and got[0] == m and abs(got[1] - 3000) < N.GRID.T, got
        assert N.detect(x, 2700, 3300, 0.0, N.family(77, 1, 5, 6)) is None  # another turn's family
    # a reserved meaning is no nod
    x = np.concatenate([pad, N.waveform(fam[N.N_MEANINGS]), pad])
    assert N.detect(x, 2700, 3300, 0.0, fam) is None
    for s in range(20):
        assert N.detect(np.random.default_rng(s).normal(size=12000), 2700, 3300, 0.0, fam) is None


class FullRx:
    def __init__(self, burst):
        self.burst, self.submode, self.n_cw = burst, burst.submode, len(burst.slots)

    def decode(self, i, mask_id, rv, key):
        s = self.burst.slots[i]
        return s.payload if mask_id == s.mask_id and rv == s.rv else None

    def forget(self, key):
        pass


@pytest.fixture
def nodding(monkeypatch):
    monkeypatch.setattr(L, "NOD", True)


# real mode names (the C++ station looks them up): two CPM, one OFDM; payload bytes, RV cycle
NOD_MODES = {"fsk16r25-r1/2": (40, 4), "fsk8r50-r1/2": (60, 4), "qpsk-r1/2": (22, 4)}
CPM = ("fsk16r25-r1/2", "fsk8r50-r1/2", L.NOD_MODE)


class AskingPolicy(L.Policy):
    """test_arq.RandomPolicy on real mode names, whose recommendation
    moves: a step or two in band 0, now and then a size hint or reply mode
    that differs."""

    def __init__(self, rng, change=0.2, modes=tuple(NOD_MODES), max_cw=6, p_kept=0.1):
        self.rng, self.change, self.modes, self.max_cw = rng, change, modes, max_cw
        self.dup_rng = random.Random(max_cw * 1000 + int(change * 100))
        self.mode = rng.choice(modes)
        self.idx, self.hint, self.reply, self.p_kept = 5, 1, None, p_kept

    def choose(self, station, escalation):
        if escalation:
            return max(self.modes, key=lambda m: NOD_MODES[m][0]), 2
        if self.rng.random() < self.change:
            self.mode = self.rng.choice(self.modes)
        return self.mode, self.rng.randint(1, self.max_cw)

    def payload_bytes(self, m):
        return NOD_MODES[m][0] if m in NOD_MODES else CTL_BYTES

    def ctl_payload_bytes(self, m):
        return CTL_BYTES if m in CPM else NOD_MODES[m][0]

    def max_ctl(self, m):
        return 1 if m in CPM else 4

    @property
    def want_dup(self):
        return self.dup_rng.random() < 0.3

    def rv_cycle(self, m):
        return NOD_MODES[m][1]

    def recommend(self, station):
        if self.rng.random() < 0.3:
            self.idx = min(15, max(0, self.idx + self.rng.choice((-2, -1, 1, 2))))
        if self.rng.random() < self.p_kept:
            self.hint = self.rng.randrange(4)
        if self.rng.random() < self.p_kept:
            self.reply = self.rng.choice((None, 3, 7))
        return self.idx, self.hint, self.reply


def _reply(ext):
    return ext[F.T_REPLY][0] if ext.get(F.T_REPLY) else None


def _ctl(burst):
    return F.Control.unpack([burst.slots[0].payload])


def run(seed, a_nods=True, b_nods=True, max_turns=3000):
    """Lockstep, 15% of bursts lost: -> (a, b, nods expanded, nods taken)."""
    rng = random.Random(seed)
    data_a, data_b = payload(rng, 1500, "mixed"), payload(rng, 100, "mixed")
    a = L.Station(0, AskingPolicy(random.Random(seed + 1)), master=True, nod=a_nods)
    b = L.Station(1, AskingPolicy(random.Random(seed + 2)), nod=b_nods)
    a.write(data_a)
    b.write(data_b)
    got_a, got_b, nods = bytearray(), bytearray(), [0, 0]
    burst, sender = a.build(nod=True), a
    for _ in range(max_turns):
        receiver = b if sender is a else a
        ok = False
        if rng.random() >= 0.15:
            if burst.nod is not None:
                meaning, fam = burst.nod
                assert fam == receiver.nod_family()
                p = receiver.nod_payload(meaning)
                assert p is not None
                want, have = _ctl(burst), F.Control.unpack([p])
                assert want.core == have.core, (want.core, have.core)
                for t in (F.T_DUPCTL, F.T_CHAT):
                    assert (t in want.ext) == (t in have.ext)
                assert _reply(want.ext) == _reply(have.ext)
                nods[0] += 1
                ok = receiver.handle(receiver.nod_rx(meaning))
                nods[1] += ok
            else:
                ok = receiver.handle(FullRx(burst))
        got_a += a.read()
        got_b += b.read()
        assert bytes(got_b) == data_a[:len(got_b)] and bytes(got_a) == data_b[:len(got_a)]
        assert a.state != L.FAILED and b.state != L.FAILED, (a.fail_reason, b.fail_reason)
        if got_a == data_b and got_b == data_a and not a.tx.pending() and not b.tx.pending():
            return a, b, nods
        if ok:
            burst, sender = receiver.build(nod=True), receiver
            receiver.answered()
        else:
            burst, sender = a.on_timeout(), a
            assert burst is not None, a.fail_reason
    raise AssertionError("stuck")


@pytest.mark.parametrize("seed", range(6))
def test_nods_stand_for_their_controls(nodding, seed):
    a, b, nods = run(seed)
    assert nods[1] >= 3 and nods[1] == nods[0], nods
    assert a.nod_peer and b.nod_peer and a._nods_told and b._nods_told


@pytest.mark.parametrize("seed", range(3))
def test_no_nods_to_a_peer_that_never_said(nodding, seed):
    """b without nods (a v5 peer, or DATA2G_NOD=0) never sends T_NODS: a
    never nods to it (it stops telling once b acted on it, as with any
    peer); b's own replies stay full."""
    a, b, nods = run(seed, b_nods=False)
    assert nods == [0, 0] and not a.nod_peer and a._nods_told and b.nod_peer


def test_t_nods_until_acted_on(nodding):
    modes = ("fsk16r25-r1/2",)
    a = L.Station(0, AskingPolicy(random.Random(1), 0.0, modes, p_kept=0.0), master=True, nod=True)
    b = L.Station(1, AskingPolicy(random.Random(2), 0.0, modes, p_kept=0.0), nod=True)
    a.write(payload(random.Random(3), 2000, "mixed"))
    x = a.build(nod=True)
    assert F.T_NODS in _ctl(x).ext
    a.on_timeout()  # lost: still unacknowledged
    x = a.on_timeout(allow_repeat=False)
    assert F.T_NODS in _ctl(x).ext and not a._nods_told
    assert b.handle(FullRx(x)) and b.nod_peer
    r = b.build(nod=True)
    assert r.nod is None and F.T_NODS in _ctl(r).ext  # a has not told b yet: b sends in full
    assert a.handle(FullRx(r)) and a._nods_told and a.nod_peer
    x = a.build(nod=True)
    assert F.T_NODS not in _ctl(x).ext


def test_a_kept_field_change_goes_in_full(nodding):
    """A control-only reply whose size_hint (or T_REPLY) differs from my
    previous burst's goes in full: a nod would leave the peer's old one."""
    modes = ("fsk16r25-r1/2",)
    a = L.Station(0, AskingPolicy(random.Random(1), 0.0, modes, p_kept=0.0), master=True, nod=True)
    b = L.Station(1, AskingPolicy(random.Random(2), 0.0, modes, p_kept=0.0), nod=True)
    a.write(payload(random.Random(3), 2000, "mixed"))
    for _ in range(3):  # a's data, b's control-only reply: nods once anchored
        assert b.handle(FullRx(a.build(nod=True)))
        r = b.build(nod=True)
        assert a.handle(FullRx(r))
    assert r.nod is not None
    for field, value in (("hint", 3), ("reply", 7)):
        assert b.handle(FullRx(a.build(nod=True)))
        setattr(b.policy, field, value)
        r = b.build(nod=True)
        assert r.nod is None
        assert a.handle(FullRx(r))
        assert (a.peer_size_hint, a.peer_reply_recommend) == (b.policy.hint, b.policy.reply)
        assert b.handle(FullRx(a.build(nod=True)))  # unchanged again: nods
        r = b.build(nod=True)
        assert r.nod is not None
        assert F.Control.unpack([a.nod_payload(r.nod[0])]).core.size_hint == b.policy.hint
        assert a.handle(a.nod_rx(r.nod[0]))


def test_cpp_detects_as_python(native, reference):
    """The C++ family is the Python one, and its detector decides the same
    (meaning and start; the score to rounding) on clean, weak and absent nods."""
    family, detect = reference(N, "family"), reference(N, "detect")
    for key in ((0x1234ABCD, 1, 5, 2), (7, 0, 0, 7)):
        assert (np.asarray(native.arq.nod_family(*key)) == family(*key)).all()
    fam = family(77, 1, 4, 6)
    rng = np.random.default_rng(9)
    pad = np.zeros(6000)
    hits = 0
    for trial in range(24):
        m = trial % N.M  # reserved meanings too
        x = np.concatenate([pad, N.waveform(fam[m]) if trial % 6 else np.zeros(N.span()), pad])
        x = x + rng.normal(0, [0.3, 1.5, 2.5][trial % 3], len(x))
        lo, hi = (5700, 6300) if trial % 2 else (1200, 10800)  # +-0.6 s: nod.thr raised
        want, got = detect(x, lo, hi, 1.5, fam), native.arq.nod_detect(x, lo, hi, 1.5, fam)
        assert (want is None) == (got is None), (trial, want, got)
        if want is not None:
            hits += 1
            assert want[:2] == tuple(got[:2]) and abs(want[2] - got[2]) < 1e-6 * max(1.0, abs(want[2])), (want, got)
    assert hits >= 6
