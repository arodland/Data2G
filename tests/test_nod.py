"""Nods (docs/arq.md §3b): keyed c16r25 sequences standing for a control-only
reply or poll. Detection on audio, and at link level, stations that nod
whenever the codebook covers the control: every nod a listener takes
expands to the control it stands for (deferred fields aside), and both
streams arrive whole through lost bursts and lost nods."""

import random

import numpy as np
import pytest

from data2g.arq import frames as F
from data2g.arq import link as L
from data2g.arq import nod as N

from test_arq import CPM_LIKE, MODES, RandomPolicy, payload


def test_codebook():
    assert len(set(N.CODEBOOK)) == N.N_MEANINGS <= N.M
    a, b = N.family(1, 0, 2, 3), N.family(1, 1, 2, 3)
    assert a.shape == (N.M, N.L) and (a == N.family(1, 0, 2, 3)).all() and not (a == b).all()


def test_detect():
    rng = np.random.default_rng(5)
    fam = N.family(77, 1, 4, 6)
    pad = np.zeros(3000)
    for m in (0, 25):
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
    monkeypatch.setattr(L, "NOD_MODE", "c40")
    monkeypatch.setattr(L, "MODES", {m: m in CPM_LIKE for m in MODES})
    monkeypatch.setattr(L, "is_cpm", bool)


@pytest.mark.parametrize("seed", range(6))
def test_nods_stand_for_their_controls(nodding, seed):
    rng = random.Random(seed)
    data_a, data_b = payload(rng, 1500, "mixed"), payload(rng, 100, "mixed")
    modes = ("c40", "c60", "m22")
    a = L.Station(0, RandomPolicy(random.Random(seed + 1), 0.2, modes, 6), master=True, nod=True)
    b = L.Station(1, RandomPolicy(random.Random(seed + 2), 0.2, modes, 6), nod=True)
    a.write(data_a)
    b.write(data_b)
    got_a, got_b, nods = bytearray(), bytearray(), [0, 0]
    burst, sender, timed_out = a.build(nod=True), a, False
    for _ in range(3000):
        receiver = b if sender is a else a
        ok = False
        if rng.random() >= 0.15:
            if burst.nod is not None:
                meaning, fam = burst.nod
                kind = N.CODEBOOK[meaning][0]
                assert (kind, fam) in receiver.nod_families() or kind == "poll" and timed_out
                if (kind, fam) in receiver.nod_families():
                    p = receiver.nod_payload(kind, meaning)
                    assert p is not None
                    want, have = F.Control.unpack([burst.slots[0].payload]), F.Control.unpack([p])
                    w, h = want.core, have.core
                    w.size_hint = h.size_hint  # deferred: as the peer's last control said
                    assert w == h, (w, h)
                    assert (F.T_DUPCTL in want.ext) == (F.T_DUPCTL in have.ext)
                    nods[0] += 1
                    ok = receiver.handle(receiver.nod_rx(kind, meaning))
                    nods[1] += ok
            else:
                ok = receiver.handle(FullRx(burst))
        got_a += a.read()
        got_b += b.read()
        assert bytes(got_b) == data_a[:len(got_b)] and bytes(got_a) == data_b[:len(got_a)]
        assert a.state != L.FAILED and b.state != L.FAILED, (a.fail_reason, b.fail_reason)
        if got_a == data_b and got_b == data_a and not a.tx.pending() and not b.tx.pending():
            break
        if ok:
            burst, sender, timed_out = receiver.build(nod=True), receiver, False
            receiver.answered()
        else:
            burst, sender, timed_out = a.on_timeout(nod=True), a, True
            assert burst is not None, a.fail_reason
    else:
        raise AssertionError("stuck")
    assert nods[1] >= 3 and nods[1] == nods[0], nods
