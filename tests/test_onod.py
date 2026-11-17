"""OFDM nods (docs/arq.md §3c): a keyed burst where an OFDM header would be,
standing for a control-only reply. Detection on audio, and at link level:
T_ONODS until acted on, OFDM nods only to a peer that sent it and only where
the policy takes them to be heard, and every one a listener takes expands to
the control it stands for, both streams arriving whole through lost bursts
and lost nods."""

import random

import numpy as np
import pytest

from data2g import modem
from data2g.arq import frames as F
from data2g.arq import link as L
from data2g.arq import onod as O
from data2g.config import FS, LEADIN_SAMPLES

from test_arq import CTL_BYTES, payload
from test_nod import AskingPolicy, FullRx, _ctl, _reply

FAMILY_PIN = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]


def test_family():
    for band, k in O.THR:
        a, b = O.family(band, k, 1, 0, 2, 3), O.family(band, k, 1, 1, 2, 3)
        assert a.shape == (O.M, 2 * modem.BANDS[band].nc * k) and set(np.unique(a)) == {-1.0, 1.0}
        assert (a == O.family(band, k, 1, 0, 2, 3)).all() and not (a == b).all()
        # no valid header word of the band matches a sequence's head (what a stock receiver reads there)
        hdr = modem._valid_signs(band)
        assert float(np.max(hdr @ a[:, :hdr.shape[1]].T)) / hdr.shape[1] <= O.SCREEN
    # splitmix64 and the screen, as native onod::family (pinned: a change is a wire change)
    assert O.family("w", 9, 0x1234ABCD, 1, 5, 2)[0, :12].tolist() == FAMILY_PIN
    assert O.band_of("ack-4f") == O.band_of("ack-1f") == "w" and O.band_of("n10-ack-4f") == "n10"
    assert O.band_of("qpsk-r1/2") is None and O.band_of("fsk16r25-r1/3") is None and O.band_of("?") is None


@pytest.mark.parametrize("band", sorted(O.K))
def test_detect(band):
    k = O.K[band]
    rng = np.random.default_rng(5)
    fam = O.family(band, k, 77, 1, 4, 6)
    p0 = 3000 + LEADIN_SAMPLES
    for m in (0, O.N_MEANINGS - 1):
        x = np.concatenate([np.zeros(3000), O.waveform(band, k, fam[m]), np.zeros(FS)])
        assert len(O.waveform(band, k, fam[m])) == round(O.seconds(band, k) * FS)
        x = x + rng.normal(0, 0.3, len(x))
        got = O.detect(x, p0 - 1200, p0 + 1200, 0.0, band, k, fam)
        assert got is not None and got[0] == m and abs(got[1] - p0) <= 2, got
        assert O.detect(x, p0 - 1200, p0 + 1200, 0.0, band, k, O.family(band, k, 77, 1, 5, 6)) is None  # another turn's
        assert O.detect(x, p0 + 2400, p0 + 4800, 0.0, band, k, fam) is None  # not in the slot
    # a reserved meaning is no nod
    x = np.concatenate([np.zeros(3000), O.waveform(band, k, fam[O.N_MEANINGS]), np.zeros(FS)])
    assert O.detect(x, p0 - 1200, p0 + 1200, 0.0, band, k, fam) is None
    for s in range(10):
        assert O.detect(np.random.default_rng(s).normal(size=3 * FS), 1800, 4200, 0.0, band, k, fam) is None


def test_no_header_where_a_nods_symbols_are():
    """The sequences are screened against the band's valid header words: read
    where its header would be, a clean nod scores under HEADER_MIN_SCORE. (A
    stock receiver's search also reads at starts a repeat or two off, where
    nothing is screened: scripts/onod_study.py --false counts those locks.)"""
    from data2g.waveform import dsp

    for band, k in O.K.items():
        fam = O.family(band, k, 9, 0, 1, 2)
        for m in range(O.N_MEANINGS):
            x = np.concatenate([np.zeros(2000), O.waveform(band, k, fam[m]), np.zeros(FS)])
            r = modem._read_header(dsp.to_baseband(x), 2000 + LEADIN_SAMPLES, band)
            assert r["hdr"] is None and r["score"] < modem.HEADER_MIN_SCORE[band], (band, m, r["score"])


MODES = {"ack-4f": (4, 4), "n10-ack-4f": (4, 4), "qpsk-r1/2": (22, 4)}  # payload bytes, RV cycle


class OfdmPolicy(AskingPolicy):
    """AskingPolicy on OFDM modes: data in qpsk-r1/2, control-only replies in an ACK mode."""

    ok = True

    def __init__(self, rng, change=0.2, p_kept=0.1):
        super().__init__(rng, change, tuple(MODES), 6, p_kept)

    def choose(self, station, escalation):
        if escalation or not station.tx.pending():
            return ("ack-4f" if self.rng.random() < 0.7 else "n10-ack-4f"), 4
        return "qpsk-r1/2", self.rng.randint(2, self.max_cw)

    def payload_bytes(self, m):
        return MODES[m][0] if m in MODES else CTL_BYTES

    def ctl_payload_bytes(self, m):
        return MODES[m][0] if m in MODES else CTL_BYTES

    def max_ctl(self, m):
        return 4

    def rv_cycle(self, m):
        return MODES[m][1]

    def ofdm_nod_ok(self):
        return self.ok


def run(seed, a_nods=True, b_nods=True, max_turns=3000):
    """test_nod.run on OFDM modes: -> (a, b, nods expanded, nods taken)."""
    rng = random.Random(seed)
    data_a, data_b = payload(rng, 1500, "mixed"), payload(rng, 100, "mixed")
    a = L.Station(0, OfdmPolicy(random.Random(seed + 1)), master=True, nod=a_nods)
    b = L.Station(1, OfdmPolicy(random.Random(seed + 2)), nod=b_nods)
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
                assert O.band_of(burst.submode) and fam == receiver.nod_family()
                p = receiver.nod_payload(meaning)
                assert p is not None
                want, have = F.Control.unpack([s.payload for s in burst.slots]), F.Control.unpack([p])
                # (the nod stands for the control whatever codewords it took in its own mode)
                for f in ("ftype", "burst_seq", "acted_on", "cum", "reply_lost", "k", "recommend", "size_hint"):
                    assert getattr(want.core, f) == getattr(have.core, f), (f, want.core, have.core)
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


@pytest.fixture
def nodding(monkeypatch):
    monkeypatch.setattr(L, "NOD", True)
    monkeypatch.setattr(L, "ONOD", True)


@pytest.mark.parametrize("seed", range(6))
def test_ofdm_nods_stand_for_their_controls(nodding, seed):
    a, b, nods = run(seed)
    assert nods[1] >= 3 and nods[1] == nods[0], nods
    assert a.onod_peer and b.onod_peer and a._onods_told and b._onods_told


@pytest.mark.parametrize("seed", range(3))
def test_no_ofdm_nods_to_a_peer_that_never_said(nodding, seed):
    """b without nods (a peer before T_ONODS) never sends it: a never nods to
    it and stops telling once b acted on its own; b's replies stay full."""
    a, b, nods = run(seed, b_nods=False)
    assert nods == [0, 0] and not a.onod_peer and a._onods_told and b.onod_peer


def test_no_ofdm_nods_where_the_policy_holds_them(nodding, monkeypatch):
    monkeypatch.setattr(OfdmPolicy, "ok", False)
    a, b, nods = run(1)
    assert nods == [0, 0] and a.onod_peer and b.onod_peer


def test_gate():
    """GearShifter.ofdm_nod_ok: none before a measurement; then all but a
    spreading path under the gate's SNR (medians)."""
    from data2g.arq import policy as G

    g = G.GearShifter()
    assert not g.ofdm_nod_ok()
    spread, snr = G.ONOD_GATE
    for spreads, snrs, ok in (([0.0, 0.1, 2.0], [-6.0, -5.0, -4.0], True),
                              ([spread, spread, 0.0], [snr - 1, snr - 2, snr + 5], False),
                              ([spread, spread, 0.0], [snr, snr + 1, snr - 5], True)):
        g.spreads, g.snrs = spreads, snrs
        assert g.ofdm_nod_ok() is ok


def test_cpp_matches(native, pure):
    """native onod::family, waveform, scores and detect against onod.py (the
    family bit for bit: it is the wire format)."""
    a = native.arq
    rng = np.random.default_rng(8)
    for band, k in O.THR:
        for key in ((77, 1, 4, 6), (0x1234ABCD, 0, 7, 0)):
            fam = O.family(band, k, *key)
            assert (a.onod_family(band, k, *key) == fam).all()
        p0 = 3000 + LEADIN_SAMPLES
        for m, sigma in ((3, 0.3), (O.N_MEANINGS - 1, 1.5), (O.N_MEANINGS, 0.3)):
            x = O.waveform(band, k, fam[m])
            assert a.onod_waveform(band, k, fam[m]) == pytest.approx(x, abs=1e-9)
            y = np.concatenate([np.zeros(3000), x, np.zeros(FS)])
            y = y + rng.normal(0, sigma, len(y))
            want, got = O.scores(y, p0 - 1200, p0 + 1200, 2.0, band, k, fam), a.onod_scores(y, p0 - 1200, p0 + 1200, 2.0, band, k, fam)
            assert got[0] == pytest.approx(want[0], rel=1e-4, abs=1e-6)
            strong = want[0] > 0.2  # (a weak sequence's best read is a near tie between starts)
            assert (got[1][strong] == want[1][strong]).all() and got[2][strong] == pytest.approx(want[2][strong])
            want, got = O.detect(y, p0 - 1200, p0 + 1200, 2.0, band, k, fam), a.onod_detect(y, p0 - 1200, p0 + 1200, 2.0, band, k, fam)
            assert (got is None) == (want is None)
            if want is not None:
                assert got[:2] == want[:2] and got[2:] == pytest.approx(want[2:], rel=1e-4)

