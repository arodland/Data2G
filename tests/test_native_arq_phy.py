"""native phy and kisslink (native/core/arq/phy, native/core/kisslink)
against data2g/arq/phy.py and data2g/kisslink.py.

Decisions are compared for equality: which slots decode, under which
masks, with or without DD, and every KISS burst built. Floats (audio, soft
bits, stored buffers, measurements) to stated tolerances. DD runs with no
wall-clock budget on both sides: with the real clock C++ finishes more DD
passes than numpy within DD_BUDGET_S, so results may legitimately differ.
"""

import numpy as np
import pytest

import conftest
from data2g import codes, cpm, hfchannel, modem
from data2g import kisslink as KL
from data2g.arq import phy as PHY
from data2g.arq.link import Slot, TxBurst, ctl_mask, data_mask
from data2g.config import FS, SUBMODES
from data2g.hfchannel import FadingPreset

TOL = 1e-9  # audio, soft bits, buffers: relative to the largest value
MEAS_TOL = 1e-6
# soft bits stored after DD: a rescued slot's estimate stays, and it was
# refined from a failed decode's posterior (float32, numpy's tanh/log an
# ULP off: test_native_ldpc compares those to 1e-3)
BUF_TOL = 1e-4  # measurements (effective MI sums over float32-rounded tables)


@pytest.fixture
def pure(monkeypatch):
    """The Python modules as written, whether or not --native substituted them."""
    for (module, attr), fn in conftest._originals.items():
        monkeypatch.setattr(module, attr, fn)


def close(got, want, tol=TOL):
    want = np.asarray(want)
    s = max(float(np.max(np.abs(want))) if want.size else 0.0, 1.0)
    np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=tol * s)


def payloads(spec, n, rng):
    return [bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8)) for _ in range(n)]


def ofdm_burst(name, n, key=7, rv=0, seed=0):
    pl = payloads(SUBMODES[name], n, np.random.default_rng(seed))
    return TxBurst(name, [Slot((key, 0, i), rv, p) for i, p in enumerate(pl)], 0), pl


def hear(b, snr=None, seed=0, fading=None):
    x = np.concatenate([np.zeros(2400), PHY.tx_audio(b), np.zeros(2400)])
    if snr is not None:
        x = hfchannel.apply_channel(x, snr_db=snr, seed=seed, fading_preset=fading)
    return modem.receive(x)


def rx_pair(native, r, dd=True):
    """(Python's ModemRx, the native one) over r, no DD budget, each its own store."""
    py = PHY.ModemRx({k: v for k, v in r.items() if k != "_soft"}, {})
    nat = native.phy.ModemRx({k: v for k, v in r.items() if k != "_soft"}, {}, None, dd)
    return py, nat


def test_mask_value(native, pure):
    rng = np.random.default_rng(0)
    for _ in range(500):
        m = (int(rng.integers(0, 1 << 16)), int(rng.integers(0, 2)), int(rng.integers(0, 132)))
        assert native.phy.mask_value(m) == PHY.mask_value(m)
    assert native.phy.mask_value((0, 1, 5)) == 0


def test_tx_audio(native, pure):
    for name, n in (("qpsk-r1/2", 3), ("w48-16qam-r1/2", 5), ("n4-qpsk-r1/3", 2), ("ack-4f", 1)):
        b, _ = ofdm_burst(name, n, rv=1)
        close(native.phy.tx_audio(b), PHY.tx_audio(b))
    for name in ("fsk8r50-r1/2", "fsk32r62-r1/3"):
        spec = cpm.SPECS[name]
        ctl = bytes(codes.payload_bytes(cpm.CTL[spec.grid]))
        for dup, cap in ((False, 0), (True, 0), (False, 1), (True, 2)):  # each cap's filter (cpm.TX_FILTERS)
            slots = [Slot(ctl_mask(0, 0, 3), 0, ctl)] + ([Slot(ctl_mask(0, 0, 3), 1, ctl)] if dup else [])
            slots += [Slot(data_mask(0, i, 3), 0, bytes([i]) * codes.payload_bytes(spec)) for i in range(2)]
            b = TxBurst(spec.name, slots, 0, cap)
            close(native.phy.tx_audio(b), PHY.tx_audio(b))
    for name in PHY.MODES:
        for cap in range(3):
            assert native.arq.peak_db(name, cap) == PHY.peak_db(name, cap)


def test_soft_bits_and_measure(native, pure):
    b, _ = ofdm_burst("w48-qpsk-r1/2", 6)
    r = hear(b, snr=4.0, seed=2, fading=FadingPreset("mpd", 2.0, 4.0))
    close(native.phy.soft_bits(r), PHY.soft_bits(r))
    want, got = PHY.measure(r), native.phy.measure(r)
    assert set(got) == set(want)
    for k in want:
        assert got[k] == pytest.approx(want[k], rel=MEAS_TOL, abs=MEAS_TOL), k
    r = _cpm_received()
    for g, w in zip(native.phy.soft_bits(r), PHY.soft_bits(r)):
        close(g, w)
    want, got = PHY.measure(r), native.phy.measure(r)
    for k in want:
        assert got[k] == pytest.approx(want[k], rel=MEAS_TOL, abs=MEAS_TOL), k


def _cpm_received(seed=1, noise=0.05):
    spec = cpm.SPECS["fsk8r50-r1/2"]
    b = TxBurst(spec.name, [Slot(ctl_mask(0, 0, 3), 0, bytes(codes.payload_bytes(cpm.CTL[spec.grid])))]
                + [Slot(data_mask(0, i, 3), 0, bytes([i]) * codes.payload_bytes(spec)) for i in (1, 2)], 0)
    y = np.concatenate([np.zeros(2000), PHY.tx_audio(b), np.zeros(2000)])
    y += np.random.default_rng(seed).normal(0, noise, len(y))
    return cpm.receive(y, cpm.find(cpm.GRIDS[spec.grid], y))


def test_dd_rescue_decisions_match(native, pure):
    """test_dd's burst: codeword 1 fails on the pilot estimate, DD rescues it."""
    from test_dd import _burst

    r, mids, pays = _burst()
    for dd in (False, True):
        PHY.DD = dd
        try:
            py, nat = rx_pair(native, r, dd)
            got = [nat.decode(i, m, 0, None) for i, m in enumerate(mids)]
            assert got == [py.decode(i, m, 0, None) for i, m in enumerate(mids)]
        finally:
            PHY.DD = True
        assert got == (pays if dd else [pays[0], None])


# draws where DD rescues (6 -> 8, 1 -> 4, 0 -> 1 of the codewords) and one where every slot fails
@pytest.mark.parametrize("name,n,snr,seed", [("w48-qpsk-r1/2", 8, 6.0, 8), ("qpsk-r1/2", 4, 2.5, 3),
                                             ("w48-16qam-r1/2", 4, 9.0, 5), ("w48-qpsk-r1/2", 8, 4.0, 3)])
def test_keyed_decodes_and_store_match(native, pure, name, n, snr, seed):
    """Slot by slot under the session's masks with keys (DD on failures,
    undo, the store), then the plain decodes, then a resend combined."""
    b, pl = ofdm_burst(name, n, seed=seed)
    r = hear(b, snr=snr, seed=seed, fading=FadingPreset("mpd", 2.0, 4.0))
    sp, sn = {}, {}
    py, nat = PHY.ModemRx(dict(r), sp), native.phy.ModemRx(dict(r), sn, None, True)
    want = [py.decode(i, (7, 0, i), 0, ("p", i)) for i in range(n)]
    assert [nat.decode(i, (7, 0, i), 0, ("p", i)) for i in range(n)] == want
    assert sp.keys() == sn.keys()
    for k in sp:
        close(sn[k][0], sp[k][0], BUF_TOL)
        assert sn[k][1:] == sp[k][1:]
    assert [nat.decode(i, (8, 0, i), 0, None) for i in range(n)] == [py.decode(i, (8, 0, i), 0, None) for i in range(n)]
    resend = TxBurst(name, [Slot((7, 0, i), 1, pl[i]) for i in range(n) if want[i] is None], 0)
    if resend.slots:
        r2 = hear(resend, snr=snr, seed=seed + 100, fading=FadingPreset("mpd", 2.0, 4.0))
        py, nat = PHY.ModemRx(r2, sp), native.phy.ModemRx(r2, sn, None, True)
        keys = [s.mask_id[2] for s in resend.slots]
        assert ([nat.decode(j, (7, 0, i), 1, ("p", i)) for j, i in enumerate(keys)]
                == [py.decode(j, (7, 0, i), 1, ("p", i)) for j, i in enumerate(keys)])


def test_budget_spent_and_store_mismatch(native, pure):
    from test_dd import _burst

    r, mids, pays = _burst()
    nat = native.phy.ModemRx(r, {}, 0.0, True)
    assert [nat.decode(i, m, 0, None) for i, m in enumerate(mids)] == [pays[0], None]
    # an injected clock: DD stops once it passes the deadline
    t = [0.0]
    nat = native.phy.ModemRx(dict(r), {}, 1.0, True, lambda: t[0])
    t[0] = 2.0
    assert [nat.decode(i, m, 0, None) for i, m in enumerate(mids)] == [pays[0], None]
    store = {("p", 1): (np.zeros((1, 10)), 0, "w-other", (0, 0, (7, 1, 1)))}
    with pytest.raises(AssertionError, match="stored in w-other"):
        native.phy.ModemRx(dict(r), store, None, True).decode(1, mids[1], 0, ("p", 1))


def test_cpm_decodes_match(native, pure):
    r = _cpm_received(noise=0.3)
    py, nat = rx_pair(native, r)
    probes = [(0, ctl_mask(0, 0, 3), 0, None), (1, data_mask(0, 1, 3), 0, None), (2, data_mask(0, 2, 3), 0, None),
              (1, ctl_mask(0, 0, 3), 0, None), (0, ctl_mask(1, 0, 3), 0, ("ctl", 0)), (1, ctl_mask(1, 0, 3), 1, ("ctl", 0))]
    assert [nat.decode(*p) for p in probes] == [py.decode(*p) for p in probes]


# --- kisslink -------------------------------------------------------------------------

def _frames():
    from test_kiss import frame

    return [frame("KC2G", "W1AW", 0x00, b"connected hello"), frame("W1AW", "KC2G", 0x21),
            frame("KC2G", "W1AW", 0x22, b"x" * 300), frame("KC2G", "W1AW", 0x03, b"UI"), b"not ax.25",
            frame("KC2G", "W1AW", 0x00, b"x", digis=[("RELAY", True), ("WIDE2-1", False)]), b"\x01\x02"]


def test_ax25(native, pure):
    for f in _frames():
        t = native.kisslink.parse_ax25(f)
        want = KL.parse_ax25(f)
        assert (None if t is None else KL.Ax25(*t)) == want
        if want is not None:
            assert native.kisslink.station_hash(want.sender) == KL.station_hash(want.sender)


def test_kiss_links_match(native, pure):
    """Two Python TNCs and two native ones exchange the same frames over the
    same channel: every burst and every frame heard is the same."""
    from data2g.tnc import receive_any

    t = [0.0]
    clock = lambda: t[0]  # noqa: E731
    py = [KL.KissLink(clock=clock), KL.KissLink(clock=clock)]
    nat = [native.kisslink.KissLink(clock=clock), native.kisslink.KissLink(clock=clock)]
    f = _frames()
    script = [(0, f[0]), (1, f[1]), (0, f[2]), (0, f[3]), (1, f[4]), (0, f[5]), (0, f[2] + f[2])]
    for k, (who, frame) in enumerate(script):
        out = []
        for a, b in (py, nat):
            src, dst = (a, b) if who == 0 else (b, a)
            src.enqueue(frame)
            burst = src.next_burst()
            x = PHY.tx_audio(burst)
            y = np.concatenate([np.zeros(FS // 2), x, np.zeros(FS // 2)])
            y = hfchannel.awgn(y, 8.0, seed=k, s_power=hfchannel.active_power(x))
            r = receive_any(y, lead=FS)
            out.append((burst, dst.on_burst(r) if r is not None else None, src.queue, src.n_sent))
        assert out[1] == out[0], k
        t[0] += 3
    for n, p in zip(nat, py):
        assert n.me == p.me
    with pytest.raises(ValueError):
        native.kisslink.KissLink(cap=0, broadcast="w48-qpsk-r1/5")
    k = native.kisslink.KissLink()
    k.command(2, bytes([127]))
    k.command(3, bytes([200]))
    assert (k.persist, k.slot_s) == (127, 2.0)
