"""Decision-directed re-estimation (arq.phy DATA2G_DD): a burst whose second
codeword fails on the pilot estimate decodes once the first's bits and
its own decoder posterior join the pilots."""

import numpy as np

from data2g import codes, hfchannel, modem
from data2g.arq import phy
from data2g.config import SUBMODES
from data2g.hfchannel import FadingPreset


def _burst():
    spec = SUBMODES["qpsk-r1/2"]
    rng = np.random.default_rng(4)
    mids = [(7, 1, 0), (7, 1, 1)]
    pays = [rng.bytes(codes.payload_bytes(spec)) for _ in mids]
    bits = np.stack([codes.encode(spec, q, 0, phy.mask_value(m), i) for i, (q, m) in enumerate(zip(pays, mids))])
    x = np.concatenate([np.zeros(3000), modem.modulate_bits(codes.spread(bits, spec.bits_per_cu), spec), np.zeros(3000)])
    # channel seed 54: a draw where codeword 1 fails without DD (which draws
    # do depends on the code's shift tables and the scrambling)
    r = modem.receive(hfchannel.apply_channel(x, snr_db=3.0, fading_preset=FadingPreset("mpd", 2.0, 4.0), seed=54))
    return r, mids, pays


def test_dd_rescues_a_codeword(monkeypatch):
    r, mids, pays = _burst()
    got = {}
    for dd in (False, True):
        monkeypatch.setattr(phy, "DD", dd)
        rx = phy.ModemRx({k: v for k, v in r.items() if k != "_soft"}, {})
        got[dd] = [rx.decode(i, m, 0, None) for i, m in enumerate(mids)]
    assert got[False] == [pays[0], None]
    assert got[True] == pays


def test_dd_budget_spent():
    """A live receiver past its DD budget decodes on the pilot estimate alone."""
    r, mids, pays = _burst()
    rx = phy.ModemRx(r, {}, dd_budget=0.0)
    assert [rx.decode(i, m, 0, None) for i, m in enumerate(mids)] == [pays[0], None]
