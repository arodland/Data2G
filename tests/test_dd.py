"""Decision-directed re-estimation (arq.phy DATA2G_DD): a burst whose second
codeword fails on the pilot estimate decodes once the first's bits and
its own decoder posterior join the pilots."""

import numpy as np

from data2g import codes, hfchannel, modem
from data2g.arq import phy
from data2g.config import SUBMODES
from data2g.hfchannel import FadingPreset


def test_dd_rescues_a_codeword(monkeypatch):
    spec = SUBMODES["qpsk-r1/2"]
    rng = np.random.default_rng(4)
    mids = [(7, 1, 0), (7, 1, 1)]
    pays = [rng.bytes(codes.payload_bytes(spec)) for _ in mids]
    bits = np.stack([codes.encode(spec, q, 0, phy.mask_value(m)) for q, m in zip(pays, mids)])
    x = np.concatenate([np.zeros(3000), modem.modulate_bits(codes.spread(bits, spec.bits_per_cu), spec), np.zeros(3000)])
    r = modem.receive(hfchannel.apply_channel(x, snr_db=3.0, fading_preset=FadingPreset("mpd", 2.0, 4.0), seed=4))
    got = {}
    for dd in (False, True):
        monkeypatch.setattr(phy, "DD", dd)
        rx = phy.ModemRx({k: v for k, v in r.items() if k != "_soft"}, {})
        got[dd] = [rx.decode(i, m, 0, None) for i, m in enumerate(mids)]
    assert got[False] == [pays[0], None]
    assert got[True] == pays
