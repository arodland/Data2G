"""native/core/hand against data2g/hand.py: the frozen tables, Hand audio
(including a payload with every bit set), and a two-channel window's decodes
(payloads, SNR, landing time, both --hand-decoder grids, and a round under the
wrong tag)."""

import numpy as np

from data2g import hand, hfchannel


def test_channels_and_audio(native):
    for c in (0, 2, 19, 39):  # 2: f0 rounds half to even
        assert native.hand.channel_freq(c) == hand.channel_freq(c)
    for p in (0x1234567890ABC, 2 ** 63 - 1, 2 ** 62):  # bit 62 set: all 63 payload bits
        np.testing.assert_allclose(native.hand.modulate(0x5A, p, hand.channel_freq(2)),
                                   hand.modulate(0x5A, hand.payload_bits(p), hand.channel_freq(2)), atol=1e-9)
        np.testing.assert_allclose(native.hand.tx_audio(0x5A, p, 20, 3.7), hand.tx_audio(0x5A, p, 20, 3.7), atol=1e-9)


def test_decode_window(native):
    rng = np.random.default_rng(11)
    tag, late = 0x5A, 777  # 777: a start a fraction of a symbol off, so the nailing decides
    y = np.zeros(hand.WINDOW_SAMPLES)
    sent = {5: int(rng.integers(0, 2 ** hand.PAYLOAD)), 7: int(rng.integers(0, 2 ** hand.PAYLOAD))}
    for c, p in sent.items():
        x = hand.modulate(tag, hand.payload_bits(p), hand.channel_freq(c))
        y[hand.SEARCH_SAMPLES + late:hand.SEARCH_SAMPLES + late + len(x)] += x
    y = hfchannel.awgn(hfchannel.freq_shift(y, -1.5), 4.0, seed=5)
    # both --hand-decoder grids, in both languages: the same two Hands, the same SNR
    # and the same landing time to within 1e-6 s. cheap searches every T/4 and
    # nails on the search's own CFO grid; the T/16 nail still lands on one nailed
    # sample, so a cheap round and a shipped one differ by a step, never a symbol.
    for rnd in ("full", "cheap"):
        py = hand.decode_window(y, tag, [5, 6, 7], hand.SEARCH_SAMPLES, hand.ROUNDS[rnd])
        nat = native.hand.decode_window(y, tag, [5, 6, 7], hand.SEARCH_SAMPLES, round=rnd)
        assert [(r["c"], r["payload"]) for r in py] == [(c, p) for c, p in sent.items()]
        assert [(r["c"], r["payload"]) for r in nat] == [(r["c"], r["payload"]) for r in py]
        for a, b in zip(nat, py):
            assert abs(a["snr"] - b["snr"]) < 1e-6 and abs(a["dt_ms"] - b["dt_ms"]) < 1e-6
    # nailed before the soft bits, in both languages: 1e-6 s is far below the
    # 10 ms one nailed sample can be from the next, so the two land on the same
    # one. And another net's request: a round decoded with the CRC seeded from
    # 0x5B, not the 0x5A these were sent under, answers nothing in either language.
    assert hand.decode_window(y, 0x5B, [5, 6, 7], hand.SEARCH_SAMPLES) == []
    assert native.hand.decode_window(y, 0x5B, [5, 6, 7], hand.SEARCH_SAMPLES) == []
