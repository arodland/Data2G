import numpy as np

from data2g import bulk, codes, hfchannel, modem
from data2g.arq import phy as PHY
from data2g.config import FS

TEXT = open(__file__.replace("tests/test_bulk.py", "docs/arq.md"), "rb").read()[:3000]


def test_blocks_stand_alone():
    blocks = bulk.pack(TEXT + bytes(range(256)) * 2, 46)  # the tail doesn't deflate: stored blocks
    assert all(len(b) == 46 for b in blocks)
    text, lost = bulk.unpack(dict(enumerate(blocks)), len(blocks))
    assert text == TEXT + bytes(range(256)) * 2 and lost == []
    have = {i: b for i, b in enumerate(blocks) if i not in (3, 4, 9)}
    text, lost = bulk.unpack(have, len(blocks))
    assert lost == [(3, 4), (9, 9)] and b"blocks 3-4 of" in text


def test_round_trip_with_a_header_lost():
    mode, h = "qpsk-r1/2", 6
    bs = bulk.bursts(TEXT, mode, h, 1)
    lay = bulk.Layout(bulk.MODES[mode], len(bulk.pack(TEXT, codes.payload_bytes(bulk.MODES[mode]))), h)
    assert [len(PHY.tx_audio(b)) for b in bs] == [lay.length(g) for g in range(len(bs))]
    lead = FS // 2
    y = np.concatenate([np.zeros(lead), bulk.tx_audio(bs), np.zeros(FS)])
    s = lead + lay.offset(1)
    y[s:s + FS] = 0  # burst 1's preamble, header and header copy
    y = hfchannel.apply_channel(y, snr_db=15, freq_offset_hz=20, ppm=30, seed=1)
    rx = bulk.receive(y[i:i + FS] for i in range(0, len(y), FS))
    (st,) = rx.streams.values()
    assert st.stats["headerless"] == 1
    # burst 1's own blocks come from its headerless receive (an r1/2 copy at RV 1 can't decode alone)
    assert bulk.unpack(st.blocks, st.lay.n) == (TEXT, [])
    assert modem.LEADIN_SAMPLES < FS
