import numpy as np
import pytest

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


@pytest.mark.parametrize("lost", [0, 1])  # 0: before any control is heard
def test_round_trip_with_a_header_lost(lost):
    mode, h = "qpsk-r1/2", 6
    bs = bulk.bursts(TEXT, mode, h, 1)
    lay = bulk.Layout(bulk.MODES[mode], len(bulk.pack(TEXT, codes.payload_bytes(bulk.MODES[mode]))), h)
    assert [len(PHY.tx_audio(b)) for b in bs] == [lay.length(g) for g in range(len(bs))]
    lead = FS // 2
    y = np.concatenate([np.zeros(lead), bulk.tx_audio(bs), np.zeros(FS)])
    s = lead + lay.offset(lost)
    y[s:s + FS] = 0  # the burst's preamble, header and header copy
    y = hfchannel.apply_channel(y, snr_db=15, freq_offset_hz=20, ppm=30, seed=1)
    rx = bulk.receive(y[i:i + FS] for i in range(0, len(y), FS))
    (st,) = rx.streams.values()
    assert st.stats["headerless"] == 1
    # the burst's own blocks come from its headerless receive (an r1/2 copy at RV 1 can't decode alone)
    assert bulk.unpack(st.blocks, st.lay.n) == (TEXT, [])
    assert modem.LEADIN_SAMPLES < FS


def test_cpm_round_trip_with_a_front_lost():
    """fsk32r62-r1/2: the control in its two polar slots, timing from the CPM
    layout (not cpm.burst_seconds, which counts the ramps on top), and a
    burst whose front sync and first header are gone received by its timing."""
    mode, h = "fsk32r62-r1/2", 2
    text = TEXT[:900]
    bs = bulk.bursts(text, mode, h, 1)
    lay = bulk.Layout(bulk.MODES[mode], len(bulk.pack(text, codes.payload_bytes(bulk.MODES[mode]))), h)
    assert [len(bulk.cpm_audio(b)) for b in bs] == [lay.length(g) for g in range(len(bs))]
    assert not np.array_equal(bulk.cpm_audio(bs[1]), PHY.tx_audio(bs[1]))  # data tones dealt over the burst
    lead = FS // 2
    y = np.concatenate([np.zeros(lead), bulk.tx_audio(bs), np.zeros(FS)])
    s = lead + lay.offset(1)
    y[s:s + FS] = 0  # burst 1's front sync block, first header copy, part of its control
    y = hfchannel.apply_channel(y, snr_db=10, freq_offset_hz=20, ppm=30, seed=1)
    rx = bulk.receive((y[i:i + FS] for i in range(0, len(y), FS)), mode)  # back to back at 30 ppm: tnc.CPM_TAIL
    (st,) = rx.streams.values()
    assert st.stats["headerless"] == 1
    assert bulk.unpack(st.blocks, st.lay.n) == (text, [])


def test_listen_prints_in_order_as_blocks_finalize(monkeypatch):
    """Printer (data2g-bulk listen): text in block order while audio still
    arrives; a burst lost whole (an r1/2 copy at RV 1 can't decode alone)
    shows as its blocks' markers. The DD cap reaches every ModemRx."""
    import io

    from data2g.arq import frames as F

    mode, h = "qpsk-r1/2", 6
    blocks = bulk.pack(TEXT, codes.payload_bytes(bulk.MODES[mode]))
    lay = bulk.Layout(bulk.MODES[mode], len(blocks), h)
    lead = FS // 2
    y = np.concatenate([np.zeros(lead), bulk.tx_audio(bulk.bursts(TEXT, mode, h, 1)), np.zeros(4 * FS)])
    y[lead + lay.offset(1):lead + lay.offset(2)] = 0  # burst 1, all of it
    y = hfchannel.apply_channel(y, snr_db=15, freq_offset_hz=20, ppm=30, seed=1)
    out, seen = io.BytesIO(), []
    rx = bulk.Rx(dd_cap=0.25)
    budgets, real = [], PHY.ModemRx  # the budget as passed: --native's ModemRx keeps it in C++
    monkeypatch.setattr(PHY, "ModemRx", lambda r, store, dd_budget=None: budgets.append(dd_budget) or
                        real(r, store, dd_budget))
    printer = bulk.Printer(out)
    chunks = [y[i:i + FS] for i in range(0, len(y), FS)]
    bulk.receive(chunks, rx=rx, on_chunk=lambda r: (printer(r), seen.append(len(out.getvalue()))))
    want = b"".join(b"[block %d lost]" % k if 6 <= k < 12 else F.inflate(b"", b) for k, b in enumerate(blocks))
    assert out.getvalue() == want
    assert 0 < seen[len(chunks) // 2] < len(want)  # printing began mid-transfer
    assert budgets and all(b is not None for b in budgets)
