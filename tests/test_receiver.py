"""The streaming receiver (data2g.receiver) without audio hardware: a
burst through the 48 kHz audio path, false locks, BUSY, the header-copy
path, and what Accept lets through."""

import numpy as np
import pytest
from scipy import signal as sps

from data2g import audio, codes, hfchannel, modem, receiver
from data2g.config import FS, SUBMODES
from data2g.kisslink import pack_frames, unpack_frames


def test_burst_through_the_audio_path_and_streaming_receiver():
    spec = SUBMODES["qpsk-r1/5"]
    packets = [b"CQ CQ de TEST" * 3, bytes(range(200))[:90]]
    x = modem.modulate(pack_frames(packets, codes.payload_bytes(spec)), spec)
    x = np.concatenate([np.zeros(FS), x, np.zeros(FS)])
    y = hfchannel.apply_channel(x, snr_db=15, freq_offset_hz=20.0, ppm=5, seed=1)
    card = sps.resample_poly(y, 6, 1)  # as the TNC's 48 kHz device would carry it
    dec, rx, got = audio.Decimator(48000), receiver.Receiver(modem.Accept.of(["qpsk-r1/5"], min_score=0.35)), []
    for i in range(0, len(card), 24000):  # half-second chunks
        for kind, ev in rx.feed(dec(card[i : i + 24000])):
            if kind == "burst":
                b = modem.decode_received(ev["rx"])
                got += unpack_frames(b.payloads, b.crc_ok)[0]
    assert got == packets


def test_digital_silence_is_not_searched():
    rx = receiver.Receiver(modem.Accept.of(["qpsk-r1/5"]))
    assert rx.feed(np.zeros(5 * FS)) == [] and not rx.busy


def test_accept_caps_codewords_by_burst_length():
    a = modem.Accept.of(["qpsk-r1/5", "w48-qpsk-r1/2"], max_secs=10)
    lim = dict(a.max_cw)
    # qpsk-r1/5: 8 frames (1.15 s) a codeword after ~0.4 s of preamble and header
    assert lim["qpsk-r1/5"] == 8 and a.bands == ["w", "w48"]
    assert "qpsk-r1/5" not in dict(modem.Accept.of(["qpsk-r1/5"], max_secs=1).max_cw)


def test_rx_rejects_headers_outside_accept():
    spec = SUBMODES["qpsk-r1/5"]
    rng = np.random.default_rng(0)
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec)) for _ in range(6)], spec)
    x = np.concatenate([np.zeros(FS), x, np.zeros(FS)])
    assert modem.find_burst(x, accept=modem.Accept.of(["qpsk-r1/5"]))["n_cw"] == 6
    for accept in (modem.Accept.of(["qpsk-r1/5"], max_secs=5, min_score=0.35),  # claims 7 s
                   modem.Accept.of(["qpsk-r1/3"], min_score=0.35)):  # another submode
        try:
            modem.find_burst(x, accept=accept)
            raise AssertionError(f"accepted under {accept}")
        except modem.SyncError:
            pass


def _whole_buffer_search(monkeypatch):
    """The receiver's old search (every start of the buffer, the buffer's
    own noise level). The incremental search takes its noise level from the
    recent audio, so data whose preamble was lost no longer false-locks;
    these tests need a false lock to test what follows one."""
    monkeypatch.setattr(receiver.Receiver, "_stats", lambda self, w0=0: None)
    monkeypatch.setattr(receiver.Receiver, "_searched", lambda self: None)


def test_a_false_lock_on_data_does_not_cost_the_next_burst(monkeypatch):
    """Data whose preamble was lost can read as a burst with a random header
    (w floor 0.25: about half the time), claiming up to a long burst's
    length. A real burst arriving meanwhile must still be received: the
    receiver keeps searching and a better header supersedes."""
    from data2g.config import BANDS, LEADIN_SAMPLES

    monkeypatch.setitem(modem.HEADER_MIN_SCORE, "w", 0.25)  # 0.33 now rejects the false locks this needs
    _whole_buffer_search(monkeypatch)
    spec = SUBMODES["qpsk-r1/2"]
    cut = LEADIN_SAMPLES + BANDS[spec.sync_band].preamble_samples + modem.header_samples(spec.sync_band)
    got_any = 0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        pl = [bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8)) for _ in range(8)]
        real = [bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8)) for _ in range(2)]
        x = np.concatenate([np.zeros(FS), modem.modulate(pl, spec)[cut:], np.zeros(FS // 4),
                            modem.modulate(real, spec), np.zeros(2 * FS)])
        y = hfchannel.awgn(x, 12, seed=seed, s_power=1.0)
        rx = receiver.Receiver(modem.Accept.of(None, 16.0))
        bursts = [ev for i in range(0, len(y), FS // 10) for k, ev in rx.feed(y[i:i + FS // 10]) if k == "burst"]
        ok = [modem.decode_received(b["rx"]).payloads == real for b in bursts if b["rx"] is not None]
        assert any(ok), (seed, [(b["header"]["spec"].name, b["header"]["score"]) for b in bursts])
        got_any += len(bursts) > 1
    assert got_any  # at least one seed actually false-locked first (the case under test)


def test_a_half_arrived_header_is_waited_for_not_misread():
    """A search landing while a real burst's header is still arriving must
    wait for it: reading the shifted positions that fit read garbage (score
    ~0.25, a random submode and length) that passed the w floor and hid the
    real header (the audio loopback missed replies this way)."""
    for seed in range(24):
        # w, and n10 whose longer header let the w detector's shorter one
        # (read off the same narrow preamble) commit first (a CQ frame at 500)
        spec = SUBMODES["ack-1f" if seed < 12 else "n10-qpsk-r1/3"]
        rng = np.random.default_rng(seed)
        pl = [bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8)) for _ in range(2)]
        lead = int(rng.uniform(1.6, 2.2) * FS)  # the receiver's first search at 1.6 s + k * 0.5 s
        x = np.concatenate([np.zeros(lead), modem.modulate(pl, spec), np.zeros(2 * FS)])
        # near-digital silence around it, as a sound-card loopback (or a
        # muted receiver) gives: the case the loopback hit
        y = x + np.random.default_rng(seed).normal(0, 1e-6, len(x)) if seed % 2 else hfchannel.awgn(x, 20, seed=seed,
                                                                                                    s_power=1.0)
        rx = receiver.Receiver(modem.Accept.of(None, 16.0))
        events = [(k, ev) for i in range(0, len(y), FS // 10) for k, ev in rx.feed(y[i:i + FS // 10])]
        heads = [ev for k, ev in events if k == "header"]
        assert heads and all(h["spec"].name == spec.name and h["score"] > 0.9 for h in heads), (
            seed, [(h["spec"].name, round(h["score"], 2)) for h in heads])


def test_busy_ends_with_the_signal_not_a_false_headers_claim(monkeypatch):
    """A burst whose preamble was missed reads as false headers claiming up
    to 15 s past where the audio ends. BUSY (Receiver.channel_busy, what the
    host reports) follows the pilots and the in-band energy, so it ends with
    the signal; the internal hold (Receiver.busy) may run on. (The w header
    floor, 0.33, now rejects these false headers; at 0.25 they got through,
    which this reproduces to test BUSY on its own.)"""
    import sys
    from pathlib import Path

    import numpy as np

    from data2g import hfchannel, modem
    from data2g.arq import phy as PHY
    from data2g.arq.engine import Engine
    from data2g.config import BANDS, FS
    from data2g.receiver import Receiver

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import outcome_data as O

    monkeypatch.setitem(modem.HEADER_MIN_SCORE, "w", 0.25)
    _whole_buffer_search(monkeypatch)
    rng = np.random.default_rng(1)
    x = PHY.tx_audio(O.burst("qpsk-r1/2", 20, rng))
    cut = x[BANDS["w"].preamble_samples + modem.header_samples("w") + 800 + int(rng.integers(0, 3000)):]
    y = np.concatenate([np.zeros(FS * 20), cut, np.zeros(FS * 12)])
    y = hfchannel.awgn(y, 20.0, seed=1, s_power=hfchannel.active_power(x))
    r = Receiver(Engine("W1AW", seed=1).accept)
    end = FS * 20 + len(cut)
    busy_after = held_after = 0.0
    for i in range(0, len(y), FS // 50):
        r.feed(y[i:i + FS // 50])
        if i >= end:
            busy_after += r.channel_busy / 50
            held_after += r.busy / 50
    assert busy_after <= 1.0 < held_after, (busy_after, held_after)


def _false_lock_clip(starts):
    """On air (recordings/20261024-190009, t 1652.0): a false n4-qpsk-r1/5
    header (score 0.33) claimed 52 s, and the FSK bursts inside it were never
    searched for. Whether a false header is met depends on where the searches
    fall (each hop, 0.26 s here, and the search span): the clip from each of
    `starts` (samples). -> how many met a suspect header first."""
    from pathlib import Path

    from data2g import cpm
    from data2g.arq.engine import Engine

    y = np.load(Path(__file__).parent / "data" / "kc2g_n4_false_lock_fsk.npz")["audio"].astype(np.float64)
    blk, false_locks = FS // 50, 0
    for o in starts:
        rx = receiver.Receiver(Engine("KC2G", seed=1).accept, cpm_grids=tuple(cpm.GRIDS))
        evs = [ev for i in range(o, len(y), blk) for ev in rx.feed(y[i:i + blk])]
        heads = [(ev["spec"].name, ev["score"]) for kind, ev in evs if kind == "header"]
        false_locks += heads[0][0] != "fsk16r25-r1/2" and heads[0][1] < receiver.Receiver.SUSPECT_SCORE
        got = [ev for kind, ev in evs if kind == "burst" and ev["rx"] is not None]
        assert [ev["header"]["spec"].name for ev in got] == ["fsk16r25-r1/2"], (o, heads)
    return false_locks


def test_a_cpm_burst_replaces_a_suspect_header_claiming_its_span():
    """A CPM lock replaces a suspect header while it is pending, and a CPM
    front waiting for its second header copy survives the suspect burst's
    trim. At 180 ms: the n4-qpsk-r1/5 header as on air; at 20 ms a
    polar-k96-f8 one, under which the FSK front waits (its first header copy
    is weak). The starts move with the search span (re-pick them from the
    slow test's sweep)."""
    assert _false_lock_clip([1440, 160]) == 2, "the clip no longer meets its false headers: re-pick the starts"


@pytest.mark.slow
def test_a_cpm_burst_survives_a_suspect_header_at_every_search_phase():
    assert _false_lock_clip(range(0, 13 * 160, 160))


def test_a_burst_whose_head_faded_is_found_from_its_header_copy():
    """Preamble and header lost (a fade): the streaming receiver finds the
    burst from its frame pilots and mid-burst header copy (modem.find_copy)
    and decodes it; noise alone finds nothing."""
    import sys
    from pathlib import Path

    from data2g import cpm
    from data2g.arq import phy as PHY
    from data2g.arq.modes import MODES
    from data2g.config import BANDS, LEADIN_SAMPLES

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import outcome_data as O

    acc = modem.Accept.of(None, 16.0)
    for name, n in (("qpsk-r1/5", 2), ("ack-4f", 1), ("w48-qpsk-r1/2", 3)):
        b = O.burst(name, n, np.random.default_rng(3))
        x = PHY.tx_audio(b)
        sb = MODES[name].sync_band
        head = LEADIN_SAMPLES + BANDS[sb].preamble_samples + modem.header_samples(sb)
        x = np.concatenate([np.zeros(2 * FS), np.zeros(head), x[head:], np.zeros(2 * FS)])
        y = hfchannel.awgn(hfchannel.freq_shift(x, 37.0), 10, seed=2, s_power=hfchannel.active_power(x))
        rx = receiver.Receiver(acc, cpm_grids=tuple(cpm.GRIDS))
        got = [ev for i in range(0, len(y), FS // 10) for k, ev in rx.feed(y[i:i + FS // 10]) if k == "burst"]
        ok = [ev for ev in got if ev["rx"] is not None and ev["rx"]["spec"].name == name
              and all(PHY.ModemRx(ev["rx"], {}).decode(i, s.mask_id, 0, None) == s.payload
                      for i, s in enumerate(b.slots))]
        assert ok and "copy" in ok[0]["header"], (name, [(ev["header"]["spec"].name, "copy" in ev["header"]) for ev in got])
    y = np.random.default_rng(4).normal(size=60 * FS)
    rx = receiver.Receiver(acc, cpm_grids=tuple(cpm.GRIDS))
    assert not [k for i in range(0, len(y), FS // 10) for k, _ in rx.feed(y[i:i + FS // 10])]


def test_a_cpm_burst_whose_first_header_copy_faded_locks_on_both():
    """Front sync clear, first header copy lost (a fade): cpm.find waits for
    the second copy (a candidate before it, a lock from both after), and the
    streaming receiver decodes the burst from that late lock. (Not c8r50:
    with its first copy blanked, a tile alignment two periods early reads
    the front's tiles as a header and early-locks a wrong word.)"""
    import sys
    from pathlib import Path

    from data2g import cpm
    from data2g.arq import phy as PHY

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import outcome_data as O

    acc = modem.Accept.of(None, 16.0)
    for name, n in (("fsk16r25-r1/2", 2), ("fsk32r62-r1/2", 1), ("fsk32r62-r1/2", 4)):
        g = cpm.GRIDS[cpm.SPECS[name].grid]
        b = O.burst(name, n, np.random.default_rng(3))
        x = PHY.tx_audio(b).copy()
        rows = cpm.layout(g.name, cpm.stream_symbols(g.name, n - 1, False)).hdr_rows[0]
        x[rows[0] * g.T:(rows[-1] + 1) * g.T] = 0  # copy 1 faded out
        lead = 2 * FS
        x = np.concatenate([np.zeros(lead), x, np.zeros(2 * FS)])
        y = hfchannel.awgn(hfchannel.freq_shift(x, 37.0), 10, seed=2, s_power=hfchannel.active_power(x))
        cand = cpm.find(g, y[:lead + cpm.hdr_span(g.name, 1) + FS], wait=True)
        assert cand is not None and cand.get("wait"), (name, cand)
        assert cpm.find(g, y[:lead + cpm.hdr_span(g.name, 1) + FS]) is None, name
        lock = cpm.find(g, y)
        assert lock is not None and lock["spec"].name == name and lock["n_data"] == n - 1, (name, lock)
        assert lock["header_end"] - lock["start"] == (cpm.layout(g.name, cpm.stream_symbols(g.name, n - 1, False))
                                                      .hdr_rows[1][-1] + 1) * g.T, name
        rx = receiver.Receiver(acc, cpm_grids=tuple(cpm.GRIDS))
        got = [ev for i in range(0, len(y), FS // 10) for k, ev in rx.feed(y[i:i + FS // 10]) if k == "burst"]
        ok = [ev for ev in got if ev["rx"] is not None and ev["rx"]["spec"].name == name
              and all(PHY.ModemRx(ev["rx"], {}).decode(i, s.mask_id, 0, None) == s.payload
                      for i, s in enumerate(b.slots))]
        assert ok, (name, [ev["header"]["spec"].name for ev in got])


def test_a_cpm_burst_whose_front_faded_locks_on_its_mid_block():
    """Front sync block and first header copy lost (a fade): cpm.find
    misses, find_mid locks from the mid block and both header copies (a
    short burst's mid block after its data), and the streaming receiver
    decodes the burst; noise alone finds nothing."""
    import sys
    from pathlib import Path

    from data2g import cpm
    from data2g.arq import phy as PHY

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import outcome_data as O

    acc = modem.Accept.of(None, 16.0)
    for name, n in (("fsk16r25-r1/2", 3), ("fsk8r50-r1/2", 2), ("fsk32r62-r1/2", 1), ("fsk32r62-r1/2", 4)):
        g = cpm.GRIDS[cpm.SPECS[name].grid]
        b = O.burst(name, n, np.random.default_rng(3))
        x = PHY.tx_audio(b).copy()
        lay = cpm.layout(g.name, cpm.stream_symbols(g.name, n - 1, False))
        x[:(lay.hdr_rows[0][-1] + 1) * g.T] = 0  # front and copy 1 faded out
        lead = 2 * FS
        x = np.concatenate([np.zeros(lead), x, np.zeros(2 * FS)])
        y = hfchannel.awgn(hfchannel.freq_shift(x, 37.0), 10, seed=2, s_power=hfchannel.active_power(x))
        assert cpm.find(g, y) is None, name
        lock = cpm.find_mid(g, y)
        assert lock is not None and lock["spec"].name == name and lock["n_data"] == n - 1, (name, lock)
        assert abs(lock["start"] - lead) < g.T // 4, (name, lock["start"])
        rx = receiver.Receiver(acc, cpm_grids=tuple(cpm.GRIDS))
        got = [ev for i in range(0, len(y), FS // 10) for k, ev in rx.feed(y[i:i + FS // 10]) if k == "burst"]
        ok = [ev for ev in got if ev["rx"] is not None and ev["rx"]["spec"].name == name
              and all(PHY.ModemRx(ev["rx"], {}).decode(i, s.mask_id, 0, None) == s.payload
                      for i, s in enumerate(b.slots))]
        assert ok, (name, [ev["header"]["spec"].name for ev in got])
    y = np.random.default_rng(4).normal(size=60 * FS)
    assert all(cpm.find_mid(g, y[i:i + 2 * FS]) is None for g in cpm.GRIDS.values() for i in range(0, len(y), 2 * FS))
