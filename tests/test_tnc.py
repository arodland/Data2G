"""The TNC's pieces without audio hardware: KISS, framing, and a burst
through the 48 kHz audio path into the streaming receiver."""

import numpy as np
from scipy import signal as sps

from data2g import codes, hfchannel, modem, tnc
from data2g.config import FS, SUBMODES


def test_kiss_roundtrip_with_escapes_across_reads():
    frames = [bytes([0xC0, 0xDB, 0xDC, 0xDD, 1, 2]), b"hello", bytes([0xDB, 0xDC])]
    wire = b"".join(tnc.kiss_encode(f) for f in frames)
    dec, got = tnc.KissDecoder(), []
    for i in range(0, len(wire), 3):  # frames split across reads
        got += dec.feed(wire[i : i + 3])
    assert got == [(0, f) for f in frames]


def test_framing_loses_only_what_a_bad_codeword_touches():
    spec = SUBMODES["qpsk-r1/5"]  # 46-byte payloads
    packets = [bytes([i]) * n for i, n in enumerate((10, 60, 20, 5))]
    payloads = tnc.pack(packets, spec)
    assert all(len(p) == codes.payload_bytes(spec) for p in payloads)
    assert tnc.unpack(payloads, [True] * len(payloads)) == (packets, 0)
    # codeword 1 (bytes 46-91) holds the middle of packet 1 and the length
    # of packet 2: packet 0 survives, packet 1 is lost, parsing stops there
    ok = [True] * len(payloads)
    ok[1] = False
    assert tnc.unpack(payloads, ok) == (packets[:1], 2)


def test_decimator_is_seamless_across_chunks():
    x = np.random.default_rng(0).normal(size=48000)
    whole = tnc.Decimator(48000)(x)
    d = tnc.Decimator(48000)
    parts = np.concatenate([d(x[i : i + 1234]) for i in range(0, len(x), 1234)])
    np.testing.assert_allclose(parts, whole, atol=1e-12)


def test_burst_through_the_audio_path_and_streaming_receiver():
    spec = SUBMODES["qpsk-r1/5"]
    packets = [b"CQ CQ de TEST" * 3, bytes(range(200))[:90]]
    x = modem.modulate(tnc.pack(packets, spec), spec)
    x = np.concatenate([np.zeros(FS), x, np.zeros(FS)])
    y = hfchannel.apply_channel(x, snr_db=15, freq_offset_hz=20.0, ppm=5, seed=1)
    audio = sps.resample_poly(y, 6, 1)  # as the TNC's 48 kHz device would carry it
    dec, rx, got = tnc.Decimator(48000), tnc.Receiver(modem.Accept.of(["qpsk-r1/5"], min_score=0.35)), []
    for i in range(0, len(audio), 24000):  # half-second chunks
        for kind, ev in rx.feed(dec(audio[i : i + 24000])):
            if kind == "burst":
                b = modem.decode_received(ev["rx"])
                got += tnc.unpack(b.payloads, b.crc_ok)[0]
    assert got == packets


def test_digital_silence_is_not_searched():
    rx = tnc.Receiver(modem.Accept.of(["qpsk-r1/5"]))
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


def test_a_false_lock_on_data_does_not_cost_the_next_burst():
    """Data whose preamble was lost can read as a burst with a random header
    (w floor 0.25: about half the time), claiming up to a long burst's
    length. A real burst arriving meanwhile must still be received: the
    receiver keeps searching and a better header supersedes."""
    from data2g.config import BANDS, LEADIN_SAMPLES

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
        rx = tnc.Receiver(modem.Accept.of(None, 16.0))
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
        rx = tnc.Receiver(modem.Accept.of(None, 16.0))
        events = [(k, ev) for i in range(0, len(y), FS // 10) for k, ev in rx.feed(y[i:i + FS // 10])]
        heads = [ev for k, ev in events if k == "header"]
        assert heads and all(h["spec"].name == spec.name and h["score"] > 0.9 for h in heads), (
            seed, [(h["spec"].name, round(h["score"], 2)) for h in heads])
