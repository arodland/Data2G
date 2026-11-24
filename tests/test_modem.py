"""data2g.modem end to end: burst loopback per submode, the header code,
framing, HARQ through the modem, timing slips and the header copy."""

import numpy as np
import pytest

from data2g import codes, hfchannel, modem
from data2g.config import ACQUIRE_REACH_HZ, BANDS, FS, SUBMODES

# A sound card's offset: 37 Hz on HF; FM audio has none (an RF one comes out
# as DC), and its search reaches +-12.5 Hz, so less there.
CFO_HZ = min(37.0, 0.8 * ACQUIRE_REACH_HZ)


def _payloads(spec, n, seed):
    rng = np.random.default_rng(seed)
    return [rng.bytes(codes.payload_bytes(spec)) for _ in range(n)]


# f108 at 36: SNR here is in 2500 Hz and f108 spreads over 5350, 3.3 dB less per
# carrier, near its top rung's (256-point r0.874) threshold at 30
LOOPBACK = [(s, "awgn", 36 if SUBMODES[s].band == "f108" else 30) for s in SUBMODES] + [
    (s, "mpp", 20) for s in ("ack-1f", "polar-k96-f4", "qpsk-r1/2", "n4-ack-2f") if s in SUBMODES  # n4: n10's sync and header (HF)
]


@pytest.mark.parametrize("sub,channel,snr", LOOPBACK)
def test_burst_loopback(sub, channel, snr):
    """Every submode bit-exact well above its threshold, through the full
    numpy modem with CFO and clock error; a few robust ones on fading."""
    spec = SUBMODES[sub]
    sent = _payloads(spec, 2, spec.index)
    # FM's f55/f108 on the flat path: their top rungs (256-point r0.83-0.88) are
    # flat-radio rungs; the mic path clips them at 6 dB, near their threshold here
    # (test_fm_bands round-trips both paths clean)
    path = "flat" if spec.band in ("f55", "f108") else None
    x = modem.modulate(sent, sub, tx_path=path)
    x = np.concatenate([np.zeros(3000), x, np.zeros(3000)])
    y = hfchannel.apply_channel(
        x, snr_db=snr, freq_offset_hz=CFO_HZ, ppm=10,
        fading_preset=None if channel == "awgn" else channel, seed=3,
    )
    b = modem.demodulate(y)
    assert b.submode == spec
    assert b.payloads == sent and all(b.crc_ok)


@pytest.mark.parametrize("band", sorted(BANDS))
def test_header_roundtrip_every_field(band):
    """Every defined submode and codeword count decodes to itself with a
    full score; ML runs over valid words only, so an undefined submode
    is never sent and never decoded."""
    for (b, sub), spec in modem.BY_INDEX.items():
        if b != band:
            continue
        top = modem.max_codewords(band)
        for n in (1, 2, top - 1, top):
            _, got, score = modem.decode_header(1.0 - 2.0 * modem.header_bits(sub, n, band), band)
            assert got == (spec, n) and score > 0.999
        with pytest.raises(ValueError):
            modem.header_bits(sub, top + 1, band)


def test_header_of_another_version_is_rejected(monkeypatch):
    """The CRC is seeded with PROTOCOL_VERSION: a clean header sent by
    another version is d_min from every word of this one, so it scores
    under the floor (narrow bands; the wide ones have none)."""
    monkeypatch.setattr(modem, "PROTOCOL_VERSION", modem.PROTOCOL_VERSION + 1)
    sent = [1.0 - 2.0 * modem.header_bits(s.index, 5, "n10") for s in SUBMODES.values() if s.band == "n10"]
    monkeypatch.undo()
    assert sent and all(modem.decode_header(soft, "n10")[2] < modem.HEADER_MIN_SCORE["n10"] for soft in sent)


def test_header_corrects_heavy_noise():
    rng = np.random.default_rng(0)
    ok = 0
    specs = [s for s in SUBMODES.values() if s.band == "w"]
    for i in range(100):
        spec, n = specs[i % len(specs)], int(rng.integers(1, 65))
        x = 1.0 - 2.0 * modem.header_bits(spec.index, n)
        _, got, _ = modem.decode_header(x + rng.normal(scale=1.6, size=x.shape))  # -4 dB per bit
        ok += got == (spec, n)
    assert ok >= 99


@pytest.mark.parametrize("sub", ["ack-1f", "qpsk-r1/2"])
def test_crc_flags_garbage(sub):
    spec = SUBMODES[sub]
    p = _payloads(spec, 1, 9)[0]
    soft = 1.0 - 2.0 * codes.encode(spec, p)
    assert codes.decode(spec, soft) == (p, True)
    got, ok = codes.decode(spec, np.random.default_rng(1).normal(size=soft.shape))
    assert not ok


def test_spread_round_trips_and_keeps_symbols_whole():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, (3, 5, 48))  # batch, codewords, bits
    s = codes.spread(x, 4)
    assert np.array_equal(codes.despread(s, 5, 4), x)
    # each 4-bit symbol comes from one codeword, dealt round-robin
    assert np.array_equal(s[0, :4], x[0, 0, :4]) and np.array_equal(s[0, 4:8], x[0, 1, :4])


def test_every_submode_is_frozen():
    """Each submode's interleaver (and polar info set) comes from its
    committed file (tools/freeze_format.py), not from whatever this numpy
    computes: a submode changed without re-freezing fails here."""
    from data2g.codes import frozen

    stale = [s.name for s in SUBMODES.values() if frozen(s) is None]
    assert not stale, f"re-run tools/freeze_format.py: {stale}"


def test_resend_at_rv1_combines_through_the_modem():
    """A clean RV 1 burst alone is not a codeword decode can read, but it
    combines with RV 0's soft bits into one that decodes."""
    spec = SUBMODES["w48-qpsk-r1/2"]
    rng = np.random.default_rng(3)
    pl = [bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8)) for _ in range(2)]
    pad = np.zeros(2400)
    b0 = modem.demodulate(np.concatenate([pad, modem.modulate(pl, spec), pad]))
    b1 = modem.demodulate(np.concatenate([pad, modem.modulate(pl, spec, [1, 1]), pad]))
    assert b0.payloads == pl and not any(b1.crc_ok)
    buf = codes.combine(spec, codes.combine(spec, None, b0.soft, 0), b1.soft, 1)
    assert [p for p, ok in codes.decode_buffer(spec, buf, max_rv=1) if ok] == pl


def test_zero_payloads_decode_like_random_ones():
    """Zero-padded payloads (control codewords, a stream's tail) must not
    code to a lopsided bit mix the clipper wrecks: the scrambler."""
    spec = SUBMODES["w48-qpsk-r3/4"]
    pl = [bytes(codes.payload_bytes(spec))] * 3
    bits = np.stack([codes.encode(spec, p) for p in pl])
    assert 0.4 < bits.mean() < 0.6
    for seed in range(3):
        x = np.concatenate([np.zeros(2400), modem.modulate(pl, spec), np.zeros(2400)])
        b = modem.demodulate(hfchannel.apply_channel(x, snr_db=8, seed=seed))
        assert b.payloads == pl and all(b.crc_ok)


@pytest.mark.parametrize("slip", [85 * FS // 8000, -64 * FS // 8000])  # sound-card buffers, in time
def test_timing_slip_recovered(slip):
    """Samples inserted (a sender's playback underrun: recordings/20261013-234857
    had 85 of silence, a 512-frame period at 48 kHz) or dropped mid-burst: the
    frames after are demodulated where they moved to, and the burst decodes."""
    spec = SUBMODES["qpsk-r1/5"]
    sent = _payloads(spec, 3, 5)
    x = np.concatenate([np.zeros(3000), modem.modulate(sent, spec), np.zeros(3000)])
    at = len(x) // 2
    x = np.concatenate([x[:at], np.zeros(slip), x[at:]]) if slip > 0 else np.concatenate([x[:at], x[at - slip:]])
    r = modem.receive(hfchannel.apply_channel(x, snr_db=10, freq_offset_hz=CFO_HZ, ppm=10, seed=4))
    found = r["slips"][r["slips"] != 0]
    assert len(found) == 1 and abs(found[0] - slip) <= 3
    b = modem.decode_received(r)
    assert b.payloads == sent and all(b.crc_ok)


@pytest.mark.parametrize("channel", ["mpp", "mpd"])
def test_no_slip_from_fading(channel):
    """A path fading in or out moves the pilots' timing by its delay (mpd's
    4 ms): not a slip (SLIP_MIN), nor a deep fade's noise (the acceptance)."""
    spec = SUBMODES["qpsk-r1/5"]
    for seed in range(4):
        x = np.concatenate([np.zeros(3000), modem.modulate(_payloads(spec, 3, seed), spec), np.zeros(3000)])
        r = modem.receive(hfchannel.apply_channel(x, snr_db=5, freq_offset_hz=CFO_HZ, ppm=10, fading_preset=channel, seed=seed))
        assert not r["slips"].any()


def test_no_burst_from_a_tone():
    """SSTVAE's 1b05ba4 false lock: a steady carrier read as a burst with a
    random header before the wide band had a header score floor. (Data
    whose preamble was lost may still read as one at the 0.25 floor; that
    is receiver.Receiver's supersede search's job: tests/test_receiver.py.)"""
    from data2g.config import FS

    for seed in range(3):
        tone = np.sin(2 * np.pi * (900 + 300 * seed) * np.arange(40000) / FS)
        for x in (tone,):
            y = hfchannel.awgn(np.concatenate([np.zeros(4000), x, np.zeros(4000)]), 15, seed=seed, s_power=0.5)
            with pytest.raises(modem.SyncError):
                modem.demodulate(y)


@pytest.mark.parametrize("band", ["w", "w48", "n10"] + (["n4"] if any(s.band == "n4" for s in SUBMODES.values()) else []))
def test_a_steady_tone_does_not_read_as_pilots(band):
    """A steady narrowband signal repeats from frame to frame, so the
    coherence alone read it as a burst's frame pilots (a tone's leakage turns
    by one phase in every carrier). The pattern (the pilot divided out,
    adjacent carriers agree) is what a tone lacks: on a real burst's frame
    grid, the burst's own pilots pass both checks and a strong tone or
    Hand-like 8-FSK at 1530 Hz doesn't (scripts/pilot_metric_study.py)."""
    from data2g import cpm
    from data2g.arq.policy import slots_for

    s = SUBMODES[{"w": "qpsk-r1/2", "w48": "w48-qpsk-r1/2", "n10": "n10-qpsk-r1/2", "n4": "n4-qpsk-r1/3"}[band]]
    rng = np.random.default_rng(3)
    x = modem.modulate([rng.bytes(codes.payload_bytes(s)) for _ in range(slots_for(s, 6.0))], s)
    y = np.concatenate([np.zeros(FS), x, np.zeros(FS)])
    lock = modem.find_burst(y, [s.sync_band])

    def passes(sig):
        c, pat = modem.pilot_coherence(sig, lock, modem.PILOT_PAIRS, latest=True, pattern=True)
        return np.mean(c) > modem.PILOT_NOISE[band] and pat > modem.PILOT_PATTERN[band]

    assert passes(hfchannel.awgn(y, 10, seed=1))
    t = np.arange(len(y)) / FS
    hand = cpm.tones(cpm.Grid("hand1530", 8, 6.25, center=1530.0), rng.integers(0, 8, 66))
    for sig in (np.sin(2 * np.pi * 1530.0 * t), np.pad(hand, (0, len(y)))[:len(y)]):
        assert not passes(hfchannel.awgn(sig, 20, seed=2, s_power=hfchannel.active_power(sig)))


@pytest.mark.parametrize("sub,n_cw", [("ack-1f", 1), ("qpsk-r1/5", 1), ("w48-qpsk-r1/5", 2)])
def test_header_copy_rescues_a_lost_first_copy(sub, n_cw):
    """The 4-symbol headers (w, w48) carry a second copy in a frame of its
    own (after data frame 2, or the last on a shorter burst: ack-1f's 1
    frame). With the first copy wiped, the burst still decodes; the copy's
    frame is not data."""
    spec = SUBMODES[sub]
    rng = np.random.default_rng(7)
    payloads = [bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8)) for _ in range(n_cw)]
    x = modem.modulate(payloads, spec)
    assert modem.frames_on_air(spec, n_cw) == n_cw * spec.frames_per_cw + 1
    h0 = modem.LEADIN_SAMPLES + BANDS[spec.sync_band].preamble_samples
    x[h0:h0 + modem.header_samples(spec.sync_band)] = rng.normal(0, 1, modem.header_samples(spec.sync_band))
    y = hfchannel.apply_channel(np.concatenate([x, np.zeros(4000)]), snr_db=15, seed=3)
    b = modem.demodulate(y)
    assert b.submode.name == sub and b.payloads == payloads


def test_valid_signs_are_the_valid_rows_of_the_full_table():
    for band in modem.SYNC_BANDS:
        assert np.array_equal(modem._valid_signs(band), modem._header_signs(band)[modem._valid_words(band)])


def test_a_narrower_bands_burst_is_no_copy_lock():
    """find_copy asked only that a lock's frame pilots be coherent. An n10
    burst's pilots are (same frame grid, a tenth of the carriers), and its
    data read as a wide-band header copy over the floor: a 16qam-r1/3 x63
    lock holding BUSY for a minute. The pilot's pattern is not there
    (COPY_PATTERN)."""
    s = SUBMODES["n10-qpsk-r1/2"]
    rng = np.random.default_rng(4)
    x = modem.modulate([rng.bytes(codes.payload_bytes(s)) for _ in range(6)], s)
    y = hfchannel.awgn(np.pad(x, (FS // 2, FS)), 10, seed=4, s_power=hfchannel.active_power(x))
    assert modem.find_copy(y, "w") is None


def test_symbol_noise_scale_flags_a_click_hit_symbol():
    """Gaussian noise: every symbol's factor near 1. One symbol with 20x
    the noise (a click across all its carriers): that one raised, by about
    its excess, and the rest left alone."""
    from data2g import constellation

    pts = constellation.load("gray-qam16")
    rng = np.random.default_rng(7)
    n_rows, nc, var = 50, 48, 0.01
    h = np.exp(2j * np.pi * rng.uniform(size=(n_rows, nc)))
    x = pts[rng.integers(0, len(pts), size=(n_rows, nc))]
    noise = np.sqrt(var / 2) * (rng.normal(size=(n_rows, nc)) + 1j * rng.normal(size=(n_rows, nc)))
    noise[17] *= np.sqrt(20)
    s = modem.symbol_noise_scale((h * x + noise)[None], h[None], np.full((1, n_rows, nc), var), pts)[0, :, 0]
    assert s[17] > 8 and np.delete(s, 17).max() < 1.5


@pytest.mark.parametrize("group", [1, 2, 3, 7])
def test_windowed_spread_round_trips_and_keeps_groups_together(group):
    """codes.spread with a group: the inverse is despread with the same
    group, and a group's symbols sit in its own stretch of the burst."""
    rng = np.random.default_rng(0)
    coded = rng.integers(0, 2, size=(7, 24))
    x = codes.spread(coded, 2, group)
    assert np.array_equal(codes.despread(x, 7, 2, group), coded)
    pos = codes.despread(np.arange(7 * 12), 7, 1, group)  # each codeword's symbol positions
    for a in range(0, 7, group):
        g = pos[a:a + group]
        assert g.min() == a * 12 and g.max() == min(a + group, 7) * 12 - 1


def test_windowed_burst_loops_back_and_slot_soft_matches(monkeypatch):
    from data2g import config

    monkeypatch.setattr(config, "SPREAD_WINDOW", 8)  # 16-QAM r1/2: 4 frames a codeword, groups of 2
    spec = SUBMODES["16qam-r1/2"]
    sent = _payloads(spec, 5, 9)
    x = np.concatenate([np.zeros(FS // 2), modem.modulate(sent, spec), np.zeros(FS // 2)])
    r = modem.receive(hfchannel.apply_channel(x, snr_db=20, seed=3))
    b = modem.decode_received(r)
    assert b.payloads == sent and all(b.crc_ok)
    soft = modem.burst_soft(r)
    for s in range(5):
        np.testing.assert_allclose(modem.slot_soft(r, r["est"], s), soft[s])


def test_a_preamble_cfo_off_by_a_pilot_alias_is_put_right(monkeypatch):
    """Fading's random FM puts the preamble's CFO estimate past half the
    pilot rate now and then (1 burst in 10 on mpp, 6.8 Hz off); the
    pilots measure it only modulo 6.94 Hz, and the header's coarse estimate
    picks the alias (modem.receive). A local named `known` once shadowed
    the argument of that name, so the alias nearest no change was taken
    instead and such bursts were lost (Python only; native was right)."""
    spec = SUBMODES["qpsk-r1/2"]
    sent = _payloads(spec, 2, 5)
    x = np.concatenate([np.zeros(FS // 2), modem.modulate(sent, spec), np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=20, seed=5)
    best = modem._best_header

    def off(*a, **k):
        hd, acq, z = best(*a, **k)
        acq.freq_offset += 5.5
        return hd, acq, modem.dsp.freq_correct(z, 5.5)

    monkeypatch.setattr(modem, "_best_header", off)
    b = modem.decode_received(modem.receive(y))
    assert b.payloads == sent and all(b.crc_ok)


def test_stream_products_are_pair_products_of_its_outputs():
    """StreamDetector.products_from, extended chunk by chunk and trimmed as
    the receiver does, is pair_products of the kept outputs (find_copy reads
    it instead of recomputing)."""
    from data2g.config import FRAME_SAMPLES
    from data2g.waveform import dsp, ofdm, sync

    rng = np.random.default_rng(5)
    y = rng.normal(size=6 * FS)
    d = sync.StreamDetector(ofdm.band("w"))
    fed = 0
    while fed < len(y):
        n = int(rng.integers(300, 5000))
        d.feed(dsp.to_baseband(y[fed:fed + n], fed))
        fed += n
        if rng.random() < 0.3:
            d.trim(fed - int(rng.integers(FS, 3 * FS)))
        if rng.random() < 0.5:
            a, D = d.products_from(d.c0 + 7, FRAME_SAMPLES, modem.COPY_PAIRS)
            _, C = d.outputs_from(d.c0)
            want = sync.pair_products(C, FRAME_SAMPLES, modem.COPY_PAIRS)[:, a - d.c0:]
            assert a == d.c0 + 7 and D.shape == want.shape
            # (bit for bit in Python; native rounds its products its own way)
            np.testing.assert_allclose(D, want, rtol=0, atol=1e-12 * max(1.0, float(np.abs(want).max(initial=0.0))))


def test_find_copy_on_audio_shorter_than_its_pairs():
    """Fewer matched filter outputs than COPY_PAIRS frames: nothing found,
    not an error (receive_any runs it on whatever it is given)."""
    rng = np.random.default_rng(1)
    for n in (1000, 3000):
        assert modem.find_copy(rng.normal(size=n), "w") is None


def _fold_walk(seed):
    """Pair products with bursts 40 dB up, and windows that mostly slide
    forward (the receiver's hops), sometimes jump back, change length."""
    F = modem.FRAME_SAMPLES
    rng = np.random.default_rng(seed)
    D = (rng.normal(size=(25, 60 * F)) + 1j * rng.normal(size=(25, 60 * F))) * np.where(
        rng.random(60 * F) < 0.3, 100.0, 1.0)
    lo, steps = 0, []
    for _ in range(300):
        m = int(rng.integers(2, 10))
        lo = (min(lo + int(rng.integers(0, 2500)), 60 * F - m * F) if rng.random() > 0.05
              else int(rng.integers(0, 50 * F)))
        steps.append((max(0, lo - int(rng.integers(0, 3 * F))), lo, lo + m * F))
    return D, steps


def test_fold_track_is_the_fold_of_its_window():
    """modem.FoldTrack, kept running (adds, subtractions, fresh sums every
    RENEW), rolled to its window's start: _pilot_fold's, to rounding."""
    F = modem.FRAME_SAMPLES
    D, steps = _fold_walk(3)
    t = modem.FoldTrack()
    for a, lo, hi in steps:
        R = t.fold(D[:, a:], a, lo, hi)
        ref = modem._pilot_fold(D, lo, (hi - lo) // F)
        np.testing.assert_allclose(np.roll(R, -(lo % F), axis=1), ref, rtol=0, atol=1e-12 * np.abs(ref).max())
