import numpy as np
import pytest

from data2g import codes, hfchannel, modem
from data2g.config import BANDS, SUBMODES


def _payloads(spec, n, seed):
    rng = np.random.default_rng(seed)
    return [rng.bytes(codes.payload_bytes(spec)) for _ in range(n)]


LOOPBACK = [(s, "awgn", 30) for s in SUBMODES] + [
    (s, "mpp", 20) for s in ("ack-1f", "polar-k96-f4", "qpsk-r1/2", "n4-ack-2f")  # n4: n10's sync and header
]


@pytest.mark.parametrize("sub,channel,snr", LOOPBACK)
def test_burst_loopback(sub, channel, snr):
    """Every submode bit-exact well above its threshold, through the full
    numpy modem with CFO and clock error; a few robust ones on fading."""
    spec = SUBMODES[sub]
    sent = _payloads(spec, 2, spec.index)
    x = modem.modulate(sent, sub)
    x = np.concatenate([np.zeros(3000), x, np.zeros(3000)])
    y = hfchannel.apply_channel(
        x, snr_db=snr, freq_offset_hz=37.0, ppm=10,
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
        for n in (1, 2, 63, 64):
            _, got, score = modem.decode_header(1.0 - 2.0 * modem.header_bits(sub, n, band), band)
            assert got == (spec, n) and score > 0.999


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


def test_no_burst_from_a_tone():
    """SSTVAE's 1b05ba4 false lock: a steady carrier read as a burst with a
    random header before the wide band had a header score floor. (Data
    whose preamble was lost may still read as one at the 0.25 floor; that
    is tnc.Receiver's supersede search's job: tests/test_tnc.py.)"""
    from data2g.config import FS

    for seed in range(3):
        tone = np.sin(2 * np.pi * (900 + 300 * seed) * np.arange(40000) / FS)
        for x in (tone,):
            y = hfchannel.awgn(np.concatenate([np.zeros(4000), x, np.zeros(4000)]), 15, seed=seed, s_power=0.5)
            with pytest.raises(modem.SyncError):
                modem.demodulate(y)


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
