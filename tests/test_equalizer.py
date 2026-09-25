"""Pins for the receiver faults found while replacing SSTVAE's EQ.
Each case failed on the old receiver; see README "Equalizer"."""

import numpy as np
from scipy import signal

from data2g import codes, modem
from data2g.config import SUBMODES

SPEC = SUBMODES["qpsk-r1/2"]  # 8-frame codewords


def _burst(seed, n=3):
    rng = np.random.default_rng(seed)
    sent = [rng.bytes(codes.payload_bytes(SPEC)) for _ in range(n)]
    x = np.concatenate([np.zeros(3000), modem.modulate(sent, SPEC), np.zeros(3000)])
    return x, codes.spread(np.stack([codes.encode(SPEC, p, index=i) for i, p in enumerate(sent)]), SPEC.bits_per_cu)


def _raw_ber(y, bits):
    r = modem.receive(y)
    est = r["est"]
    soft = modem.soft_bits(r["raw"], est["h"], modem.noise_var(est["h"], est) + est["mse"], SPEC)
    return np.mean((soft < 0) != (bits == 1))


def test_late_dominant_path_is_not_synced_to():
    """Static two-path, 24 samples apart, late path 2x the early one's
    amplitude, no noise. SSTVAE's first-path rule syncs to the late path
    here and pushes the early one out of the CP: 0.44% raw BER. Window
    placement from the delay profile puts both inside it."""
    x, bits = _burst(0)
    z = signal.hilbert(x)
    y = np.real(z + 2.0 * np.exp(1j) * np.concatenate([np.zeros(24), z[:-24]]))
    assert _raw_ber(y, bits) == 0


def test_alias_resolution_picks_the_alias_nearest_the_coarse_estimate():
    """With 8 preamble repeats no fading burst measured pulls acquisition
    past the alias boundary (mpd: worst 2.5 Hz of 200); this is the
    safety net for when one does (SSTVAE's 4 repeats: 3.7 Hz, mpp)."""
    rate = 1 / modem.equalizer.FRAME_S  # 6.94 Hz
    for true in (-9.0, -3.7, 0.2, 3.7, 9.0):
        fine = (true + rate / 2) % rate - rate / 2  # what the pilots see
        assert abs(modem.resolve_alias(fine, true + 1.5) - true) < 1e-9


def test_a_tone_on_one_carrier_does_not_sink_the_burst():
    """A steady carrier 6 dB under the whole signal lands ~8 dB over one
    wide-band carrier. With one band-wide noise level that carrier's bits
    were confidently wrong (a codeword lost in most bursts); per-carrier
    noise marks them unreliable and LDPC fills them in."""
    from data2g import hfchannel
    from data2g.config import FS

    for seed in range(3):
        rng = np.random.default_rng(seed)
        sent = [rng.bytes(codes.payload_bytes(SPEC)) for _ in range(6)]
        x = np.concatenate([np.zeros(4000), modem.modulate(sent, SPEC), np.zeros(4000)])
        p = hfchannel.active_power(x)
        y = hfchannel.awgn(x, 6, seed=seed, s_power=p)
        y = y + np.sqrt(2 * p * 10 ** (-6 / 10)) * np.sin(2 * np.pi * 1440.0 * np.arange(len(y)) / FS + seed)
        b = modem.demodulate(y)
        assert b.payloads == sent and all(b.crc_ok)


def test_per_carrier_noise_is_the_band_level_on_a_clean_channel():
    rng = np.random.default_rng(0)
    for m in (5, 20, 60):
        p = rng.gamma(m, 1 / m, size=(200, 24))  # 200 bursts of 24 carriers, unit noise
        n0 = np.array([modem.equalizer.per_carrier_noise(row, m) for row in p])
        assert np.mean(n0 != n0[:, :1]) < 0.04  # 1-3% flagged by chance (the median reference is noisy too)
        assert abs(np.median(n0) - 1) < 0.05
