"""Pins for the receiver faults found while replacing SSTVAE's EQ.
Each case failed on the old receiver; see the old README, "Equalizer"."""

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


def _two_path_pilots(seed, n0, delay=24, late=0.5, n_p=17):
    """n10's LS pilots on a static channel of two paths `delay` samples
    apart, the late one `late` x the first's amplitude, unit power, plus
    noise of variance n0 per carrier."""
    from data2g.waveform import ofdm

    bb = ofdm.band("n10").bb.astype(float)
    rng = np.random.default_rng(seed)
    h = 1 + late * np.exp(-2j * np.pi * bb * delay / modem.FS + 0.7j)
    h /= np.sqrt(np.mean(np.abs(h) ** 2))
    noise = rng.normal(size=(n_p, len(bb))) + 1j * rng.normal(size=(n_p, len(bb)))
    return h, h[None] + np.sqrt(n0 / 2) * noise, bb


def test_support_order_pads_for_a_path_a_narrow_band_cannot_resolve():
    """n10 at 30 dB on two paths 3 ms apart, the late one 6 dB down: the
    delay profile reads one, the tight projection drops the other (an
    error floor near -9 dB), the padded one keeps it (-32 dB), and
    support_slack picks the padded one."""
    from data2g import equalizer

    h, hp, bb = _two_path_pilots(0, 1e-3)
    sup = equalizer.delay_support(hp, bb=bb)
    assert equalizer.support_slack(hp, sup, bb, n0_pre=1e-3) == equalizer.PADDED_SLACK
    err = {s: np.mean(np.abs(equalizer.freq_smooth(hp, sup, bb, s)[0] - h) ** 2)
           for s in (equalizer.SUPPORT_SLACK, equalizer.PADDED_SLACK)}
    assert err[equalizer.PADDED_SLACK] < err[equalizer.SUPPORT_SLACK] / 10
    est = equalizer.estimate(hp, sup, bb=bb, n0_pre=1e-3)
    assert est["slack"] == equalizer.PADDED_SLACK


def test_support_order_stays_tight_on_a_flat_channel_in_noise():
    """One path at -5 dB: the padded dimensions hold only noise, so the
    tight projection (more noise averaging) is kept, burst after burst."""
    from data2g import equalizer

    picks = []
    for seed in range(50):
        _, hp, bb = _two_path_pilots(seed, 3.0, late=0.0)
        sup = equalizer.delay_support(hp, bb=bb)
        picks.append(equalizer.support_slack(hp, sup, bb, n0_pre=3.0))
    assert picks.count(equalizer.SUPPORT_SLACK) >= 49
