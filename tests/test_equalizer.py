"""Pins for the receiver faults found while replacing SSTVAE's EQ.
Each case failed on the old receiver; see the old README, "Equalizer"."""

import numpy as np
import pytest
from scipy import signal

from data2g import codes, config, modem
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


@pytest.mark.hf_only("window placement for echoes most of a 4 ms CP; FM's CP (2 ms, provisional) and DEMOD_BACKOFF "
                     "are M4's: at 16 kHz a 1.25-1.5 ms echo leaves 1 bit error in 5760")
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


def test_delay_support_noise_gate_keeps_a_lone_path_narrow():
    """One path at -3 dB per carrier: the profile's noise floor is ~10 dB
    under it, so ungated noise peaks spread the support; gated on n0 it
    stays on the path."""
    from data2g import equalizer

    wide, narrow = [], []
    for seed in range(20):
        _, hp, bb = _two_path_pilots(seed, 2.0, late=0.0)
        d0, d1 = equalizer.delay_support(hp, bb=bb)
        g0, g1 = equalizer.delay_support(hp, bb=bb, n0=2.0)
        wide.append(d1 - d0)
        narrow.append(g1 - g0)
    # the premise (ungated noise spreads it) needs HF's 4 ms CP of search: FM's 2 ms holds too few bins
    assert config.PROFILE != "hf" or np.median(wide) > 40
    assert max(narrow) <= 8


def test_dd_time_pass_low_rank_matches_the_exact_solve(monkeypatch):
    """refine's time pass in its reduced-rank form (DD_LOWRANK_TOL) against the exact solve (a negative
    tolerance: full rank, which the cost rule sends to the exact solve), on a real fading receive with half
    the codewords known and then all: well under the estimate's own error."""
    import copy

    from data2g import equalizer, hfchannel
    from data2g.arq import phy
    from data2g.hfchannel import FadingPreset

    spec = SUBMODES["w48-qpsk-r1/2"]
    rng = np.random.default_rng(16)
    sent = [rng.bytes(codes.payload_bytes(spec)) for _ in range(16)]
    x = np.concatenate([np.zeros(3000), modem.modulate(sent, spec), np.zeros(3000)])
    y = hfchannel.apply_channel(x, snr_db=10.0, fading_preset=FadingPreset("mpd", 2.0, 2.0), seed=16)
    calls, refine = [], equalizer.refine
    monkeypatch.setattr(equalizer, "refine", lambda *a, **k: calls.append(copy.deepcopy((a, k))) or refine(*a, **k))
    r = modem.receive(y)
    bits = np.stack([codes.encode(spec, p, index=i) for i, p in enumerate(sent)])
    post = {i: 30.0 * (1 - 2.0 * bits[i]) for i in range(0, len(sent), 2)}
    phy._dd_estimate(r, post)
    phy._dd_estimate(r, {i: 30.0 * (1 - 2.0 * bits[i]) for i in range(len(sent))})
    assert len(calls) == 2
    for a, k in calls:
        h1, m1 = refine(*a, **k)
        monkeypatch.setattr(equalizer, "DD_LOWRANK_TOL", -1.0)
        h0, m0 = refine(*a, **k)
        monkeypatch.undo()
        m0 = np.maximum(m0, 1e-12 * a[6]["p_sig"])
        assert np.max(np.abs(h1 - h0) ** 2 / m0) < 1e-6
        assert np.max(np.abs(m1 - m0) / m0) < 1e-3


def test_snr_noise_ignores_what_repeats_in_every_pilot():
    """A fixed weak echo past the support (a radio passband's tail) and a
    fixed per-carrier pattern (n4) read as noise in n0, not in n0_snr."""
    from data2g import equalizer
    from data2g.config import FS
    from data2g.waveform import ofdm

    rng = np.random.default_rng(5)
    # past the support (n0 reads it) and inside its SNR_SLACK widening (n0_snr drops it): 26 samples
    # at HF's 8 kHz; at FM's 16 kHz band w's support reaches 26 and the window is 34-38
    d = 36 if FS == 16000 else 26
    for band, extra in (("w", lambda bb: 0.1 * np.exp(-2j * np.pi * bb * d / FS)),  # -20 dB at d samples
                        ("n4", lambda bb: 0.1 * np.exp(2j * np.pi * rng.uniform(size=len(bb))))):
        bb = ofdm.band(band).bb.astype(float)
        n_p, n0 = 40, 1e-3  # 30 dB
        h = np.ones((n_p, len(bb)), complex) + extra(bb)
        h += np.sqrt(n0 / 2) * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
        est = equalizer.estimate(h, (0, 0), bb, n0_pre=n0)
        assert est["n0"] > 3 * n0, band
        assert abs(10 * np.log10(est["n0_snr"] / n0)) < 1.0, band
