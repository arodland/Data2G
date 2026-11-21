"""receiver.NoiseProfile: the passband's noise between bursts, per sub-band."""

import numpy as np
import pytest

from data2g.config import FS
from data2g.receiver import NoiseProfile


def feed(p, x, t0=0.0, step=FS // 25):
    for i in range(0, len(x), step):
        p.feed(x[i:i + step], t0 + i / FS)


def test_predictor_weighs_the_same_sub_bands():
    """The outcome model's noise inputs (predictor.noise_features) are per these sub-bands."""
    from data2g.arq import predictor as P

    assert P.NOISE_BANDS_HZ == tuple(map(tuple, NoiseProfile.BANDS_HZ))


def test_white_noise_is_flat():
    p = NoiseProfile()
    feed(p, np.random.default_rng(1).normal(0, 0.1, 30 * FS))
    s = p.snapshot()
    assert s["noise_blocks"] == 270  # 30 s less COMMIT_S
    assert np.ptp(s["noise_db"]) < 0.5
    assert max(s["noise_tail_db"]) < 2  # a 0.1 s block's own spread: 1.0-1.4 dB
    assert s["impulses_per_min"] < 1  # Gaussian peaks over 80 samples stay under 5x


def test_intermittent_edge_carrier_shows_in_its_band_only():
    """A 2390 Hz carrier on 20% of the time (as on air): the 2100-2700 Hz
    band's tail stands out, its median and the other bands don't move."""
    rng = np.random.default_rng(2)
    n = 60 * FS
    x = rng.normal(0, 0.1, n)
    t = np.arange(n) / FS
    on = (t % 5.0) < 1.0
    x += on * 0.3 * np.sin(2 * np.pi * 2390 * t)
    p = NoiseProfile()
    feed(p, x)
    s = p.snapshot()
    tails = s["noise_tail_db"]
    assert tails[4] > 10 and max(tails[:4]) < 3
    assert np.ptp(s["noise_db"]) < 1.0


def test_impulses_are_counted():
    """Clicks at 2 per second, 25 dB over the noise (on air: ~22 dB, 1-3 ms):
    about 120 a minute, the bands' medians unmoved."""
    from data2g import hfchannel

    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.1, 60 * FS)
    p = NoiseProfile()
    feed(p, hfchannel.clicks(x, 2.0, 25.0, seed=6, s_power=0.01))
    s = p.snapshot()
    assert 80 < s["impulses_per_min"] < 160
    assert np.ptp(s["noise_db"]) < 0.5


def test_marked_spans_and_gaps_are_not_noise():
    """A burst (loud, in band) marked as heard is left out; so is the radio's
    recovery after a transmission, and a gap (transmitting) starts a new block."""
    rng = np.random.default_rng(3)
    p = NoiseProfile()
    x = rng.normal(0, 0.1, 40 * FS)
    x[10 * FS:14 * FS] += rng.normal(0, 3.0, 4 * FS)  # a burst at 10-14 s
    p.mark(10.0, 14.0)
    feed(p, x[:20 * FS])
    p.mark(22.0, 22.0 + NoiseProfile.RECOVER_S)  # we transmitted 20-22 s
    feed(p, x[22 * FS:], 22.0)
    s = p.snapshot()
    assert np.ptp(s["noise_db"]) < 0.5 and max(s["noise_tail_db"]) < 3
    assert s["noise_blocks"] == 200 - 40 + 150 - 6  # 0-20 s less the burst; 22-37 s (COMMIT_S) less recovery


def muted(seed=7):
    """60 s of noise, muted 15.03-45.07 s (60 dB down, as another program
    sharing the radio does while it transmits) and zeroed 50.0-50.3 s."""
    x = np.random.default_rng(seed).normal(0, 0.1, 60 * FS)
    x[round(15.03 * FS):round(45.07 * FS)] *= 1e-3
    x[50 * FS:round(50.3 * FS)] = 0.0
    return x


def test_mutes_are_not_noise():
    """The muted blocks and MUTE_PAD_S either side are left out: the floor,
    tail and impulse count are the noise's alone. (Kept, the mute set the
    floor, and the edge blocks' few noise pieces read as impulses over the
    mute's.)"""
    p, ref = NoiseProfile(), NoiseProfile()
    x = muted()
    feed(p, x)
    feed(ref, x[:15 * FS])
    s, r = p.snapshot(), ref.snapshot()
    # 0-57 s (COMMIT_S) less 14.1-46.0 s (blocks 15.1-45.0 s are under 20 dB, the edges' are not) and 49.0-51.3 s
    assert s["noise_blocks"] == 570 - 319 - 23
    np.testing.assert_allclose(s["noise_db"], r["noise_db"], atol=0.3)
    assert max(s["noise_tail_db"]) < 2 and s["impulses_per_min"] < 1


def test_too_little_audio_is_none():
    p = NoiseProfile()
    feed(p, np.zeros(4 * FS))
    assert p.snapshot() is None


@pytest.mark.parametrize("native_side", [False, True])
def test_native_matches(native, native_side):
    """The C++ profile on the same audio and marks: the same blocks kept,
    the same numbers to FFT tolerance."""
    rng = np.random.default_rng(4)
    n = 30 * FS
    t = np.arange(n) / FS
    x = rng.normal(0, 0.1, n) * (1 + 0.5 * (t > 15)) + ((t % 4) < 1) * 0.2 * np.sin(2 * np.pi * 600 * t)
    x[rng.integers(0, n, 200)] += 3.0  # impulses
    x[round(8.05 * FS):round(11.0 * FS)] *= 1e-3  # a mute
    outs, ps = [], []
    for cls in (NoiseProfile, native.receiver.NoiseProfile):
        p = cls()
        p.mark(5.0, 7.5)
        feed(p, x)
        outs.append(p.snapshot())
        ps.append(p)
    py, cpp = outs
    assert cpp["noise_blocks"] == py["noise_blocks"]
    np.testing.assert_allclose(cpp["noise_db"], py["noise_db"], atol=1e-9)
    np.testing.assert_allclose(cpp["noise_tail_db"], py["noise_tail_db"], atol=1e-9)
    assert cpp["impulses_per_min"] == pytest.approx(py["impulses_per_min"]) and py["impulses_per_min"] > 0
    for start, end, lo, hi in ((16.0, 19.5, 400, 900), (25.0, 28.5, 950, 2100)):  # the 600 Hz tone's band, and not
        zp = ps[0].span_z(ps[0].span_spectra(start, end), lo, hi)
        assert ps[1].span_z(start, end, lo, hi) == pytest.approx(zp, rel=1e-9)
    assert ps[0].span_z(ps[0].span_spectra(16.0, 19.5), 400, 900) > 5 > ps[0].span_z(ps[0].span_spectra(25.0, 28.5), 950, 2100)


def test_recordings_carry_it(tmp_path):
    """Engine recordings: every rx event has a noise profile (None until
    MIN_BLOCKS of idle audio)."""
    import json

    from test_engine import link

    from data2g.arq import engine as E
    from data2g.arq import session as S

    a, b = E.Engine("W1AW", seed=1, record_dir=tmp_path / "a"), E.Engine("K2XYZ", seed=2)
    b.listen()
    a.connect("K2XYZ", 2)
    assert link(a, b, 12, 60, lambda: a.session.state == S.CONNECTED)
    ev = [json.loads(line) for line in open(tmp_path / "a" / "events.jsonl")]
    rx = [e for e in ev if e["kind"] == "rx"]
    assert rx and all("noise" in e for e in rx)
    got = [e["noise"] for e in rx if e["noise"]]
    assert all(len(g["noise_db"]) == len(g["noise_tail_db"]) == len(NoiseProfile.BANDS_HZ) for g in got)


def agc_stream(agc: bool):
    """40 s of white noise with a 5 s burst (1300-1750 Hz noise, 15 dB over
    the floor in SNR_REF_BW_HZ) from 30 s, through a receiver AGC or not."""
    from data2g import agc as AGC

    rng = np.random.default_rng(7)
    n, sigma = 40 * FS, 0.1
    x = rng.normal(0, sigma, n)
    b = np.fft.rfft(rng.normal(size=5 * FS))
    f = np.fft.rfftfreq(5 * FS, 1 / FS)
    b[(f < 1300) | (f >= 1750)] = 0
    s = np.fft.irfft(b, 5 * FS)
    s *= np.sqrt(sigma ** 2 * 2500 / (FS / 2) * 10 ** 1.5 / np.mean(s ** 2))
    x[30 * FS:35 * FS] += s
    return AGC.apply(AGC.Spec(0.0, 30.0, 0.005, 0.5) if agc else None, x, sigma ** 2)


def test_span_snr_gauges_agc():
    """Receiver AGC drops the gain under a strong burst (issue #257): the
    out-of-band gauge reads the burst's SNR as without it; the floor alone
    reads it ~10 dB low."""
    got = {}
    for agc in (False, True):
        p = NoiseProfile()
        p.mark(30.0, 35.0)
        feed(p, agc_stream(agc))
        got[agc] = p.span_snr_db(p.span_spectra(30.0, 35.0), 1300, 1750)
        if agc:
            p.GAUGE_HZ = (0.0, 0.0)
            got["ungauged"] = p.span_snr_db(p.span_spectra(30.0, 35.0), 1300, 1750)
    assert abs(got[True] - got[False]) < 1.0, got
    assert got["ungauged"] < got[False] - 8, got


def test_span_snr_native_matches(native):
    """The C++ span_snr_db (the AGC gauge with it) on the same audio and marks."""
    for agc in (False, True):
        ps = [NoiseProfile(), native.receiver.NoiseProfile()]
        for p in ps:
            p.mark(30.0, 35.0)
            feed(p, agc_stream(agc))
        for start, end, lo, hi in ((30.0, 35.0, 1300, 1750), (30.0, 35.0, 325, 2725), (10.0, 14.0, 950, 2100)):
            py = ps[0].span_snr_db(ps[0].span_spectra(start, end), lo, hi)
            assert ps[1].span_snr_db(start, end, lo, hi) == pytest.approx(py, rel=1e-9, abs=1e-9)
