"""tnc.NoiseProfile: the passband's noise between bursts, per sub-band."""

import numpy as np
import pytest

from data2g.config import FS
from data2g.tnc import NoiseProfile


def feed(p, x, t0=0.0, step=FS // 25):
    for i in range(0, len(x), step):
        p.feed(x[i:i + step], t0 + i / FS)


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
    outs = []
    for cls in (NoiseProfile, native.tnc.NoiseProfile):
        p = cls()
        p.mark(5.0, 7.5)
        feed(p, x)
        outs.append(p.snapshot())
    py, cpp = outs
    assert cpp["noise_blocks"] == py["noise_blocks"]
    np.testing.assert_allclose(cpp["noise_db"], py["noise_db"], atol=1e-9)
    np.testing.assert_allclose(cpp["noise_tail_db"], py["noise_tail_db"], atol=1e-9)
    assert cpp["impulses_per_min"] == pytest.approx(py["impulses_per_min"]) and py["impulses_per_min"] > 0


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
