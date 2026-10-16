"""Rate conversion and impulse blanking (data2g.audio)."""

import numpy as np

from data2g import audio, hfchannel
from data2g.config import FS


def test_decimator_is_seamless_across_chunks():
    x = np.random.default_rng(0).normal(size=48000)
    whole = audio.Decimator(48000)(x)
    d = audio.Decimator(48000)
    parts = np.concatenate([d(x[i : i + 1234]) for i in range(0, len(x), 1234)])
    np.testing.assert_allclose(parts, whole, atol=1e-12)


def test_blanker_removes_clicks_and_leaves_noise_and_level_steps():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=20 * FS)
    b = audio.Blanker()
    assert np.array_equal(b(noise), noise) and b.n_blanked == 0
    y = hfchannel.clicks(noise, 10, 20, seed=1, s_power=1.0)
    z = audio.Blanker()(y)
    assert np.mean(z**2) < 1.1 < np.mean(y**2) / 1.5
    # a strong station keying up (+30 dB) is a level step, not a click
    step = np.concatenate([noise[:FS], 30 * noise[FS:2 * FS]])
    b = audio.Blanker()
    assert np.array_equal(b(step), step)
    assert np.array_equal(audio.Blanker()(np.zeros(FS)), np.zeros(FS))
