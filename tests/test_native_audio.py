"""native.audio (Decimator, Interpolator, Blanker) against data2g.tnc and
data2g.host, chunk by chunk, so carried state is compared too. The filters
agree to rounding, not bits: scipy's lfilter with a = 1 is np.convolve,
summed by BLAS."""

import numpy as np
import pytest

from data2g import hfchannel, host, tnc
from data2g.config import FS


def chunks(n, rng):
    i = 0
    while i < n:
        k = int(rng.integers(1, 3000))
        yield slice(i, i + k)
        i += k


@pytest.mark.parametrize("rate", [8000, 16000, 48000, 96000])
def test_decimator(native, reference, rate):
    rng = np.random.default_rng(rate)
    py, nat = reference(tnc, "Decimator")(rate), native.audio.Decimator(rate)
    np.testing.assert_array_equal(nat.taps, py.taps)
    x = rng.normal(size=rate * 2)
    for s in chunks(len(x), rng):
        np.testing.assert_allclose(nat(x[s]), py(x[s]), rtol=0, atol=1e-12)


@pytest.mark.parametrize("rate", [8000, 48000])
def test_interpolator(native, reference, rate):
    rng = np.random.default_rng(rate + 1)
    py, nat = reference(host, "Interpolator")(rate), native.audio.Interpolator(rate)
    np.testing.assert_array_equal(nat.taps, py.taps)
    x = rng.normal(size=FS * 2)
    for s in chunks(len(x), rng):
        np.testing.assert_allclose(nat(x[s]), py(x[s]), rtol=0, atol=1e-12)


def test_blanker(native, reference):
    rng = np.random.default_rng(3)
    y = hfchannel.clicks(rng.normal(size=20 * FS), 10, 20, seed=1, s_power=1.0)
    # leading digital silence, and a +30 dB level step
    y = np.concatenate([np.zeros(FS // 3), y[:5 * FS], 30 * y[5 * FS:10 * FS], y[10 * FS:]])
    py, nat = reference(tnc, "Blanker")(), native.audio.Blanker()
    for s in chunks(len(y), rng):
        np.testing.assert_array_equal(nat(y[s]), py(y[s]))
        assert nat.env == py.env and nat.n_blanked == py.n_blanked
    assert py.n_blanked > 0
