import numpy as np
import pytest

from data2g import polar


def test_transform_is_an_involution():
    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, (5, 64)).astype(np.uint8)
    assert np.array_equal(polar.transform(polar.transform(u)), u)


@pytest.mark.parametrize("e", [240, 480, 720])
def test_scl_decodes(e):
    code = polar.PolarCode(48, e, design_snr_db=-4.0)
    dec = polar.SCLDecoder(code, list_size=8)
    rng = np.random.default_rng(e)
    bits = rng.integers(0, 2, (64, 48))
    x = 1.0 - 2.0 * code.encode(bits)
    sig = 0.8
    y = x + rng.normal(scale=sig, size=x.shape)
    u, pm = dec.decode(2 * y / sig**2)
    assert u.shape == (64, 8, 48)
    assert (u[:, 0] == bits).all(axis=1).mean() > 0.95
