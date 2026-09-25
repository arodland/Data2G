import numpy as np
import pytest

from data2g import polar


def test_transform_is_an_involution():
    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, (5, 64)).astype(np.uint8)
    assert np.array_equal(polar.transform(polar.transform(u)), u)


def test_ga_construction_agrees_with_nr_reliability_order():
    """With nothing punctured, GA at a moderate SNR should pick nearly
    the same info set as TS 38.212's reliability sequence."""
    q = np.genfromtxt(polar.__file__.replace("polar.py", "codes_data/polar_5G.csv"), delimiter=";")
    order = q[:, 1].astype(int)  # bit indices, least reliable first
    n, k = 256, 64
    nr = set(order[order < n][-k:])
    ours = set(polar.PolarCode(k, n, design_snr_db=0.0).info_pos)
    assert len(nr & ours) >= 0.9 * k


@pytest.mark.parametrize("e", [240, 480, 720])
def test_scl_decodes(e):
    torch = pytest.importorskip("torch")
    code = polar.PolarCode(48, e, design_snr_db=-4.0)
    dec = polar.SCLDecoder(code, list_size=8)
    rng = np.random.default_rng(e)
    bits = rng.integers(0, 2, (64, 48))
    x = 1.0 - 2.0 * code.encode(bits)
    sig = 0.8
    y = x + rng.normal(scale=sig, size=x.shape)
    u, pm = dec.decode(torch.tensor(2 * y / sig**2, dtype=torch.float32))
    assert u.shape == (64, 8, 48)
    assert (u[:, 0].numpy() == bits).all(axis=1).mean() > 0.95
