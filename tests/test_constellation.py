import numpy as np
import pytest

from data2g import constellation as C


@pytest.mark.parametrize("m", [2, 4, 6, 8])
def test_gray_qam_is_gray_and_unit_power(m):
    p = C.gray_qam(m)
    assert abs(np.mean(np.abs(p) ** 2) - 1) < 1e-12
    dmin = np.min(np.abs(p[:, None] - p[None, :]) + 9 * np.eye(len(p)))
    lb = C.label_bits(m)
    for i in range(len(p)):
        nn = np.flatnonzero(np.abs(np.abs(p - p[i]) - dmin) < 1e-9)
        assert all(np.sum(lb[i] != lb[j]) == 1 for j in nn)


def test_llr_matches_closed_form_for_qpsk():
    """Gray QPSK's exact LLR is linear: 2 sqrt(2) Re/Im(y h*) / var, up
    to which axis a bit sits on and its sign."""
    rng = np.random.default_rng(0)
    p = C.gray_qam(2)
    y = rng.normal(size=50) + 1j * rng.normal(size=50)
    h = rng.normal(size=50) + 1j * rng.normal(size=50)
    var = rng.uniform(0.5, 2, 50)
    l = C.llr(y, h, var, p).reshape(-1, 2)
    mrc = y * np.conj(h) / var
    np.testing.assert_allclose(l[:, 0], -2 * np.sqrt(2) * mrc.real, rtol=1e-9)
    np.testing.assert_allclose(l[:, 1], -2 * np.sqrt(2) * mrc.imag, rtol=1e-9)


@pytest.mark.parametrize("m", [4, 6])
def test_llr_signs_at_high_snr(m):
    rng = np.random.default_rng(m)
    p = C.gray_qam(m)
    bits = rng.integers(0, 2, 200 * m)
    x = C.modulate(bits, p)
    l = C.llr(x, np.ones_like(x), np.full(len(x), 1e-3), p)
    assert np.all((l < 0) == (bits == 1))
