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


def _ir(e=240, k=56, design=-6.0):
    base = polar.PolarCode(k, e, design_snr_db=-3.0)
    return base, polar.IRPolarCode(base, design)


def test_ir_keeps_rv0_and_extends():
    base, ir = _ir()
    assert (ir.n, ir.e, ir.k) == (2 * base.n, 2 * base.e, base.k)
    assert len(ir.copies)
    src, dst = ir.copies.T
    assert (src < base.n).all() and np.isin(dst, ir.info_pos).all()
    bits = np.random.default_rng(0).integers(0, 2, (20, 56))
    x = ir.encode(bits)
    np.testing.assert_array_equal(x[:, : base.e], base.encode(bits))  # RV 0 unchanged on air
    # it is the length-2N polar code [v, u] with v holding the copies
    u = np.zeros((20, ir.n), np.uint8)
    u[:, ir.info_pos] = bits
    u[:, src] = u[:, dst]
    np.testing.assert_array_equal(polar.transform(u)[:, ir.sent], x)


def test_ir_design_copies_only_upward():
    """Every copy is to a position the combined code protects better."""
    base, ir = _ir()
    mean = np.zeros(ir.n)
    mean[ir.sent] = 4 * 10 ** (-6.0 / 10)
    rel = polar.de_reliability(mean)
    assert (rel[ir.copies[:, 0]] > rel[ir.copies[:, 1]]).all()


def test_ir_beats_chase():
    """RV 0 + RV 1 decodes where RV 0 twice (Chase) mostly fails."""
    base, ir = _ir(e=80, design=-6.0)
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, (200, 56))
    x = 1.0 - 2.0 * ir.encode(bits)
    sig = np.sqrt(1 / (2 * 10 ** (-4.5 / 10)))
    noisy = lambda: 2 * (x + rng.normal(scale=sig, size=x.shape)) / sig**2  # noqa: E731
    ir_llr, again = noisy(), noisy()
    chase = ir_llr[:, : base.e] + again[:, : base.e]
    ok = lambda dec, llr: (dec.decode(llr)[0][:, 0] == bits).all(1).mean()  # noqa: E731
    p_ir, p_chase = ok(polar.SCLDecoder(ir), ir_llr), ok(polar.SCLDecoder(base), chase)
    assert p_ir > p_chase + 0.1, (p_ir, p_chase)
