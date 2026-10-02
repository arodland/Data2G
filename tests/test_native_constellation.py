"""native.constellation against data2g/constellation.py, side by side.
Skips if the module isn't built; `pytest --native` errors instead.

Tables, modulate and ace_project must match bitwise (the C++ rounds complex
products through fma as numpy's SIMD loop does on FMA hardware). LLRs go
through exp/log and sums in a different order (numpy's SIMD exp, BLAS matmul
on the general path), so they match to 1e-12 of the call's largest |LLR|:
each LLR is a difference of log-sums whose terms reach |y - h x|^2 / var,
so rounding scales with that, not with the LLR itself. Measured worst
case: 8e-14 of it (learned sets, general path, var 1e-3); Gray QPSK is
bitwise, square QAM within an ulp.
"""

import numpy as np
import pytest

from data2g import constellation as C

NAMES = [f"gray-qam{4**k}" for k in range(1, 5)] + sorted(p.stem for p in C.DIR.glob("*.npy"))


def bits_of(a):
    return np.asarray(a).view(np.uint64)


def test_names(native):
    assert native.constellation.names() == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_tables(native, reference, name):
    load, ace_dirs = reference(C, "load"), reference(C, "ace_dirs")
    pts = load(name)
    np.testing.assert_array_equal(bits_of(native.constellation.points(name)), bits_of(pts))
    np.testing.assert_array_equal(bits_of(native.constellation.ace_dirs(name)), bits_of(ace_dirs(name)))
    assert native.constellation.bits_per_symbol(name) == C.bits_per_symbol(pts)


@pytest.mark.parametrize("name", NAMES)
def test_modulate(native, reference, name):
    pts = reference(C, "load")(name)
    m = C.bits_per_symbol(pts)
    bits = np.random.default_rng(m).integers(0, 2, 1000 * m)
    np.testing.assert_array_equal(bits_of(native.constellation.modulate(bits.astype(np.uint8), name)),
                                  bits_of(reference(C, "modulate")(bits, pts)))
    with pytest.raises(ValueError):
        native.constellation.modulate(np.zeros(m + 1, np.uint8), name)
    with pytest.raises(KeyError):
        native.constellation.modulate(np.zeros(m, np.uint8), "nope")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("var_scale", [1e-3, 1.0, 30.0])
def test_llr(native, reference, name, var_scale):
    pts = reference(C, "load")(name)
    rng = np.random.default_rng(len(pts))
    shape = (40, 50)  # 2-D as modem passes it: flattened in C order
    y, h = rng.normal(size=(2, *shape)) + 1j * rng.normal(size=(2, *shape))
    h[0, :5] = 0  # dead carriers
    var = rng.uniform(0.01, 2, shape) * var_scale
    want = reference(C, "llr")(y, h, var, pts)
    got = native.constellation.llr(y, h, var, name)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * max(1.0, np.max(np.abs(want))))


@pytest.mark.parametrize("name", NAMES)
def test_ace_project(native, reference, name):
    pts, dirs = reference(C, "load")(name), reference(C, "ace_dirs")(name)
    rng = np.random.default_rng(len(pts) + 1)
    idx = rng.integers(0, len(pts), 2000)
    got = rng.normal(size=2000) + 1j * rng.normal(size=2000)
    want = 0.9 * pts[idx]
    np.testing.assert_array_equal(bits_of(native.constellation.ace_project(got, want, dirs[idx])),
                                  bits_of(reference(C, "ace_project")(got, want, dirs[idx])))
