"""native/core/polar against data2g/polar.py over every polar submode and
the CPM control codeword: codes and encodings bitwise, SCL decisions, path
order and float32 metrics exactly."""

import numpy as np
import pytest

from data2g import codes, config, cpm, polar

SPECS = [s for s in config.SUBMODES.values() if s.code == "polar"] + [next(iter(cpm.CTL.values()))]


def _pair(native, reference, spec):
    py = reference(codes, "polar_code")(spec)
    if spec.name in config.SUBMODES:
        nat = native.polar.polar_code(spec.name)
    else:
        nat = native.polar.PolarCode(spec.k, spec.coded_bits)
    return py, nat


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_code_and_encode(native, reference, spec):
    py, nat = _pair(native, reference, spec)
    assert (nat.k, nat.e, nat.n) == (py.k, py.e, py.n)
    for attr in ("info_pos", "sent", "punctured"):
        np.testing.assert_array_equal(getattr(nat, attr), getattr(py, attr))
    bits = np.random.default_rng(spec.coded_bits).integers(0, 2, (50, spec.k))
    np.testing.assert_array_equal(nat.encode(bits), py.encode(bits))
    np.testing.assert_array_equal(nat.encode(bits[0]), py.encode(bits[0]))


def test_transform(native):
    u = np.random.default_rng(0).integers(0, 2, (7, 256)).astype(np.uint8)
    np.testing.assert_array_equal(native.polar.transform(u), polar.transform(u))


def _llrs(spec, py, rng, b):
    """Noisy LLRs from easy to hopeless, plus exact-tie cases (quantised,
    all zero) that exercise the stable sort orders."""
    x = 1.0 - 2.0 * py.encode(rng.integers(0, 2, (b, spec.k)))
    sig = np.geomspace(0.5, 3.0, b)[:, None]
    llr = 2 * (x + rng.normal(size=x.shape) * sig) / sig**2
    yield "noisy", llr.astype(np.float32)
    yield "float64", llr  # decode_buffer passes a float64 buffer
    yield "quantised", np.round(llr / 2).astype(np.float32)
    yield "zeros", np.zeros((3, spec.coded_bits), np.float32)
    yield "huge", np.clip(llr * 1e30, -3e38, 3e38).astype(np.float32)


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_scl_decode(native, reference, spec):
    py, nat = _pair(native, reference, spec)
    py_dec, nat_dec = polar.SCLDecoder(py, codes.POLAR_LIST), native.polar.SCLDecoder(nat, codes.POLAR_LIST)
    rng = np.random.default_rng(spec.k * 1000 + spec.coded_bits)
    for what, llr in _llrs(spec, py, rng, 48):
        pu, ppm = py_dec.decode(llr)
        nu, npm = nat_dec.decode(llr)
        assert nu.dtype == np.uint8 and npm.dtype == np.float32, what
        np.testing.assert_array_equal(nu, pu, err_msg=what)
        np.testing.assert_array_equal(npm, ppm, err_msg=what)  # bitwise, inf included
