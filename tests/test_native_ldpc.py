"""native ldpc (native/core/ldpc) against data2g/ldpc.py, side by side, on
every LDPC code on air and every incremental-redundancy extent.

Tolerances. Construction, encoding and the min-sum decoder are bitwise:
integer work, and float32 + - * in numpy's order. Sum-product, the default,
also sums each check's _phi values (in numpy's order, which depends on the
batch size) and goes through _phi's float32 tanh and log. C++ rounds each correctly
(but ~1 in 10^5); numpy's SIMD float32 tanh and log are an ULP off in about
1 result in 5, and that alone moves posteriors by whole steps of log(j)
where a check message floors (_phi then works on a few float32 ULPs). So:
- against Python with _phi made correctly rounded, everything matches:
  flags and decisions exactly, posteriors to 1e-3 + 0.1% (C++'s rare
  misroundings, drifting a little in failed decodes);
- against Python as it is, converged flags and converged decisions are
  exact, signs agree, failed decodes' decisions agree to 99.9%, and
  posteriors are not compared elementwise beyond that.
"""

import numpy as np
import pytest

from data2g import codes, config, cpm, ldpc

SPECS = [s for s in config.SUBMODES.values() if s.code == "ldpc"] + list(cpm.SPECS.values())


def _extents(spec):
    """codes.decode_buffer's extents beyond RV 0, up to the whole buffer."""
    m = ldpc.qc_code(spec.k, spec.coded_bits).mother()
    return sorted({min(m.n, (r + 1) * spec.coded_bits) for r in range(1, 4)})


def _pairs(native, reference, spec, whole=True):
    """(Python, C++) codes: the submode's, its IR extents, and with `whole`
    the whole mother code."""
    qc = reference(ldpc, "qc_code")
    p, c = qc(spec.k, spec.coded_bits), native.ldpc.qc_code(spec.k, spec.coded_bits)
    yield p, c
    ns = _extents(spec)
    if whole:
        ns = sorted({*ns, p.mother().n})
    for n in ns:
        yield p.mother(n), c.mother(n)


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_code_and_encoder(native, reference, spec):
    rng = np.random.default_rng(spec.k)
    bits = rng.integers(0, 2, (5, spec.k))
    assert native.ldpc.layout(spec.k, spec.coded_bits) == ldpc.layout(spec.k, spec.coded_bits)
    for p, c in _pairs(native, reference, spec):
        for attr in ("z", "kb", "k", "n", "mb", "n_cols"):
            assert getattr(c, attr) == getattr(p, attr), attr
        for attr in ("base", "full_base", "sent"):
            np.testing.assert_array_equal(getattr(c, attr), getattr(p, attr))
        np.testing.assert_array_equal(c.edges[0], p.edges[0])
        np.testing.assert_array_equal(c.edges[1], p.edges[1])
        full = c.encode_full(bits)
        np.testing.assert_array_equal(full, p.encode_full(bits))
        np.testing.assert_array_equal(c.encode(bits), p.encode(bits))
        assert c.syndrome_ok(full).all()
        full[:, 3] ^= 1
        np.testing.assert_array_equal(c.syndrome_ok(full), p.syndrome_ok(full))


def _llrs(code, rng, b=8):
    """BPSK at noise levels from clean to hopeless, so a batch mixes
    converged and failed decodes."""
    x = 1.0 - 2.0 * code.encode(rng.integers(0, 2, (b, code.k)))
    sigma = np.linspace(0.5, 1.4, b)[:, None]
    return (2 * (x + sigma * rng.normal(size=x.shape)) / sigma**2).astype(np.float32)


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_min_sum_bitwise(native, reference, spec):
    rng = np.random.default_rng(spec.k + 1)
    for i, (p, c) in enumerate(_pairs(native, reference, spec, whole=False)):
        llr = _llrs(p, rng)
        for alpha in [(0.8, np.linspace(0.6, 0.9, 40))[i % 2]]:  # one number, or one per iteration
            want = reference(ldpc, "MinSumDecoder")(p).decode(llr, iters=40, alpha=alpha, posterior=True)
            got = native.ldpc.MinSumDecoder(c).decode(llr, iters=40, alpha=alpha, posterior=True)
            for g, w in zip(got, want):
                assert g.dtype == w.dtype
                np.testing.assert_array_equal(g, w)


def _phi_correctly_rounded(x):
    """ldpc._phi with each float32 step correctly rounded."""
    x = np.clip(x, np.float32(1e-7), np.float32(30.0))
    t = np.tanh((x / np.float32(2)).astype(np.float64)).astype(np.float32)
    return (-np.log(t.astype(np.float64))).astype(np.float32)


def test_phi(native):
    x = np.concatenate([np.float32(1e-9) * np.arange(300, dtype=np.float32),
                        np.linspace(0, 40, 200001, dtype=np.float32), [1e4]]).astype(np.float32)
    got, want = native.ldpc.phi(x), _phi_correctly_rounded(x)
    assert got.dtype == np.float32
    assert (got != want).sum() <= len(x) // 10**5
    np.testing.assert_allclose(got, want, rtol=2e-7)
    assert np.mean(got == ldpc._phi(x)) > 0.5  # numpy's own float32 tanh/log


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_sum_product_matches_with_exact_phi(native, reference, spec, monkeypatch):
    monkeypatch.setattr(ldpc, "_phi", _phi_correctly_rounded)
    rng = np.random.default_rng(spec.k + 2)
    for p, c in _pairs(native, reference, spec, whole=False):
        llr = _llrs(p, rng)
        for rows in (llr, llr[:1]):  # B = 1: numpy sums each check pairwise
            want = reference(ldpc, "MinSumDecoder")(p).decode(rows, iters=40, posterior=True)
            got = native.ldpc.MinSumDecoder(c).decode(rows, iters=40, posterior=True)
            for g, w in zip(got, want):
                assert g.dtype == w.dtype
            np.testing.assert_array_equal(got[0], want[0])
            np.testing.assert_array_equal(got[1], want[1])
            np.testing.assert_allclose(got[2], want[2], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_sum_product(native, reference, spec):
    rng = np.random.default_rng(spec.k + 3)
    agree = total = 0
    for p, c in _pairs(native, reference, spec, whole=False):
        llr = _llrs(p, rng)
        out, ok, post = reference(ldpc, "MinSumDecoder")(p).decode(llr, iters=40, posterior=True)
        c_out, c_ok, c_post = native.ldpc.MinSumDecoder(c).decode(llr, iters=40, posterior=True)
        np.testing.assert_array_equal(c_ok, ok)
        np.testing.assert_array_equal(c_out[ok], out[ok])
        assert np.array_equal(np.sign(c_post[ok]), np.sign(post[ok]))
        agree += (c_out[~ok] == out[~ok]).sum()
        total += out[~ok].size
    assert agree >= 0.999 * total


def test_default_iterations_and_early_stop(native, reference):
    """30 iterations by default, and a batch runs until every codeword
    satisfies H: one hopeless row keeps the converged rows iterating."""
    spec = config.SUBMODES["w48-qpsk-r1/2"]
    p, c = reference(ldpc, "qc_code")(spec.k, spec.coded_bits), native.ldpc.qc_code(spec.k, spec.coded_bits)
    rng = np.random.default_rng(3)
    llr = (8 * (1.0 - 2.0 * p.encode(rng.integers(0, 2, (3, spec.k))))).astype(np.float32)
    llr[1] = rng.normal(size=spec.coded_bits)
    for kw in ({}, {"iters": 1}, {"iters": 40}):
        want = reference(ldpc, "MinSumDecoder")(p).decode(llr, posterior=True, **kw)
        got = native.ldpc.MinSumDecoder(c).decode(llr, posterior=True, **kw)
        np.testing.assert_array_equal(got[1], want[1])
        assert list(want[1]) == [True, False, True]
        np.testing.assert_array_equal(got[0][[0, 2]], want[0][[0, 2]])


def test_float64_input_and_clamp(native, reference):
    """float64 LLRs round to float32 first, then clip at CH_CLAMP."""
    spec = config.SUBMODES["qpsk-r1/5"]
    p, c = reference(ldpc, "qc_code")(spec.k, spec.coded_bits), native.ldpc.qc_code(spec.k, spec.coded_bits)
    llr = 100 * (1.0 - 2.0 * p.encode(np.zeros((1, spec.k), int)))
    llr[0, :50] = -1e-3
    want = reference(ldpc, "MinSumDecoder")(p).decode(llr, iters=40, posterior=True)
    got = native.ldpc.MinSumDecoder(c).decode(llr, iters=40, posterior=True)
    np.testing.assert_array_equal(got[1], want[1])
    np.testing.assert_array_equal(got[0], want[0])
    assert native.ldpc.CH_CLAMP == ldpc.CH_CLAMP and native.ldpc.BIG == reference(ldpc, "MinSumDecoder").BIG


def test_missing_shift_table(native):
    with pytest.raises(KeyError, match="no shift table"):
        native.ldpc.qc_code(8000, 24000)
    with pytest.raises(ValueError):
        native.ldpc.qc_code(500, 300, 2)  # n below the info bits


def test_codes_level_round_trip(native, reference):
    """codes.decode_buffer's path, mother(extent) included, on native codes."""
    spec = config.SUBMODES["w48-16qam-r2/3"]
    c = native.ldpc.qc_code(spec.k, spec.coded_bits)
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, (4, spec.k))
    extent = _extents(spec)[1]
    m = c.mother(extent)
    llr = (6 * (1.0 - 2.0 * m.encode(bits))).astype(np.float32)
    out, ok = native.ldpc.MinSumDecoder(m).decode(llr, iters=40)
    assert ok.all() and np.array_equal(out, bits)
    assert codes.buffer_len(spec) == c.mother().n
