"""native codes' codec (encode, spread, combine, decode) against
data2g/codes.py, on every submode, CPM mode and CPM control codeword.

Integer work (encode, interleave, CRCs, scrambling) is bitwise, and so is
combine (float64 adds in np.add.at's order). Decodes go through each side's
decoder: with --native both use the C++ ones and everything is exact; without
it Python uses numpy's, which round sum-product a little differently
(tests/test_native_ldpc.py), so ok flags and payloads are compared where
the decode converged.
"""

import numpy as np
import pytest

from data2g import codes, config, cpm

SPECS = [*config.SUBMODES.values(), *cpm.SPECS.values(), *cpm.CTL.values()]
IDS = [s.name for s in SPECS]
CODEC = ("crc_bits", "payload_bytes", "rv_cycle", "buffer_len", "rv_positions", "info_bits", "encode", "encode_info",
         "flip", "spread", "despread", "combine", "decode_buffer", "_payloads", "decode_many", "decode_llrs",
         "decode_raw", "descramble", "check", "crc_ok", "interleaver")


@pytest.fixture
def py(monkeypatch, reference):
    """data2g.codes with its own codec functions, even under --native."""
    for attr in CODEC:
        monkeypatch.setattr(codes, attr, reference(codes, attr))
    return codes


def _payloads(spec, rng, n):
    return [rng.bytes(codes.payload_bytes(spec)) for _ in range(n)]


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_tables_and_encode(native, py, spec):
    n = native.codes
    rng = np.random.default_rng(spec.coded_bits + spec.k)
    for attr in ("crc_bits", "payload_bytes", "rv_cycle", "buffer_len"):
        assert getattr(n, attr)(spec.name) == getattr(py, attr)(spec), attr
    np.testing.assert_array_equal(n.interleaver(spec.name), py.interleaver(spec))
    for rv in range(5):
        np.testing.assert_array_equal(n.rv_positions(spec.name, rv), py.rv_positions(spec, rv))
    for (p,), rv, mask, index in zip([_payloads(spec, rng, 1) for _ in range(4)], (0, 1, 2, 3), (0, 1, 0xDEADBEEF, 7),
                                     (0, 5, codes.PLAIN, 63)):
        np.testing.assert_array_equal(n.info_bits(spec.name, p, mask, index), py.info_bits(spec, p, mask, index))
        np.testing.assert_array_equal(n.encode(spec.name, p, rv, mask, index), py.encode(spec, p, rv, mask, index))
        np.testing.assert_array_equal(n.flip(spec.name, index, rv), py.flip(spec, index, rv))
    bits = rng.integers(0, 2, (3, spec.k), dtype=np.uint8)
    for rv in (0, 1, 4):
        np.testing.assert_array_equal(n.encode_info(spec.name, bits, rv), py.encode_info(spec, bits, rv))
    with pytest.raises(ValueError):
        n.encode(spec.name, b"x" * (codes.payload_bytes(spec) + 1))


def test_spread(native, py):
    rng = np.random.default_rng(0)
    for dtype in (np.uint8, np.float64):
        for n_cw, N, m in ((1, 12, 2), (3, 24, 4), (5, 18, 6)):
            x = rng.integers(0, 2, (2, n_cw, N)).astype(dtype)
            s = native.codes.spread(x.reshape(2, -1), n_cw, m)
            np.testing.assert_array_equal(s, py.spread(x, m).reshape(2, -1))
            assert s.dtype == dtype
            np.testing.assert_array_equal(native.codes.despread(s, n_cw, m), py.despread(s, n_cw, m).reshape(2, -1))


def _noisy(spec, coded, rng, sigmas):
    """BPSK-on-AWGN LLRs per row at noise sigmas[row], mapping order."""
    s = np.asarray(sigmas)[:, None]
    y = (1 - 2.0 * coded) + s * rng.normal(size=coded.shape)
    return 2 * y / s**2


def _same(got, want):
    assert [ok for _, ok in got] == [ok for _, ok in want]
    assert [p for p, ok in got if ok] == [p for p, ok in want if ok]


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_decode(native, py, spec):
    n = native.codes
    rng = np.random.default_rng(spec.k)
    pls = _payloads(spec, rng, 6)
    masks = np.array([0, 1, 2, 0xFFFFFFFF, 12345, 0])
    index = np.array([0, 1, 2, 3, codes.PLAIN, 9])
    coded = np.stack([py.encode(spec, p, 0, int(m), int(i)) for p, m, i in zip(pls, masks, index)])
    sigmas = np.linspace(0.5, 1.4, 6)
    soft = _noisy(spec, coded, rng, sigmas)

    _same(n.decode_many(spec.name, soft, masks, index), py.decode_many(spec, soft, masks, index))
    _same(n.decode_many(spec.name, soft), py.decode_many(spec, soft))
    bits, ok = n.decode_llrs(spec.name, soft, 40, masks, index)
    pbits, pok = py.decode_llrs(spec, soft, crc_mask=masks, index=index)
    np.testing.assert_array_equal(ok, pok)
    np.testing.assert_array_equal(bits[ok], pbits[pok])
    np.testing.assert_array_equal(n.crc_ok(spec.name, pbits, masks, index), py.crc_ok(spec, pbits, masks, index))
    _same(n.payloads(spec.name, pbits, pok.astype(np.uint8), masks, index), py._payloads(spec, pbits, pok, masks, index))
    np.testing.assert_array_equal(n.descramble(spec.name, pbits, 5), py.descramble(spec, pbits, 5))

    cands, usable = n.decode_raw(spec.name, soft, index)
    pc, pu = py.decode_raw(spec, soft, index)
    assert cands.shape == pc.shape
    np.testing.assert_array_equal(usable, pu)
    for b, m in enumerate(masks):
        want = py.check(spec, pc[b], pu[b], int(m))
        assert n.check(spec.name, pc[b], pu[b], int(m)) == want
        if want is not None:
            assert n.check(spec.name, cands[b], usable[b], int(m)) == want

    # HARQ: a weak RV 0 plus an RV 1 resend, scrambled per slot and flipped back
    re = np.stack([py.encode(spec, p, 1, int(m), int(i)) for p, m, i in zip(pls, masks, index)])
    soft1 = _noisy(spec, re, rng, sigmas)
    fl0 = np.stack([py.flip(spec, int(i), 0) for i in index])
    fl1 = np.stack([py.flip(spec, int(i), 1) for i in index])
    buf = n.combine(spec.name, None, fl0 * soft, 0)
    pbuf = py.combine(spec, None, fl0 * soft, 0)
    np.testing.assert_array_equal(buf, pbuf)
    same = n.combine(spec.name, buf, fl1 * soft1, [1] * 6)
    assert same is buf  # in place, as np.add.at
    np.testing.assert_array_equal(buf, py.combine(spec, pbuf, fl1 * soft1, [1] * 6))
    for top in (0, 1, 3):
        _same(n.decode_buffer(spec.name, buf, top, masks, codes.PLAIN),
              py.decode_buffer(spec, pbuf, top, masks, index=codes.PLAIN))
