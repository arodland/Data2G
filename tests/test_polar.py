"""data2g.polar: the transform and the CA-SCL decoder."""

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


def test_polar_payload_enters_reversed():
    """codes.polar_order: the payload's last bit on the code's first info
    position, the CRC in order after it; encode and every decode agree."""
    from data2g import codes, cpm

    spec = cpm.CTL["c8r50"]
    nb = 8 * codes.payload_bytes(spec)
    order = codes.polar_order(spec)
    assert order[0] == nb - 1 and order[nb - 1] == 0 and (order[nb:] == np.arange(nb, spec.k)).all()
    pay = bytes([1]) + bytes(codes.payload_bytes(spec) - 1)
    bits = codes.info_bits(spec, pay, 0, codes.PLAIN)[None]
    u = np.zeros((1, codes.polar_code(spec).n), np.uint8)
    u[:, np.asarray(codes.polar_code(spec).info_pos)] = bits[:, order]
    np.testing.assert_array_equal(codes.encode(spec, pay, 0, 0, codes.PLAIN),
                                  polar.transform(u)[0, codes.polar_code(spec).sent][codes.interleaver(spec)])
    soft = 4.0 * (1 - 2.0 * codes.encode(spec, pay, 0, 0x55, 3))
    assert codes.decode_many(spec, soft[None], 0x55, index=[3])[0] == (pay, True)


def test_known_polar_bits_are_frozen():
    """codes.decode_raw_pinned on polar: with every payload bit known only
    the CRC is decoded, and it decodes far below the plain decode; a known
    bit given wrong fails the CRC instead of passing."""
    from data2g import codes, cpm

    spec = cpm.CTL["c16r25"]
    rng = np.random.default_rng(4)
    pb = codes.payload_bytes(spec)
    pay = rng.bytes(12) + bytes(pb - 12)
    sigma = np.sqrt(1 / (2 * 10 ** (-6 / 10)))
    soft = 2 * ((1 - 2.0 * codes.encode(spec, pay, 0, 0x1234, 1)) + sigma * rng.normal(size=spec.coded_bits)) / sigma**2
    cands, usable = codes.decode_raw(spec, soft[None], index=[1])
    assert codes.check(spec, cands[0], usable[0], 0x1234) is None
    known = np.full(spec.k, -1, np.int8)
    known[:8 * pb] = np.unpackbits(np.frombuffer(pay, np.uint8))
    cands, usable = codes.decode_raw_pinned(spec, soft[None], known, 1)
    assert codes.check(spec, cands[0], usable[0], 0x1234) == pay
    known[3] ^= 1
    cands, usable = codes.decode_raw_pinned(spec, soft[None], known, 1)
    assert codes.check(spec, cands[0], usable[0], 0x1234) is None
