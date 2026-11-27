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


def test_polar_payload_enters_by_reliability():
    """codes.polar_order: CRC sits on the most reliable channels, payload
    bits follow in decreasing GA reliability (so freezing any byte tail
    reduces the code to the GA-optimal shorter polar code), and encode
    and every decode agree."""
    from data2g import codes, cpm

    spec = cpm.CTL["c8r50"]
    nb = 8 * codes.payload_bytes(spec)
    order = codes.polar_order(spec)
    assert sorted(order.tolist()) == list(range(spec.k))
    # Freezing the tail past any K payload bytes leaves the exact info_pos
    # of a fresh GA-designed PolarCode of length 8*K + crc_bits.
    code = codes.polar_code(spec)
    cb = codes.crc_bits(spec)
    for k_bytes in (4, 6, 8, 12, 16):
        keep_info = set(range(8 * k_bytes)) | set(range(nb, nb + cb))
        active = sorted(code.info_pos[j] for j in range(spec.k) if int(order[j]) in keep_info)
        fresh = polar.PolarCode(8 * k_bytes + cb, spec.coded_bits,
                                design_snr_db=codes.POLAR_DESIGN_SNR_DB)
        assert active == list(fresh.info_pos)
    pay = bytes([1]) + bytes(codes.payload_bytes(spec) - 1)
    bits = codes.info_bits(spec, pay, 0, codes.PLAIN)[None]
    u = np.zeros((1, code.n), np.uint8)
    u[:, np.asarray(code.info_pos)] = bits[:, order]
    np.testing.assert_array_equal(codes.encode(spec, pay, 0, 0, codes.PLAIN),
                                  polar.transform(u)[0, code.sent][codes.interleaver(spec)])
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
