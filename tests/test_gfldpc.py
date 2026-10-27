"""GF(M) LDPC (data2g.gfldpc) and the CPM data codec through data2g.codes."""

import numpy as np
import pytest

from data2g import codes, cpm, gfldpc

SPECS = list(cpm.SPECS.values())


def energies(g, tones, rng, esn0_db=30.0):
    """Tone energies of `tones` (AWGN, N0 = 1)."""
    es = 10 ** (esn0_db / 10)
    y = (rng.normal(size=(len(tones), g.m)) + 1j * rng.normal(size=(len(tones), g.m))) / np.sqrt(2)
    y[np.arange(len(tones)), tones] += np.sqrt(es)
    return np.abs(y) ** 2


def test_field():
    for p in (3, 4, 5):
        gf = gfldpc.GF(p)
        a = np.arange(1, gf.q)
        assert (gf.mul[a, gf.inv[a]] == 1).all()
        assert sorted(gf.mul[3, a]) == list(a)  # a permutation of the non-zero elements


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_tables(spec):
    c = codes.gf_code(spec)
    p = spec.bits_per_cu
    assert c.k == -(-spec.k // p) and c.core_n == codes.n_sym(spec)
    assert c.n * p == 6 * spec.k  # mother rate 1/6, as the binary LDPC's
    u = np.random.default_rng(1).integers(0, c.q, (3, c.k))
    assert c.syndrome_ok(c.encode(u)).all()


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_codec(spec):
    """A payload through tones and symbol_ll decodes; scrambling, plain and IR combine."""
    g = cpm.GRIDS[spec.grid]
    rng = np.random.default_rng(2)
    payload = bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8))
    soft = cpm.symbol_ll(g, energies(g, cpm.to_tones(g, codes.encode(spec, payload, index=3)), rng))
    assert soft.shape == (codes.soft_len(spec),)
    assert codes.decode_many(spec, soft[None], index=[3])[0] == (payload, True)
    assert codes.check(spec, *(x[0] for x in codes.decode_raw(spec, soft[None], index=[3])), 0) == payload
    # plain: the same codeword unscrambled, decoded at PLAIN
    assert codes.decode_many(spec, codes.plain(spec, soft, 3)[None], index=[codes.PLAIN])[0] == (payload, True)
    # RV 0 at burst position 3 and RV 1 at 5, combined in the plain buffer
    buf = None
    for rv, idx in ((0, 3), (1, 5)):
        t = cpm.to_tones(g, codes.encode(spec, payload, rv=rv, index=idx))
        buf = codes.combine(spec, buf, codes.plain(spec, cpm.symbol_ll(g, energies(g, t, rng)), idx, rv)[None], rv)
    assert codes.decode_buffer(spec, buf, 1, index=codes.PLAIN)[0] == (payload, True)


def test_ir_gains():
    """Two receptions at a weak SNR: RV 0 + RV 1 (IR) decode where RV 0 alone does not."""
    spec = cpm.SPECS["fsk32r62-r1/3"]
    g = cpm.GRIDS[spec.grid]
    rng = np.random.default_rng(3)
    alone = both = 0
    for _ in range(20):
        payload = bytes(rng.integers(0, 256, codes.payload_bytes(spec), dtype=np.uint8))
        bufs = []
        for rv in (0, 1):
            t = cpm.to_tones(g, codes.encode(spec, payload, rv=rv, index=codes.PLAIN))
            bufs.append(cpm.symbol_ll(g, energies(g, t, rng, 3.5)))
        alone += codes.decode_many(spec, bufs[0][None], index=[codes.PLAIN])[0] == (payload, True)
        buf = codes.combine(spec, codes.combine(spec, None, bufs[0][None], 0), bufs[1][None], 1)
        both += codes.decode_buffer(spec, buf, 1, index=codes.PLAIN)[0] == (payload, True)
    assert alone <= 10 and both == 20
