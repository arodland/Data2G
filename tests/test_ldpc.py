import numpy as np
import pytest

from data2g import ldpc


def _ldpc_specs():
    from data2g import cpm
    from data2g.config import SUBMODES

    return [s for s in SUBMODES.values() if s.code == "ldpc"] + list(cpm.SPECS.values())


def test_codewords_satisfy_h():
    """Every LDPC code on air (OFDM and CPM): codewords satisfy H, and the
    systematic part that is sent really is the info bits."""
    from data2g import codes

    for s in _ldpc_specs():
        code = codes.ldpc_code(s)
        rng = np.random.default_rng(s.k)
        bits = rng.integers(0, 2, (4, s.k))
        cw = code.encode(bits)
        assert cw.shape == (4, s.coded_bits)
        assert code.syndrome_ok(code.encode_full(bits)).all(), s.name
        sent_info = code.sent[code.sent < s.k]
        assert np.array_equal(cw[:, : len(sent_info)], bits[:, sent_info]), s.name


def test_shift_tables_cover_every_ldpc_submode():
    """Every LDPC code on air has a table on its graph's mask, without
    4-cycles, whose mother code's codewords satisfy H."""
    from data2g import codes

    rng = np.random.default_rng(2)
    for s in _ldpc_specs():
        code = codes.ldpc_code(s)
        bg = 1 if code.kb == ldpc.KB[1] else 2
        assert np.array_equal(code.full_base >= 0, ldpc.mask(bg)), s.name
        b, z = code.full_base, code.z
        for r1 in range(b.shape[0]):
            for r2 in range(r1 + 1, b.shape[0]):
                cols = np.flatnonzero((b[r1] >= 0) & (b[r2] >= 0))
                d = (b[r1, cols] - b[r2, cols]) % z
                assert len(set(d)) == len(d), f"{s.name}: 4-cycle in rows {r1},{r2}"
        m = code.mother()
        bits = rng.integers(0, 2, (2, s.k))
        assert m.syndrome_ok(m.encode_full(bits)).all(), s.name


def test_a_lifting_size_without_a_table_is_an_error():
    with pytest.raises(KeyError, match="no shift table"):
        ldpc.qc_code(8000, 24000)


def test_min_sum_decodes_noise_free_and_corrects_errors():
    code = ldpc.qc_code(500, 1000)
    dec = ldpc.MinSumDecoder(code)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, (8, 500))
    x = 1.0 - 2.0 * code.encode(bits)
    y = x + rng.normal(scale=0.7, size=x.shape)  # BPSK, Es/N0 ~3 dB at rate 1/2
    out, ok = dec.decode(2 * y / 0.49)
    assert ok.all()
    assert np.array_equal(out, bits)


def test_mother_code_starts_with_the_codeword():
    from data2g import codes
    from data2g.config import SUBMODES

    rng = np.random.default_rng(1)
    for name in ("w48-qpsk-r1/2", "qpsk-r1/5", "w48-16qam-r2/3", "n10-qpsk-r1/3"):
        s = SUBMODES[name]
        c = codes.ldpc_code(s)
        b = rng.integers(0, 2, (2, s.k))
        assert (c.mother().encode(b)[:, : c.n] == c.encode(b)).all()
        assert (codes.encode_info(s, b, 0) == codes.encode_info(s, b)).all()


def test_incremental_redundancy_decodes_what_one_rv_cannot():
    """BPSK soft bits at Es/N0 -5 dB: a rate-1/2 codeword alone fails, the
    same codeword's RV 0 + RV 1 (rate ~1/4) combined decodes."""
    from data2g import codes
    from data2g.config import SUBMODES

    s = SUBMODES["w48-qpsk-r1/2"]
    rng = np.random.default_rng(2)
    payloads = [bytes(rng.integers(0, 256, codes.payload_bytes(s), dtype=np.uint8)) for _ in range(8)]
    sigma = 10 ** (2 / 20)  # Es/N0 = 1 / (2 sigma^2)

    def soft(rv):
        x = 1.0 - 2.0 * np.stack([codes.encode(s, p, rv, index=i) for i, p in enumerate(payloads)])
        return 2 * (x + sigma * rng.normal(size=x.shape)) / sigma**2

    s0 = soft(0)
    alone = codes.decode_buffer(s, codes.combine(s, None, s0, 0))
    assert sum(ok for _, ok in alone) <= 1
    both = codes.decode_buffer(s, codes.combine(s, codes.combine(s, None, s0, 0), soft(1), 1), max_rv=1)
    assert all(ok and p == q for (p, ok), q in zip(both, payloads))


def test_a_crc_match_needs_a_converged_decode():
    """A failed LDPC decode's guess can carry a matching CRC16 (1 in 65536):
    it is still a failure."""
    import numpy as np

    from data2g import codes
    from data2g.config import SUBMODES

    spec = SUBMODES["qpsk-r1/5"]
    bits = codes.info_bits(spec, bytes(range(codes.payload_bytes(spec))))[None]
    assert codes._payloads(spec, bits, [True], index=[0])[0][1]
    assert not codes._payloads(spec, bits, [False], index=[0])[0][1]
    soft = (1 - 2 * codes.encode_info(spec, bits).astype(int)) * 4.0
    assert codes.decode_many(spec, soft)[0] == (bytes(range(codes.payload_bytes(spec))), True)
