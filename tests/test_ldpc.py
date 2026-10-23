"""data2g.ldpc: every on-air code's H, the decoder, IR extension and the CRC
gate."""

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
    dec = ldpc.Decoder(code)
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
    assert codes.payloads(spec, bits, [True], index=[0])[0][1]
    assert not codes.payloads(spec, bits, [False], index=[0])[0][1]
    soft = (1 - 2 * codes.encode_info(spec, bits).astype(int)) * 4.0
    assert codes.decode_many(spec, soft)[0] == (bytes(range(codes.payload_bytes(spec))), True)


@pytest.mark.parametrize("name", ["qpsk-r1/5", "w48-qpsk-r2/3", "w48-16qam-r3/4", "n10-qpsk-r3/4"])
def test_pinned_bits_are_the_sent_bits(name):
    """codes.pin sets each known, sent info bit to the clamp with the sign it
    went on air with (scrambled, interleaved), and touches nothing else."""
    from data2g import codes
    from data2g.config import SUBMODES

    spec = SUBMODES[name]
    rng = np.random.default_rng(1)
    payload = rng.bytes(20) + bytes(codes.payload_bytes(spec) - 20)
    known = codes.zero_tail(spec, 20)
    pos = codes.systematic_positions(spec)
    for index in (0, 3):
        sent = codes.encode(spec, payload, 0, 0x1234, index)
        soft = rng.normal(size=spec.coded_bits)
        pinned = codes.pin(spec, soft, known, index)
        at = pos[(known >= 0) & (pos >= 0)]
        assert len(at) > 0
        np.testing.assert_array_equal(pinned[at], ldpc.CH_CLAMP * (1 - 2.0 * sent[at]))
        rest = np.setdiff1d(np.arange(spec.coded_bits), at)
        np.testing.assert_array_equal(pinned[rest], soft[rest].astype(np.float32))


def test_systematic_positions():
    """LDPC: every info bit but the punctured first 2Z is sent, each once;
    polar sends none."""
    from data2g import codes, cpm
    from data2g.config import SUBMODES

    spec = SUBMODES["w48-qpsk-r2/3"]
    pos = codes.systematic_positions(spec)
    z = codes.ldpc_code(spec).z
    assert (pos[: 2 * z] == -1).all() and (pos[2 * z:] >= 0).all()
    assert len(set(pos[2 * z:])) == spec.k - 2 * z
    ctl = next(iter(cpm.CTL.values()))
    assert (codes.systematic_positions(ctl) == -1).all()
    soft = np.arange(ctl.coded_bits, dtype=float)
    np.testing.assert_array_equal(codes.pin(ctl, soft, codes.zero_tail(ctl, 4), 0), soft)


def test_pinned_buffer_bits_are_the_sent_bits():
    """codes.pin_buffer: a combined IR buffer (code order, unscrambled) with
    each known, sent info bit at the clamp, its sign the bit's; the rest as
    it was."""
    from data2g import codes
    from data2g.config import SUBMODES

    spec = SUBMODES["w48-qpsk-r2/3"]
    rng = np.random.default_rng(2)
    payload = rng.bytes(20) + bytes(codes.payload_bytes(spec) - 20)
    sent = [codes.encode(spec, payload, rv, 0, i) for i, rv in ((0, 0), (1, 1))]
    buf = codes.combine(spec, None, codes.flip(spec, 0, 0) * (1 - 2.0 * sent[0]), 0)
    buf = codes.combine(spec, buf, codes.flip(spec, 1, 1) * (1 - 2.0 * sent[1]), 1)
    known = codes.zero_tail(spec, 20)
    pinned = codes.pin_buffer(spec, buf * 0.1, known)
    info = codes.info_bits(spec, payload, 0, codes.PLAIN)  # unscrambled
    z = codes.ldpc_code(spec).z
    at = np.flatnonzero(known[2 * z:] >= 0)  # info bit 2z + j sits at buffer position j
    np.testing.assert_array_equal(pinned[0, at], ldpc.CH_CLAMP * (1 - 2.0 * info[2 * z + at]))
    rest = np.setdiff1d(np.arange(buf.shape[1]), at)
    np.testing.assert_array_equal(pinned[0, rest], buf[0, rest] * 0.1)


def test_layered_decodes_what_flooding_does_in_fewer_iterations():
    code = ldpc.qc_code(2560, 3840)
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, (64, 2560))
    x = 1.0 - 2.0 * code.encode(bits)
    sigma = 0.66
    llr = 2 * (x + rng.normal(scale=sigma, size=x.shape)) / sigma**2
    out, ok = ldpc.Decoder(code).decode(llr, iters=40)
    l_out, l_ok, conv, first = ldpc.LayeredDecoder(code).decode(llr, iters=20, record=True)
    assert ok.all() and l_ok.all()
    assert np.array_equal(l_out, bits) and np.array_equal(first, bits)
    assert conv.max() <= 20 and conv.mean() < 10  # flooding needs ~12 here
