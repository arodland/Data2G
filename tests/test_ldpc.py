import numpy as np
import pytest

from data2g import ldpc


@pytest.mark.parametrize("k,n", [(48, 240), (120, 480), (500, 1000), (1320, 2880), (4000, 4800), (8000, 24000)])
def test_nr_codewords_satisfy_h(k, n):
    code = ldpc.nr_code(k, n)
    rng = np.random.default_rng(k)
    bits = rng.integers(0, 2, (4, k))
    cw = code.encode(bits)
    assert cw.shape == (4, n)
    assert code.syndrome_ok(code.encode_full(bits)).all()
    # the systematic part that is sent really is the info bits
    sent_info = code.sent[code.sent < k]
    assert np.array_equal(cw[:, : len(sent_info)], bits[:, sent_info])


def test_min_sum_decodes_noise_free_and_corrects_errors():
    torch = pytest.importorskip("torch")
    code = ldpc.nr_code(500, 1000)
    dec = ldpc.MinSumDecoder(code)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, (8, 500))
    x = 1.0 - 2.0 * code.encode(bits)
    y = x + rng.normal(scale=0.7, size=x.shape)  # BPSK, Es/N0 ~3 dB at rate 1/2
    out, ok = dec.decode(torch.tensor(2 * y / 0.49, dtype=torch.float32))
    assert ok.all()
    assert np.array_equal(out.numpy(), bits)


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
