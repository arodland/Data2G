"""data2g.tnc: C++ (native/core/tnc) against the Python reference. Skips if
the module isn't built; `pytest --native` errors instead.

The streaming Receiver must give the same events for the same audio in the
same chunks: kinds, order, submodes, positions exactly; scores and receive
results to the modem port's tolerances (they come from modem / cpm, whose
ports agree with Python to float32 ulps in the header correlation)."""

import sys
from pathlib import Path

import conftest
import numpy as np
import pytest

from data2g import codes, cpm, hfchannel, modem, tnc
from data2g.config import BANDS, FS, LEADIN_SAMPLES, SUBMODES
from data2g.waveform import sync

TOL = 1e-6
CHUNK_TOL = 1e-2


def same(a, b, path="ev", tol=TOL):
    """Deep equality: exact for ints, strings, specs and shapes, close for floats."""
    if isinstance(a, sync.Acquisition):
        a, b = vars(a), vars(b)
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, sorted(a), sorted(b))
        for k in a:
            same(a[k], b[k], f"{path}.{k}", tol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]", tol)
    elif isinstance(a, np.ndarray) or isinstance(a, float):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, path
        if a.size:
            s = max(float(np.max(np.abs(b))), 1.0)
            np.testing.assert_allclose(a, b, rtol=0, atol=tol * s, err_msg=path)
    else:
        assert a == b, (path, a, b)


def _mixed(seed=0, seconds=60):
    """Noise with bursts of every family in it: OFDM on each sync band, a
    burst whose head faded (the header-copy path), a CPM burst, and a
    stretch of digital silence."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import outcome_data as O

    from data2g.arq import phy as PHY

    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.05, seconds * FS)
    y[2 * FS:3 * FS] = 0.0  # muted input
    at = 4 * FS
    for name, n in (("qpsk-r1/2", 3), ("n10-qpsk-r1/3", 2), ("fsk8r50-r1/2", 2), ("w48-qpsk-r1/2", 3),
                    ("ack-4f", 1), ("qpsk-r1/5", 2), ("fsk16r25-r1/3", 2), ("n4-qpsk-r1/5", 1)):
        x = PHY.tx_audio(O.burst(name, n, rng))
        if name == "qpsk-r1/5":  # its preamble and header faded
            head = LEADIN_SAMPLES + BANDS["w"].preamble_samples + modem.header_samples("w")
            x = np.concatenate([np.zeros(head), x[head:]])
        x = hfchannel.freq_shift(x, float(rng.uniform(-40, 40))) * 0.3
        if at + len(x) > len(y):
            break
        y[at:at + len(x)] += x
        at += len(x) + int(rng.uniform(0.3, 2.0) * FS)
    return y


def _events(rx, y, chunk):
    return [ev for i in range(0, len(y), chunk) for ev in rx.feed(y[i:i + chunk])]


def _wrapper(native):
    """The --native Receiver wrapper (tests/conftest.py), with or without --native."""
    for p in conftest._PROVIDERS:
        if p.__name__ == "_tnc_substitutions":
            return p(native)[tnc, "Receiver"]
    raise AssertionError("no tnc provider")


@pytest.fixture(scope="module")
def mixed():
    return _mixed()


@pytest.mark.parametrize("chunk", [FS // 10, 333])
def test_receiver_events_match(native, reference, mixed, chunk):
    want = _events(reference(tnc, "Receiver")(modem.Accept.of(None, 16.0), cpm_grids=tuple(cpm.GRIDS)), mixed, chunk)
    got = _events(_wrapper(native)(modem.Accept.of(None, 16.0), cpm_grids=tuple(cpm.GRIDS)), mixed, chunk)
    kinds = [(k, ev["spec"].name if k == "header" else ev["header"]["spec"].name) for k, ev in want]
    assert len(kinds) >= 10 and {"header", "burst"} <= {k for k, _ in kinds}, kinds
    assert any(k == "burst" and "copy" in ev["header"] for k, ev in want)  # the copy path ran
    assert any(k == "burst" and ev["header"].get("family") == "cpm" for k, ev in want)
    same(got, want)


def test_receiver_chunking_independent(native, reference, mixed):
    """Chunks that divide HOP and the blanker's block land the searches at
    the same samples: the Python receiver's decisions are then the same,
    and the C++ one's must be too. Floats only to CHUNK_TOL: the detectors'
    FFT lengths follow the chunks, and a copy lock's CFO comes from their
    matched filter outputs (0.0035 Hz apart, its header score 3.5e-6, in
    both receivers)."""
    acc = modem.Accept.of(None, 16.0)

    def drop(d):
        # CPM positions in buffer indices follow the chunking: a lock's
        # header_end (found at), rx's preamble_start and header_end
        # (received at; an OFDM rx is in its audio segment's), and a header's
        # stream_end (how much had been fed when it was found)
        cpm_ = d is not None and d.get("family") == "cpm"
        return d and {k: v for k, v in d.items()
                      if k != "stream_end" and not (cpm_ and k in ("header_end", "preamble_start"))}

    def run(R, chunk):
        return [(k, drop(ev)) if k == "header" else (k, dict(ev, header=drop(ev["header"]), rx=drop(ev["rx"])))
                for k, ev in _events(R(acc, cpm_grids=tuple(cpm.GRIDS)), mixed, chunk)]

    R, P = _wrapper(native), reference(tnc, "Receiver")
    base = run(R, 400)
    same(run(P, 2000), run(P, 400), tol=CHUNK_TOL)
    same(run(R, 2000), base, tol=CHUNK_TOL)
    same(run(R, 80), base, tol=CHUNK_TOL)


def test_receiver_busy_matches(native, reference):
    """BUSY (pilots, in-band energy) and the hold, sampled every 20 ms."""
    rng = np.random.default_rng(5)
    spec = SUBMODES["qpsk-r1/2"]
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec)) for _ in range(6)], spec)
    y = np.concatenate([np.zeros(8 * FS), x, np.zeros(6 * FS)])
    y = hfchannel.awgn(y, 6.0, seed=5, s_power=1.0)
    acc = modem.Accept.of(None, 16.0)
    a, b = reference(tnc, "Receiver")(acc), _wrapper(native)(acc)
    for i in range(0, len(y), FS // 50):
        same(b.feed(y[i:i + FS // 50]), a.feed(y[i:i + FS // 50]))
        assert (b.busy, b.channel_busy, b.on_air) == (a.busy, a.channel_busy, a.on_air), i
    b.reset()
    assert not b.busy and b.feed(np.zeros(FS)) == []


def test_receive_any_matches(native, reference):
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import outcome_data as O

    from data2g.arq import phy as PHY

    py_any = reference(tnc, "receive_any")
    for name, n, seed in (("qpsk-r1/3", 2, 1), ("fsk32r62-r1/2", 2, 2), ("w48-qpsk-r1/2", 2, 3)):
        rng = np.random.default_rng(seed)
        x = PHY.tx_audio(O.burst(name, n, rng))
        y = hfchannel.awgn(np.concatenate([np.zeros(FS // 2), x, np.zeros(FS // 2)]), 8, seed=seed,
                           s_power=hfchannel.active_power(x))
        want = py_any(y, lead=FS)
        got = native.tnc.receive_any(y, FS, None)
        assert want is not None and got is not None
        if want.get("family") == "cpm":
            got = dict(got, spec=cpm.SPECS[got["spec"]])
        else:
            s, f, m, alts = got["acq"]
            got = dict(got, spec=SUBMODES[got["spec"]],
                       acq=sync.Acquisition(preamble_start=s, freq_offset=f, metric=m, alternatives=alts))
        same(got, want)
    assert native.tnc.receive_any(np.random.default_rng(9).normal(size=4 * FS), 0, None) is None
    assert py_any(np.random.default_rng(9).normal(size=4 * FS)) is None


def test_framing_matches(native, reference):
    T = native.tnc
    rng = np.random.default_rng(0)
    frames = [bytes(rng.integers(0, 256, int(n), dtype=np.uint8)) for n in rng.integers(0, 300, 20)]
    frames += [bytes([0xC0, 0xDB, 0xDC, 0xDD] * 5)]
    wire = b"".join(T.kiss_encode(f, p % 16) for p, f in enumerate(frames))
    assert wire == b"".join(reference(tnc, "kiss_encode")(f, p % 16) for p, f in enumerate(frames))
    d, pd = T.KissDecoder(), reference(tnc, "KissDecoder")()
    for i in range(0, len(wire), 7):
        assert d.feed(wire[i:i + 7]) == pd.feed(wire[i:i + 7])
    for name in ("qpsk-r1/5", "n10-qpsk-r1/2", "w48-qpsk-r1/2"):
        spec = SUBMODES[name]
        assert T.capacity(name, 3) == reference(tnc, "capacity")(spec, 3)
        packets = [bytes(rng.integers(0, 256, int(n), dtype=np.uint8)) for n in rng.integers(1, 90, 5)]
        payloads = T.pack(packets, name)
        assert payloads == reference(tnc, "pack")(packets, spec)
        for _ in range(10):
            ok = list(rng.random(len(payloads)) > 0.3)
            assert T.unpack(payloads, ok) == reference(tnc, "unpack")(payloads, ok)
        with pytest.raises(ValueError):
            T.pack([bytes(T.capacity(name, 64))], name)
    for bands, grids in ((["w"], []), (["w", "n10", "w48"], list(cpm.GRIDS)), ([], ["c8r50"])):
        assert T.search_span(bands, grids) == reference(tnc, "search_span")(bands, grids)
