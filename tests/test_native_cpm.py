"""C++ data2g::cpm against data2g/cpm.py. Skips if the module isn't built;
`pytest --native` errors instead (conftest.py)."""

import numpy as np
import pytest

from data2g import codes, cpm
from data2g.config import FS
from data2g.modem import _crc6

GRIDS = list(cpm.GRIDS.values())
ALL_SPECS = list(cpm.SPECS.values()) + list(cpm.CTL.values())
RTOL = 1e-12  # anything through exp, FFT or a reduction


def ref(reference, name):
    return reference(cpm, name)


def test_header_tables_are_the_rng_draws(native, reference):
    """Every frozen header row is numpy PCG64's draw for that word."""
    header_symbols = ref(reference, "header_symbols")
    for g in GRIDS:
        for v in range(1024):
            np.testing.assert_array_equal(native.cpm.header_symbols(g.name, v),
                                          header_symbols(g.name, (v << 6) | _crc6(v)), err_msg=f"{g.name} {v}")
        for s in cpm.grid_specs(g.name):
            for d in (False, True):
                for n in range(cpm.MAX_DATA + 1):
                    assert native.cpm.header_value(s.index, n, d) << 6 | _crc6(native.cpm.header_value(s.index, n, d)) \
                        == cpm.header_word(s.index, n, d)


def test_grid_and_spec_tables(native):
    rows = native.cpm.grids()
    assert [r["name"] for r in rows] == list(cpm.GRIDS)
    for r, g in zip(rows, GRIDS):
        assert (r["m"], r["rate"], r["center"], r["bp"], r["clip_db"], r["T"], r["bits"], r["f0"]) == \
            (g.m, g.rate, g.center, g.bp, g.clip_db, g.T, g.bits, g.f0)
        assert (r["sync_threshold"], r["header_threshold"]) == (cpm.SYNC_THRESHOLD[g.name], cpm.HEADER_THRESHOLD[g.name])
        assert (r["hdr_len"], r["costas_len"]) == (cpm.hdr_len(g), len(cpm.costas(g.m)))
        np.testing.assert_array_equal(r["preamble"], cpm.preamble_pattern(g))
        np.testing.assert_array_equal(r["mid_block"], cpm.mid_block(g))
    assert native.cpm.specs() == [dict(name=s.name, grid=s.grid, code=s.code, index=s.index, k=s.k,
                                       coded_bits=s.coded_bits, n_sym=s.n_sym) for s in ALL_SPECS]


def test_layout_and_timing(native, reference):
    layout, stream_symbols, burst_seconds = (ref(reference, n) for n in ("layout", "stream_symbols", "burst_seconds"))
    for g in GRIDS:
        for n_sym in [0, 1, 7, *range(50, 3000, 97)]:
            a, b = native.cpm.layout(g.name, n_sym), layout(g.name, n_sym)
            assert (a["n"], a["front"]) == (b.n, b.front)
            for k in ("sync_rows", "sync_tones", "data_rows"):
                np.testing.assert_array_equal(a[k], getattr(b, k))
            assert len(a["hdr_rows"]) == len(b.hdr_rows)
            for x, y in zip(a["hdr_rows"], b.hdr_rows):
                np.testing.assert_array_equal(x, y)
        for n_data in range(cpm.MAX_DATA + 1):
            for dup in (False, True):
                assert native.cpm.stream_symbols(g.name, n_data, dup) == stream_symbols(g.name, n_data, dup)
    for s in cpm.SPECS.values():
        for n_cw in range(0, cpm.MAX_DATA + 3):
            for dup in (False, True):
                assert native.cpm.burst_seconds(s.name, n_cw, dup) == burst_seconds(s, n_cw, dup)


def _burst(spec, n_data, dup, rng):
    g = cpm.GRIDS[spec.grid]
    ctl = cpm.CTL[g.name]
    coded = [rng.integers(0, 2, ctl.coded_bits).astype(np.uint8)]
    if dup:
        coded.append(coded[0])
    coded += [rng.integers(0, 2, spec.coded_bits).astype(np.uint8) for _ in range(n_data)]
    return coded


def _channel(x, rng, lead=2000, cfo=0.0, sigma=0.3):
    y = np.concatenate([np.zeros(lead), x, np.zeros(lead)])
    y = y * np.cos(2 * np.pi * cfo * np.arange(len(y)) / FS) * np.sqrt(2) if cfo else y
    return y + rng.normal(0, sigma, len(y))


def test_modulate(native, reference):
    rng = np.random.default_rng(3)
    for spec in cpm.SPECS.values():
        g = cpm.GRIDS[spec.grid]
        for n_data, dup in ((0, False), (2, True), (3, False)):
            coded = _burst(spec, n_data, dup, rng)
            stream = np.concatenate([ref(reference, "to_tones")(g, c) for c in coded])
            np.testing.assert_array_equal(np.concatenate([native.cpm.to_tones(g.name, c) for c in coded]), stream)
            sym = rng.integers(0, g.m, 50)
            np.testing.assert_allclose(native.cpm.tones(g.name, sym), ref(reference, "tones")(g, sym), rtol=0,
                                       atol=1e-12)
            want = ref(reference, "modulate")(spec, coded, dup)
            got = cpm.bandpass(g, native.cpm.modulate(spec.name, coded, dup))
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


# (spec, n_data, dup, cfo Hz, noise sigma)
CASES = [("fsk16r25-r1/2", 1, False, 0.0, 0.3), ("fsk8r50-r1/3", 2, True, 17.0, 0.5),
         ("fsk32r62-r1/2", 1, True, -40.0, 0.4), ("fsk8r50-r1/2", 0, False, 3.0, 1.0)]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_receive_chain(native, reference, case):
    name, n_data, dup, cfo, sigma = case
    spec = cpm.SPECS[name]
    g = cpm.GRIDS[spec.grid]
    rng = np.random.default_rng(CASES.index(case))
    coded = _burst(spec, n_data, dup, rng)
    y = _channel(ref(reference, "modulate")(spec, coded, dup), rng, lead=1777, cfo=cfo, sigma=sigma)

    E_ref = ref(reference, "_energies")(g, y, 1700, 60, 5.0, 2)
    np.testing.assert_allclose(native.cpm.energies(g.name, y, 1700, 60, 5.0, 2), E_ref, rtol=RTOL, atol=1e-12)
    np.testing.assert_allclose(native.cpm.energies(g.name, y, -50, 10, 0.0), ref(reference, "_energies")(g, y, -50, 10, 0.0),
                               rtol=RTOL, atol=1e-12)
    np.testing.assert_allclose(native.cpm.shares(E_ref), ref(reference, "_shares")(E_ref), rtol=1e-15, atol=0)

    detect = ref(reference, "detect")
    for kw in (dict(), dict(front_only=True), dict(fine=False, n_sym=200), dict(reach_hz=60.0, floor=0.99)):
        a, b = native.cpm.detect(g.name, y, **kw), detect(g, y, **kw)
        assert a[1:] == b[1:], kw
        assert a[0] == pytest.approx(b[0], rel=RTOL)
    score, s0, f = b

    for copies in (1, 2):
        a, b = native.cpm.read_header(g.name, y, s0, f, copies), ref(reference, "read_header")(g, y, s0, f, copies)
        assert a[:3] == (b[0].name, *b[1:3])
        assert a[3:] == pytest.approx(b[3:], rel=RTOL)
    assert native.cpm.peak_ratio(g.name, y, s0, f) == pytest.approx(ref(reference, "_peak_ratio")(g, y, s0, f), rel=RTOL)

    for kw in (dict(), dict(threshold=0.05), dict(front_only=False), dict(lo=1000, hi=3000), dict(lo=5000)):
        a, b = native.cpm.find(g.name, y, **kw), ref(reference, "find")(g, y, **kw)
        assert (a is None) == (b is None), kw
        if b is not None:
            assert a["spec"] == b["spec"].name
            for k in ("n_data", "dup", "start", "end", "header_end", "cfo"):
                assert a[k] == b[k], (kw, k)
            for k in ("score", "header_score", "header_margin"):
                assert a[k] == pytest.approx(b[k], rel=RTOL, abs=1e-12), (kw, k)

    slots, E = native.cpm.soft(g.name, y, s0, f, n_data, dup)
    slots_ref, E_ref = ref(reference, "soft")(g, spec, y, s0, f, n_data, dup)
    np.testing.assert_allclose(E, E_ref, rtol=RTOL, atol=1e-12)
    assert len(slots) == len(slots_ref)
    for a, b in zip(slots, slots_ref):
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=1e-9)
        np.testing.assert_allclose(native.cpm.llrs(g.name, E_ref[:len(a) // g.bits]),
                                   ref(reference, "llrs")(g, E_ref[:len(a) // g.bits]), rtol=RTOL, atol=1e-9)

    m, m_ref = native.cpm.measure(g.name, E_ref, len(E_ref)), ref(reference, "measure")(g, E_ref, len(E_ref))
    for k in ("snr_est", "spread_est", "frames"):
        assert m[k] == pytest.approx(m_ref[k], rel=1e-9, abs=1e-12), k


def test_llrs_zero_padded(native, reference):
    """A codeword past the audio's end (all-zero energies) reads as erasures in both."""
    g = cpm.GRIDS["c8r50"]
    E = np.zeros((40, g.m))
    np.testing.assert_array_equal(native.cpm.llrs(g.name, E), ref(reference, "llrs")(g, E))


def test_codes_roundtrip_through_native_modulate(native):
    """A real control + data burst (data2g.codes), native modulation, Python's receiver."""
    rng = np.random.default_rng(5)
    spec = cpm.SPECS["fsk8r50-r1/2"]
    g = cpm.GRIDS[spec.grid]
    ctl = codes.encode(cpm.CTL[g.name], bytes(codes.payload_bytes(cpm.CTL[g.name])), 0, 0)
    data = codes.encode(spec, bytes(range(codes.payload_bytes(spec))), 0, 0, 1)
    y = _channel(cpm.bandpass(g, native.cpm.modulate(spec.name, [ctl, data], False)), rng, sigma=0.1)
    lock = cpm.find(g, y)
    assert lock is not None and lock["n_data"] == 1 and lock["spec"] is spec
