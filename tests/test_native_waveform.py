"""data2g.waveform: C++ (native/core/waveform) against the Python reference,
every band. Skips if the module isn't built; `pytest --native` errors instead.

Tolerances. Phasor tables are exact turns through the same sin/cos: equal
here, held to 1e-15 so another libm passes. Anything through BLAS, an FFT
or numpy's SIMD complex multiply sums in its own order: 1e-13 of its scale
(measured worst 1e-14). Decisions (found or not, sample positions, the
chosen CFO bin, the alternatives' order) must match exactly.
"""

import numpy as np
import pytest
from scipy import fft, signal

from data2g import codes, config, modem
from data2g.hfchannel import freq_shift
from data2g.waveform import dsp, ofdm, sync

BANDS = list(config.BANDS)
PHASOR_TOL = 1e-15
SUM_TOL = 1e-13


@pytest.fixture
def py(reference):
    """The Python functions, whether --native substituted them or not."""
    class Py:
        band = staticmethod(reference(ofdm, "band"))
        to_baseband = staticmethod(reference(dsp, "to_baseband"))
        freq_correct = staticmethod(reference(dsp, "freq_correct"))
        tx_condition = staticmethod(reference(dsp, "tx_condition"))
        papr_db = staticmethod(reference(dsp, "papr_db"))
        repeat_corr = staticmethod(reference(sync, "_repeat_corr"))
        repeat_corrs = staticmethod(reference(sync, "_repeat_corrs"))
        raw_stat = staticmethod(reference(sync, "_raw_stat"))
        detection_stat = staticmethod(reference(sync, "detection_stat"))
        first_path = staticmethod(reference(sync, "first_path"))
        refine = staticmethod(reference(sync, "_refine"))
        crossings = staticmethod(reference(sync, "_crossings"))
        acquire = staticmethod(reference(sync, "acquire"))
        StreamDetector = reference(sync, "StreamDetector")
    return Py


def close(got, want, tol, scale=None):
    want = np.asarray(want)
    s = np.max(np.abs(want)) if scale is None else scale
    np.testing.assert_allclose(got, want, rtol=0, atol=tol * max(s, 1.0))


@pytest.mark.parametrize("name", BANDS)
def test_band_tables(native, py, name):
    b, n = py.band(name), native.waveform.band(name)
    assert n.name == name and n.nc == b.nc and n.preamble_samples == b.spec.preamble_samples
    assert n.preamble_threshold == b.spec.preamble_threshold and n.tx_bandpass == b.spec.tx_bandpass
    np.testing.assert_array_equal(n.freqs, b.freqs)
    np.testing.assert_array_equal(n.bb, b.bb)
    for attr in ("mod", "demod", "pilot"):
        close(getattr(n, attr), getattr(b, attr), PHASOR_TOL)
    close(n.preamble_template, b.preamble_template(), SUM_TOL)
    close(n.preamble_waveform(), b.preamble_waveform(), SUM_TOL)


@pytest.mark.parametrize("name", BANDS)
def test_modulate_and_demod_window(native, py, name):
    rng = np.random.default_rng(1)
    b, n = py.band(name), native.waveform.band(name)
    s = rng.normal(size=(9, b.nc)) + 1j * rng.normal(size=(9, b.nc))
    close(n.modulate_symbols(s), b.modulate_symbols(s), SUM_TOL)
    z = rng.normal(size=1500) + 1j * rng.normal(size=1500)
    # inside, off the end (zero-padded), and negative starts (Python's slice wraps)
    for start, backoff in [(100, 6), (32, 0), (1400, 6), (1500, 0), (3, 6), (-150, 0), (-160, 0), (-1200, 0), (-5000, 0)]:
        close(n.demod_window(z, start, backoff), b.demod_window(z, start, backoff), SUM_TOL, scale=10.0)


def test_baseband_and_freq_correct(native, py):
    rng = np.random.default_rng(2)
    x = rng.normal(size=7001)
    for n0 in (0, 1, 15, 12345, 10**12 + 7):
        np.testing.assert_array_equal(native.waveform.to_baseband(x, n0), py.to_baseband(x, n0))
    z = py.to_baseband(x)
    for f in (0.0, 37.3, -143.0, 612.5):
        close(native.waveform.freq_correct(z, f), py.freq_correct(z, f), SUM_TOL)


def test_numpy_primitives(native):
    d = native.dsp
    rng = np.random.default_rng(3)
    for n in (1, 5, 7, 8, 9, 100, 128, 129, 300, 1000, 4097, 20000):
        a = rng.normal(size=n) * 10 ** rng.uniform(-5, 5, size=n)
        c = a + 1j * rng.normal(size=n)
        assert d.pairwise_sum(a) == np.sum(a)
        assert d.pairwise_sum_complex(c) == np.sum(c)
        assert d.quantile(a, 0.2) == np.quantile(a, 0.2)
    assert all(d.next_fast_len(n) == fft.next_fast_len(n) for n in range(1, 20000))
    for lo, hi in [config.TX_BANDPASS] + [config.BANDS[b].tx_bandpass for b in BANDS] + [(612.5, 1987.25)]:
        np.testing.assert_array_equal(d.firwin_bandpass(201, lo, hi, config.FS),
                                      signal.firwin(201, (lo, hi), fs=config.FS, pass_zero=False))
    x = rng.normal(size=999)
    close(d.hilbert(x), signal.hilbert(x), SUM_TOL)
    close(d.convolve_same(x, x[:201]), np.convolve(x, x[:201], mode="same"), SUM_TOL)


@pytest.mark.parametrize("name", BANDS)
def test_tx_condition(native, py, name):
    rng = np.random.default_rng(4)
    b = py.band(name)
    x = b.modulate_symbols(np.exp(2j * np.pi * rng.random((30, b.nc))))
    x = np.concatenate([np.zeros(800), x, np.zeros(800)])
    bp = b.spec.tx_bandpass
    want = py.tx_condition(x, 1.0, (1.0, 1.5, 2.0), active=slice(800, len(x) - 800), bandpass=bp)
    got = native.waveform.tx_condition(x, 1.0, [1.0, 1.5, 2.0], 800, len(x) - 800, bp)
    close(got, want, SUM_TOL)
    # the ACE hook: a projector after each overshoot pass, then the closing passes
    calls = []

    def project(v):
        calls.append(len(v))
        return v * 1.01 + 0.001 * np.sin(np.arange(len(v)))

    want = py.tx_condition(x, 0.5, (1.0, 1.5), bandpass=bp, project=project, closing=(1.2,))
    n_py = len(calls)
    got = native.waveform.tx_condition(x, 0.5, [1.0, 1.5], 0, len(x), bp, project, [1.2])
    assert len(calls) == 2 * n_py == 4
    close(got, want, SUM_TOL)
    assert native.waveform.tx_condition(np.zeros(500), 1.0, [1.0], 0, 500, bp).tolist() == [0.0] * 500
    assert abs(native.waveform.papr_db(x) - py.papr_db(x)) < 1e-12


def _received(name, offset_hz, snr_db, seed, lead=2500, tail=3000):
    """A burst's start (preamble and header) on the sync band, CFO and AWGN."""
    rng = np.random.default_rng(seed)
    b = ofdm.band(name)
    x = np.concatenate([np.zeros(lead), b.preamble_waveform(), b.modulate_symbols(
        np.exp(2j * np.pi * rng.random((6, b.nc)))), np.zeros(tail)])
    x = freq_shift(x, offset_hz)
    x = x + rng.normal(scale=np.sqrt(np.mean(x[x != 0] ** 2) * 10 ** (-snr_db / 10)), size=len(x))
    return dsp.to_baseband(x)


@pytest.mark.parametrize("name", BANDS)
def test_matched_filter_and_statistic(native, py, name):
    W = native.waveform
    b = py.band(name)
    z = _received(name, 37.0, 5.0, 5)
    t = b.preamble_template()[config.PREAMBLE_CP:config.PREAMBLE_CP + config.M]
    t = t / np.linalg.norm(t)
    close(W.unit_template(name), t, PHASOR_TOL)
    freqs = list(sync._cfo_grid()) + list(sync.NOISE_REF_HZ)
    np.testing.assert_array_equal(W.cfo_grid(), sync._cfo_grid())
    np.testing.assert_array_equal(W.cfo_grid(50.0), sync._cfo_grid(50.0))
    close(W.repeat_corrs(z, t, freqs), py.repeat_corrs(z, t, freqs), SUM_TOL)
    for f in (-150.0, 37.5, 0.0):
        close(W.repeat_corr(z, t, f), py.repeat_corr(z, t, f), SUM_TOL)
    for kw in ({}, {"levels_from": 3000}, {"reach": 50.0, "repeats": 5}):
        outs = []
        S, q, f = py.raw_stat(z, b, outs=outs, **kw)
        S2, q2, f2, c2 = W.raw_stat(z, name, kw.get("reach", config.ACQUIRE_REACH_HZ), kw.get("repeats", 0),
                                     kw.get("levels_from"), True)
        close(S2, S, SUM_TOL)
        close(q2, q, SUM_TOL)
        close(c2, np.array(outs), SUM_TOL)
        np.testing.assert_array_equal(f2, f)
    S, f = py.detection_stat(z, b)
    S2, f2 = W.detection_stat(z, name)
    close(S2, S, SUM_TOL)


def test_first_path_and_crossings(native, py):
    W = native.waveform
    rng = np.random.default_rng(6)
    for _ in range(300):
        p = rng.random(int(rng.integers(3, 80))) ** 4
        peak = int(rng.integers(0, len(p)))
        for cyclic in (False, True):
            for search, frac in ((32, 0.5), (5, 0.1)):
                assert W.first_path(p, peak, search, frac, cyclic) == py.first_path(p, peak, search, frac, cyclic)
        D = rng.random(500) * 30
        for thr, span, limit in ((25.0, 40, 5), (29.9, 3, 2), (31.0, 10, 5), (0.0, 600, 3)):
            assert W.crossings(D, thr, span, limit) == py.crossings(D, thr, span, limit)


CASES = [(name, f, snr, seed) for name in BANDS
         for f, snr, seed in [(0.0, 20.0, 1), (6.0, 0.0, 2), (-37.5, -3.0, 3), (143.0, -6.0, 4), (-121.3, 3.0, 5)]]


@pytest.mark.parametrize("name,offset,snr,seed", CASES)
def test_acquire_decisions_match(native, py, name, offset, snr, seed):
    W = native.waveform
    b = py.band(name)
    z = _received(name, offset, snr, seed)
    try:
        want = py.acquire(z, band=b)
    except sync.SyncError as e:
        with pytest.raises(W.SyncError, match="no preamble"):
            W.acquire(z, name)
        assert "no preamble" in str(e)
        return
    start, f, metric, alts = W.acquire(z, name)
    assert start == want.preamble_start
    assert abs(f - want.freq_offset) < 1e-9
    assert abs(metric - want.metric) < 1e-12 * want.metric
    assert [a[0] for a in alts] == [a[0] for a in want.alternatives]
    np.testing.assert_allclose([a[1] for a in alts], [a[1] for a in want.alternatives], rtol=0, atol=1e-9)
    st, ff = W.refine(z, name, want.preamble_start + 3, 25.0)
    st2, ff2 = py.refine(z, b, want.preamble_start + 3, 25.0)
    assert st == st2 and abs(ff - ff2) < 1e-9
    # a search window, and a precomputed statistic with starts masked out
    lo, hi = want.preamble_start - 100, want.preamble_start + 100
    assert W.acquire(z, name, None, config.ACQUIRE_REACH_HZ, (lo, hi))[0] == py.acquire(z, band=b, search=(lo, hi)).preamble_start
    S, _ = py.detection_stat(z, b)
    S[:, :lo] = -1
    got, want = W.acquire(z, name, 10.0, config.ACQUIRE_REACH_HZ, None, S), py.acquire(z, 10.0, band=b, S=S)
    assert got[0] == want.preamble_start and [a[0] for a in got[3]] == [a[0] for a in want.alternatives]
    np.testing.assert_allclose([a[1] for a in got[3]], [a[1] for a in want.alternatives], rtol=0, atol=1e-9)


def test_acquire_errors(native, py):
    W = native.waveform
    z = py.to_baseband(np.random.default_rng(2).normal(size=4 * config.FS))
    with pytest.raises(W.SyncError, match="no preamble"):
        W.acquire(z, "w")
    with pytest.raises(W.SyncError, match="too short"):
        W.acquire(z[:1000], "w")
    with pytest.raises(W.SyncError, match="empty search window"):
        W.acquire(z, "w", None, config.ACQUIRE_REACH_HZ, (10**6, 10**6 + 5))
    with pytest.raises(KeyError):
        W.band("nope")


@pytest.mark.parametrize("name", ["w", "n10", "w48"])
def test_stream_detector(native, py, name):
    """Fed the same chunks (as tnc feeds them: real audio through
    to_baseband with the stream index), trimmed alike: the same state."""
    rng = np.random.default_rng(7)
    sm = config.SUBMODES[{"w": "ack-4f", "n10": "n10-ack-4f", "w48": "w48-qpsk-r1/5"}[name]]
    burst = modem.modulate([rng.bytes(codes.payload_bytes(sm))], sm)
    x = np.concatenate([rng.normal(size=6000) * 0.1, burst, rng.normal(size=9000) * 0.1])
    b = py.band(name)
    a, c = py.StreamDetector(b), native.waveform.StreamDetector(name)
    assert c.span == a.span and c.CHUNKS == a.CHUNKS
    fed = 0
    for k, n in enumerate([700, 2000, 2000, 1, 5000, 2000, 2000, 2000, 2000, 2000, 2000, 333, 2000]):
        chunk = x[fed:fed + n]
        za = py.to_baseband(chunk, fed)
        a.feed(za)
        c.feed(za)
        fed += n
        assert (c.fed, c.s0, c.c0) == (a.fed, a.s0, a.c0)
        assert c.S.shape == a.S.shape and c.C.shape == a.C.shape and len(c.levels) == len(a.levels)
        if a.S.shape[1]:
            close(c.S, a.S, SUM_TOL)
            close(c.C, a.C, SUM_TOL)
            close(np.array(c.levels), np.array(a.levels), SUM_TOL)
        np.testing.assert_array_equal(c.tail, a.tail)
        assert (c.level() is None) == (a.level() is None)
        if a.level() is not None:
            assert abs(c.level() - a.level()) <= SUM_TOL * a.level()
            lo = a.s0 - 50
            close(c.stat(lo, fed), a.stat(lo, fed), SUM_TOL)
        if k == 7:
            a.trim(fed - 5000)
            c.trim(fed - 5000)
        if k == 10:  # tnc: trimmed away, start over
            a.reset()
            c.reset()
            a.fed = c.fed = fed
