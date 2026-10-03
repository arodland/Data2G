"""data2g.modem: C++ (native/core/modem) against the Python reference.
Skips if the module isn't built; `pytest --native` errors instead.

Decisions must match exactly: header words and their order, which
hypothesis and band win, lock positions, CFO aliases, kc, shift, steps,
support, decoded payloads. Floats: the header correlation is float32 in
both, but numpy's BLAS sums in its own order, so values agree to a few
float32 ulps (CORR_TOL relative); everything downstream of the FFT-based
sync and the equalizer to 1e-9 of its scale (measured worst ~1e-12).
"""

import contextlib

import numpy as np
import pytest

from data2g import codes, config, hfchannel, modem
from data2g.config import BANDS, SUBMODES

CORR_TOL = 1e-6
TOL = 1e-9
# modem's own functions: restored to Python together, so a reference call
# never reaches the native ones through module globals (--native)
OWN = ("_crc6", "header_bits", "_signs", "_valid_words", "_valid_signs", "decode_header", "modulate",
       "modulate_bits", "burst_waveform", "ace_cells", "_bin_phase_step", "_demod_frames", "_read_header",
       "_copy_llr", "_best_header", "find_burst", "pilot_coherence", "find_copy", "_cfo_aliases", "_copy_header",
       "receive", "resolve_alias", "data_channel", "noise_var", "soft_bits", "demodulate", "decode_received")


@pytest.fixture
def py(reference):
    """py.<name>(...): modem's Python function, with every modem function
    it calls Python too."""
    class Py:
        def __getattr__(self, name):
            def call(*a, **k):
                with python_modem(reference):
                    return getattr(modem, name)(*a, **k)
            return call
    return Py()


@contextlib.contextmanager
def python_modem(reference):
    saved = {n: getattr(modem, n) for n in OWN}
    try:
        for n in OWN:
            setattr(modem, n, reference(modem, n))
        yield
    finally:
        for n, f in saved.items():
            setattr(modem, n, f)


def close(got, want, tol=TOL):
    want = np.asarray(want)
    s = max(float(np.max(np.abs(want))) if want.size else 0.0, 1.0)
    np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=tol * s)


def _payloads(spec, n, seed):
    rng = np.random.default_rng(seed)
    return [rng.bytes(codes.payload_bytes(spec)) for _ in range(n)]


def _burst(name, n_cw=2, snr=12.0, preset=None, seed=1, cfo=23.0, pad=3000):
    spec = SUBMODES[name]
    x = np.concatenate([np.zeros(pad), modem.modulate(_payloads(spec, n_cw, seed), spec), np.zeros(pad)])
    return hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=cfo, ppm=10, fading_preset=preset, seed=seed)


# --- header -----------------------------------------------------------------

def test_header_tables(native, py):
    N = native.modem
    assert [N.crc6(v) for v in range(1024)] == [py._crc6(v) for v in range(1024)]
    for band in modem.SYNC_BANDS:
        np.testing.assert_array_equal(N.valid_words(band), py._valid_words(band))
        np.testing.assert_array_equal(N.signs(py._valid_words(band), band), py._valid_signs(band))
        for (b, sub) in modem.BY_INDEX:
            if b == band:
                for n in (1, 7, 64):
                    np.testing.assert_array_equal(N.header_bits(sub, n, band), py.header_bits(sub, n, band))
    acc = modem.Accept.of(["qpsk-r1/2", "ack-1f", "w48-qpsk-r1/2"], max_secs=2.5, min_score=0.2)
    for band in modem.SYNC_BANDS:
        np.testing.assert_array_equal(N.valid_words(band, acc), py._valid_words(band, acc))


@pytest.mark.parametrize("band", modem.SYNC_BANDS)
def test_decode_header_matches(native, py, band):
    """Random headers at -6..+3 dB per bit and pure noise: the same word,
    the score to float32 rounding. Near-ties (top two within 4 float32
    ulps, where the summation order could decide) occur in a few of these,
    mostly the noise-only reads, and are decided alike."""
    rng = np.random.default_rng(5)
    acc = modem.Accept.of(None, 4.0, 0.0)
    for i in range(300):
        words = py._valid_words(band)
        w = int(words[rng.integers(len(words))])
        x = py._signs(np.array([w]), band)[0].astype(float)
        sigma = [0.0, 0.7, 1.4, 2.0, 1e3][i % 5]
        soft = x + rng.normal(scale=sigma, size=x.shape) if sigma < 1e3 else rng.normal(size=x.shape)
        a = acc if i % 3 == 0 else None
        word, (spec, n), score = py.decode_header(soft, band, a)
        nw, (nname, nn), nscore = native.modem.decode_header(soft, band, a)
        assert (nw, nname, nn) == (word, spec.name, n)
        assert abs(nscore - score) <= CORR_TOL * max(abs(score), 1e-3)
        c_py = py._valid_signs(band, a) @ soft.astype(np.float32)
        c_n = native.modem.header_corr(soft, band, a)
        close(c_n, c_py, CORR_TOL)


# --- transmit -----------------------------------------------------------------

@pytest.mark.parametrize("name", list(SUBMODES))
def test_modulate_matches(native, py, name):
    spec = SUBMODES[name]
    pl = _payloads(spec, 2, spec.index + 1)
    want = py.modulate(pl, spec, [0, 1])
    close(native.modem.modulate(pl, name, [0, 1]), want, 1e-9)
    n_f = 2 * spec.frames_per_cw
    win, full = py.ace_cells(spec, n_f)
    np.testing.assert_array_equal(native.modem.ace_cells(name, n_f), full)


def test_burst_waveform_matches(native, py):
    rng = np.random.default_rng(2)
    for name in ("ack-1f", "qpsk-r1/2", "n4-ack-2f", "n10-qpsk-r1/5", "w48-16qam-r1/2"):
        spec = SUBMODES[name]
        nc = BANDS[spec.band].nc
        data = np.exp(2j * np.pi * rng.random((3 * spec.frames_per_cw, 5, nc)))
        close(native.modem.burst_waveform(data, name), py.burst_waveform(data, spec), 1e-12)


# --- receive ------------------------------------------------------------------

RX_CASES = [("ack-1f", 1, 10.0, None), ("qpsk-r1/2", 2, 6.0, "mpp"), ("n4-ack-2f", 2, 0.0, None),
            ("n10-qpsk-r1/3", 1, 4.0, "mpp"), ("w48-16qam-r1/2", 3, 14.0, None), ("w48-64l-r1/2", 2, 20.0, "mpd"),
            ("polar-k96-f4", 3, -3.0, "mpd")]


def same_received(n, r):
    for k in ("n_cw", "kc", "p0", "shift", "preamble_start", "band"):
        assert n[k] == r[k], k
    assert n["spec"] == r["spec"].name
    assert tuple(n["support"]) == tuple(r["support"])
    np.testing.assert_array_equal(n["steps"], r["steps"])
    s, f, metric, alts = n["acq"]
    assert (s, [a[0] for a in alts]) == (r["acq"].preamble_start, [a[0] for a in r["acq"].alternatives])
    close([f, metric, n["cfo"], n["phi_ref"]] + [a[1] for a in alts],
          [r["acq"].freq_offset, r["acq"].metric, r["cfo"], r["phi_ref"]] + [a[1] for a in r["acq"].alternatives])
    close(n["score"], r["score"], CORR_TOL)
    for k in ("raw", "hp"):
        close(n[k], r[k])
    for k in ("h", "mse", "n0", "n0_k", "p_sig", "spread_hz", "clip_ratio", "gain"):
        close(n["est"][k], r["est"][k], 1e-8)
    assert n["est"]["band"] == r["est"]["band"]


@pytest.mark.parametrize("name,n_cw,snr,preset", RX_CASES)
def test_receive_matches(native, py, name, n_cw, snr, preset):
    y = _burst(name, n_cw, snr, preset, seed=n_cw + 3)
    r = py.receive(y)
    same_received(native.modem.receive(y), r)
    # the streaming path: head-limited search
    head = 3000 + 2 * config.LEADIN_SAMPLES + modem.head_samples(SUBMODES[name].sync_band)
    same_received(native.modem.receive(y, [SUBMODES[name].sync_band], None, head), py.receive(
        y, [SUBMODES[name].sync_band], None, head))
    b, nb = py.decode_received(r), native.modem.decode_received(r)
    assert nb[1] == b.payloads and nb[2] == b.crc_ok
    close(nb[6], b.soft, 1e-8)
    close([nb[3], nb[5]], [b.freq_offset, b.snr_db], 1e-8)
    var = py.noise_var(r["est"]["h"], r["est"]) + r["est"]["mse"]
    close(native.modem.noise_var(r["est"]["h"], r["est"]["n0_k"], r["est"]["clip_ratio"]) + r["est"]["mse"], var)
    close(native.modem.soft_bits(r["raw"], r["est"]["h"], var, name), py.soft_bits(r["raw"], r["est"]["h"], var, r["spec"]))


def test_search_parts_match(native, py):
    """_best_header (complete and streaming), _read_header at each shift,
    the copy LLRs, _demod_frames with replayed steps, data_channel."""
    y = _burst("qpsk-r1/3", 1, 2.0, "mpd", seed=9)
    z0 = modem.to_baseband(y)
    hd, acq, z = py._best_header(z0)
    nhd, nacq, nz = native.modem.best_header(z0)
    assert nhd["word"] == hd["word"] and nhd["start"] == hd["start"] and nacq[0] == acq.preamble_start
    close(nz, z)
    for k in (-2, 0, 1):
        a, b = native.modem.read_header(z, hd["start"] + k * config.M, "w"), py._read_header(z, hd["start"] + k * config.M, "w")
        assert (a["word"], a["pending_copy"]) == (b["word"], b["pending_copy"])
        assert (a["hdr"] is None) == (b["hdr"] is None)
        close(a["score"], b["score"], CORR_TOL)
        for f in ("y", "y_all", "h_pre", "h_first", "n0_pre", "n0_pre_k"):
            close(a[f], b[f])
    close(native.modem.copy_llr(z, hd["p0"] + config.FRAME_SAMPLES, "w", 4),
          py._copy_llr(z, hd["p0"] + config.FRAME_SAMPLES, "w", 4))
    assert native.modem.copy_llr(z, len(z), "w", 4) is None
    raw, hp, steps = py._demod_frames(z, hd["p0"], 9, 3, 0.4)
    nraw, nhp, nsteps = native.modem.demod_frames(z, hd["p0"], 9, 3, 0.4)
    np.testing.assert_array_equal(nsteps, steps)
    close(nraw, raw)
    close(nhp, hp)
    replay = np.array([0, 1, 1, 0, -1, 0, 0, 2, 2, 2], float)
    close(native.modem.demod_frames(z, hd["p0"], 9, 0, 0.0, replay)[0], py._demod_frames(z, hd["p0"], 9, 0, 0.0, replay)[0])
    est = py.data_channel(hp, (-3, 20), "w", 0.1, None, None, 8)
    nest = native.modem.data_channel(hp, (-3, 20), "w", 0.1, None, None, 8)
    for k in ("h", "mse", "n0", "gain", "clip_ratio"):
        close(nest[k], est[k], 1e-8)
    # a streaming buffer cut inside the header: both wait
    cut = hd["start"] + 1500
    for f in (py._best_header, native.modem.best_header):
        with pytest.raises(Exception, match="no preamble|still arriving|header decode"):
            f(z0[:cut], None, False)


def test_find_burst_and_noise(native, py):
    y = _burst("n10-qpsk-r1/5", 1, 3.0, None, seed=4)
    a, b = native.modem.find_burst(y[:9000]), py.find_burst(y[:9000])
    assert {k: v for k, v in a.items() if k not in ("score", "cfo")} == \
        {k: (v.name if k == "spec" else v) for k, v in b.items() if k not in ("score", "cfo")}
    close([a["score"], a["cfo"]], [b["score"], b["cfo"]], CORR_TOL)
    lock = dict(b, spec=b["spec"].name)
    close(native.modem.pilot_coherence(y, lock, 8, True), py.pilot_coherence(y, b, 8, True))
    rng = np.random.default_rng(11)
    for band in modem.SYNC_BANDS:
        noise = rng.normal(size=16000)
        errs = []
        for f in (py.find_burst, native.modem.find_burst):
            try:
                f(noise, [band])
                errs.append(None)
            except Exception as e:  # modem.SyncError / native SyncError
                errs.append(str(e))
        assert errs[0] == errs[1]


@pytest.mark.parametrize("name,n_cw", [("ack-4f", 1), ("qpsk-r1/5", 1), ("w48-qpsk-r1/2", 3)])
def test_find_copy_matches(native, py, reference, name, n_cw):
    """Preamble and first header wiped: the copy lock, its CFO alias and
    the receive() it feeds."""
    spec = SUBMODES[name]
    x = modem.modulate(_payloads(spec, n_cw, 3), spec)
    h0 = config.LEADIN_SAMPLES
    x[h0:h0 + BANDS[spec.sync_band].preamble_samples + modem.header_samples(spec.sync_band)] = 0
    y = hfchannel.apply_channel(np.concatenate([np.zeros(2000), x, np.zeros(3000)]), snr_db=10, freq_offset_hz=-41.0,
                                seed=2)
    lock = py.find_copy(y, spec.sync_band)
    nlock, peak = native.modem.find_copy(y, spec.sync_band)
    assert lock is not None and nlock is not None
    for k in ("n_cw", "start", "end", "p0", "band", "copy"):
        assert nlock[k] == lock[k], k
    close([nlock["score"], nlock["cfo"], peak], [lock["score"], lock["cfo"], reference(modem, "find_copy").peak], CORR_TOL)
    same_received(native.modem.receive(y, None, None, None, nlock), py.receive(y, copy=lock))
    centre = 12.5 * np.round(lock["cfo"] / 12.5)
    d = np.exp(2j * np.pi * lock["cfo"] * config.FRAME_SAMPLES / config.FS)
    close(native.modem.cfo_aliases(d, centre), py._cfo_aliases(d, centre))


def test_demodulate_matches(native, py):
    y = _burst("w48-qpsk-r1/3", 4, 1.0, "mpp", seed=6)
    b, n = py.demodulate(y), native.modem.demodulate(y)
    assert n[0] == b.submode.name and n[1] == b.payloads and n[2] == b.crc_ok and n[4] == b.preamble_start
