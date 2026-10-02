"""native equalizer (native/core/equalizer) against data2g/equalizer.py.

Integer decisions (support, window shift, rank, hot carriers) must match
exactly. Floats are compared relative to the array's scale at RTOL: the
C++ solves, eigen-solve and sums round in a different order from LAPACK's
and numpy's pairwise sums, and the LMMSE solves amplify that by their
condition number. Measured worst 5e-12 (estimate mse at 30 dB); RTOL leaves
20x margin."""

import copy

import numpy as np
import pytest
from conftest import EQUALIZER_NATIVE

from data2g import codes, equalizer, hfchannel, modem
from data2g.arq import phy
from data2g.config import BANDS, FS, NCP, SUBMODES
from data2g.hfchannel import FadingPreset
from data2g.waveform import ofdm

RTOL = 1e-10


@pytest.fixture
def py(reference, monkeypatch):
    """data2g.equalizer with every --native substitution undone."""
    for name in EQUALIZER_NATIVE:
        monkeypatch.setattr(equalizer, name, reference(equalizer, name))
    return equalizer


def close(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, what
    scale = max(np.max(np.abs(b)) if b.size else 0.0, 1e-300)
    err = np.max(np.abs(a - b)) / scale if b.size else 0.0
    assert err <= RTOL, f"{what}: relative error {err:.3g}"


def pilots(bb, n_p, kind, seed, snr_db=10.0):
    """(n_p, nc) LS pilot estimates of a synthetic channel: one path ("awgn"),
    two static paths, or two Rayleigh-fading paths (2 Hz, mpd-like)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_p) * equalizer.FRAME_S
    paths = {"awgn": [(4, 1.0)], "static": [(2, 1.0), (22, 0.6)], "fading": [(3, 1.0), (19, 1.0)]}[kind]
    h = np.zeros((n_p, len(bb)), complex)
    for d, a in paths:
        g = np.full(n_p, a * np.exp(2j * np.pi * rng.uniform()))
        if kind == "fading":
            f = rng.normal(0, 1.0, 8)
            ph = rng.uniform(0, 2 * np.pi, 8)
            g = a * np.sum(np.exp(1j * (2 * np.pi * f[:, None] * t + ph[:, None])), axis=0) / np.sqrt(8)
        h += g[:, None] * np.exp(-2j * np.pi * np.outer(np.ones(n_p), bb) * d / FS)
    n0 = np.mean(np.abs(h) ** 2) * 10 ** (-snr_db / 10)
    return h + np.sqrt(n0 / 2) * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))


CASES = [(b, n_p, kind, snr) for b in BANDS for n_p in (2, 5, 9, 33) for kind in ("awgn", "static", "fading")
         for snr in (-3.0, 10.0, 30.0)]


@pytest.mark.parametrize("band,n_p,kind,snr", CASES)
def test_pilot_path(native, py, band, n_p, kind, snr):
    bb = ofdm.band(band).bb
    ne = native.equalizer
    h = pilots(bb.astype(float), n_p, kind, seed=n_p * 7 + len(bb), snr_db=snr)
    assert ne.residual_cfo(h) == pytest.approx(py.residual_cfo(h), rel=RTOL, abs=1e-12)
    close(ne.delay_profile(h, bb), py.delay_profile(h, bb), "delay_profile")
    sup = py.delay_support(h, bb=bb)
    assert ne.delay_support(h, bb=bb) == sup
    shift = py.window_shift(sup)
    assert ne.window_shift(sup) == shift
    sup = (sup[0] - shift, sup[1] - shift)
    a, b = ne._freq_smooth(h, sup, bb), py._freq_smooth(h, sup, bb)
    close(a[0], b[0], "hs")
    assert a[2] == b[2]
    assert a[1] == pytest.approx(b[1], rel=RTOL)  # inf == inf on narrow bands
    close(a[3], b[3], "keep")
    reps = pilots(bb.astype(float), 8, "awgn", seed=3, snr_db=snr)
    assert ne.preamble_noise(reps) == pytest.approx(py.preamble_noise(reps), rel=RTOL)
    close(ne.preamble_noise_k(reps), py.preamble_noise_k(reps))
    kw = {"bb": bb, "n0_pre": py.preamble_noise(reps), "n0_pre_k": py.preamble_noise_k(reps)}
    want, got = py.estimate(h, sup, **kw), ne.estimate(h, sup, **kw)
    assert set(got) == set(want)
    for k in ("h", "mse", "n0_k"):
        close(got[k], want[k], k)
    for k in ("n0", "p_sig", "spread_hz"):
        assert got[k] == pytest.approx(want[k], rel=RTOL), k
    steps = np.arange(n_p) * 0.37 - 3
    close(ne.time_shift_phase(steps, bb), py.time_shift_phase(steps, bb))


def test_narrow_band_without_preamble_noise_raises(native, py):
    bb = ofdm.band("n4").bb
    h = pilots(bb.astype(float), 9, "static", 1)
    with pytest.raises(ValueError):
        py.estimate(h, (0, 20), bb)
    with pytest.raises(ValueError):
        native.equalizer.estimate(h, (0, 20), bb)


@pytest.mark.parametrize("band", list(BANDS))
def test_support_rank_and_subspace(native, band):
    """Rank from a Jacobi eigen-solve on B B^H against the SVD's, at every
    support width the receiver can produce (the delay grid's 0..4 NCP) and
    off-centre: the 1e-2 threshold's nearest miss is 4e-4 decades
    (n10 width 93), far above rounding."""
    bb = ofdm.band(band).bb.astype(np.float64)
    py_basis = equalizer._support_basis.__wrapped__
    for width in range(4 * NCP + 1):
        for d0 in {(NCP - width) // 2, -7}:
            U, r, r_full = py_basis(bb.tobytes(), d0, d0 + width)
            u, r2, r_full2, keep = native.equalizer.support_basis(bb, d0, d0 + width)
            assert (r2, r_full2) == (r, r_full), (width, d0)
            close(u @ u.conj().T, U @ U.conj().T, f"projector {width} {d0}")
            close(keep, 1 - np.sum(np.abs(U) ** 2, axis=1), "keep")


def test_per_carrier_noise(native, py):
    rng = np.random.default_rng(5)
    for n in (1, 2, 7, 9, 33, 65, 257, 1538, 2048):
        for _ in range(20):
            p = rng.gamma(n, 1 / n, size=int(rng.choice([4, 10, 24, 48])))
            p[rng.integers(len(p))] *= rng.choice([1, 3, 30])
            close(native.equalizer.per_carrier_noise(p, n), py.per_carrier_noise(p, n), f"n={n}")
    with pytest.raises(IndexError):
        native.equalizer.per_carrier_noise(np.ones(4), 2049)


def test_doppler_and_spread(native, py):
    dt = np.linspace(-1, 1, 41).reshape(-1, 1) - np.linspace(0, 0.5, 3)
    close(native.equalizer._doppler_corr(dt, 1.3), py._doppler_corr(dt, 1.3))
    for kind in ("awgn", "fading"):
        hs = pilots(ofdm.band("w").bb.astype(float), 20, kind, 2)
        for n0_s in (0.0, 0.05, 100.0):
            assert native.equalizer.measure_spread(hs, n0_s) == pytest.approx(py.measure_spread(hs, n0_s), rel=RTOL)


def _received(submode, n_cw, snr, seed):
    """A real receive: modem.receive's result and the estimate/refine calls it
    and arq.phy's DD pass make, captured with their arguments."""
    spec = SUBMODES[submode]
    rng = np.random.default_rng(seed)
    sent = [rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)]
    x = np.concatenate([np.zeros(3000), modem.modulate(sent, spec), np.zeros(3000)])
    y = hfchannel.apply_channel(x, snr_db=snr, fading_preset=FadingPreset("mpd", 2.0, 2.0), seed=seed)
    r = modem.receive(y)
    bits = np.stack([codes.encode(spec, p, index=i) for i, p in enumerate(sent)])
    # half the codewords known: rows know a comb of carriers, as mid-burst
    post = {i: 30.0 * (1 - 2.0 * bits[i]) for i in range(0, n_cw, 2)}
    return r, post


def _capture(monkeypatch, name):
    """Record (args, kwargs, Python's result) of every call to equalizer.<name>."""
    calls, fn = [], getattr(equalizer, name)

    def rec(*a, **k):
        out = fn(*a, **k)
        calls.append(copy.deepcopy((a, k, out)))  # the caller rescales est in place
        return out

    monkeypatch.setattr(equalizer, name, rec)
    return calls


@pytest.mark.parametrize("submode,n_cw,snr", [("qpsk-r1/2", 3, 6.0), ("w48-qpsk-r1/2", 16, 10.0),
                                             ("n10-qpsk-r1/2", 2, 8.0), ("n4-qpsk-r1/2", 1, 10.0),
                                             ("w48-16qam-r1/2", 5, 30.0)])
def test_live_estimate_and_refine(native, py, monkeypatch, submode, n_cw, snr):
    est_calls, ref_calls = _capture(monkeypatch, "estimate"), _capture(monkeypatch, "refine")
    r, post = _received(submode, n_cw, snr, seed=n_cw)
    assert r is not None and est_calls
    phy._dd_estimate(r, post)  # every other codeword known
    phy._dd_estimate(r, {i: post[0] for i in range(n_cw)})  # all known
    assert len(ref_calls) == 2
    for a, k, want in est_calls:
        got = native.equalizer.estimate(*a, **k)
        for key in ("h", "mse", "n0_k"):
            close(got[key], want[key], key)
        for key in ("n0", "p_sig", "spread_hz"):
            assert got[key] == pytest.approx(want[key], rel=RTOL), key
    for a, k, want in ref_calls:
        got = native.equalizer.refine(*a, **k)
        close(got[0], want[0], "refine h")
        close(got[1], want[1], "refine mse")
