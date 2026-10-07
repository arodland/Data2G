"""C++ modem timing, data2g.arq.modes, .predictor and .policy against the
Python reference. Skips if the module isn't built; `pytest --native` errors
instead (conftest.py).

Tolerances. Integers and burst lengths are exact. np.interp (capacity) is
exact on linux x86-64 and 1e-15 relative elsewhere: macOS arm64's numpy
was 1 ulp off at 3 of 2261 points (plan, Findings: portability). The MI
features go through log10 and numpy's pairwise mean: 1e-14. The outcome
model's logits go through BLAS dot products, tanh and exp: 1e-12 relative
(atol 1e-12 near zero). Every shifter decision (modes, size hints, slots,
want_dup) must match exactly.
"""

import platform
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import conftest
from data2g import codes, config, modem
from data2g.arq import modes
from data2g.arq import policy as G
from data2g.arq import predictor as P

LOGIT_TOL = 1e-12
MI_TOL = 1e-14
# The bitwise reference platform (plan, Findings: portability).
CAP_RTOL = 0.0 if (sys.platform == "linux" and platform.machine() == "x86_64") else 1e-15


@pytest.fixture
def pure(monkeypatch):
    """Every --native substitution undone for the test: the reference is all
    Python (its callees included), the native side called directly."""
    for (module, attr), fn in conftest._originals.items():
        monkeypatch.setattr(module, attr, fn)


@pytest.fixture
def A(native):
    return native.arq


def test_timing(native, pure):
    T = native.timing
    for band in config.BANDS:
        np.testing.assert_array_equal(T.header_layout(band), modem.header_layout(band))
        assert (T.hosts(band), T.header_samples(band), T.head_samples(band)) == \
            (modem._hosts(band), modem.header_samples(band), modem.head_samples(band))
        for n_f in range(12):
            assert T.copy_frame(band, n_f) == modem.copy_frame(band, n_f)
    for s in config.SUBMODES.values():
        for n in range(config.MAX_CODEWORDS + 1):
            assert T.frames_on_air(s.name, n) == modem.frames_on_air(s, n)
            assert T.burst_end(1234, s.name, n) == modem.burst_end(1234, s, n)
            assert T.burst_seconds(s.name, n) == modem.burst_seconds(s, n)


def test_modes(A, pure):
    assert A.modes() == list(modes.MODES)
    for name, s in modes.MODES.items():
        assert A.is_cpm(name) == modes.is_cpm(s)
        assert (A.payload_bytes(name), A.ctl_payload_bytes(name), A.max_ctl(name), A.rv_cycle(name)) == \
            (codes.payload_bytes(s), modes.ctl_payload_bytes(s), modes.max_ctl(s), codes.rv_cycle(s))
        assert [A.min_cw(name, d) for d in (False, True)] == [modes.min_cw(s, d) for d in (False, True)]
        for n in range(1, 20):
            for dup in (False, True):
                assert A.burst_seconds(name, n, dup) == modes.burst_seconds(s, n, dup), (name, n, dup)


def test_capacity_and_effective_mi(A, pure):
    rng = np.random.default_rng(1)
    snr = np.concatenate([rng.uniform(-30, 60, 2000), np.load(P.DATA / "capacity_tables.npz")["grid"]])
    for c in P.CONSTS + ("c64-w48-r12", "c256-w48-r58"):
        assert A.const_family(c) == P.const_family(c)
        np.testing.assert_allclose(A.capacity(snr, c), P.capacity(snr, c), rtol=CAP_RTOL, atol=0)
        for shape in ((7,), (5, 24), (3, 48)):
            h = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * 10 ** rng.uniform(-2, 1)
            var = rng.uniform(0.01, 1, shape)
            want = P.effective_mi(h, var, c)
            assert A.effective_mi(h.reshape(-1), var.reshape(-1), c) == pytest.approx(want, rel=MI_TOL, abs=MI_TOL)
        x = np.sqrt(rng.uniform(0, 100, 30))  # cpm.measure: real amplitudes
        assert A.effective_mi(x.astype(complex), np.ones(30), c) == \
            pytest.approx(P.effective_mi(x, np.ones_like(x), c), rel=MI_TOL, abs=MI_TOL)


def random_measured(rng, band="w", noise=False):
    nc = {"w": 24, "n10": 10, "n4": 4, "w48": 48}.get(band, 10)
    snr = rng.uniform(-8, 25)
    m = dict(snr_est=snr, spread_est=rng.choice([0.0, rng.uniform(0, 3)]), delay_est_ms=rng.uniform(0, 4),
             headroom=float(rng.choice([0.0, 1.0, 3.0])), frames=int(rng.choice([4, 8, 16, 48])))
    for c in P.CONSTS:
        m[f"mi_{c}"] = float(P.capacity(snr + 10 * np.log10(50 / nc) + rng.normal(0, 2), c))
    if rng.uniform() < 0.3:  # cpm.measure's and older callers' dicts
        del m["headroom"]
        m["frames"] = float(m["frames"])
    if noise and rng.uniform() < 0.5:  # the receiver's noise profile, often one loud sub-band (the noise rule)
        m["noise"] = random_noise(rng)
    return m


def random_noise(rng):
    db = rng.normal(0, 0.3, 5)
    tail = rng.uniform(0.9, 2.0, 5)
    if rng.uniform() < 0.7:
        k = rng.integers(5)
        db[k] += rng.uniform(0, 20)
        tail[k] += rng.uniform(0, 8)
    return {"noise_db": [float(v) for v in db], "noise_tail_db": [float(v) for v in tail],
            "impulses_per_min": float(rng.choice([0.0, 30.0]))}


def test_outcome_model(A, pure):
    model = P.outcome_model()
    assert A.outcome_modes() == list(model.modes) and A.outcome_bands() == list(model.bands)
    rng = np.random.default_rng(2)
    worst = 0.0
    for i in range(300):
        band = rng.choice(model.bands)
        m = random_measured(rng, band)
        if model.energy and i % 4:  # the energy inputs: none yet, or some
            m = dict(m, energy=[float(rng.uniform(-12, 30)), float(rng.choice([0.25, 0.5, 1.0])), 1.0])
        prev = None if i % 3 == 0 else (random_measured(rng), rng.choice(model.bands), rng.uniform(0, 30))
        gap, sec = rng.uniform(1, 5), round(rng.uniform(0.5, 50), 2)
        x = P.outcome_inputs(m, band, gap, sec, prev, model.bands, False, False, model.energy)
        xn = A.outcome_inputs(m, band, gap, sec, prev, list(model.bands), bool(model.energy))
        np.testing.assert_allclose(xn, x, rtol=1e-15, atol=1e-15)
        z, zn = model(x), A.outcome_logits(x)
        np.testing.assert_allclose(zn, z, rtol=LOGIT_TOL, atol=LOGIT_TOL)
        worst = max(worst, float(np.max(np.abs(zn - z) / np.maximum(np.abs(z), 1))))
        want = P.predict_outcome(m, band, gap, sec, None, prev)
        got = A.predict_outcome(m, band, gap, sec, prev)
        for k, (pb, pc) in want.items():
            assert got[k] == pytest.approx((pb, pc), rel=LOGIT_TOL, abs=LOGIT_TOL), k
    assert worst < LOGIT_TOL
    for name in modes.MODES:
        assert A.outcome_knows(name) == P.outcome_knows(name)


def test_policy_helpers(A, pure):
    for cap in G.CAP_HZ:
        assert A.allowed(cap) == [s.name for s in G.allowed(cap)]
        assert (A.cap_hz(cap), A.fallback(cap)) == (G.CAP_HZ[cap], G.FALLBACK[cap])
        assert [A.connect_mode(cap, t) for t in (0, 1, 2)] == [G.GearShifter().connect_mode(cap, t) for t in (0, 1, 2)]
    for rec in range(64):
        assert A.decode(rec) == G.decode(rec)
    for name, s in modes.MODES.items():
        assert (A.encode(name), A.width_hz(name), A.ctl_slots(name)) == (G.encode(name), G.width_hz(s), G.ctl_slots(s))
        for sec in (*G.SIZE_S, 0.1, 2.0, 30.0, 100.0):
            for data in (False, True):
                for dup in (False, True):
                    assert A.slots_for(name, sec, data, dup) == G.slots_for(s, sec, data, dup), (name, sec, data, dup)


def station(rng, cap, rec=None):
    pending = bool(rng.random() < 0.7)
    return SimpleNamespace(cap=cap, chat=rng.random() < 0.2, peer_chat=rng.random() < 0.1,
                           peer_queued=rng.choice([0, 0, 150, 900, 5000]), rx=SimpleNamespace(buf=dict.fromkeys(
                               range(rng.choice([0, 0, 1, 5])))), tx=SimpleNamespace(pending=lambda: pending),
                           peer_recommend=rec[0] if rec else None, peer_size_hint=rec[1] if rec else 1,
                           peer_reply_recommend=rec[2] if rec else None, peer_wants_dup=rng.random() < 0.3)


@pytest.mark.parametrize("seed", range(11))
def test_shifter_decisions_match(A, pure, seed):
    """Random sessions: observations, recommendations, outcomes, choices.
    Every decision identical; probabilities and biases to LOGIT_TOL. Seeds 6-8:
    flat, under 12 dB, so the gate's model (predictor.GATE) decides; 9-10:
    under -4 dB, so the CPM floor (policy.CPM_FLOOR) holds; energy readings throughout."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    kw = dict(gap_s=rng.choice([2.5, 1.5, 4.0]), use_cpm=seed % 3 != 2, min_success=rng.choice([0.0, 0.0, 0.5]))
    ref, nat = G.GearShifter(**kw), A.GearShifter()
    nat.gap_s, nat.use_cpm, nat.min_success = kw["gap_s"], kw["use_cpm"], kw["min_success"]
    cap, now, rec = seed % 3, 0.0, None
    assert nat.choose(station(rng, cap), 0) == ref.choose(station(rng, cap), 0)
    for step in range(60):
        if rng.random() < 0.05:
            cap = rng.choice(list(G.CAP_HZ))
        heard = rng.choice(G.allowed(cap)).name
        m = random_measured(nrng, modes.MODES[heard].band, noise=True)
        if 6 <= seed <= 8:
            m = dict(m, spread_est=0.05, snr_est=min(m["snr_est"], 10.0))
        if seed >= 9:
            m = dict(m, snr_est=min(m["snr_est"], -5.0))
        now += rng.choice([1.0, 5.0, 12.0, 35.0])
        ref.observe(m, heard, now)
        nat.observe(m, heard, now)
        e = float(nrng.uniform(-12, 30))
        ref.observe_energy(e, now)
        nat.observe_energy(e, now)
        assert nat.gate() == (ref.gate() is not None) and nat.cpm_floor() == ref.cpm_floor()
        st = station(rng, cap)
        want, got = ref.recommend(st), nat.recommend(st)
        assert got == want, (seed, step, [G.decode(r) for r in (want[0], want[2])], [A.decode(r) for r in (got[0], got[2])])
        assert nat.want_dup == ref.want_dup and nat.predicted.keys() == ref.predicted.keys()
        for k, v in ref.predicted.items():
            assert nat.predicted[k] == pytest.approx(v, rel=LOGIT_TOL, abs=LOGIT_TOL)
        rec = want
        sent = rng.choice([0, 1, 4, 8])
        decoded = rng.randint(0, sent)
        usable = rng.choice([None, True, False])
        for mode in (A.decode(want[0]), A.decode(want[2]), heard):
            ref.outcome(mode, decoded, sent, usable)
            nat.outcome(mode, decoded, sent, usable)
        for b in ("bias", "bias_burst"):
            r, n = getattr(ref, b), getattr(nat, b)
            assert n.keys() == r.keys() and all(n[k] == pytest.approx(r[k], rel=LOGIT_TOL, abs=LOGIT_TOL) for k in r)
        st = station(rng, cap, rec)
        esc = int(rng.random() < 0.1)
        assert nat.choose(st, esc) == ref.choose(st, esc)
        assert nat.next_capacity(st) == ref.next_capacity(st)
    assert nat.log == ref.log


def test_native_shifter_state_roundtrips(A):
    g = A.GearShifter()
    m = dict(snr_est=3.0, spread_est=0.5, delay_est_ms=1.0, headroom=0.0, frames=8.0,
             **{f"mi_{c}": 0.5 for c in P.CONSTS})
    g.observe(m, "qpsk-r1/2", 1.0)
    g.observe(m, "n4-qpsk-r1/3", 2.0)
    assert g.measured == m and g.measured_band == "n4" and g.prev == (m, "w", 1.0)
    g.predicted = {"ack-4f": (0.25, 0.5)}
    assert g.predicted == {"ack-4f": (0.25, 0.5)}
    g.log = [("ack-4f", 1, "ack-4f")]
    assert g.log == [("ack-4f", 1, "ack-4f")]


def test_noise_rule_helpers_match(A, pure):
    """predictor.band_span_hz, noise_shift_db and shifted, C++ against Python."""
    from data2g import cpm

    for b in sorted({s.band for s in modes.MODES.values()} | set(cpm.GRIDS)):
        assert A.band_span_hz(b) == pytest.approx(P.band_span_hz(b)), b
    rng = np.random.default_rng(3)
    bands = sorted({s.band for s in modes.MODES.values()})
    for _ in range(300):
        n = random_noise(rng)
        mb, b = rng.choice(bands), rng.choice(bands)
        w = float(rng.choice([0.5, 1.0]))
        assert A.noise_shift_db(n, mb, b, w) == pytest.approx(P.noise_shift_db(n, mb, b, w), abs=1e-9)
        m = random_measured(rng)
        shift = float(rng.uniform(0, 12))
        want, got = P.shifted(m, shift), A.shifted(m, shift)
        assert got["snr_est"] == pytest.approx(want["snr_est"])
        assert all(got[f"mi_{c}"] == pytest.approx(want[f"mi_{c}"], abs=1e-12) for c in P.CONSTS)
    assert A.noise_shift_db(None, "w", "w48") == 0.0
