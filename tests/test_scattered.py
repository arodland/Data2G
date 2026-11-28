"""HF scattered pilots (config.SC_BANDS): fk3d2 intra-frame combs on w/w48, gold on n10/n4."""

import numpy as np
import pytest

from data2g import codes, hfchannel, modem
from data2g.arq import phy
from data2g.config import SUBMODES

pytestmark = pytest.mark.hf_only("scattered pilots are defined on the 8 kHz HF bands (w, w48, n10, n4)")


@pytest.mark.parametrize("name,n_cw", [("qpsk-r1/2", 1), ("16qam-r1/2", 3), ("w48-16qam-r1/2", 4), ("n10-qpsk-r1/2", 2),
                                       ("n4-qpsk-r1/2", 1), ("n10-16qam-r1/2", 3), ("ack-1f", 1)])
def test_round_trip(name, n_cw):
    spec = SUBMODES[name]
    rng = np.random.default_rng(5)
    pl = [rng.bytes(codes.payload_bytes(spec)) for _ in range(n_cw)]
    x = modem.modulate(pl, name)
    assert len(x) == round(modem.burst_seconds(spec, n_cw) * 8000)
    y = hfchannel.awgn(np.concatenate([np.zeros(2000), x, np.zeros(2000)]), 25, seed=1, s_power=1.0)
    r = modem.receive(y)
    assert ("sc" in r) == modem.sc_spec(spec, n_cw * spec.frames_per_cw)
    b = modem.decode_received(r)
    assert b.payloads == pl and all(b.crc_ok)
    assert r["est"]["h"].shape == (n_cw * spec.frames_per_cw, 5, modem.BANDS[spec.band].nc)
    if "sc" in r:
        # a streaming lock's pilot check reads each carrier's own pilot
        lock = dict(spec=spec, n_cw=n_cw, p0=r["p0"], cfo=r["cfo"])
        c, pat = modem.pilot_coherence(y, lock, pattern=True)
        assert min(c) > 0.6 and pat > 0.6  # the pilots clip with the data (12 dB): not 1
        # decision-directed: every codeword known -> an estimate no worse than the pilots'
        post = {i: 30.0 * (1 - 2.0 * codes.encode(spec, p, 0, index=i)) for i, p in enumerate(pl)}
        est = phy._dd_estimate(r, post)
        assert est["h"].shape == r["est"]["h"].shape and np.mean(est["mse"]) < np.mean(r["est"]["mse"])


def test_the_header_does_not_read_as_its_own_copy():
    """A scattered-pilot copy frame closes on copy_close, not the pilot: closed on the pilot it had the first
    header's own shape (the preamble's last repeat, the header, the header pilot), so a preamble hypothesis two
    frames and a symbol early read the real header as its copy and held the receiver (test_receiver's false
    lock on data, seed 0)."""
    from data2g.config import BANDS, FRAME_SAMPLES, LEADIN_SAMPLES, NSYM
    from data2g.waveform import dsp

    spec = SUBMODES["qpsk-r1/2"]
    rng = np.random.default_rng(1)
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec)) for _ in range(2)], spec)
    z = dsp.to_baseband(hfchannel.awgn(x, 20, seed=2, s_power=1.0))
    start = LEADIN_SAMPLES
    p0 = start + BANDS["w"].preamble_samples + modem.header_samples("w")
    kc = modem.copy_frame("w", 2 * spec.frames_per_cw)
    floor = modem.HEADER_MIN_SCORE["w"]
    assert modem.decode_header(modem._copy_llr(z, p0 + kc * FRAME_SAMPLES, "w", 4), "w")[2] > 0.9  # the copy
    at_header = start + BANDS["w"].preamble_samples - NSYM  # where the header's shape starts
    assert modem.decode_header(modem._copy_llr(z, at_header, "w", 4), "w")[2] < floor


def test_find_copy_on_a_band_without_copies():
    """A scattered-pilot band whose bursts carry no header copy (n10): find_copy finds none. It read a copy band's
    frame there and raised on a strong tone (test_native_modem's pair-products signals)."""
    from data2g.config import FS

    n = 3 * FS
    y = np.sin(2 * np.pi * 1510.0 * np.arange(n) / FS) + np.random.default_rng(7).normal(size=n)
    assert modem.find_copy(y, "n10") is None


def test_a_copy_lock_reads_its_phase_reference_off_copy_open():
    """receive(copy=...) takes the timing tracker's phase reference from the copy frame's first symbol, which on a
    scattered-pilot band is copy_open: divided by the pilot instead, the reference was off by the pilots' own phases,
    and the tracker stepped the window 8 samples over a 3-codeword burst the preamble-locked receive held at 2."""
    spec = SUBMODES["qpsk-r1/2"]
    rng = np.random.default_rng(1)
    x = modem.modulate([rng.bytes(codes.payload_bytes(spec)) for _ in range(3)], spec)
    y = hfchannel.apply_channel(np.concatenate([np.zeros(2000), x, np.zeros(2000)]), snr_db=15, fading_preset="mpp", seed=3)
    r0 = modem.receive(y)
    lock = modem.find_copy(y, "w")
    assert lock is not None
    r1 = modem.receive(y, copy=lock)
    assert abs(np.angle(np.exp(1j * (r1["phi_ref"] - r0["phi_ref"])))) < 0.3
    assert np.max(np.abs(r1["steps"])) <= np.max(np.abs(r0["steps"])) + 2


@pytest.mark.parametrize("tone_hz", [2200.0, 2150.0, 825.0])
def test_a_tone_raises_its_own_carriers_noise_alone(tone_hz):
    """A steady tone 25 dB over the floor in 50 Hz (interference.PRESETS["carriers"]' level), on an odd carrier
    (2200 Hz), an even carrier (2150 Hz), or between two (825 Hz): the band's noise stays the floor's, the tone's
    carriers get their own, and the burst decodes."""
    from data2g.config import FS, LEADIN_SAMPLES

    spec = SUBMODES["w48-qpsk-r1/3"]
    rng = np.random.default_rng(3)
    pl = [rng.bytes(codes.payload_bytes(spec)) for _ in range(4)]
    x = np.concatenate([np.zeros(2400), modem.modulate(pl, spec), np.zeros(2400)])
    sigma = np.sqrt(np.mean(x**2) / 10**0.5 / (2500 / (FS / 2)))  # 5 dB in 2500 Hz
    y = x + np.random.default_rng(4).normal(scale=sigma, size=len(x))
    known = dict(start=2400 + LEADIN_SAMPLES, cfo=0.0, spec=spec, n_cw=4)
    n0_clean = modem.receive(y, known=known)["est"]["n0"]
    amp = np.sqrt(2 * sigma**2 * 50.0 / (FS / 2) * 10 ** 2.5)
    y = y + amp * np.cos(2 * np.pi * tone_hz * np.arange(len(y)) / FS + 0.3)
    r = modem.receive(y, known=known)
    assert r["est"]["n0"] < 2 * n0_clean
    nk = np.asarray(r["est"]["n0_k"])
    assert nk.shape == r["est"]["h"].shape and nk.max() > 10 * r["est"]["n0"]
    b = modem.decode_received(r)
    assert b.payloads == pl


def test_find_copy_reads_past_a_steady_tone():
    """Two tones just outside the w band, 25 dB over the floor in 50 Hz (interference.PRESETS["carriers"]): a
    comb's matched filter lets them through, and their frame-apart products, the same at every frame phase, held
    every grid's place, so the copy of a burst whose preamble was lost was never read in a receiver's 2 s window
    (PR #308 review, item 8; at -2 dB 0 of 10 seeds found, 10 with each CFO bin's mean over the phases out)."""
    from data2g.config import BANDS, FS, LEADIN_SAMPLES

    spec = SUBMODES["qpsk-r1/2"]
    rng = np.random.default_rng(0)
    pl = [rng.bytes(codes.payload_bytes(spec)) for _ in range(3)]
    x = np.concatenate([np.zeros(8000), modem.modulate(pl, spec), np.zeros(8000)])
    start = 8000 + LEADIN_SAMPLES
    x[start:start + BANDS["w"].preamble_samples + modem.header_samples("w")] = 0  # the head lost
    sigma = np.sqrt(np.mean(x[x != 0] ** 2) / 10 ** -0.2 / (2500 / (FS / 2)))  # -2 dB in 2500 Hz
    y = x + np.random.default_rng(100).normal(scale=sigma, size=len(x))
    t = np.arange(len(y)) / FS
    y += np.sqrt(2 * sigma**2 * 50.0 / (FS / 2) * 10**2.5) * (np.cos(2 * np.pi * 800 * t) + np.cos(2 * np.pi * 2200 * t + 1))
    lock = modem.find_copy(y[start - 400:start - 400 + 16704], "w")
    assert lock is not None and lock["spec"] == spec and lock["n_cw"] == 3


def test_copy_row_weights_only_where_a_tone_spreads_the_bins():
    """Bins within SC_COPY_BIN_SPREAD of each other (noise alone): no weights, the search as before. Past it each
    row (n_combs x the bins) is the quietest bin's level over its own."""
    assert modem._copy_row_weights(np.array([1.0, 1.1, 1.05]), 3) is None
    w = modem._copy_row_weights(np.array([1.0, 4.0, 2.0]), 3)
    assert np.allclose(w, np.tile([1.0, 0.25, 0.5], 3))


def test_find_copy_weighs_the_bins_a_tone_leaves_noisy():
    """fk3d2's comb products next to in-band tones (800 and 2200 Hz, +25 dB over the floor in 50 Hz) on mpp at
    -4 dB: the tone's cross terms with the noise, largest in the bins it leaks into, took the search's grid
    places, and this burst's copy (its preamble and first header lost) was not found; weighed by each bin's noise
    level (_copy_row_weights), it is."""
    from data2g.config import BANDS, FS, LEADIN_SAMPLES

    spec = SUBMODES["qpsk-r1/2"]
    rng = np.random.default_rng(9)
    pl = [rng.bytes(codes.payload_bytes(spec)) for _ in range(3)]
    x = np.concatenate([np.zeros(8000), modem.modulate(pl, spec), np.zeros(8000)])
    start = 8000 + LEADIN_SAMPLES
    x[start:start + BANDS["w"].preamble_samples + modem.header_samples("w")] = 0  # the head lost
    y = hfchannel.apply_channel(x, snr_db=-4, fading_preset="mpp", seed=7)
    sigma = np.std(y[:4000])
    t = np.arange(len(y)) / FS
    y = y + np.sqrt(2 * sigma**2 * 50.0 / (FS / 2) * 10**2.5) * (np.cos(2 * np.pi * 800 * t) + np.cos(2 * np.pi * 2200 * t + 1))
    lock = modem.find_copy(y[start - 400:start - 400 + 16704], "w")
    assert lock is not None and lock["spec"] == spec and lock["n_cw"] == 3


def test_a_slow_channel_is_estimated_separably_and_alike(monkeypatch):
    """On wide delay supports (> 52 samples) or narrow bands, spreads up to SC_SEP_HZ get the separable estimate;
    on w48 mpp (delay span <= 52 samples) Slepian handles all spreads and agrees closely with separable."""
    from data2g.config import LEADIN_SAMPLES

    spec = SUBMODES["w48-qpsk-r1/2"]
    rng = np.random.default_rng(2)
    pl = [rng.bytes(codes.payload_bytes(spec)) for _ in range(4)]
    x = np.concatenate([np.zeros(2400), modem.modulate(pl, spec), np.zeros(2400)])
    y = hfchannel.apply_channel(x, snr_db=15, fading_preset="mpp", seed=3)
    known = dict(start=2400 + LEADIN_SAMPLES, cfo=0.0, spec=spec, n_cw=4)
    r = modem.receive(y, known=known)
    assert r["est"]["filter_spread_hz"] <= modem.SC_SEP_HZ
    assert modem.decode_received(r).payloads == pl


def test_a_separable_spread_is_reported_as_the_joint_search_read_it():
    """The report (spread_hz) is capped at SC_SEP_REPORT_HZ when filter_spread_hz <= SC_SEP_HZ."""
    from data2g.config import LEADIN_SAMPLES

    spec = SUBMODES["w48-qpsk-r1/2"]
    rng = np.random.default_rng(2)
    x = np.concatenate([np.zeros(2400), modem.modulate([rng.bytes(codes.payload_bytes(spec)) for _ in range(4)], spec),
                        np.zeros(2400)])
    y = hfchannel.apply_channel(x, snr_db=15, fading_preset="mpp", seed=6)
    est = modem.receive(y, known=dict(start=2400 + LEADIN_SAMPLES, cfo=0.0, spec=spec, n_cw=4))["est"]
    assert est["filter_spread_hz"] <= modem.SC_SEP_HZ and est["spread_hz"] <= modem.SC_SEP_REPORT_HZ


@pytest.mark.python_only("reads Python's weight cache on narrow-band joint estimate")
def test_a_2_hz_channel_is_estimated_from_2_frames_either_side():
    """From SC_JOINT_REACH2_HZ up the joint estimate (n10/n4) reaches 2 frames either side."""
    from data2g.config import LEADIN_SAMPLES

    spec = SUBMODES["n10-qpsk-r1/2"]
    rng = np.random.default_rng(2)
    x = np.concatenate([np.zeros(2400), modem.modulate([rng.bytes(codes.payload_bytes(spec)) for _ in range(4)], spec),
                        np.zeros(2400)])
    y = hfchannel.apply_channel(x, snr_db=15, fading_preset="mpd", seed=3)
    modem._SC_WEIGHTS.clear()
    est = modem.receive(y, known=dict(start=2400 + LEADIN_SAMPLES, cfo=0.0, spec=spec, n_cw=4))["est"]
    if est["filter_spread_hz"] >= modem.SC_JOINT_REACH2_HZ:
        q = round(np.log2(est["filter_spread_hz"]) * 8)
        spans = [np.max(np.abs(np.frombuffer(k[7]) - 2.5)) for k in modem._SC_WEIGHTS if k[4] == q]
        assert spans and max(spans) <= 2 * modem.SYMS_PER_FRAME + 3


def test_w48_ace_closing_passes_bound_pep():
    """Scattered-pilot ACE modes restore pilots after projection, so two closing
    clip passes are needed to keep envelope PEP under control (w48-16qam)."""
    from data2g.arq import phy

    for name, ceiling in [("w48-16qam-r1/2", 5.15), ("w48-16qam-r2/3", 5.25), ("w48-16qam-r3/4", 5.95)]:
        assert len(SUBMODES[name].ace) >= 2
        assert phy.peak_db(name) <= ceiling

