"""Scattered pilots (DATA2G_PILOTS=k3d2, config.SC_BANDS): the study flag on."""

import numpy as np
import pytest

from data2g import codes, hfchannel, modem
from data2g.arq import phy
from data2g.config import SUBMODES


LAYOUTS = {"k3d2": ("w", "w48"), "gold": ("w", "w48", "n10", "n4"), "mix": ("w", "w48", "n10", "n4")}


def _layout(monkeypatch, layout):
    """The layout, patched in unless DATA2G_PILOTS set it (a patched one keeps --native's calls in Python)."""
    if modem.SC_LAYOUT != layout:
        monkeypatch.setattr(modem, "SC_BANDS", LAYOUTS[layout])
        monkeypatch.setattr(modem, "SC_LAYOUT", layout)


@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("name,n_cw", [("qpsk-r1/2", 1), ("16qam-r1/2", 3), ("w48-16qam-r1/2", 4), ("n10-qpsk-r1/2", 2),
                                       ("n4-qpsk-r1/2", 1), ("n10-16qam-r1/2", 3), ("ack-1f", 1)])
def test_round_trip(monkeypatch, layout, name, n_cw):
    _layout(monkeypatch, layout)
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


def test_the_header_does_not_read_as_its_own_copy(monkeypatch):
    """A scattered-pilot copy frame closes on copy_close, not the pilot: closed on the pilot it had the first
    header's own shape (the preamble's last repeat, the header, the header pilot), so a preamble hypothesis two
    frames and a symbol early read the real header as its copy and held the receiver (test_receiver's false
    lock on data, seed 0)."""
    from data2g.config import BANDS, FRAME_SAMPLES, LEADIN_SAMPLES, NSYM
    from data2g.waveform import dsp

    _layout(monkeypatch, "mix")
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


def test_find_copy_on_a_band_without_copies(monkeypatch):
    """A scattered-pilot band whose bursts carry no header copy (n10): find_copy finds none. It read a copy band's
    frame there and raised on a strong tone (test_native_modem's pair-products signals)."""
    from data2g.config import FS

    _layout(monkeypatch, "mix")
    n = 3 * FS
    y = np.sin(2 * np.pi * 1510.0 * np.arange(n) / FS) + np.random.default_rng(7).normal(size=n)
    assert modem.find_copy(y, "n10") is None


def test_a_copy_lock_reads_its_phase_reference_off_copy_open(monkeypatch):
    """receive(copy=...) takes the timing tracker's phase reference from the copy frame's first symbol, which on a
    scattered-pilot band is copy_open: divided by the pilot instead, the reference was off by the pilots' own phases,
    and the tracker stepped the window 8 samples over a 3-codeword burst the preamble-locked receive held at 2."""
    _layout(monkeypatch, "mix")
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


@pytest.mark.parametrize("tone_hz", [2200.0, 825.0])
def test_a_tone_raises_its_own_carriers_noise_alone(monkeypatch, tone_hz):
    """A steady tone 25 dB over the floor in 50 Hz (interference.PRESETS["carriers"]' level), on a carrier (2200 Hz)
    or between two (825 Hz): the band's noise stays the floor's, the tone's carriers get their own, and the burst
    decodes. Summed into one excess the tone told the decoder ~7 dB too much noise on every carrier (2200 Hz:
    sc_tone 0.42 of codewords); in the preamble's noise (825 Hz) it put the pilots' power under it (0.53)."""
    from data2g.config import FS, LEADIN_SAMPLES

    _layout(monkeypatch, "mix")
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
