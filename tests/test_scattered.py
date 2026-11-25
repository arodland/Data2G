"""Scattered pilots (DATA2G_PILOTS=k3d2, config.SC_BANDS): the study flag on."""

import numpy as np
import pytest

from data2g import codes, hfchannel, modem
from data2g.arq import phy
from data2g.config import SUBMODES


LAYOUTS = {"k3d2": ("w", "w48"), "gold": ("w", "w48", "n10", "n4"), "mix": ("w", "w48", "n10", "n4")}


@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("name,n_cw", [("qpsk-r1/2", 1), ("16qam-r1/2", 3), ("w48-16qam-r1/2", 4), ("n10-qpsk-r1/2", 2),
                                       ("n4-qpsk-r1/2", 1), ("n10-16qam-r1/2", 3), ("ack-1f", 1)])
@pytest.mark.python_only("scattered pilots (DATA2G_PILOTS) are not in native")
def test_round_trip(monkeypatch, layout, name, n_cw):
    monkeypatch.setattr(modem, "SC_BANDS", LAYOUTS[layout])
    monkeypatch.setattr(modem, "SC_LAYOUT", layout)
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


@pytest.mark.python_only("scattered pilots")
def test_the_header_does_not_read_as_its_own_copy(monkeypatch):
    """A scattered-pilot copy frame closes on copy_close, not the pilot: closed on the pilot it had the first
    header's own shape (the preamble's last repeat, the header, the header pilot), so a preamble hypothesis two
    frames and a symbol early read the real header as its copy and held the receiver (test_receiver's false
    lock on data, seed 0)."""
    from data2g.config import BANDS, FRAME_SAMPLES, LEADIN_SAMPLES, NSYM
    from data2g.waveform import dsp

    monkeypatch.setattr(modem, "SC_BANDS", ("w", "w48", "n10", "n4"))
    monkeypatch.setattr(modem, "SC_LAYOUT", "mix")
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
