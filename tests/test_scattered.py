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
    assert ("sc" in r) == modem.sc_spec(spec)
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
