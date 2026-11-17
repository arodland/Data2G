"""Native parity of the FM ping / level check (data2g.arq.ping, native/core/arq/ping)."""

import numpy as np
import pytest

from data2g import config, fmchannel, modem
from data2g.arq import ping as PG
from data2g.config import FS

pytestmark = pytest.mark.skipif(config.PROFILE != "fm", reason="FM only")

M = dict(snr_db=21.3, rms_dbfs=-18.2, peak_dbfs=-3.1, clip_permille=2, thirds_db=[24.0, 20.5, 15.0], dist_db=0.7,
         noise_tilt_db=2.5)


def test_wire_and_level(native, pure):
    for rep in (None, PG.pack_report(M)):
        body = PG.ping_body("N0CALL", "W1AW", rep, 55)
        assert native.ping.ping_body("N0CALL", "W1AW", rep, 55) == body
        assert native.ping.unpack_ping(body) == PG.unpack_ping(body)
    assert native.ping.pack_report(M) == PG.pack_report(M)
    got, want = native.ping.unpack_report(PG.pack_report(M)), PG.unpack_report(PG.pack_report(M))
    assert {k: list(v) if k == "thirds_db" else v for k, v in got.items()} == want
    x = np.random.default_rng(0).normal(0, 0.2, 5000)
    x[7] = 1.0
    assert native.ping.level(x) == pytest.approx(PG.level(x), rel=1e-12)
    assert native.ping.tx_level("mic") == PG.tx_level("mic") and native.ping.tx_level("flat") == 1.0
    assert native.ping.fmt(M) == PG.fmt(M)
    for far, path, band in ((M, "mic", "f55"), (dict(M, dist_db=3.0), "flat", "f55"), (dict(M, snr_db=8.2), "mic", "f108")):
        assert native.ping.verdict(far, path, band) == PG.verdict(far, path, band)


@pytest.mark.parametrize("radio, path", [("mic", "flat"), ("flat", "mic")])
def test_measure(native, pure, radio, path):
    from test_fm_ping import _ping

    from data2g.arq.phy import tx_audio

    b, payloads = _ping(1)
    x = np.concatenate([np.zeros(FS // 2), tx_audio(b, tx_path=path, level=1.0), np.zeros(FS // 2)])
    drive = {"mic": 1460.0, "flat": 2350.0}[radio] * PG.tx_level(path)
    y = fmchannel.FmChannel("static", radio, 20, seed=1, duration_s=len(x) / FS + 2, drive_dev_hz=drive).apply(x)
    r = modem.receive(y)
    lvl = PG.level(0.1 * y)
    want, got = PG.measure(r, payloads, lvl), native.ping.measure(r, payloads, lvl)
    for k in ("snr_db", "dist_db", "noise_tilt_db"):
        assert abs(got[k] - want[k]) < 1e-6, (k, got[k], want[k])
    np.testing.assert_allclose(got["thirds_db"], want["thirds_db"], atol=1e-6)
