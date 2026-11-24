"""Native parity of the FM ping and its sounding probe (data2g.arq.ping, data2g.arq.probe; native/core/arq/ping, probe)."""

import numpy as np
import pytest

from data2g import config, fmchannel
from data2g.arq import ping as PG
from data2g.arq import probe as PR
from data2g.config import FS

pytestmark = pytest.mark.skipif(config.PROFILE != "fm", reason="FM only")

M = dict(lo_hz=300.0, hi_hz=3350.0, snr_nb=24.5, snr_top=-11.0, snr_tilt=0.5, noise_tilt=-1.0,
         snr=[36.0, 34.5, 29.0, 23.0, 17.0, 34.0], sinad=[14.5, 30.0, 35.0, 28.0, 22.5, 30.0],
         rms_dbfs=-21.5, peak_dbfs=-8.0, clip_permille=0)
REPORTS = [M, dict(M, hi_hz=6050.0, sinad=[12.5, 12.5, 30.5, 34.5, 30.0, 12.5]),
           dict(M, sinad=[38.0, 34.0, 28.5, 22.0, 16.5, 34.0]), dict(M, sinad=[14.5, 22.0, 35.0, 28.0, 22.5, 27.0]),
           dict(M, hi_hz=0.0), dict(M, clip_permille=3, snr_tilt=8.0, noise_tilt=4.5)]


def test_wire_and_level(native, pure):
    rep = PR.pack_report(M)
    for r in (None, rep):
        body = PG.ping_body("N0CALL", "W1AW", r)
        assert native.ping.ping_body("N0CALL", "W1AW", r) == body
        assert native.ping.unpack_ping(body) == PG.unpack_ping(body)
    x = np.random.default_rng(0).normal(0, 0.2, 5000)
    x[7] = 1.0
    assert native.ping.level(x) == pytest.approx(PG.level(x), rel=1e-12)
    assert native.ping.tx_level("mic") == PG.tx_level("mic") and native.ping.tx_level("flat") == 1.0


def test_report_and_verdict(native, pure):
    assert list(native.probe.tones()) == list(PR.TONES) and list(native.probe.holes()) == list(PR.HOLES)
    assert native.probe.n_samples() == PR.n_samples()
    for m in REPORTS:
        b = PR.pack_report(m)
        assert native.probe.pack_report(m) == b
        got, want = native.probe.unpack_report(b), PR.unpack_report(b)
        assert {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in got.items()} == want
        assert native.probe.fmt(want) == PR.fmt(want)
        assert native.probe.best_step(want) == PR.best_step(want)
        for path in ("mic", "flat"):
            assert native.probe.verdict(want, path) == PR.verdict(want, path)


@pytest.mark.parametrize("path", ["mic", "flat"])
def test_waveform(native, pure, path):
    want = PR.waveform(path, PG.tx_level(path))
    got = np.asarray(native.probe.waveform(path, PG.tx_level(path)))
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=1e-9)


@pytest.mark.parametrize("radio, path, hot", [("mic", "mic", 1.0), ("mic", "mic", 2.0), ("flat", "flat", 1.0)])
def test_measure(native, pure, radio, path, hot):
    x = np.concatenate([np.zeros(FS // 2), PR.waveform(path, 1.0), np.zeros(FS // 2)])
    drive = {"mic": 1460.0, "flat": 2350.0}[radio] * hot * PG.tx_level(path)
    y = fmchannel.FmChannel("static", radio, 25, seed=1, duration_s=len(x) / FS + 2, drive_dev_hz=drive).apply(x)
    y = 0.1 * y[FS // 2:FS // 2 + PR.n_samples()]
    lvl = PG.level(y)
    want, got = PR.summary(PR.analyse(y), lvl), native.probe.measure(y, lvl)
    for k in ("lo_hz", "hi_hz", "snr_nb", "snr_top", "snr_tilt", "noise_tilt"):
        assert abs(got[k] - want[k]) < 1e-6, (k, got[k], want[k])
    np.testing.assert_allclose(got["snr"], want["snr"], atol=1e-6)
    np.testing.assert_allclose(got["sinad"], want["sinad"], atol=1e-6)
    assert native.probe.pack_report(got) == PR.pack_report(want)


@pytest.mark.parametrize("heard", [0.0, 0.5])
def test_measure_exact_zeros(native, pure, heard):
    y = PR.waveform("flat", 1.0) + np.random.default_rng(0).normal(0, 0.01, PR.n_samples())
    y[int(heard * len(y)):] = 0.0
    lvl = PG.level(0.1 * y)
    assert native.probe.pack_report(native.probe.measure(y, lvl)) == PR.pack_report(PR.summary(PR.analyse(y), lvl))
