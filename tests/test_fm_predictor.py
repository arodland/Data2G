"""The FM gear shifter's predictor (arq.predictor.fm_outcome) and its regime
input (arq.phy.n0_spread_db): what they do, and native parity (FM profile)."""

import numpy as np
import pytest

from data2g import codes, config, fmchannel, modem
from data2g.arq import phy as PHY
from data2g.arq import predictor as P

pytestmark = pytest.mark.skipif(config.PROFILE != "fm", reason="FM profile only")


def test_steady_noise_uses_the_static_threshold_and_fading_the_mobile_one():
    name = "f55-64l-r67"
    static, _, mobile, _ = P._FM_THR[name]
    assert P.fm_outcome(static, name, "f55", 0.0) == pytest.approx(0.9)
    assert P.fm_outcome(mobile, name, "f55", 2.0) == pytest.approx(0.9)
    # steady: a steeper waterfall than fading's
    assert P.fm_outcome(static - 2, name, "f55", 0.0) < 0.4 < 0.7 < P.fm_outcome(mobile - 2, name, "f55", 2.0)


def test_a_reading_on_another_band_moves_by_the_band_offsets():
    name = "f108-qpsk-r41"
    t = P._FM_THR[name][0]
    off = P.FM_BAND_OFFSET_DB
    # n10 and f108 read alike on the same channel; w about 4.7 dB higher than f108
    t -= P.FM_STEADY_OFFSET_DB.get("f108", 0.0)  # steady noise: f108 from another band reads that much lower
    assert P.fm_outcome(t, name, "n10") == pytest.approx(0.9)
    assert P.fm_outcome(t + off["w"] - off["f108"], name, "w") == pytest.approx(0.9)


def _f55_received(cond="mobile-urban", seed=3):
    spec = config.SUBMODES["f55-qpsk-r40"]
    rng = np.random.default_rng(seed)
    sent = [rng.bytes(codes.payload_bytes(spec)) for _ in range(24 // spec.frames_per_cw)]
    x = np.concatenate([np.zeros(config.FS // 2), modem.modulate(sent, spec, tx_path="mic"), np.zeros(config.FS)])
    ch = fmchannel.FmChannel(cond, "mic", 22.0, seed=seed, duration_s=len(x) / config.FS + 1, drive_dev_hz=1460.0)
    return modem.receive(ch.apply(x))


def test_the_noise_spread_tells_a_fading_path_from_a_steady_one():
    assert PHY.measure(_f55_received("static"))["n0_spread_db"] < 0.7 <= PHY.measure(_f55_received())["n0_spread_db"]


def test_native_measure_and_predictor_match(native, pure):
    r = _f55_received()
    want, got = PHY.measure(r), native.phy.measure(r)
    assert got["n0_spread_db"] == pytest.approx(want["n0_spread_db"], rel=1e-6, abs=1e-6)
    for snr in (0.0, 9.5, 17.25, 30.0):
        for spread in (0.0, 0.99, 1.0, 3.0):
            for band in ("w", "n10", "f55", "f108"):
                m = dict(want, snr_est=snr, n0_spread_db=spread)
                py = P.predict_outcome(m, band, 0.5, 3.0)
                nat = native.arq.predict_outcome(m, band, 0.5, 3.0, None)
                for k, v in py.items():
                    assert nat[k][0] == pytest.approx(v[0], rel=1e-9, abs=1e-12), (k, snr, spread, band)



def test_f108_from_another_band_is_a_db_lower_on_steady_noise_only():
    """predictor.FM_STEADY_OFFSET_DB: f55 readings overstate f108 by about 1 dB at
    low CNR on static paths; mobile has its own column."""
    name = "f108-qpsk-r41"
    t, mobile = P._FM_THR[name][0], P._FM_THR[name][2]
    shift = P.FM_BAND_OFFSET_DB["f108"] - P.FM_BAND_OFFSET_DB["f55"]
    assert P.fm_outcome(t - shift + 1.0, name, "f55", 0.0) == pytest.approx(0.9)
    assert P.fm_outcome(mobile - shift, name, "f55", 2.0) == pytest.approx(0.9)
    assert P.fm_outcome(t, name, "f108", 0.0) == pytest.approx(0.9)  # its own band's reading: as is


def test_fm_data_mer_drops_distorted_and_rolled_off_f108_but_keeps_healthy_flat():
    """Issue #290: on FM flat f108, healthy CNR 32 still rates f108-256l-r87 >= 0.9,
    while TX soft-clipping or low-frequency rolloff drops it well below 0.5."""
    from scipy import signal

    spec = config.SUBMODES["f108-256l-r87"]
    rng = np.random.default_rng(11)
    sent = [rng.bytes(codes.payload_bytes(spec)) for _ in range(8)]
    x = modem.modulate(sent, spec, tx_path="flat")
    pad = np.zeros(config.FS // 2)

    def eval_tx(tx_wave):
        y = np.concatenate([pad, tx_wave, pad])
        ch = fmchannel.FmChannel("static", "flat", 32.0, seed=5, duration_s=len(y) / config.FS + 1, drive_dev_hz=2000.0)
        r = modem.receive(ch.apply(y))
        m0 = PHY.measure(r)
        rx = PHY.ModemRx(r, {}, dd_budget=0.0)
        for s in range(r["n_cw"]):
            rx.converged(s)
        m1 = rx.measure_data(m0) or m0
        return m0, m1, P.predict_outcome(m1, "f108", 2.5, 6.0)

    _, _, p_ok = eval_tx(x)
    assert p_ok["f108-256l-r87"][0] >= 0.9

    # TX peak clipping: low-PAPR pilots (~1.4x RMS) stay >29 dB, data symbols clip
    rms = float(np.sqrt(np.mean(x[np.abs(x) > 0] ** 2)))
    m0_c, m1_c, p_clip = eval_tx(np.clip(x, -2.0 * rms, 2.0 * rms))
    assert m0_c["snr_est"] > 29.0
    assert m1_c["mer_db"] < m0_c["snr_est"] - 3.0
    assert p_clip["f108-256l-r87"][0] < 0.4

    # Severe LF rolloff below 1.2 kHz: pilot snr_est stays high, MIESM drops f108-256l-r87
    sos = signal.butter(4, 1200.0, "highpass", fs=config.FS, output="sos")
    m0_l, _, p_lfr = eval_tx(signal.sosfilt(sos, x))
    assert m0_l["snr_est"] > 28.0
    assert p_lfr["f108-256l-r87"][0] < 0.4
    assert p_lfr["f108-64l-r75"][0] > 0.8
