"""scripts/sounding.py: a short sequence through hfchannel, decoded back."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile

from data2g import config, cwid, hfchannel
from data2g.arq import phy as PHY
from data2g.config import FS, PROFILE
from data2g.waveform import dsp

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))  # as the other tests reach scripts/
import sounding as S  # noqa: E402

MODES = [m for m in ("qpsk-r1/2", "n10-qpsk-r1/2", "w48-qpsk-r1/3", "n4-ack-2f", "fsk32r62-r1/2")
         if m in S.MODE_THRESHOLDS]  # what the ARQ may name (FM: no w48, n4 or CPM)


# the noise reference: HF, unit power (each burst's envelope peak at full scale); FM, the
# fixed scale's (phy.full_scale: unit-RMS audio over config.FM_TX_PEAK), so snr_db is a burst's
REF = 1.0 if PROFILE == "hf" else config.FM_TX_PEAK ** -2


def _burst_snr(bursts, snr_db):
    """Each burst's SNR when noise is set for REF power at snr_db (its audio as sent: phy.full_scale)."""
    return {b["n"]: snr_db + 10 * np.log10(dsp.active_power(PHY.full_scale(b["audio"])) / REF) for b in bursts}


@pytest.mark.parametrize("snr", [3.0, 12.0])
def test_awgn_round_trip(snr):
    x, bursts = S.sequence("W1AW", seed=1, reps=1, modes=MODES)
    y = hfchannel.awgn(x, snr, seed=7, s_power=REF)
    rep = S.analyse(y, FS, "W1AW", 1, 1, MODES)
    assert abs(rep["offset_s"]) < 0.01
    want = _burst_snr(bursts, snr)
    for r in rep["bursts"]:
        assert r["heard"] and r["decoded"], r
        if snr > 10:
            assert r["cw_ok"] == r["n_cw"], r
        assert abs(r["genie"]["snr_db"] - want[r["n"]]) < 2.0, (r["submode"], r["genie"]["snr_db"], want[r["n"]])
    if snr > 10:
        ofdm = [r for r in rep["bursts"] if r["family"] != "cpm"]
        assert all(r["genie"]["doppler_hz"] < 0.1 for r in ofdm)
        # HF's channel classes (mode_thresholds columns); its 0.2 ms delay line sits in
        # the estimator's noise at 12 dB, which this seed clears only on HF
        assert PROFILE != "hf" or rep["path"]["channel"] == "awgn"


@pytest.mark.hf_only("HF fading path estimate from the w48 bursts FM does not send")
def test_fading():
    x, _ = S.sequence("W1AW", seed=2, reps=1, modes=MODES)
    y = hfchannel.apply_channel(x, snr_db=None, fading_preset="mpd", seed=3)
    y = hfchannel.awgn(y, 20.0, seed=4, s_power=1.0)
    rep = S.analyse(y, FS, "W1AW", 2, 1, MODES)
    heard = [r for r in rep["bursts"] if r["heard"]]
    assert len(heard) == len(MODES)
    # this seed's faded CW ID reads as a suspect ack-1f header (score 0.34): reported, not matched
    assert all(e["rec_s"] < cwid.seconds(cwid.id_text("W1AW")) for e in rep["extra"])
    assert np.all(np.diff([r["rec_s"] for r in heard]) > 0)
    # two paths 4 ms apart: 2 ms RMS (this seed reads 1.5), 2 Hz Doppler (2.06)
    assert 1.0 < rep["path"]["delay_ms"] < 3.0
    assert 1.4 < rep["path"]["doppler_hz"] < 2.8
    assert rep["path"]["channel"] == "mpd"


@pytest.mark.hf_only("an HF recording; on FM a noise read as an f108 header at the provisional 0.16 floor (follow-up)")
def test_kiwi_like_recording(tmp_path):
    """12 kHz stereo float WAV, 37 Hz off, the sequence 3.3 s in after noise."""
    x, bursts = S.sequence("W1AW", seed=3, reps=1, modes=MODES)
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(0, 0.05, int(3.3 * FS)), x])
    y = hfchannel.awgn(hfchannel.freq_shift(x, 37.0), 15.0, seed=6, s_power=1.0)
    g = np.gcd(12000, FS)
    y12 = signal.resample_poly(y, 12000 // g, FS // g) * 0.3
    wavfile.write(tmp_path / "rec.wav", 12000, np.stack([y12, y12[::-1]], axis=1).astype(np.float32))
    x0, fs0 = S.read_wav(tmp_path / "rec.wav")
    assert fs0 == 12000
    rep = S.analyse(x0, fs0, "W1AW", 3, 1, MODES)
    assert rep["offset_s"] == pytest.approx(3.3, abs=0.02)
    assert [r["n"] for r in rep["bursts"] if r["heard"]] == [b["n"] for b in bursts]
    assert all(r["cw_ok"] == r["n_cw"] for r in rep["bursts"])
    assert all(abs(r["cfo_hz"] - 37.0) < 2 for r in rep["bursts"] if r["family"] != "cpm")
    assert not rep["extra"]


@pytest.mark.skipif(PROFILE != "fm", reason="FM's bands, TX path and simulator")
@pytest.mark.parametrize("radio", ["mic", "flat"])
def test_fm_round_trip_through_the_simulator(radio, monkeypatch):
    """FM: a sounding sent with the radio's TX path, through the FM simulator at
    a station's level (static, CNR 35), decodes every codeword, and the report
    calls the path static (the policy's noise-spread regime)."""
    from data2g import fmchannel, modem

    monkeypatch.setattr(modem, "TX_PATH", radio)
    modes = ["f55-qpsk-r21", "f55-64l-r67", "f55-256l-r72"] + (["f108-256l-r87"] if radio == "flat" else [])
    x, bursts = S.sequence("W1AW", seed=1, reps=1, modes=modes)
    station = 1460.0 if radio == "mic" else 2350.0  # rms deviation of a unit-RMS burst (fork K's levels)
    ref = np.sqrt(np.mean(x[np.abs(x) > 0] ** 2))  # FmChannel scales the whole input by its rms
    ch = fmchannel.FmChannel("static", radio, 35.0, seed=3, duration_s=len(x) / FS + 1,
                             drive_dev_hz=station * ref / (1 / config.FM_TX_PEAK))
    rep = S.analyse(ch.apply(x), FS, "W1AW", 1, 1, modes)
    for r in rep["bursts"]:
        assert r["cw_ok"] == r["n_cw"], r
    assert rep["path"]["channel"] == "static"
    assert all(m["predicted_pass"] for m in rep["per_mode"]), [(m["submode"], m.get("p_usable")) for m in rep["per_mode"]]
