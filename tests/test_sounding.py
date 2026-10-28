"""scripts/sounding.py: a short sequence through hfchannel, decoded back."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile

from data2g import cwid, hfchannel
from data2g.config import FS
from data2g.waveform import dsp

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))  # as the other tests reach scripts/
import sounding as S  # noqa: E402

MODES = ["qpsk-r1/2", "n10-qpsk-r1/2", "w48-qpsk-r1/3", "n4-ack-2f", "fsk32r62-r1/2"]


def _burst_snr(bursts, snr_db):
    """Each burst's SNR when noise is set for unit power at snr_db (its peak-1.0 audio's active power)."""
    return {b["n"]: snr_db + 10 * np.log10(dsp.active_power(b["audio"] / np.max(dsp.envelope(b["audio"]))))
            for b in bursts}


@pytest.mark.parametrize("snr", [3.0, 12.0])
def test_awgn_round_trip(snr):
    x, bursts = S.sequence("W1AW", seed=1, reps=1, modes=MODES)
    y = hfchannel.awgn(x, snr, seed=7, s_power=1.0)
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
        assert rep["path"]["channel"] == "awgn"


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


def test_kiwi_like_recording(tmp_path):
    """12 kHz stereo float WAV, 37 Hz off, the sequence 3.3 s in after noise."""
    x, bursts = S.sequence("W1AW", seed=3, reps=1, modes=MODES)
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(0, 0.05, int(3.3 * FS)), x])
    y = hfchannel.awgn(hfchannel.freq_shift(x, 37.0), 15.0, seed=6, s_power=1.0)
    y12 = signal.resample_poly(y, 3, 2) * 0.3
    wavfile.write(tmp_path / "rec.wav", 12000, np.stack([y12, y12[::-1]], axis=1).astype(np.float32))
    x0, fs0 = S.read_wav(tmp_path / "rec.wav")
    assert fs0 == 12000
    rep = S.analyse(x0, fs0, "W1AW", 3, 1, MODES)
    assert rep["offset_s"] == pytest.approx(3.3, abs=0.02)
    assert [r["n"] for r in rep["bursts"] if r["heard"]] == [b["n"] for b in bursts]
    assert all(r["cw_ok"] == r["n_cw"] for r in rep["bursts"])
    assert all(abs(r["cfo_hz"] - 37.0) < 2 for r in rep["bursts"] if r["family"] != "cpm")
    assert not rep["extra"]
