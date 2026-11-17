"""scripts/fm_if_analyse.py on synthetic USB IF: a known FM signal up-converted
to the 12 kHz real IF the IC-9700 and TH-D75 give, with noise and carrier-off
stretches either side, with and without a radio-like 15 kHz IF filter."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from data2g import fmchannel, modem
from data2g.config import FS, PROFILE

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import fm_if_analyse as A  # noqa: E402
import sounding as S  # noqa: E402

pytestmark = pytest.mark.skipif(PROFILE != "fm", reason="FM's simulator and modes")
RF = fmchannel.rf_rate(FS)  # 48 kHz: a USB audio device's rate


def _if(x, radio, drive, filt, seed=0):
    """Audio -> (a 12 kHz real IF at RF, CNR 27 dB, 1 s of noise alone either side; the true deviation, Hz)."""
    m = x / np.sqrt(np.mean(x[np.abs(x) > 0] ** 2)) * drive
    dev = signal.resample_poly(fmchannel.tx_audio(m, fmchannel.RADIOS[radio], FS), RF // FS, 1)
    pad = np.zeros(RF)
    s = np.concatenate([pad, np.exp(2j * np.pi * np.cumsum(dev) / RF), pad])
    s[:RF] = s[-RF:] = 0
    rng = np.random.default_rng(seed)
    n0 = 1e-3 / fmchannel.CNR_REF_BW_HZ
    s = s + np.sqrt(n0 * RF / 2) * (rng.normal(size=len(s)) + 1j * rng.normal(size=len(s)))
    y = np.real(s * np.exp(2j * np.pi * A.CENTRE_HZ * np.arange(len(s)) / RF))
    if filt:  # the radio's FM IF filter: about 15 kHz, soft skirts
        y = signal.sosfilt(signal.butter(4, [4_500, 19_500], "bandpass", fs=RF, output="sos"), y)
    return y, dev


@pytest.mark.parametrize("radio,drive", [("mic", 1460.0), ("flat", 2350.0)])
def test_deviation_width_and_filter(radio, drive, monkeypatch):
    monkeypatch.setattr(modem, "TX_PATH", radio)
    mode = "f55-256l-r72" if radio == "mic" else "f108-256l-r87"
    x = S.sequence("W1AW", seed=1, reps=1, modes=[mode])[1][0]["audio"]  # the burst alone
    y, dev = _if(x, radio, drive, False)
    rep, _ = A.analyse(y, RF, tx_path=radio)
    seg = max(rep["segments"], key=lambda g: g["end_s"] - g["start_s"])
    true = fmchannel.occupied_bandwidth(np.exp(2j * np.pi * np.cumsum(dev) / RF), RF)
    assert abs(seg["db26_hz"] / true["db26"] - 1) < 0.05, (seg, true)
    assert abs(seg["rms_dev_hz"] / np.sqrt(np.mean(dev ** 2)) - 1) < 0.07, seg  # noise adds 1-5% at CNR 27
    assert abs(rep["centre_found_hz"] - A.CENTRE_HZ) < 100
    rf = A.analyse(_if(x, radio, drive, True)[0], RF, tx_path=radio)[0]
    assert 13_000 < rf["filter_db3_hz"] < 16_500, rf  # the radio's filter, seen in its noise


@pytest.mark.parametrize("radio", ["mic", "flat"])
def test_demodulated_audio_decodes(radio, monkeypatch):
    """Our own discriminator's audio, through the radio-like IF filter, decodes as a sounding."""
    monkeypatch.setattr(modem, "TX_PATH", radio)
    modes = ["f55-qpsk-r21", "f55-64l-r67"]
    x, _ = S.sequence("W1AW", seed=1, reps=1, modes=modes)
    y, _ = _if(x, radio, 1460.0 if radio == "mic" else 2350.0, True)
    _, audio = A.analyse(y, RF, tx_path=radio)
    rep = S.analyse(audio, FS, "W1AW", 1, 1, modes)
    assert all(r["cw_ok"] == r["n_cw"] for r in rep["bursts"]), [(r["submode"], r["cw_ok"], r["n_cw"]) for r in rep["bursts"]]
