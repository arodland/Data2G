"""FM transmit settings measured after the provisional ladder, as one opt-in
switch for the ladder screen and the streamed harness (FM_TX=1; off: today's TX):

- flat radio: a fixed +3 dB/octave TX tilt (scripts/fm_tilt_study.tilt_fir,
  after TX conditioning, rms kept), drive re-picked for 15 kHz 26 dB down:
  1.4-2.9 dB on f108 (5d82823);
- flat radio, 16-QAM and up: no TX clipping (scripts/fm_clip_study.py,
  cbf586f): clipping buys no deviation on FM, 1.2-1.4 dB on 64/256-point at
  CNR 16; the receiver keeps its spec's clip constants (6 dB, a slight
  over-estimate of clip noise, as that study);
- mic radio: the drive set by the radio's limiter, not the width: 64-point
  at 70%, 256-point at 55% of the width-limited drive (limiter active under
  1%; BMI more than doubles); 16-QAM and QPSK unchanged.
"""

import os

import numpy as np

ON = os.environ.get("FM_TX") == "1"
TILT_DB_OCT = 3.0
MIC_DRIVE = {6: 0.70, 8: 0.55}  # bits per cell -> share of the width-limited drive
NOCLIP_HEADROOM = 40.0  # an envelope clip level the burst never reaches


def noclip(radio: str, bits: int) -> bool:
    return ON and radio == "flat" and bits >= 4


def shape(x: np.ndarray, band: str, radio: str) -> np.ndarray:
    """The transmitted audio after the settings' TX tilt (flat), rms kept."""
    if not (ON and radio == "flat"):
        return x
    import fm_tilt_study as T

    from data2g import config

    act = np.abs(x) > 1e-9
    y = np.convolve(x, T.tilt_fir(band, TILT_DB_OCT, config.FS), "same")
    return y * np.sqrt(np.mean(x[act] ** 2) / np.mean(y[act] ** 2))


def drive_scale(radio: str, bits: int) -> float:
    return MIC_DRIVE.get(bits, 1.0) if ON and radio == "mic" else 1.0
