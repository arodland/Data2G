"""A radio's IF recording (e.g. an IC-9700's or TH-D75's USB IF: a real IF
at some centre frequency) read with our own FM discriminator: per stretch of
modulation, the deviation (rms and peak), the occupied bandwidth (26 dB
down, fmchannel.occupied_bandwidth's definition, and 99%) and the CNR; and
the demodulated audio as a 16 kHz WAV that `sounding decode` reads (FM
plan M7 bandwidth check, M10's first look at I/Q receive).

    DATA2G_PROFILE=fm python scripts/fm_if_analyse.py if.wav [--centre HZ] [--tx-path mic|flat] [--out PREFIX]

The IC-9700 and TH-D75 put the dial frequency at 12 kHz in their USB IF
audio (--centre's default; the spectrum's own centre is reported as a
check), behind the radio's FM IF filter, about 15 kHz wide with soft
skirts: noise is read where no carrier is (the gaps between overs, the
recording's head and tail), in band, and its shape is the radio's filter
(reported: its widths 3 and 26 dB down). A 26 dB width near 15 kHz is then
partly that filter, not the sender alone. --tx-path is the sender's: mic
de-emphasises the demodulated audio, flat leaves it flat.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from data2g import fmchannel
from data2g.config import FS

HALF_BW_HZ = 10_000.0  # the channel filter around the carrier: a 15 kHz signal and its skirts
# envelope std / mean per window: noise alone 0.52 (Rayleigh); a carrier with noise sqrt(1/(2 CNR))
# about, 0.3 at about 7 dB CNR in the channel filter
CV_NOISE, CV_CARRIER = 0.45, 0.3
WIN_S = 0.1  # analysis windows: carrier presence and modulation activity
ACTIVE = 0.3  # a window is modulated when its rms deviation is this share of the loudest window's
CENTRE_HZ = 12_000.0  # IC-9700 and TH-D75 USB IF: the dial frequency
IN_BAND_HZ = 3_000.0  # noise density read within this of the centre (inside any IF filter)
AUDIO_HZ = 7_000.0  # the deviation's statistics are read within this audio band


def read(path):
    fs, x = wavfile.read(path)
    x = x[:, 0] if x.ndim > 1 else x
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / -np.iinfo(x.dtype).min
    return x.astype(np.float64), fs


def find_centre(x: np.ndarray, fs: int) -> float:
    """The FM carrier's centre: the power-weighted mean frequency of the
    spectrum's part more than 10 dB over its median (the noise)."""
    f, p = signal.welch(x, fs, nperseg=min(len(x), 1 << 14))
    on = p > 10 * np.median(p)
    return float(np.sum(f[on] * p[on]) / np.sum(p[on]))


def _widths(f: np.ndarray, p: np.ndarray, ref: float) -> dict:
    """The contiguous width around 0 Hz where p stays within 3 and 26 dB of ref."""
    o = np.argsort(f)
    f, p = f[o], p[o]
    c = np.argmin(np.abs(f))
    out = {}
    for db in (3, 26):
        ok = p >= ref * 10 ** (-db / 10)
        lo = hi = c
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        while hi < len(f) - 1 and ok[hi + 1]:
            hi += 1
        out[f"filter_db{db}_hz"] = round(float(f[hi] - f[lo]))
    return out


def analyse(x: np.ndarray, fs: int, centre: float = CENTRE_HZ, tx_path: str = "mic") -> tuple[dict, np.ndarray]:
    """-> (report, demodulated audio at FS, scaled to Hz of deviation / 4000)."""
    fc = centre
    n = np.arange(len(x))
    zr = 2 * x * np.exp(-2j * np.pi * fc * n / fs)  # the IF to complex baseband (x2: the real signal's half)
    z = signal.sosfiltfilt(signal.butter(8, HALF_BW_HZ, "lowpass", fs=fs, output="sos"), zr)
    dev = np.concatenate([[0.0], np.angle(z[1:] * np.conj(z[:-1])) * fs / (2 * np.pi)])  # Hz
    # deviation statistics within the audio band (7 kHz, just over a 9600 port's 6): the
    # discriminator's noise above it inflated the peak (99.9th percentile) by 25-60% at CNR 27 on
    # synthetic IF; 6 kHz cut a tilted flat burst's rms by 5%
    dev_a = signal.sosfiltfilt(signal.butter(6, AUDIO_HZ, "lowpass", fs=fs, output="sos"), dev)
    w = int(WIN_S * fs)
    k = len(z) // w
    env = np.abs(z[:k * w].reshape(k, w))
    pw = np.mean(env ** 2, axis=1)
    # carrier or not by the envelope, not the power: an FM carrier's is constant, noise's is
    # Rayleigh (std / mean 0.52). By power, an IC-9700 IF capture with no carrier (open
    # squelch, 2026-11-17) read as one long modulated stretch: its noise is as loud as a weak signal
    cv = env.std(axis=1) / np.maximum(env.mean(axis=1), 1e-30)
    quiet = cv > CV_NOISE
    if np.any(quiet):  # a window straddling a burst's edge varies too: noise alone is also no louder than noise
        quiet &= pw < 2 * np.median(pw[quiet])
    noise_only = bool(np.any(quiet))
    n0, filt = float("nan"), {}
    if noise_only:
        zq = np.concatenate([zr[i * w:(i + 1) * w] for i in np.flatnonzero(quiet)])
        fz, pz = signal.welch(zq, fs, nperseg=min(len(zq), 1 << 12), return_onesided=False, detrend=False)
        n0 = float(np.median(pz[np.abs(fz) <= IN_BAND_HZ]))  # power per Hz, in band
        filt = _widths(fz, pz, n0)  # the radio's IF filter, seen in its noise
        carrier = cv < CV_CARRIER
    else:
        carrier = np.ones(k, dtype=bool)  # carrier throughout: no noise reading, no CNR
    rms = np.sqrt(np.mean(dev[:k * w].reshape(k, w) ** 2, axis=1))
    loud = rms[carrier].max() if np.any(carrier) else 0.0
    act = carrier & (rms > ACTIVE * loud)
    segs, i = [], 0
    while i < k:  # runs of modulated windows (a burst, or a stretch of one)
        if act[i]:
            j = i
            while j < k and act[j]:
                j += 1
            a, b = i * w, j * w
            d = dev_a[a:b]
            ob = fmchannel.occupied_bandwidth(z[a:b], fs)
            p = float(np.mean(np.abs(z[a:b]) ** 2))
            segs.append(dict(start_s=round(a / fs, 2), end_s=round(b / fs, 2),
                             rms_dev_hz=round(float(np.sqrt(np.mean(d ** 2))), 1),
                             peak_dev_hz=round(float(np.percentile(np.abs(d), 99.9)), 1),
                             db26_hz=round(ob["db26"]), obw99_hz=round(ob["obw99"]),
                             cnr_db=round(float(10 * np.log10(max(p - n0 * 2 * HALF_BW_HZ, 1e-30)
                                                              / (n0 * fmchannel.CNR_REF_BW_HZ))), 1)
                             if noise_only else None))
            i = j
        else:
            i += 1
    y = signal.resample_poly(dev, FS, fs) if fs % FS else signal.resample_poly(dev, 1, fs // FS)
    if tx_path == "mic":  # undo the sender's pre-emphasis, as its radio's receiver would
        b, a = fmchannel._emphasis(FS, 750.0, inverse=True)
        y = signal.lfilter(b, a, y)
    rep = dict(fs=fs, centre_hz=round(fc, 1), centre_found_hz=round(find_centre(x, fs), 1),
               noise_dbhz=round(float(10 * np.log10(n0)), 1) if n0 > 0 else None, **filt,
               tx_path=tx_path, segments=segs)
    return rep, y / 4000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--centre", type=float, default=CENTRE_HZ, help="the IF's centre (Hz; IC-9700, TH-D75: 12000)")
    ap.add_argument("--tx-path", choices=("mic", "flat"), default="mic", help="the sender's TX path")
    ap.add_argument("--out", help="output prefix (default: the input's, less .wav)")
    a = ap.parse_args()
    x, fs = read(a.wav)
    rep, y = analyse(x, fs, a.centre, a.tx_path)
    out = Path(a.out) if a.out else Path(a.wav).with_suffix("")
    wavfile.write(f"{out}.demod.wav", FS, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    Path(f"{out}.if.json").write_text(json.dumps(rep, indent=1))
    print(f"centre {rep['centre_hz']} Hz (spectrum's {rep['centre_found_hz']}), noise {rep['noise_dbhz']} dB/Hz, "
          f"radio IF filter {rep.get('filter_db3_hz')} / {rep.get('filter_db26_hz')} Hz wide (3 / 26 dB down), "
          f"{len(rep['segments'])} modulated stretches")
    print("start_s end_s rms_dev peak_dev db26 obw99 cnr")
    for s in rep["segments"]:
        print(s["start_s"], s["end_s"], s["rms_dev_hz"], s["peak_dev_hz"], s["db26_hz"], s["obw99_hz"], s["cnr_db"])
    print(f"wrote {out}.demod.wav, {out}.if.json")


if __name__ == "__main__":
    main()
