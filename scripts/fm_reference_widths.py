"""How wide existing 2 m FM modes are, by the yardsticks a bandwidth target
could use (FM plan Q15), on M1's RF path:

- db26: the outermost points 26 dB under the spectrum's peak (the usual
  reading of Part 97's 97.3(a)(8), at a 100 Hz and a 300 Hz resolution);
- pr26: the band outside which the power is 26 dB under the power inside
  (the literal reading of 97.3(a)(8): 99.75% of the power inside);
- obw99: ITU's occupied bandwidth (99% inside).

References: FM voice (mic path, 5 kHz limiter) at a normal and a loud drive;
1200 baud AFSK (Bell 202 tones into the mic path, 3 kHz peak); 9600 baud
G3RUH (flat, 3 kHz peak); Data2G FM's w48 modes at the drive giving a 15 kHz db26.

Run: DATA2G_PROFILE=fm python scripts/fm_reference_widths.py
"""

import numpy as np
from scipy import signal

from data2g import codes, config, fmchannel, modem

FS = config.FS
RF = fmchannel.rf_rate(FS)
RNG = np.random.default_rng(1)
SECONDS = 20


def widths(s: np.ndarray) -> dict:
    out = {}
    for rbw in (100, 300):
        w = fmchannel.occupied_bandwidth(s, RF, rbw_hz=rbw)
        out[f"db26@{rbw}"] = w["db26"]
        out["obw99"] = w["obw99"]
    f, p = signal.welch(s, RF, nperseg=1 << 13, return_onesided=False, detrend=False)
    o = np.argsort(f)
    f, c = f[o], np.cumsum(p[o]) / p.sum()
    out["pr26"] = float(f[np.searchsorted(c, 1 - 0.00125)] - f[np.searchsorted(c, 0.00125)])
    return out


def fm(dev_rf: np.ndarray) -> np.ndarray:
    """Deviation (Hz, at RF) -> complex baseband."""
    return np.exp(2j * np.pi * np.cumsum(dev_rf) / RF)


def via_radio(dev_audio: np.ndarray, radio: str) -> tuple[np.ndarray, float]:
    """Audio as Hz of deviation through the TX radio (pre-emphasis, limiter,
    band) -> RF, and the peak deviation reached."""
    y = fmchannel.tx_audio(dev_audio, fmchannel.RADIOS[radio], FS)
    return fm(signal.resample_poly(y, RF // FS, 1)), float(np.percentile(np.abs(y), 99.9))


def scale_to_peak(x: np.ndarray, radio: str, peak_hz: float) -> np.ndarray:
    """x scaled so the TX radio's output peaks (99.9th percentile) at peak_hz."""
    lo, hi = 1e-3, 1e4
    for _ in range(50):
        g = np.sqrt(lo * hi)
        y = fmchannel.tx_audio(x * g, fmchannel.RADIOS[radio], FS)
        lo, hi = (g, hi) if np.percentile(np.abs(y), 99.9) < peak_hz else (lo, g)
    return x * lo


def voice() -> np.ndarray:
    """Speech's long-term spectrum, roughly: noise, 300-3000 Hz, falling
    6 dB/octave above 500 Hz."""
    x = RNG.standard_normal(FS * SECONDS)
    b, a = signal.butter(4, [300, 3000], "bandpass", fs=FS)
    x = signal.lfilter(b, a, x)
    b, a = signal.butter(1, 500, fs=FS)
    return signal.lfilter(b, a, x)


def afsk1200() -> np.ndarray:
    bits = RNG.integers(0, 2, 1200 * SECONDS)
    f = np.repeat(np.where(bits, 1200.0, 2200.0), FS // 1200 + 1)[: FS * SECONDS]
    return np.sin(2 * np.pi * np.cumsum(f) / FS)


def g3ruh9600(peak_hz: float = 3000.0) -> np.ndarray:
    """9600 baud NRZ, raised-cosine shaped (alpha 0.5), straight to the
    modulator (a 9600 port), made at RF rate: FS can't hold its spectrum."""
    sps = 5
    rate = 9600 * sps
    bits = 2.0 * RNG.integers(0, 2, 9600 * SECONDS) - 1
    up = np.zeros(len(bits) * sps)
    up[::sps] = bits
    t = (np.arange(-8 * sps, 8 * sps + 1)) / sps
    h = np.sinc(t) * np.cos(np.pi * 0.5 * t) / (1 - (2 * 0.5 * t) ** 2 + 1e-12)
    x = np.convolve(up, h, "same")
    x = signal.resample_poly(x, RF, rate)
    return x / np.percentile(np.abs(x), 99.9) * peak_hz


def data2g(name: str) -> np.ndarray:
    spec = config.SUBMODES[name]
    n = max(1, 28 // spec.frames_per_cw)
    x = modem.modulate([RNG.bytes(codes.payload_bytes(spec)) for _ in range(n)], spec)
    return np.tile(x, int(np.ceil(SECONDS * FS / len(x))))[: SECONDS * FS]


def row(name: str, s: np.ndarray, note: str):
    w = widths(s)
    print(f"| {name} | {note} | " + " | ".join(f"{w[k] / 1000:.1f}" for k in ("db26@100", "db26@300", "pr26", "obw99")) + " |")


print("| signal | drive | db26 @100 Hz | db26 @300 Hz | pr26 (literal) | obw99 |")
print("|---|---|---|---|---|---|")
v = voice()
s, pk = via_radio(scale_to_peak(v, "mic", 4000.0), "mic")
row("FM voice", s, f"normal, {pk / 1000:.1f} kHz peak")
s, pk = via_radio(scale_to_peak(v, "mic", 4000.0) * 4, "mic")
row("FM voice", s, f"loud (+12 dB, limiter), {pk / 1000:.1f} kHz peak")
a = afsk1200()
for p in (3000.0, 4000.0):
    s, pk = via_radio(scale_to_peak(a, "mic", p), "mic")
    row("1200 AFSK (mic input)", s, f"{pk / 1000:.1f} kHz peak")
for p in (3000.0, 4000.0):
    row("9600 G3RUH (9600 port)", fm(g3ruh9600(p)), f"{p / 1000:.1f} kHz peak")
for name in ("w48-qpsk-r1/2", "w48-16qam-r1/2"):
    x = data2g(name)
    for radio in ("mic", "flat"):
        d, _ = fmchannel.drive_for_bandwidth(x, radio, 15_000.0, "db26", FS)
        s, rf = fmchannel.rf_signal(x, radio, d, FS)
        row(f"Data2G {name} ({radio})", s, f"{d:.0f} Hz rms, db26 15 kHz")
