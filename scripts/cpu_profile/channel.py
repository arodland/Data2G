"""Two-path Watterson fading from one loopback sink to another, unprofiled:
records <src>.monitor, plays the faded audio into <dst> (where noise.py's
noise joins it), both through pacat. hfchannel's Gaussian taps, one
continuous process for the whole run: generated once, circularly, over
PERIOD_S, so it repeats seamlessly rather than ending. Analytic signal by a
255-tap FIR Hilbert transformer (image < -58 dB above 300 Hz; data2g uses
350..2700 Hz). Taps are unit power in expectation, so noise.py's SNR is the
average SNR. Adds ~2.6 ms of filter delay plus the pacat buffering.
    channel.py <mpg|mpp|mpd|mps | doppler_hz:delay_ms> <seed> <src sink> <dst sink>
    channel.py check   # self-check of the DSP, no audio devices
"""
import subprocess
import sys

import numpy as np
from scipy import signal

from data2g.hfchannel import FADING_PRESETS

FS, BLK, PERIOD_S = 48000, 960, 3600


class Fader:
    def __init__(self, dop: float, dly_ms: float, seed: int):
        rng = np.random.default_rng(seed)
        self.rate = max(64 * dop, 8.0)
        self.n_low = n_low = int(PERIOD_S * self.rate)
        f = np.fft.fftfreq(n_low, 1 / self.rate)
        shape = np.exp(-(f**2) / (4 * (dop / 2) ** 2))
        self.taps = [np.append(g, g[0]) for g in  # interpolate across the wrap
                     (np.fft.ifft(np.fft.fft(rng.normal(size=n_low) + 1j * rng.normal(size=n_low)) * shape)
                      / np.sqrt(2 * np.mean(shape**2)) for _ in range(2))]
        self.hil = -signal.remez(255, [300, 23700], [1], type="hilbert", fs=FS)  # remez: cos -> -sin
        self.zi = np.zeros(len(self.hil) - 1)
        self.xhist = np.zeros(len(self.hil) // 2)  # the FIR's group delay, to align the real part
        self.zhist = np.zeros(int(round(dly_ms * 1e-3 * FS)), complex)  # path 2's delay
        self.n = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        im, self.zi = signal.lfilter(self.hil, 1.0, x, zi=self.zi)
        xb = np.concatenate([self.xhist, x])
        self.xhist = xb[len(x):]
        z1 = xb[:len(x)] + 1j * im
        zb = np.concatenate([self.zhist, z1])
        self.zhist = zb[len(x):]
        z2 = zb[:len(x)]
        tl = ((self.n + np.arange(len(x))) * self.rate / FS) % self.n_low
        i = tl.astype(int)
        fr = tl - i
        g1, g2 = (g[i] * (1 - fr) + g[i + 1] * fr for g in self.taps)
        self.n += len(x)
        return np.real((z1 * g1 + z2 * g2) / np.sqrt(2))


def check():
    """Tone through mpp-like fading: unit mean power, fades present."""
    fd = Fader(20.0, 2.0, 1)
    x = 0.5 * np.cos(2 * np.pi * 1500 * np.arange(FS * 60) / FS)
    y = np.concatenate([fd(b) for b in np.split(x, 60 * FS // BLK)])[FS:]
    p = np.mean(y**2) / np.mean(x**2)
    env = np.sqrt(np.convolve(y**2, np.ones(480) / 480, "valid") * 2) / 0.5
    print(f"power ratio {p:.3f}, envelope 5/50/95%: {np.percentile(env, [5, 50, 95]).round(2)}")
    assert 0.85 < p < 1.15 and np.percentile(env, 5) < 0.3


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check()
        sys.exit()
    spec, seed, src, dst = sys.argv[1:5]
    dop, dly = (map(float, spec.split(":")) if ":" in spec else
                (FADING_PRESETS[spec].doppler_hz, FADING_PRESETS[spec].delay_ms))
    fade = Fader(dop, dly, int(seed))
    rec = subprocess.Popen(["pacat", "--record", f"--device={src}.monitor", "--format=float32le", "--rate=48000",
                            "--channels=1", "--latency-msec=20"], stdout=subprocess.PIPE)
    play = subprocess.Popen(["pacat", "--playback", f"--device={dst}", "--format=float32le", "--rate=48000",
                             "--channels=1", "--latency-msec=40"], stdin=subprocess.PIPE)
    while b := rec.stdout.read(BLK * 4):
        play.stdin.write(fade(np.frombuffer(b, np.float32).astype(np.float64)).astype(np.float32).tobytes())
        play.stdin.flush()
