"""Two-path Watterson fading from one loopback sink to another, unprofiled:
records <src>.monitor, plays the faded audio into <dst> (where noise.py's
noise joins it), both through pacat. As ARSFI's HFSimulator (IONOS, firmware
2.03, github.com/ARSFI/HFSimulator) makes it: two equal paths, the second
delayed; each path's tap complex Gaussian noise through its 128-tap Gaussian
FIR (designed at 64 Hz: -9.1 dB at 1 Hz, so 2 sigma ~= the spread), clocked
at 64 x the spread and held between updates (MPG: every 156 ms). One
continuous process for the whole run, generated once, circularly, over
PERIOD_S, so it repeats seamlessly rather than ending. Analytic signal by a
255-tap FIR Hilbert transformer (image < -58 dB above 300 Hz; data2g uses
350..2700 Hz). Taps are unit power in expectation, so the average power is
the input's (noise.py references the input's peak, as HFSimulator does).
Adds ~2.6 ms of filter delay plus the pacat buffering.
    channel.py <mpg|mpp|mpd|mps | doppler_hz:delay_ms> <seed> <src sink> <dst sink>
    channel.py check   # self-check of the DSP, no audio devices
"""
import subprocess
import sys

import numpy as np
from scipy import signal

from data2g.hfchannel import FADING_PRESETS

FS, BLK, PERIOD_S = 48000, 960, 3600
# HFSimulator's Doppler filter (gaus_fir_coeffs: "128 Tap Adj Gauss LPF
# Rev2", Iowa Hills, 64 Hz), run at 64 x the spread
GAUSS_FIR = np.array([
    1.175559e-11, 2.0188e-10, 1.723633e-09, 9.815423e-09, 4.21982e-08, 1.469343e-07, 4.338504e-07, 1.122119e-06,
    2.604092e-06, 5.522713e-06, 1.085739e-05, 2.001141e-05, 3.489307e-05, 5.798248e-05, 9.237679e-05, 0.0001418077,
    0.0002106267, 0.0003037562, 0.0004266051, 0.0005849522, 0.0007847989, 0.001032198, 0.001333065, 0.001692977,
    0.002116971, 0.002609341, 0.003173461, 0.003811607, 0.004524824, 0.005312813, 0.00617385, 0.007104756,
    0.008100886, 0.009156172, 0.01026319, 0.01141329, 0.0125967, 0.0138027, 0.01501986, 0.01623617, 0.0174393,
    0.01861682, 0.01975639, 0.02084602, 0.02187425, 0.02283034, 0.02370443, 0.02448774, 0.02517264, 0.02575277,
    0.02622313, 0.02658011, 0.02682151, 0.02694653, 0.02695577, 0.02685111, 0.02663571, 0.02631388, 0.02589095,
    0.0253732, 0.02476771, 0.02408221, 0.02332495, 0.02250458, 0.02163, 0.02071021, 0.01975421, 0.01877087,
    0.01776882, 0.01675638, 0.01574142, 0.01473135, 0.013733, 0.01275264, 0.01179586, 0.01086765, 0.009972282,
    0.0091134, 0.008293975, 0.007516342, 0.006782226, 0.006092768, 0.005448573, 0.004849742, 0.004295926,
    0.003786367, 0.003319954, 0.002895265, 0.002510623, 0.002164137, 0.001853753, 0.001577294, 0.001332499,
    0.001117067, 0.0009286798, 0.0007650424, 0.0006239026, 0.0005030762, 0.0004004651, 0.0003140728, 0.0002420166,
    0.0001825359, 0.0001339988, 9.490464e-05, 6.388528e-05, 3.970379e-05, 2.125141e-05, 7.543009e-06,
    -2.288719e-06, -8.999993e-06, -1.324345e-05, -1.557613e-05, -1.646781e-05, -1.630935e-05, -1.54211e-05,
    -1.406102e-05, -1.243258e-05, -1.069219e-05, -8.956196e-06, -7.307342e-06, -5.800659e-06, -4.468779e-06,
    -3.326665e-06, -2.375753e-06, -1.607534e-06, -1.006599e-06, -5.531754e-07, -2.252074e-07,
])


class Fader:
    def __init__(self, dop: float, dly_ms: float, seed: int):
        rng = np.random.default_rng(seed)
        self.rate = 64 * dop  # tap updates per second
        self.n_low = n_low = int(PERIOD_S * self.rate)
        h = np.fft.fft(GAUSS_FIR, n_low)  # circular, so the wrap is seamless
        self.taps = [np.fft.ifft(np.fft.fft(rng.normal(size=n_low) + 1j * rng.normal(size=n_low)) * h)
                     / np.sqrt(2 * np.sum(GAUSS_FIR**2)) for _ in range(2)]
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
        i = ((self.n + np.arange(len(x))) * self.rate / FS).astype(int) % self.n_low  # held between updates
        g1, g2 = (g[i] for g in self.taps)
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
    # the taps' Doppler spectrum: 2 sigma of HFSimulator's filter is 0.976 x the spread
    g = Fader(1.0, 2.0, 2).taps[0]
    f = np.fft.fftfreq(len(g), 1 / 64.0)
    psd = np.abs(np.fft.fft(g)) ** 2
    two_sigma = 2 * np.sqrt(np.sum(f**2 * psd) / np.sum(psd))
    print(f"MPP taps: 2 sigma {two_sigma:.3f} Hz")
    assert 0.93 < two_sigma < 1.02


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
