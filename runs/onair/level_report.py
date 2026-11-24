"""runs/onair/level_report.py REC.wav: per sounding burst, the in-band deviation the IC-9700's USB IF shows
(raw discriminator output, 400-3200 Hz, rms and 99.9th-percentile peak), for setting a sender's audio level:
a peak that stays put while the rms rises is clipping upstream of the modulator."""
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

fs, x = wavfile.read(sys.argv[1])
x = (x[:, 0] if x.ndim > 1 else x).astype(float) / 32768
n = np.arange(len(x))
z = signal.sosfiltfilt(signal.butter(8, 10000, fs=fs, output="sos"), 2 * x * np.exp(-2j * np.pi * 12000 * n / fs))
d = np.concatenate([[0], np.angle(z[1:] * np.conj(z[:-1])) * fs / 2 / np.pi])
inb = signal.sosfiltfilt(signal.butter(6, [400, 3200], "bandpass", fs=fs, output="sos"), d)
w = fs // 4
k = len(z) // w
e = np.abs(z[:k * w].reshape(k, w))
carrier = e.std(axis=1) / e.mean(axis=1) < 0.3
r = np.sqrt(np.mean(inb[:k * w].reshape(k, w) ** 2, axis=1))
active = carrier & (r > 100)
runs, i = [], 0
while i < k:  # stretches of in-band modulation, a burst or a CW ID
    if active[i]:
        j = i
        while j < k and active[j]:
            j += 1
        if j - i >= 3:
            s = inb[i * w:j * w]
            runs.append((i * 0.25, j * 0.25, np.sqrt(np.mean(s ** 2)), np.percentile(np.abs(s), 99.9)))
        i = j
    else:
        i += 1
for a, b, rms, pk in runs:
    print(f"{a:5.2f}-{b:5.2f} s  in-band rms {rms:6.0f} Hz  peak {pk:6.0f} Hz  peak/rms {pk / rms:4.2f}")
