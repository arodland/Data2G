"""Noise into both loopback sinks, unprofiled: white noise at 48 kHz,
low-passed to 3300 Hz (stopband from 4500 Hz), whose level wanders so the
SNR runs from ~25 dB down to ~2 dB over four minutes, with a short 5 dB dip
every 97 s; NOISE_SNR=<dB> holds it constant instead. The loopback never
carries exact zeros.

SNR as ARSFI's HFSimulator (IONOS, github.com/ARSFI/HFSimulator) defines it:
the input's PEP, a sine's power at the input's peak ((p-p / 2)^2 / 2, "to not
give any advantage of High CF signals"), over the noise in 3000 Hz. The noise
filter is its 3 kHz one. Its firmware 2.03 sets fixed noise and signal gains
calibrated at a nominal input level; here the input's peak is the host's
full scale (--output-volume 0 puts each burst's peak there). NOISE_PEAK: a
different peak (1.0)."""
import os
import subprocess
import sys
import time

import numpy as np
from scipy import signal

FS, BLK = 48000, 4800
BW_HZ = 3000  # the SNR's noise bandwidth
LP = signal.remez(255, [0, 3300, 4500, FS / 2], [1, 0], fs=FS)


class Noise:
    """One rng for every sink, drawn block by block in sink order; a filter
    state per sink. `peak`: the input's peak the SNR references."""

    def __init__(self, seed: int, sinks: int = 2, peak: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.zi = [np.zeros(len(LP) - 1) for _ in range(sinks)]
        self.pep = peak**2 / 2

    def __call__(self, snr: float) -> list[np.ndarray]:
        """The next BLK of noise for each sink, at `snr` dB."""
        # white at FS: density sigma^2 / (FS / 2) per Hz, kept by the passband
        sigma = np.sqrt(self.pep / (10 ** (snr / 10)) / (BW_HZ / (FS / 2)))
        out = []
        for k, zi in enumerate(self.zi):
            y, self.zi[k] = signal.lfilter(LP, 1.0, self.rng.normal(0, sigma, BLK), zi=zi)
            out.append(y)
        return out


if __name__ == "__main__":
    procs = [subprocess.Popen(["pacat", "--playback", f"--device={s}", "--format=float32le", "--rate=48000",
                               "--channels=1", "--latency-msec=100"], stdin=subprocess.PIPE) for s in ("d2g_ab", "d2g_ba")]
    noise = Noise(int(sys.argv[1]) if len(sys.argv) > 1 else 0, len(procs), float(os.environ.get("NOISE_PEAK", "1.0")))
    t = 0.0
    fixed = os.environ.get("NOISE_SNR")
    t0 = time.monotonic()
    log = open(sys.argv[2], "w") if len(sys.argv) > 2 else None
    while True:
        snr = 13.5 + 11.5 * np.cos(2 * np.pi * t / 240)
        if (t % 97) < 8:
            snr -= 5
        if fixed:
            snr = float(fixed)
        for p, y in zip(procs, noise(snr)):
            p.stdin.write(y.astype(np.float32).tobytes())
            p.stdin.flush()
        if log and int(t * 10) % 50 == 0:
            log.write(f"{time.time():.1f} {snr:.1f}\n")
            log.flush()
        t += BLK / FS
        lag = t - (time.monotonic() - t0)
        if lag > 0.3:
            time.sleep(lag - 0.3)
