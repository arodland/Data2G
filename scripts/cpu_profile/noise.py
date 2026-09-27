"""Noise into both loopback sinks, unprofiled: white noise at 48 kHz whose
level wanders so the in-band SNR (2500 Hz reference, a peak-1.0 burst of
RMS ~0.45) runs from ~25 dB down to ~2 dB over four minutes, with a short
5 dB dip every 97 s. The loopback never carries exact zeros."""
import subprocess
import sys
import time

import numpy as np

FS, BLK = 48000, 4800
procs = [subprocess.Popen(["pacat", "--playback", f"--device={s}", "--format=float32le", "--rate=48000",
                           "--channels=1", "--latency-msec=100"], stdin=subprocess.PIPE) for s in ("d2g_ab", "d2g_ba")]
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
t, sig_p = 0.0, 0.2
t0 = time.monotonic()
log = open(sys.argv[2], "w") if len(sys.argv) > 2 else None
while True:
    snr = 13.5 + 11.5 * np.cos(2 * np.pi * t / 240)
    if (t % 97) < 8:
        snr -= 5
    sigma = np.sqrt(sig_p / (10 ** (snr / 10)) / (2500 / 24000))
    for p in procs:
        p.stdin.write(rng.normal(0, sigma, BLK).astype(np.float32).tobytes())
        p.stdin.flush()
    if log and int(t * 10) % 50 == 0:
        log.write(f"{time.time():.1f} {snr:.1f}\n")
        log.flush()
    t += BLK / FS
    lag = t - (time.monotonic() - t0)
    if lag > 0.3:
        time.sleep(lag - 0.3)
