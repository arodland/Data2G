"""Native gear-shifter cost: predict_outcome per call and recommend per decision (single thread)."""
import random
import sys
import time

import numpy as np

sys.path.insert(0, "tests")
import data2g_native  # noqa: E402
from test_native_arq import random_measured, station  # noqa: E402

A = data2g_native.arq
rng, nrng = random.Random(1), np.random.default_rng(1)
ms = [random_measured(nrng) for _ in range(500)]
for m in ms:
    m["energy"] = [float(nrng.uniform(-12, 30)), 0.5, 1.0]
t = time.perf_counter()
for m in ms:
    A.predict_outcome(m, "w", 2.5, 6.0)
t_pred = (time.perf_counter() - t) / len(ms)
sh, times, now = A.GearShifter(), [], 0.0
for i in range(400):
    m = ms[i % len(ms)]
    now += 5.0
    sh.observe(m, "qpsk-r1/2", now)
    sh.observe_energy(m["energy"][0], now)
    st = station(rng, i % 3)
    t = time.perf_counter()
    sh.recommend(st)
    times.append(time.perf_counter() - t)
times = np.array(times[20:]) * 1e3
print(f"predict_outcome {t_pred * 1e6:.0f} us/call; recommend median {np.median(times):.2f} ms, p95 {np.percentile(times, 95):.2f} ms, max {times.max():.2f} ms")
