"""Native GearShifter.recommend wall time (the outcome model compiled in),
random observations, half of them under the CPM floor. Run under
tools/with_native.py from the worktree root."""
import random
import sys
import time

import numpy as np

sys.path.insert(0, "tests")
import data2g_native  # noqa: E402
from test_native_arq import random_measured, station  # noqa: E402

from data2g.arq import modes  # noqa: E402
from data2g.arq import policy as G  # noqa: E402

A = data2g_native.arq
rng, nrng = random.Random(1), np.random.default_rng(1)
g = A.GearShifter()
now, ts = 0.0, []
for step in range(400):
    heard = rng.choice(G.allowed(2)).name
    m = random_measured(nrng, modes.MODES[heard].band, noise=True)
    if step % 2:
        m = dict(m, snr_est=float(nrng.uniform(-12, -4)))
    now += 5.0
    g.observe(m, heard, now)
    g.observe_energy(float(nrng.uniform(-12, 30)), now)
    st = station(rng, 2)
    t = time.perf_counter()
    g.recommend(st)
    ts.append(time.perf_counter() - t)
ts = np.array(ts[20:]) * 1e3
print(f"native recommend: median {np.median(ts):.2f} ms, p90 {np.percentile(ts, 90):.2f} ms over {len(ts)}")
