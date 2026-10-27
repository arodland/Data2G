"""CPM decode latency, burst end to payloads (2026-10-27, cpm-gfm-on-acq):
an n-codeword burst at a converging and a failing SNR on AWGN, the lock
known (cpm.receive from the true start, CFO and header: refine runs, the
search does not), then every data slot decoded (phy.ModemRx, the link's
one-off decode). Single thread, best of --reps. Run under
tools/with_native.py for the C++ core's numbers.

    python tools/with_native.py scripts/cpm_latency.py [--n 8] [--reps 3]
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import cpm  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.arq.modes import MODES  # noqa: E402
from data2g.config import FS  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

# AWGN SNR (dB, 2500 Hz mean power): converging (about 3 dB over the 10%
# point) and failing (about 3 dB under it, every data codeword lost)
SNRS = {"fsk8r50": (-9.0, -15.0), "fsk16r25": (-11.0, -17.0), "fsk32r62": (-6.0, -12.0)}


def bench(mode, snr, n, reps, seed=1):
    import outcome_data as O
    import phy_session as PS

    spec = MODES[mode]
    g = cpm.GRIDS[spec.grid]
    rng = np.random.default_rng(seed)
    b = replace(O.burst(mode, n, rng), cap=2)
    y = PS.ContinuousChannel("awgn", snr, seed, 120.0).apply(PHY.tx_audio(b), 10.0)
    lock = dict(spec=spec, n_data=n - 1, dup=False, start=int(PS.PAD_S * FS), cfo=PS.CFO_HZ, band=g.name)
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = cpm.receive(y, lock)
        t1 = time.perf_counter()
        rf = cpm.refine(g, y, lock["start"], lock["cfo"], spec, n - 1, False)  # refine alone
        t2 = time.perf_counter()
        rx = PHY.ModemRx(r, {})
        ok = sum(rx.decode(i, s.mask_id, 0, None) == s.payload for i, s in enumerate(b.slots) if i)
        t3 = time.perf_counter()
        t = (t1 - t0, t2 - t1, t3 - t2)
        if best is None or sum(t) < sum(best[0]):
            best = (t, ok)
    (rcv, ref, dec), ok = best
    del rf
    print(f"{mode:15s} {snr:6.1f} {spec.code:5s} "
          f"{ok}/{n - 1}  receive {1e3 * rcv:7.1f} ms (refine {1e3 * ref:6.1f})  decode {1e3 * dec:7.1f} ms "
          f"({1e3 * dec / (n - 1):6.1f}/cw)  total {1e3 * (rcv + dec):7.1f} ms", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=[m for m in MODES if m.startswith("fsk")])
    a = ap.parse_args()
    for m in a.modes:
        for snr in SNRS[m.split("-")[0]]:
            bench(m, snr, a.n, a.reps)


if __name__ == "__main__":
    main()
