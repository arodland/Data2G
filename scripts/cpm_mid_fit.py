"""cpm.MID_THRESHOLD and MID_HEADER_THRESHOLD: what find_mid's two scores
reach without a burst to find.

- sync: the mid block's best score (shares, T/4 timing, quarter-bin CFO over
  +-150 Hz, as detect) on 1.9 s of noise or an OFDM burst through AWGN/MPG
  at 0-30 dB;
- header: find_mid with both thresholds off, streamed over 12 s of one or
  two overlapping bursts of the grid (random offsets and levels, AWGN/MPG at
  0-30 dB); a lock is false unless it is a real burst's start (+-T/4) and
  word.

Prints each one's largest and 1.1x it.

    uv run --no-sync python scripts/cpm_mid_fit.py --n 200
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cpm_header2_fit as F  # noqa: E402
from data2g import codes, cpm, hfchannel  # noqa: E402
from data2g.arq.modes import MODES, is_cpm  # noqa: E402
from data2g.config import FS  # noqa: E402


def sync_null(args):
    grid, src, seed = args
    g = cpm.GRIDS[grid]
    rng = np.random.default_rng([seed, list(cpm.GRIDS).index(grid), src == "ofdm"])
    n = int(1.9 * FS)
    if src == "noise":
        y = rng.normal(0, 1, n)
    else:
        x = np.concatenate([F.burst(rng.choice([s for s in MODES.values() if not is_cpm(s)]), rng), np.zeros(n)])[:n]
        y = hfchannel.apply_channel(x, snr_db=float(rng.choice([0, 10, 20, 30])),
                                    freq_offset_hz=float(rng.uniform(-50, 50)),
                                    fading_preset=[None, "mpg"][int(rng.integers(2))], seed=seed)
    return grid, src, cpm.detect(g, y, mid=True, fine=False)[0]


def header_null(args):
    grid, seed = args
    g = cpm.GRIDS[grid]
    cpm.MID_THRESHOLD = {k: -1.0 for k in cpm.GRIDS}
    cpm.MID_HEADER_THRESHOLD = {k: -1.0 for k in cpm.GRIDS}
    rng = np.random.default_rng([seed, list(cpm.GRIDS).index(grid), 2])
    n = 12 * FS
    y, truth = np.zeros(n), []
    for k in range(int(rng.integers(1, 3))):
        spec = rng.choice(cpm.grid_specs(grid))
        nd = int(rng.integers(0, 4))
        ctl = codes.encode(cpm.CTL[grid], rng.bytes(codes.payload_bytes(cpm.CTL[grid])), 0, 0)
        x = cpm.modulate(spec, [ctl] + [codes.encode(spec, rng.bytes(codes.payload_bytes(spec)), 0, 0)
                                       for _ in range(nd)], dup=False)
        s = int(rng.integers(-len(x) // 2, n // 2))
        lo, hi = max(0, s), min(n, s + len(x))
        if hi > lo:
            y[lo:hi] += x[lo - s:hi - s] * (1.0 if k == 0 else float(rng.uniform(0.3, 1.0)))
        truth.append((s, spec.name, nd))
    y = hfchannel.apply_channel(y, snr_db=float(rng.choice([0, 10, 20, 30])), freq_offset_hz=float(rng.uniform(-50, 50)),
                                fading_preset=[None, "mpg"][int(rng.integers(2))], seed=seed)
    out = []
    for lo in range(0, n - 4 * FS, FS // 2):  # as a stream: each mid start once
        lock = cpm.find_mid(g, y, lo=lo, hi=lo + FS // 2)
        if lock is not None:
            real = any(abs(lock["start"] - s) < g.T // 4 and lock["spec"].name == nm and lock["n_data"] == nd
                       for s, nm, nd in truth)
            out.append((real, lock["header_score"]))
    return grid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="buffers per grid and source")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    with Pool(a.jobs) as pool:
        sync, hdr = {}, {}
        for g, s, v in pool.imap_unordered(sync_null, [(g, s, i) for g in cpm.GRIDS for s in ("noise", "ofdm")
                                                       for i in range(a.n)], chunksize=4):
            sync.setdefault(g, []).append(v)
        for g, o in pool.imap_unordered(header_null, [(g, i) for g in cpm.GRIDS for i in range(a.n)], chunksize=4):
            hdr.setdefault(g, []).extend(o)
    for g in cpm.GRIDS:
        false = [h for real, h in hdr[g] if not real]
        real = [h for r, h in hdr[g] if r]
        print(f"{g}: mid sync max {max(sync[g]):.3f} -> {1.1 * max(sync[g]):.3f} (now {cpm.MID_THRESHOLD[g]}); "
              f"false mid header max {max(false, default=0):.3f} of {len(false)} -> {1.1 * max(false, default=0):.3f} "
              f"(now {cpm.MID_HEADER_THRESHOLD[g]}); real min {min(real, default=float('nan')):.3f} of {len(real)}")


if __name__ == "__main__":
    main()
