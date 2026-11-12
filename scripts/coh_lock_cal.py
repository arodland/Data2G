"""Calibrating cpm.COH_LOCK for a grid: coh_read_header's and read_header's
scores (one copy, both; best and margin) on noise at detect's placement,
and on real bursts at detect's placement, with the word read.

    python -m scripts.coh_lock_cal noise --grid c8r50 --n 1500 --seed 1 --out runs/cohcal/noise_c8r50_1.npy
    python -m scripts.coh_lock_cal signal --mode fsk8r50-r1/3 --chan awgn --snrs -14 -13 -12 --out runs/cohcal/
    python -m scripts.coh_lock_cal gates --grid c8r50 --dir runs/cohcal --coh 32 36

Columns: nc1 nc1m nc2 nc2m c1 c1m c2 c2m (scores, margins), then for
signal: nc2 word right, c2 word right. Found: data2g/cpm.py COH_LOCK's notes.
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import glob  # noqa: E402
import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import cpm  # noqa: E402
from data2g.config import FS  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))


def readings(g, y, s0, cfo, mode=None):
    out = []
    for c in (1, 2):
        h = cpm.read_header(g, y, s0, cfo, copies=c)
        out += [h[3], h[3] - h[4]]
    hc = []
    for c in (1, 2):
        h = cpm.coh_read_header(g, y, s0, cfo, copies=c)
        out += [h[3], h[3] - h[4]]
        hc.append(h)
    if mode is not None:
        h2 = cpm.read_header(g, y, s0, cfo, copies=2)
        out += [float(h2[0].name == mode and h2[1] == 1), float(hc[1][0].name == mode and hc[1][1] == 1)]
    return out


def noise(a):
    g = cpm.GRIDS[a.grid]
    rng = np.random.default_rng(a.seed)
    rows = []
    for _ in range(a.n):
        y = rng.normal(size=6 * FS)
        _, s0, cfo = cpm.detect(g, y[:3 * FS], front_only=True, floor=cpm.SYNC_THRESHOLD[g.name])
        rows.append(readings(g, y, s0, cfo))
    np.save(a.out, np.array(rows))


def signal(a):
    import outcome_data as O
    import phy_session as PS
    from cpm_floor_study import CHANS
    from data2g.arq import phy as PHY
    from data2g.arq.modes import MODES

    spec = MODES[a.mode]
    g = cpm.GRIDS[spec.grid]
    for snr in a.snrs:
        rows = []
        for seed in range(50000, 50000 + a.n):
            rng = np.random.default_rng(seed)
            b = replace(O.burst(a.mode, 2, rng), cap=2)
            ch = PS.ContinuousChannel("awgn", snr, seed, 120.0, *(CHANS[a.chan] or (None, None)))
            y = ch.apply(PHY.tx_audio(b), float(rng.uniform(5, 100)))
            s = int(PS.PAD_S * FS)
            _, s0, cfo = cpm.detect(g, y[:s + FS // 2 + 4 * FS], front_only=True, floor=cpm.SYNC_THRESHOLD[g.name])
            rows.append(readings(g, y, s0, cfo, a.mode))
        np.save(Path(a.out or a.dir) / f"sig_{g.name}_{a.chan}_{snr}.npy", np.array(rows))


def gates(a):
    nz = np.concatenate([np.load(f) for f in glob.glob(f"{a.dir}/noise_{a.grid}_*.npy")])
    nc = (cpm.LOCK_LLR[a.grid], cpm.HDR_MARGIN)
    print(f"noise windows {len(nz)}; maxima nc1 {nz[:, 0].max():.1f} nc2 {nz[:, 2].max():.1f} "
          f"c1 {nz[:, 4].max():.1f} c2 {nz[:, 6].max():.1f}; q99.9 c1 {np.quantile(nz[:, 4], 0.999):.1f} "
          f"c2 {np.quantile(nz[:, 6], 0.999):.1f}")
    c1, c2 = a.coh
    for f in sorted(glob.glob(f"{a.dir}/sig_{a.grid}_*.npy")):
        s = np.load(f)
        p_nc1 = np.mean((s[:, 0] >= nc[0][0]) & (s[:, 1] >= nc[1][0]))
        p_nc2 = np.mean((s[:, 2] >= nc[0][1]) & (s[:, 3] >= nc[1][1]) & (s[:, 8] == 1))
        p_c1 = np.mean((s[:, 4] >= c1) & (s[:, 5] >= nc[1][0]))
        p_c2 = np.mean((s[:, 6] >= c2) & (s[:, 7] >= nc[1][1]) & (s[:, 9] == 1))
        either2 = np.mean(((s[:, 6] >= c2) & (s[:, 7] >= nc[1][1]) & (s[:, 9] == 1))
                          | ((s[:, 2] >= nc[0][1]) & (s[:, 3] >= nc[1][1]) & (s[:, 8] == 1)))
        wrong = np.mean((s[:, 6] >= c2) & (s[:, 7] >= nc[1][1]) & (s[:, 9] == 0))
        print(f"{Path(f).stem[4:]:22s} one copy nc {p_nc1:.2f} coh {p_c1:.2f} | both nc {p_nc2:.2f} coh {p_c2:.2f} "
              f"either {either2:.2f} | coh wrong word {wrong:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["noise", "signal", "gates"])
    ap.add_argument("--grid")
    ap.add_argument("--mode")
    ap.add_argument("--chan", default="awgn")
    ap.add_argument("--snrs", nargs="+", type=float)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--coh", nargs=2, type=float, default=(34.0, 38.0))
    ap.add_argument("--dir", default="runs/cohcal")
    ap.add_argument("--out")
    a = ap.parse_args()
    {"noise": noise, "signal": signal, "gates": gates}[a.cmd](a)


if __name__ == "__main__":
    main()
