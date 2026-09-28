"""Polar list gates (study for the CRC16 false-accept issue): CRC-aided SCL
takes the first of L paths, best metric first, whose CRC checks, so a
failed decode passes 1 in 65536 per path tried (L=8: 1.2e-4). A gate
admits a path only if its metric is within DELTA of the best path's
(DELTA=0: the best path alone). Real-modem bursts near and below each
polar mode's 10% point (true mask); per decode, the correct path's rank
and metric gap if it is in the list at all, and every wrong path's gap.

- cost of DELTA: correct decodes lost (the correct path's gap > DELTA);
- gain: expected false accepts per failed decode = (wrong paths within
  DELTA) / 65536.

    uv run --no-sync python scripts/polar_gate_study.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
from multiprocessing import Pool

import numpy as np
import torch

from data2g import codes, hfchannel, modem
from data2g.arq import phy as PHY
from data2g.arq import policy as G
from data2g.arq.modes import MODES
from data2g.config import FS
from data2g.tnc import receive_any

NAMES = ("ack-1f", "ack-4f", "n10-ack-4f", "n4-ack-2f", "polar-k96-f4")
CHANNELS = ("awgn", "mpp", "mpg")
OFFSETS = (-2.0, -1.0, 0.0)
DELTAS = (0.0, 1.0, 2.0, 4.0, 8.0, np.inf)


def trial(args):
    name, chan, snr, seed = args
    spec = MODES[name]
    rng = np.random.default_rng(seed)
    n = min(64, max(8, G.slots_for(spec, 12.0)))
    pays = [rng.bytes(codes.payload_bytes(spec)) for _ in range(n)]
    x = np.concatenate([np.zeros(int(rng.uniform(0.3, 1.0) * FS)), modem.modulate(pays, spec), np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=float(rng.uniform(-50, 50)), ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    try:
        r = receive_any(y, lead=FS)
    except Exception:  # noqa: BLE001
        return []
    if r is None or r["spec"].name != name or r["n_cw"] != n:
        return []
    soft = torch.as_tensor(np.asarray(PHY.soft_bits(r)), dtype=torch.float32)
    deint = torch.empty_like(soft)
    deint[:, torch.as_tensor(codes.interleaver(spec))] = soft
    paths, pm = codes._decoder(spec, "cpu").decode(deint)
    paths, pm = paths.numpy(), pm.numpy()
    truth = np.stack([codes.info_bits(spec, p, 0, i) for i, p in enumerate(pays)])
    out = []
    for i in range(n):
        gap = pm[i] - pm[i, 0]
        right = (paths[i] == truth[i]).all(axis=1)
        out.append((float(gap[right][0]) if right.any() else -1.0, gap[~right & np.isfinite(gap)].tolist()))
    return out


def main():
    p10 = {r["name"]: r for r in csv.DictReader(open("runs/ladder_10pct.csv"))}
    print("mode chan snr | decodes, correct in list | correct lost at DELTA " + " ".join(f"{d:g}" for d in DELTAS)
          + " | false accepts per failed decode x1e5 at DELTA", flush=True)
    with Pool(8) as pool:
        for name in NAMES:
            for chan in CHANNELS:
                for off in OFFSETS:
                    snr = float(p10[name][chan]) + off
                    rows = [r for t in pool.map(trial, [(name, chan, snr, 104729 * k + 5) for k in range(60)]) for r in t]
                    if not rows:
                        continue
                    cg = np.array([g for g, _ in rows])
                    inlist = cg >= 0
                    lost = [int((inlist & (cg > d)).sum()) for d in DELTAS]
                    failed = [w for g, w in rows if g < 0]
                    fa = [1e5 * np.mean([sum(x <= d for x in w) for w in failed]) / 65536 if failed else float("nan")
                          for d in DELTAS]
                    print(f"{name} {chan} {snr:.2f} | {len(rows)}, {int(inlist.sum())} | " + " ".join(map(str, lost))
                          + " | " + " ".join(f"{v:.2f}" for v in fa), flush=True)


if __name__ == "__main__":
    main()
