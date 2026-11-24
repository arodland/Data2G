"""The FM training channel (data2g.fmtorch) against the full numpy chain:
BICM BMI (bits per cell) of a constellation through
  numpy: fmchannel.apply_channel -> modem.receive (acquisition, the real
         estimator, per-symbol noise) -> constellation.llr
  torch: FmBurstChannel.transmit -> channel -> receive(points) -> llr
on the same modes, radios and CNRs (static path unless --condition).

    DATA2G_PROFILE=fm python scripts/fm_bmi_check.py [--bursts 8]
"""

import argparse

import numpy as np
import torch

from data2g import config, constellation, fmchannel, modem
from data2g.channel_torch import bmi, llr
from data2g.config import DATA_SYMS_PER_FRAME, SUBMODES
from data2g.fmtorch import FmBurstChannel

torch.set_num_threads(4)
N_F = 8
CASES = {"w48-qpsk-r1/2": (6, 8, 10), "w48-16qam-r1/2": (10, 13, 16), "w48-64l-r3/4": (16, 20, 25)}


def numpy_bmi(spec, radio, cond, cnr, drive, bursts):
    pts = constellation.load(spec.constellation)
    m = int(np.log2(len(pts)))
    nc = config.BANDS[spec.band].nc
    vals, lost = [], 0
    for s in range(bursts):
        rng = np.random.default_rng(100 + s)
        bits = rng.integers(0, 2, N_F * DATA_SYMS_PER_FRAME * nc * m)
        x = np.concatenate([np.zeros(config.FS // 2), modem.modulate_bits(bits, spec), np.zeros(config.FS // 2)])
        y = fmchannel.apply_channel(x, cnr, cond, radio, s, drive_dev_hz=drive)
        try:
            r = modem.receive(y, [spec.sync_band])
        except Exception:
            lost += 1
            continue
        if r["spec"].name != spec.name:
            lost += 1
            continue
        est = r["est"]
        lam = constellation.llr(modem.data_symbols(r["raw"]), est["h"], modem._data_var(r, est), pts)
        b = bits.reshape(lam.shape)
        vals.append(1 - np.mean(np.logaddexp(0, -(1 - 2 * b) * lam)) / np.log(2))
    return m * float(np.mean(vals)) if vals else float("nan"), lost


def torch_bmi(spec, radio, cond, cnr, drive, bursts):
    """The same bursts as numpy_bmi (bits, channel seed, padding): only the
    receivers differ (genie acquisition here)."""
    ch = FmBurstChannel(spec, N_F, radio, cond, drive_dev_hz=drive, dtype=torch.float64)
    pts = constellation.load(spec.constellation)
    m = int(np.log2(len(pts)))
    bits = np.stack([np.random.default_rng(100 + s).integers(0, 2, N_F * DATA_SYMS_PER_FRAME * ch.nc * m)
                     for s in range(bursts)])
    cells = torch.tensor(constellation.modulate(bits.ravel(), pts).reshape(bursts, N_F, DATA_SYMS_PER_FRAME, ch.nc))
    pad = config.FS // 2
    with torch.no_grad():
        x = torch.nn.functional.pad(ch.transmit(cells), (pad, pad))
        y = ch.channel(x, cnr, seeds=list(range(bursts)))[:, pad:-pad]
        yd, h, var = ch.receive(y, points=pts)
        lam = llr(yd, h, var, torch.tensor(pts))
    return m * bmi(lam, torch.tensor(bits.reshape(lam.shape), dtype=lam.real.dtype)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", type=int, default=8)
    ap.add_argument("--condition", default="static")
    a = ap.parse_args()
    print("| mode | radio | CNR | numpy BMI (lost) | torch BMI | diff |")
    print("|---|---|---|---|---|---|")
    for name, cnrs in CASES.items():
        spec = SUBMODES[name]
        for radio in ("mic", "flat"):
            drive = FmBurstChannel(spec, N_F, radio).drive
            for cnr in cnrs:
                nb, lost = numpy_bmi(spec, radio, a.condition, cnr, drive, a.bursts)
                tb = torch_bmi(spec, radio, a.condition, cnr, drive, a.bursts)
                print(f"| {name} | {radio} | {cnr} | {nb:.3f} ({lost}) | {tb:.3f} | {tb - nb:+.3f} |", flush=True)


if __name__ == "__main__":
    main()
