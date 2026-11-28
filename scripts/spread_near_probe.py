"""cpm.spread's near-tone test (#312) on today's grids and the wide ones: share of control-only bursts reading
SPREAD_MAX_HZ (12 Hz), genie lock, two equal Rayleigh paths (hfchannel), 200 seeds a cell. `--lags rate`: the
wide grids' near lags at the tone spacing (g.rate) instead of the symbol rate (g.baud).

    python -m scripts.spread_near_probe --out runs/spread_near.csv [--lags rate]
"""
from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import hashlib  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, cpm, hfchannel  # noqa: E402

CELLS = [("awgn", None, 0.0, 2.0), ("flat 1 Hz", 1.0, 0.0, 2.0), ("2 ms / 0.1 Hz", 0.1, 2.0, 2.0),
         ("2 ms / 1 Hz", 1.0, 2.0, 2.0), ("2 ms / 1 Hz -6", 1.0, 2.0, -6.0), ("7 ms / 1 Hz -8", 1.0, 7.0, -8.0),
         ("4 ms / 2 Hz", 2.0, 4.0, 0.0), ("7 ms / 10 Hz", 10.0, 7.0, 0.0), ("7 ms / 30 Hz", 30.0, 7.0, 0.0)]
MODES = ["fsk16r25-r1/2", "fsk8r50-r1/3", "fsk32r62-r1/3", "fsk8w50-r1/2", "fsk16w125-r1/3"]
LAGS = "baud"


def one(args):
    mode, cell, seed = args
    if LAGS == "none":  # before #312: no near-tone test
        cpm._slow_by_near_tones = lambda *a: False
    if LAGS == "rate":
        cpm._near_lags = lambda g: range(max(1, int(np.ceil(cpm.SPREAD_NEAR_S[0] * g.rate - 1e-9))),
                                         int(np.floor(cpm.SPREAD_NEAR_S[1] * g.rate + 1e-9)) + 1)
    name, dop, delay, snr = cell
    spec = cpm.SPECS[mode]
    g, ctl = cpm.GRIDS[spec.grid], cpm.CTL[spec.grid]
    rng = np.random.default_rng([seed, MODES.index(mode), CELLS.index(cell)])
    x = cpm.modulate(spec, [codes.encode(ctl, rng.bytes(codes.payload_bytes(ctl)), index=0)], False, 2)
    x = np.concatenate([np.zeros(2000), x, np.zeros(2000)])
    p = hfchannel.active_power(x)
    if dop is not None:
        x = hfchannel.fading(x, hfchannel.FadingPreset("p", dop, delay), seed=seed)
    y = hfchannel.awgn(x, snr, seed=seed + 1, s_power=p)
    r = cpm.receive(y, dict(spec=spec, n_data=0, dup=False, start=2000, cfo=0.0, band=g.name))
    return mode, name, float(cpm.measure(g, r["E"], len(r["E"]))["spread_est"])


def main():
    global LAGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--lags", default="baud", choices=["baud", "rate", "none"])
    ap.add_argument("--modes", nargs="+", default=MODES)
    a = ap.parse_args()
    LAGS = a.lags
    jobs = [(m, c, s) for m in a.modes for c in CELLS for s in range(a.seeds)]
    with Pool(a.jobs) as p:
        out = p.map(one, jobs, chunksize=8)
    h = hashlib.sha256()
    for m in a.modes:
        cells = []
        for c in CELLS:
            v = np.array([r[2] for r in out if r[0] == m and r[1] == c[0]])
            h.update(v.tobytes())
            cells.append(f"{int((v >= cpm.SPREAD_MAX_HZ).sum()):3d}")
        print(f"{m:15s} " + " ".join(cells))
    print("cells:", " | ".join(c[0] for c in CELLS))
    print("readings hash", h.hexdigest()[:24], "lags", LAGS)


if __name__ == "__main__":
    main()
