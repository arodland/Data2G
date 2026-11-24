"""CwSense on CPM bursts (#310's QUIET_MAX was tuned on steady carriers): per grid, SNR and channel, the share of
bursts during which it reads a CW ID at all, and of those, any reading past the burst's end (what would hold a
turn after it). Control + one data codeword, 2 s of noise after.

    python -m scripts.cwsense_wide_probe
"""
from data2g import threads  # noqa: E402

threads.limit(1)

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, cpm, cwid, hfchannel  # noqa: E402
from data2g.config import FS  # noqa: E402

MODES = ["fsk8r50-r1/3", "fsk16r25-r1/3", "fsk8w50-r1/3", "fsk16w125-r1/3", "fsk32r62-r1/3"]
CELLS = [(None, -10.0), (None, 0.0), (None, 10.0), (None, 20.0), ("mpg", 10.0), ("mpp", 10.0), ("mpg", 20.0)]


def one(args):
    mode, (chan, snr), seed = args
    spec = cpm.SPECS[mode]
    g, ctl = cpm.GRIDS[spec.grid], cpm.CTL[spec.grid]
    rng = np.random.default_rng([seed, MODES.index(mode)])
    coded = [codes.encode(ctl, rng.bytes(codes.payload_bytes(ctl)), index=0),
             codes.encode(spec, rng.bytes(codes.payload_bytes(spec)), index=1)]
    x = cpm.modulate(spec, coded, False, 2)
    lead, tail = FS, 2 * FS
    y = np.concatenate([np.zeros(lead), x, np.zeros(tail)])
    y = hfchannel.apply_channel(y, snr, float(rng.uniform(-20, 20)), fading_preset=chan, seed=seed)
    s = cwid.CwSense()
    during, after = False, False
    for i in range(0, len(y), FS // 50):
        a = s.feed(y[i:i + FS // 50])
        if a and i < lead + len(x):
            during = True
        if a and i >= lead + len(x):
            after = True
    return mode, (chan, snr), during, after


def main():
    jobs = [(m, c, s) for m in MODES for c in CELLS for s in range(40)]
    with Pool(4) as p:
        out = p.map(one, jobs, chunksize=4)
    print("bursts (of 40) with a CW ID read during / after the burst")
    print(f"{'':16s}" + "".join(f"{(c or 'awgn') + ' ' + str(int(s)):>12s}" for c, s in CELLS))
    for m in MODES:
        row = [(sum(r[2] for r in out if r[0] == m and r[1] == c), sum(r[3] for r in out if r[0] == m and r[1] == c))
               for c in CELLS]
        print(f"{m:16s}" + "".join(f"{d:>8d} / {a:<2d}" for d, a in row))


if __name__ == "__main__":
    main()
