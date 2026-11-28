"""fsk32r62's data modes (r1/3, r1/2, r2/3) end to end on fixed bursts
(control + one data codeword), per channel and SNR: the burst found with
its header right, and the data codeword decoded (PHY.ModemRx: square law,
then cpm DD). ladder_study's trial (CFO +-50 Hz, 10 ppm, random lead),
with the ITU-R F.1487 severe presets and PEP-referenced SNR (noise against
the burst's envelope peak - 5 dB in 2500 Hz, as DATA2G_PEP_REF_DB=5 loss
cells). Trial k has the same seed at every SNR and in every mode (the three
bursts are the same length: the same fades).

    python -m scripts.c32_rate_study --out runs/c32_rate.csv
"""

import os

os.environ.setdefault("DATA2G_CPM_R23", "1")  # fsk32r62-r2/3 offered: not a mode by default

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import hfchannel  # noqa: E402
from data2g.arq import phy as PHY  # noqa: E402
from data2g.config import FS  # noqa: E402
from data2g.receiver import receive_any  # noqa: E402
from data2g.waveform import dsp  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import outcome_data as O  # noqa: E402

MODES = ("fsk32r62-r1/3", "fsk32r62-r1/2", "fsk32r62-r2/3")
PRESETS = {**hfchannel.FADING_PRESETS, **hfchannel.ITU_PRESETS}
PEP_REF_DB = 5.0


def trial(args):
    name, chan, snr, seed = args
    rng = np.random.default_rng(seed)
    b = O.burst(name, 2, rng)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    x = np.concatenate([np.zeros(lead), PHY.tx_audio(b), np.zeros(FS // 2)])
    s_power = np.max(dsp.envelope(x) ** 2) / 2 / 10 ** (PEP_REF_DB / 10)
    y = hfchannel.sample_clock_offset(x.astype(np.float64), 10)
    y = hfchannel.freq_shift(y, float(rng.uniform(-50, 50)))
    if chan != "awgn":
        y = hfchannel.fading(y, PRESETS[chan], seed=seed)
    y = hfchannel.awgn(y, snr, seed=seed + 1, s_power=s_power)
    try:
        r = receive_any(y, lead=FS)
    except Exception:  # noqa: BLE001 (a header read past the audio: a miss)
        r = None
    if r is None or r["spec"].name != name or r["n_cw"] != 2:
        return name, chan, snr, seed, 0, 0, 0
    rx = PHY.ModemRx(r, {})
    ok = [rx.decode(i, s.mask_id, 0, None) == s.payload for i, s in enumerate(b.slots)]
    return name, chan, snr, seed, 1, int(ok[0]), int(ok[1])


def crossing(xs, fail, target):
    """The last crossing down through `target` (a fading curve can wobble)."""
    out = float("nan")
    for i in range(len(xs) - 1):
        if fail[i] > target >= fail[i + 1]:
            out = xs[i] + (fail[i] - target) / (fail[i] - fail[i + 1]) * (xs[i + 1] - xs[i])
    return out


def summarize(path):
    rows = list(csv.DictReader(open(path)))
    snrs = sorted({float(r["snr"]) for r in rows})
    print("data codeword failure by PEP cell SNR; 10% point (dB); failures with the header right at the top SNR")
    print(f"{'':24s}" + " ".join(f"{s:5.0f}" for s in snrs) + "   10%   floor")
    for chan in dict.fromkeys(r["chan"] for r in rows):
        for m in MODES:
            fail, locked = [], []
            for s in snrs:
                rs = [r for r in rows if r["chan"] == chan and r["mode"] == m and float(r["snr"]) == s]
                fail.append(1 - np.mean([int(r["data"]) for r in rs]))
                lk = [1 - int(r["data"]) for r in rs if r["lock"] == "1"]
                locked.append(np.mean(lk) if lk else float("nan"))
            print(f"{chan:9s}{m:15s}" + " ".join(f"{f:5.2f}" for f in fail)
                  + f" {crossing(snrs, fail, 0.1):6.1f}  {locked[-1]:5.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/c32_rate.csv")
    ap.add_argument("--chans", nargs="+", default=["awgn", "mpg", "mpp", "mpd", "hilat-m", "lowlat-d", "hilat-d", "nvis-d"])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--snrs", default="-8,-6,-4,-2,0,2,4,6,8,10,14,18,22,26")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.out)
    jobs = [(m, c, float(s), 70000 + k) for c in a.chans for m in a.modes for s in a.snrs.split(",")
            for k in range(a.trials)]
    with Pool(a.jobs) as pool, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "chan", "snr", "seed", "lock", "ctl", "data"])
        for i, row in enumerate(pool.imap_unordered(trial, jobs, chunksize=8)):
            w.writerow(row)
            if i % 2000 == 0:
                f.flush()
                print(f"{i}/{len(jobs)}", file=sys.stderr, flush=True)
    summarize(a.out)


if __name__ == "__main__":
    main()
