"""FM ladder study: each mode's 10% and 1% codeword failure CNR through the
live receiver (scripts/fm_spacing_study.stream_receive: receiver.Receiver fed
0.25 s at a time after open-squelch hiss, key-up 0.2 s before the burst),
drive by the 15 kHz occupied limit (26 dB down), by bisection to 0.25 dB.

Common random numbers: trial t is the same payloads and the same channel
seed at every CNR, so failure is near-monotone in CNR and the bisection
compares like with like. A knee is the lowest CNR on the bisection's path
whose failure (lost codewords / sent, over all trials) is at or under the
level. 1% over N trials x 3 codewords is "at most one codeword in 100":
30 trials (90 codewords) make it zero failures, so it reads as a
no-failure knee there.

    DATA2G_PROFILE=fm python scripts/fm_ladder_study.py run --modes f55-qpsk-r1/2 ... \
        [--radios mic flat] [--conditions static flutter@440 mobile-rural] [--trials 30] [--workers 8] [--out NAME]
    DATA2G_PROFILE=fm python scripts/fm_ladder_study.py report [--out NAME]

Modes are config.SUBMODES names, or candidates registered by
scripts/fm_ladder_screen.register (band:constellation:frames:k:headroom
strings, e.g. f55:gray-qam16:2:1200:1). Writes runs/fm_ladder/<out>.csv.

The transmitter is the shipping one: modem.modulate with tx_path = the
radio's path ("mic" for mic radios, "flat" otherwise; config.FM_*). The
drive is a station's: one audio level, set so the full-level modes meet the
15 kHz limit, so a mode sent quieter (the mic path's top modes,
config.FM_MIC_GAIN) deviates that much less. FM_TX=1 (scripts/fm_tx.py)
re-applies the study settings on top for comparison; leave it off.
FM_RX_UNCLIPPED=1: on the flat path, the receiver assumes no clipping from
16-QAM up (clip gain 1, no clip noise) instead of the mode's headroom's
constants (what is actually sent).
"""

import os

# before numpy (and torch, via fm_ladder_screen): one BLAS thread per worker;
# unset, each worker took about 4 cores on a shared machine
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_spacing_study as S
import fm_tx as T

OUT = S.ROOT / "runs" / "fm_ladder"
LO, HI, STEP = float(os.environ.get("FM_LADDER_LO", "-2.0")), 44.0, 0.25  # FM_LADDER_LO: the floor rungs knee lower
LEVELS = (0.1, 0.01)
STATION_DRIVE = {"mic": 1460.0, "flat": 2350.0}  # rms Hz: a station's audio level (knees())


def spec_of(name: str):
    from data2g.config import SUBMODES

    if name in SUBMODES:
        return SUBMODES[name]
    import fm_ladder_screen as L

    return L.register(name)


def tx_path(radio: str) -> str:
    return "mic" if radio.startswith("mic") else "flat"


def burst(spec, seed, radio):
    """S.burst as transmitted on the radio's path (fm_tx's settings on top with FM_TX=1)."""
    from data2g import constellation, modem

    modem.TX_PATH = tx_path(radio)
    orig = modem.dsp.tx_condition
    if T.noclip(radio, constellation.bits_per_symbol(constellation.load(spec.constellation))):
        modem.dsp.tx_condition = lambda x, h, *a, **k: orig(x, T.NOCLIP_HEADROOM, *a, **k)
    try:
        payloads, x = S.burst(spec, seed)
    finally:
        modem.dsp.tx_condition = orig
    return payloads, T.shape(x, spec.band, radio)


def fail_at(spec, radio, cond, mhz, cnr, drive, trials) -> float:
    """Lost codewords / sent over `trials` paired trials at one CNR."""
    from data2g import fmchannel as F
    from data2g.config import FS

    lost = sent = 0
    for t in range(trials):
        payloads, x = burst(spec, 1000 + t, radio)
        ch = F.FmChannel(cond, S.radio_of(radio), cnr, seed=t, duration_s=len(x) / FS + 4, carrier_mhz=mhz,
                         drive_dev_hz=drive)
        n_lost, _, _ = S.stream_receive(spec, payloads, x, ch)
        lost, sent = lost + n_lost, sent + len(payloads)
    return lost / sent


def knees(args) -> dict:
    """One mode x radio x condition: bisection for each level, sharing the
    CNRs already measured."""
    name, radio, cond, trials = args
    from data2g import fmchannel as F

    spec = spec_of(name)
    c, mhz = (cond.split("@") + ["146"])[:2]
    from data2g import config, constellation

    _, x = burst(spec, 0, radio)
    bits = constellation.bits_per_symbol(constellation.load(spec.constellation))
    if os.environ.get("FM_RX_UNCLIPPED") and tx_path(radio) == "flat" and bits >= config.FM_FLAT_NOCLIP_BITS:
        config.clip_consts = lambda *a, **k: config.ClipConsts({1: 1.0}, 1.0, 0.0)  # what was sent
    # the station's level: FmChannel scales each burst to its own rms, so a mode
    # sent quieter (FM_MIC_GAIN) deviates that much less than the width-limited drive
    gain = config.FM_MIC_GAIN[bits] if tx_path(radio) == "mic" else 1.0
    if spec.band in ("f55", "f108"):
        level = F.drive_for_bandwidth(x[np.abs(x) > 0], S.radio_of(radio), S.OBW_HZ)[0]
    else:
        # w and n10 (FM's floor and ACK rungs): a narrow burst's own width limit
        # would allow far more drive than the station's one audio level, set by
        # its f55/f108 modes (runs/fm_ladder/shipped.csv: 1455-1467 Hz on mic,
        # 2309-2426 on flat, the width being rms-bound)
        level = STATION_DRIVE[tx_path(radio)]
    drive = level * gain * T.drive_scale(radio, bits)
    seen = {}

    def f(cnr):
        if cnr not in seen:
            seen[cnr] = fail_at(spec, radio, c, float(mhz), cnr, drive, trials)
        return seen[cnr]

    out = {"mode": name, "radio": radio, "condition": cond, "drive": round(drive)}
    for level in LEVELS:
        lo, hi = LO, HI
        if f(hi) > level:  # never reached within the range
            out[f"k{level}"] = float("nan")
            continue
        while hi - lo > STEP:
            mid = round((lo + hi) / 2 / STEP) * STEP
            if mid in (lo, hi):
                break
            lo, hi = (lo, mid) if f(mid) <= level else (mid, hi)
        out[f"k{level}"] = hi
    out["evals"] = len(seen)
    return out


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(m, r, c, a.trials) for m in a.modes for r in a.radios for c in a.conditions]
    path = OUT / f"{a.out}.csv"
    cols = ["mode", "radio", "condition", "drive", "k0.1", "k0.01", "evals"]
    new = not path.exists()
    with path.open("a", newline="") as fh, ProcessPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(fh, cols)
        if new:
            w.writeheader()
        for row in ex.map(knees, jobs):
            w.writerow(row)
            fh.flush()
            print(row, flush=True)


def report(a):
    with (OUT / f"{a.out}.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    print("| mode | radio | condition | 10% CNR | 1% CNR |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['mode']} | {r['radio']} | {r['condition']} | {r['k0.1']} | {r['k0.01']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report"])
    ap.add_argument("--modes", nargs="+", default=[])
    ap.add_argument("--radios", nargs="+", default=["mic"])
    ap.add_argument("--conditions", nargs="+", default=["static", "flutter@440", "mobile-rural"])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="knees")
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    run(a) if a.what == "run" else report(a)


if __name__ == "__main__":
    main()
