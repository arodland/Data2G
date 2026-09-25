"""Clipper headroom study for one band, PEP-fair.

For each headroom: measure the clip constants and post-clip PAPR
(clip_constants.measure), then re-threshold the chosen submodes with the
receiver told those constants. SNR here is average power, but a
transmitter is peak-limited, so the comparable figure is the PEP-fair
threshold: threshold + PAPR (dB), reported relative to the stock
setting. Lower is better. Appends to a CSV; resumable.

    PYTHONPATH=scripts uv run python scripts/clip_study.py --band w48 \\
        --headroom 0 1 2 3 4 5 --out runs/clip_w48.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import csv
import time

import torch

torch.set_num_threads(4)

from clip_constants import measure  # noqa: E402
from data2g.config import CLIP_OVERSHOOT, SubmodeSpec  # noqa: E402
from thresholds import Sim, threshold  # noqa: E402

BURST_FRAMES = 16

# band -> (name, code, constellation, frames per codeword, k, ladder awgn / mpd thresholds)
PANELS = {
    "w48": [
        ("qpsk-r1/5", "ldpc", "gray-qam4", 4, 384, -2.25, 1.75),
        ("qpsk-r1/2", "ldpc", "gray-qam4", 4, 960, 2.5, 7.75),
        ("16qam-r1/2", "ldpc", "gray-qam16", 4, 1920, 8.5, 14.25),
        ("16qam-r2/3", "ldpc", "gray-qam16", 4, 2560, 12.5, 20.5),
        ("16qam-r3/4", "ldpc", "gray-qam16", 4, 2880, 15.75, 30.0),
        ("64l-r1/2", "ldpc", "c64-snr18", 2, 1440, 15.75, 28.25),
    ],
    # Past the stock clipper's ceiling: the ones that never decoded
    # ("fails on every channel") and new rates above the old top.
    "w48-top": [
        ("16qam-r5/6", "ldpc", "gray-qam16", 4, 3200, 16.0, 26.0),
        ("64l-r7/12", "ldpc", "c64-snr18", 2, 1680, 16.0, 24.0),
        ("64l-r2/3", "ldpc", "c64-snr18", 2, 1920, 18.0, 26.0),
        ("64l-r3/4", "ldpc", "c64-snr18", 2, 2160, 20.0, 30.0),
        ("256l-r1/2", "ldpc", "c256-snr26", 2, 1920, 20.0, 30.0),
        ("256l-r5/8", "ldpc", "c256-snr26", 2, 2400, 24.0, 34.0),
    ],
    "w": [
        ("qpsk-r1/5", "ldpc", "gray-qam4", 8, 384, -5.5, -1.25),
        ("qpsk-r1/2", "ldpc", "gray-qam4", 8, 960, -0.75, 5.0),
        ("16qam-r1/3", "ldpc", "gray-qam16", 8, 1280, 1.75, 6.75),
        ("16qam-r1/2", "ldpc", "gray-qam16", 8, 1920, 5.25, 11.0),
        ("16qam-r2/3", "ldpc", "gray-qam16", 8, 2560, 8.75, 16.5),
        ("16qam-r3/4", "ldpc", "gray-qam16", 8, 2880, 11.5, 21.75),
    ],
    "w-top": [
        ("16qam-r5/6", "ldpc", "gray-qam16", 8, 3200, 16.0, 26.0),
        ("64l-r2/3", "ldpc", "c64-snr18", 4, 1920, 18.0, 26.0),
        ("64l-r3/4", "ldpc", "c64-snr18", 4, 2160, 20.0, 30.0),
        ("256l-r1/2", "ldpc", "c256-snr26", 3, 1440, 20.0, 30.0),
    ],
    "n10": [
        ("polar-k48-f4", "polar", "gray-qam4", 4, 48, -10.25, -6.5),
        ("qpsk-r1/5", "ldpc", "gray-qam4", 10, 200, -8.75, -3.75),
        ("qpsk-r1/3", "ldpc", "gray-qam4", 10, 336, -6.25, -1.0),
        ("qpsk-r1/2", "ldpc", "gray-qam4", 10, 496, -4.25, 2.25),
    ],
    "n4": [
        ("polar-k48-f8", "polar", "gray-qam4", 8, 48, -13.5, -9.5),
        ("qpsk-r1/5", "ldpc", "gray-qam4", 24, 192, -12.5, -8.75),
        ("qpsk-r1/3", "ldpc", "gray-qam4", 24, 320, -10.5, -6.0),
        ("16qam-r1/3", "ldpc", "gray-qam16", 24, 640, -6.25, -1.5),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="w48")
    ap.add_argument("--panel", help="PANELS key (default: the band)")
    ap.add_argument("--headroom", type=float, nargs="+", default=[0, 1, 2, 3, 4, 5])
    # headroom values must be in codes_data/clip_constants.json for a later
    # modem run; the study itself measures whatever it is given
    ap.add_argument("--overshoot", type=float, nargs="+", default=list(CLIP_OVERSHOOT))
    ap.add_argument("--channels", nargs="+", default=["awgn", "mpd"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    ov = tuple(a.overshoot)
    seen = set()
    if os.path.exists(a.out) and os.path.getsize(a.out):
        with open(a.out) as f:
            seen = {(r["headroom"], r["overshoot"], r["submode"], r["channel"]) for r in csv.DictReader(f)}
    new = not seen
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["band", "headroom", "overshoot", "submode", "channel", "threshold_db",
                        "papr_db", "peak_db", "sdr_db", "gain", "secs"])
        for hr in a.headroom:
            c = measure(a.band, hr, ov, a.device)
            (g1, _, _, _), (g8, sdr, papr, peak) = c[1], c[8]
            consts = ({1: g1}, g8, 10 ** (-sdr / 10))
            print(f"== {a.band} headroom {hr} dB: SDR {sdr:.2f} dB, PAPR {papr:.2f} dB, peak {peak:.2f} dB, "
                  f"gain {g1:.3f}/{g8:.3f}", flush=True)
            for name, code, const, frames, k, t_awgn, t_mpd in PANELS[a.panel or a.band]:
                spec = SubmodeSpec(15, name, code, const, frames, k=k, band=a.band)
                sim = None
                for chan in a.channels:
                    key = (str(hr), str(ov), name, chan)
                    if key in seen:
                        continue
                    sim = sim or Sim(spec, a.device, max(1, BURST_FRAMES // frames),
                                     batch=max(8, 256 // max(1, BURST_FRAMES // frames)),
                                     clip_setting=(hr, ov), clip_consts=consts)
                    t0 = time.time()
                    start = t_awgn if chan == "awgn" else t_mpd
                    thr = threshold(sim, chan, round(start * 4) / 4, verbose=False)
                    w.writerow([a.band, hr, ov, name, chan, thr, f"{papr:.2f}", f"{peak:.2f}",
                                f"{sdr:.2f}", f"{g8:.3f}", f"{time.time() - t0:.0f}"])
                    f.flush()
                    print(f"   {name:11s} {chan:4s} {thr:6.2f} dB  PEP-fair {thr + peak:6.2f}", flush=True)


if __name__ == "__main__":
    main()
