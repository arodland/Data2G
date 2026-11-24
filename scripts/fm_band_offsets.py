"""FM plan M6: how a band's snr_est compares with another's on the same
channel (the gear shifter predicts a mode on one band from a burst measured
on another: the robust connect is n10, data rungs f55/f108). Per radio, the
same channel draws (static, a few seeds) and CNRs carry a QPSK burst on each
band at the station's drive; the median of snr_est(band) - snr_est(f55),
synced bursts below each band's SNR ceiling (CNR <= MAX_CNR).

    DATA2G_PROFILE=fm python scripts/fm_band_offsets.py [--workers 2]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_ladder_study as L
import fm_spacing_study as S

MODES = {"w": "qpsk-r1/2", "n10": "n10-qpsk-r1/2", "f55": "f55-qpsk-r40", "f108": "f108-qpsk-r41"}
CNRS = (6, 9, 12, 15, 18, 21)
MAX_CNR = 21
TRIALS = 6


def job(args):
    radio, band = args
    from data2g import audio, modem
    from data2g import fmchannel as F
    from data2g.arq import phy as PHY
    from data2g.config import FS

    spec = L.spec_of(MODES[band])
    out = {}
    for cnr in CNRS:
        for t in range(TRIALS):
            _, x = L.burst(spec, 4000 + t, radio)
            ch = F.FmChannel("static", S.radio_of(radio), cnr, seed=900 + t, duration_s=len(x) / FS + 2,
                             drive_dev_hz=L.STATION_DRIVE[L.tx_path(radio)])
            try:
                out[(cnr, t)] = PHY.measure(modem.receive(audio.Blanker()(ch.apply(x))))["snr_est"]
            except modem.SyncError:
                pass
    return radio, band, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    jobs = [(r, b) for r in ("mic", "flat") for b in MODES if not (r == "mic" and b == "f108")]
    got = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for radio, band, out in ex.map(job, jobs):
            got[(radio, band)] = out
    for radio in ("mic", "flat"):
        ref = got[(radio, "f55")]
        for band in MODES:
            if (radio, band) not in got:
                continue
            d = [v - ref[k] for k, v in got[(radio, band)].items() if k in ref and k[0] <= MAX_CNR]
            print(f"{radio:4s} {band:5s} snr_est - f55's: median {np.median(d):+.2f} dB "
                  f"[p10 {np.percentile(d, 10):+.2f}, p90 {np.percentile(d, 90):+.2f}] n={len(d)}")


if __name__ == "__main__":
    main()
