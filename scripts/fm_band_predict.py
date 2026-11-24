"""FM plan M6: predicting another band's snr_est from the measured per-carrier
signal and noise (band_snr below) against the constant band offsets
(predictor.FM_BAND_OFFSET_DB). The same channel draw carries a QPSK burst on
each band (as scripts/fm_band_offsets.py); for each pair source -> target the
error of each prediction against the target's own snr_est, per radio and
condition, synced bursts below the bands' SNR ceiling.

    DATA2G_PROFILE=fm python scripts/fm_band_predict.py [--workers 4]

Result (2026-11-17, runs/fm_band/predict.txt, 6 seeds x 6 CNRs): the constant
is unbiased from f55 to f108 on flat (static median +0.03 dB, rms 0.79;
flutter +0.25, 1.17), and the prediction is not (+5.2 dB): not shipped. The
f108 probe that failed at CNR 10 flat was not an offset bias; policy.BAND_PROBE
makes such a try short instead.
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
PAIRS = (("f55", "f108"), ("w", "f108"), ("w", "f55"), ("n10", "f55"), ("n10", "f108"), ("f108", "f55"))
CNRS = (6, 9, 12, 15, 18, 21)
TRIALS = 6
CONDS = ("static", "flutter", "mobile-rural")


def band_snr(est, band: str, snr_est: float) -> dict:
    """Each band's snr_est predicted from one burst's per-carrier signal
    (|h|^2) and noise (n0_k), each a line in dB against log frequency, over
    the other band's carriers: snr_est times the ratio of mean signal to mean
    noise (at one station level the carrier counts cancel). Not used: see
    the result above. {} for a band under an octave wide (n10)."""
    from data2g.waveform import ofdm

    f = np.log2(ofdm.band(band).freqs.astype(float))
    if f[-1] - f[0] < 1.0 or np.ndim(est.get("n0_k")) != 3:
        return {}
    ps = np.polyfit(f, 10 * np.log10(np.mean(np.abs(est["h"]) ** 2, axis=(0, 1))), 1)
    pn = np.polyfit(f, 10 * np.log10(np.mean(np.asarray(est["n0_k"], float), axis=(0, 1))), 1)

    def ratio(b):
        g = np.log2(ofdm.band(b).freqs.astype(float))
        return np.mean(10 ** (np.polyval(ps, g) / 10)) / np.mean(10 ** (np.polyval(pn, g) / 10))

    return {b: float(snr_est + 10 * np.log10(ratio(b) / ratio(band))) for b in MODES}


def job(args):
    radio, cond, band = args
    from data2g import audio, modem
    from data2g import fmchannel as F
    from data2g.arq import phy as PHY
    from data2g.config import FS

    spec = L.spec_of(MODES[band])
    mhz = 440.0 if cond == "flutter" else 146.0
    out = {}
    for cnr in CNRS:
        for t in range(TRIALS):
            _, x = L.burst(spec, 4000 + t, radio)
            ch = F.FmChannel(cond, S.radio_of(radio), cnr, seed=900 + t, duration_s=len(x) / FS + 2,
                             drive_dev_hz=L.STATION_DRIVE[L.tx_path(radio)], carrier_mhz=mhz)
            try:
                r = modem.receive(audio.Blanker()(ch.apply(x)))
            except modem.SyncError:
                continue
            snr = PHY.measure(r)["snr_est"]
            out[(cnr, t)] = (snr, band_snr(r["est"], r["band"], snr))
    return radio, cond, band, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from data2g.arq import predictor as P

    jobs = [(r, c, b) for r in ("mic", "flat") for c in CONDS for b in MODES if not (r == "mic" and b == "f108")]
    got = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for radio, cond, band, out in ex.map(job, jobs):
            got[(radio, cond, band)] = out
    off = P.FM_BAND_OFFSET_DB
    print("error of each prediction against the target's own snr_est (dB): median [p10, p90] / rms")
    for radio in ("mic", "flat"):
        for cond in CONDS:
            for src, tgt in PAIRS:
                a_, b_ = got.get((radio, cond, src)), got.get((radio, cond, tgt))
                if not a_ or not b_:
                    continue
                ec, ep = [], []
                for k, (snr_s, pred) in a_.items():
                    if k not in b_:
                        continue
                    snr_t = b_[k][0]
                    ec.append(snr_s + off[tgt] - off[src] - snr_t)
                    if tgt in pred:
                        ep.append(pred[tgt] - snr_t)

                def fmt(e):
                    e = np.asarray(e)
                    return (f"{np.median(e):+5.2f} [{np.percentile(e, 10):+5.2f},{np.percentile(e, 90):+5.2f}] "
                            f"/ {np.sqrt(np.mean(e ** 2)):4.2f}") if len(e) else "        n/a"

                print(f"{radio:4s} {cond:12s} {src:4s}->{tgt:4s} n={len(ec):2d}  constant {fmt(ec)}   predicted {fmt(ep)}")


if __name__ == "__main__":
    main()
