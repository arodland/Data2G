"""FM plan M6: what the receiver sees that tells static, flutter and mobile
apart (HF's regime input, the Doppler spread, is about zero on FM audio),
and whether snr_est puts mic and flat on one scale.

Per burst through fmchannel (station drive per TX path, as
scripts/fm_ladder_study.py), the audio impulse-blanked as the live receiver
does, then modem.receive: snr_est (arq.phy.measure), codewords decoded, and
candidate regime inputs the receiver has for free:
    clicks     share of data symbols whose per-symbol noise factor
               (modem.symbol_noise_scale) is over CLICK_FACTOR: clicked symbols
    n0_spread  std over frames of the per-frame noise (EST_WINDOW's n0_k), dB
    blanked    share of samples the blanker touched

    DATA2G_PROFILE=fm python scripts/fm_regime_study.py run [--trials 6] [--workers 2]
    DATA2G_PROFILE=fm python scripts/fm_regime_study.py calib [--trials 4] [--workers 8]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# bursts of a few seconds, as FM sessions send: several EST_WINDOW windows each
os.environ.setdefault("FM_BURST_FRAMES", "24")

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_ladder_study as L
import fm_spacing_study as S

OUT = S.ROOT / "runs" / "fm_regime"
CLICK_FACTOR = 4.0
JOBS = [("f55-qpsk-r40", "mic"), ("f55-64l-r67", "flat"), ("f108-64l-r67", "flat"), ("n10-qpsk-r1/2", "mic")]
CONDS = ["static", "fixed", "fixed-fringe", "knife-edge", "flutter@146", "flutter@440", "pedestrian@440",
         "mobile-rural", "mobile-rural@440", "mobile-urban", "mobile-urban@440"]
CNRS = range(6, 38, 4)


def features(r) -> dict:
    from data2g import constellation, modem
    from data2g.arq import phy as PHY

    est = r["est"]
    var = np.broadcast_to(modem.noise_var(est["h"], est) + est["mse"], est["h"].shape)
    pts = constellation.load(r["spec"].constellation)
    s = modem.symbol_noise_scale(modem.data_symbols(r["raw"]), est["h"], var, pts)
    n0 = np.asarray(est.get("n0_k", est["n0"]), float)
    per_frame = 10 * np.log10(np.mean(n0, axis=tuple(range(1, n0.ndim)))) if n0.ndim == 3 else np.zeros(1)
    return {"snr_est": PHY.measure(r)["snr_est"], "clicks": float(np.mean(s > CLICK_FACTOR)), "clicks10": float(np.mean(s > 10)),
            "n0_spread": float(np.std(per_frame))}


def job(args):
    name, radio, cond, trials, *cnrs = args
    cnrs = cnrs[0] if cnrs else CNRS
    from data2g import audio, config, constellation, modem
    from data2g import fmchannel as F
    from data2g.config import FS

    spec = L.spec_of(name)
    c, mhz = (cond.split("@") + ["146"])[:2]
    bits = constellation.bits_per_symbol(constellation.load(spec.constellation))
    path = L.tx_path(radio)
    drive = L.STATION_DRIVE[path] * (config.FM_MIC_GAIN[bits] if path == "mic" else 1.0)
    rows = []
    for cnr in cnrs:
        for t in range(trials):
            sent, x = L.burst(spec, 3000 + t, radio)
            ch = F.FmChannel(c, S.radio_of(radio), cnr, seed=700 + t, duration_s=len(x) / FS + 2, carrier_mhz=float(mhz),
                             drive_dev_hz=drive)
            b = audio.Blanker()
            y = b(ch.apply(x))
            row = {"mode": name, "radio": radio, "condition": cond, "cnr": cnr, "trial": t,
                   "blanked": b.n_blanked / len(y)}
            try:
                r = modem.receive(y)
                d = modem.decode_received(r)
                row.update(features(r), ok=sum(p == q and k for p, q, k in zip(d.payloads, sent, d.crc_ok)) / len(sent))
            except modem.SyncError:
                row.update(snr_est=np.nan, clicks=np.nan, clicks10=np.nan, n0_spread=np.nan, ok=0.0)
            rows.append(row)
    return rows


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["mode", "radio", "condition", "cnr", "trial", "blanked", "snr_est", "clicks", "clicks10", "n0_spread", "ok"]
    with (OUT / "features.csv").open("w", newline="") as fh, ProcessPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for rows in ex.map(job, [(m, r, c, a.trials) for m, r in JOBS for c in CONDS]):
            w.writerows(rows)
            fh.flush()
            print(rows[0]["mode"], rows[0]["radio"], rows[0]["condition"], flush=True)


CALIB_CONDS = ("static", "fixed-fringe", "flutter@440", "mobile-rural", "mobile-urban@440")


def calib(a):
    """Per-burst data for every FM-selectable mode on each path that sends
    it, at CNRs around its static 10% knee (runs/fm_ladder/shipped.csv,
    wn10.csv) on CALIB_CONDS -> runs/fm_regime/calib.csv: the gear shifter's
    P(codeword decodes | snr_est, n0_spread) is fitted from it
    (scripts/fm_threshold_table.py fit)."""
    from data2g.arq import modes

    knee = {}
    for src in ("shipped", "wn10"):
        with (L.OUT / f"{src}.csv").open() as fh:
            for r in csv.DictReader(fh):
                if r["condition"] == "static" and r["k0.1"] != "nan":
                    knee[(r["mode"], r["radio"])] = float(r["k0.1"])
    jobs = [(m, r, c, a.trials, [round(k + d, 2) for d in range(-3, 18, 3)])
            for (m, r), k in sorted(knee.items()) if m in modes.SELECTABLE for c in CALIB_CONDS]
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["mode", "radio", "condition", "cnr", "trial", "blanked", "snr_est", "clicks", "clicks10", "n0_spread", "ok"]
    with (OUT / "calib.csv").open("w", newline="") as fh, ProcessPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for rows in ex.map(job, jobs):
            w.writerows(rows)
            fh.flush()
            print(rows[0]["mode"], rows[0]["radio"], rows[0]["condition"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "calib"])
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    run(a) if a.what == "run" else calib(a)


if __name__ == "__main__":
    main()
