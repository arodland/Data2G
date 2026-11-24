"""FM plan M6, provisional: codes_data/fm/mode_thresholds.json, FM's per-mode
thresholds in the receiver's own measure (snr_est, what the gear shifter
sees), so an FM ARQ session can choose among f55 / f108 rungs before the FM
outcome model exists.

For each f55 / f108 rung and condition, the snr_est the receiver reports at
the rung's measured 10% knee (runs/fm_ladder/shipped.csv: CNR and drive),
averaged over bursts that synced. f55 is shared by mic and flat stations and
the shifter does not yet know the peer's TX path (M6's capability per
direction), so each f55 threshold is the worse (higher) of the two paths.
Columns (policy.MODE_CHANNELS on FM): static, flutter (440 MHz), mobile
(mobile-rural), worst (their max). The floor and ACK rungs FM keeps (w, n10
up to index 15) were not measured on FM: HF's awgn, mpg, mpp, mpd thresholds
stand in for static, flutter, mobile, worst (same snr_est units).

    DATA2G_PROFILE=fm python scripts/fm_threshold_table.py measure --source shipped|wn10 [--trials 8] [--workers 8]
    DATA2G_PROFILE=fm python scripts/fm_threshold_table.py table
"""

import os

# before numpy: each worker one BLAS thread (8 workers on a shared machine)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_ladder_study as L
import fm_spacing_study as S

COLS = {"static": "static", "flutter@440": "flutter", "mobile-rural": "mobile"}


def snr_at_knee(args) -> tuple:
    """Mean snr_est of synced bursts at the knee CNR (nan: none synced)."""
    name, radio, cond, cnr, drive, trials = args
    from data2g import audio, modem
    from data2g import fmchannel as F
    from data2g.arq import phy as PHY

    spec = L.spec_of(name)
    c, mhz = (cond.split("@") + ["146"])[:2]
    vals = []
    for t in range(trials):
        _, x = L.burst(spec, 2000 + t, radio)
        ch = F.FmChannel(c, S.radio_of(radio), cnr, seed=500 + t, duration_s=len(x) / 16000 + 2, carrier_mhz=float(mhz),
                         drive_dev_hz=drive)
        try:
            vals.append(PHY.measure(modem.receive(audio.Blanker()(ch.apply(x))))["snr_est"])  # as the live receiver
        except modem.SyncError:  # not synced at the knee: says nothing about snr_est
            pass
    return name, radio, cond, float(np.mean(vals)) if vals else float("nan")


KNEES = ("shipped", "wn10")  # runs/fm_ladder/<name>.csv: f55/f108 rungs, then w and n10 (station drive)


def measure(a):
    """snr_est at every knee of a.source (10% and 1%, each path) -> runs/fm_ladder/knee_snr_<source>.csv."""
    with (L.OUT / f"{a.source}.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["condition"] in COLS]
    jobs = [(r["mode"], r["radio"], r["condition"], float(r[k]), float(r["drive"]), a.trials, k)
            for r in rows for k in ("k0.1", "k0.01") if r[k] != "nan"]
    with (L.OUT / f"knee_snr_{a.source}.csv").open("w", newline="") as fh, ProcessPoolExecutor(a.workers) as ex:
        w = csv.writer(fh)
        w.writerow(["mode", "radio", "condition", "level", "cnr", "snr_est"])
        for (name, radio, cond, v), j in zip(ex.map(snr_at_knee, [j[:6] for j in jobs]), jobs):
            w.writerow([name, radio, cond, j[6], j[3], round(v, 3)])
            fh.flush()


def table():
    """knee_snr_*.csv -> codes_data/fm/mode_thresholds.json: per mode and condition
    family, snr_est at its 10% knee, the worse (higher) of the paths that
    reach it; and the mic-flat gap on f55 (does snr_est unify the paths)."""
    from data2g import config
    from data2g.arq import modes

    got = {}
    for name in KNEES:
        with (L.OUT / f"knee_snr_{name}.csv").open() as fh:
            for r in csv.DictReader(fh):
                if r["level"] == "k0.1":
                    got.setdefault((r["mode"], COLS[r["condition"]]), {})[r["radio"]] = float(r["snr_est"])
    gaps = [v["mic"] - v["flat"] for (m, col), v in got.items() if m.startswith("f55-") and col == "static"
            and not np.isnan(v.get("mic", np.nan)) and not np.isnan(v.get("flat", np.nan))]
    if gaps:
        print(f"f55 static, snr_est at the 10% knee, mic minus flat: median {np.median(gaps):+.2f} dB, "
              f"range {min(gaps):+.2f}..{max(gaps):+.2f} over {len(gaps)} rungs")
    out = {}
    for name in (n for n in modes.MODES if n in modes.SELECTABLE):
        t = []
        for col in ("static", "flutter", "mobile"):
            v = [x for x in got.get((name, col), {}).values() if not np.isnan(x)]
            t.append(round(max(v), 2) if v else 99.0)  # no path reaches the knee: unusable there
        if all(x == 99.0 for x in t):
            continue
        out[name] = [*t, max(t)]
    path = Path(config.__file__).parent / "codes_data" / "fm" / "mode_thresholds.json"
    note = ("FM (scripts/fm_threshold_table.py): per mode, snr_est at its 10% codeword failure knee "
            "(scripts/fm_ladder_study.py: streamed, 30 paired trials; runs/fm_ladder/shipped.csv and wn10.csv), "
            "static, flutter (440 MHz), mobile (mobile-rural), worst; the worse of mic and flat where both reach it; "
            "99: never reached.")
    path.write_text(json.dumps({"note": note, "channels": ["static", "flutter", "mobile", "worst"], "modes": out},
                               indent=1) + "\n")
    print("wrote", path, len(out), "modes")


def fit():
    """runs/fm_regime/calib.csv (per burst: snr_est, n0_spread, codewords
    decoded; scripts/fm_regime_study.py calib) -> codes_data/fm/mode_thresholds.json.

    P(codeword decodes) = sigmoid(ln 9 + k_r (snr_est - t_mode - d [r = fading])),
    r = fading when n0_spread >= predictor.FM_FADING_SPREAD_DB: t_mode is the
    steady 90% point per mode (the worse path's, where mic and flat both send
    it), d and the slopes k_steady, k_fading shared. Weighted by codewords
    decoded; bursts that did not sync carry no snr_est and are left out.
    Columns as before (static, flutter: t; mobile: t + d; worst), so
    predictor.fm_outcome and the native table read them unchanged."""
    from scipy.optimize import minimize, minimize_scalar

    from data2g import config
    from data2g.arq import modes
    from data2g.arq import predictor as P

    rows = [r for r in csv.DictReader((S.ROOT / "runs" / "fm_regime" / "calib.csv").open()) if r["snr_est"] != "nan"]
    keys = sorted({(r["mode"], r["radio"]) for r in rows})
    data = {}
    for k in keys:
        rs = [r for r in rows if (r["mode"], r["radio"]) == k]
        data[k] = (np.array([float(r["snr_est"]) for r in rs]), np.array([float(r["n0_spread"]) >= P.FM_FADING_SPREAD_DB
                   for r in rs]), np.array([float(r["ok"]) for r in rs]))

    def nll_mode(t, g, x, fad, y):
        ks, kf, d = g
        z = np.log(9.0) + np.where(fad, kf * (x - t - d), ks * (x - t))
        q = np.clip(1 / (1 + np.exp(-np.clip(z, -30, 30))), 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    def best_t(g, k):
        x, fad, y = data[k]
        return minimize_scalar(lambda t: nll_mode(t, g, x, fad, y), bounds=(-20, 45), method="bounded")

    def total(g):
        if g[0] <= 0 or g[1] <= 0:
            return 1e12
        return sum(best_t(g, k).fun for k in keys)

    g = minimize(total, [2.0, 1.0, 1.0], method="Nelder-Mead", options={"xatol": 0.02, "fatol": 1.0}).x
    print(f"shared: k_steady {g[0]:.2f}/dB, k_fading {g[1]:.2f}/dB, fading offset d {g[2]:+.2f} dB "
          f"({sum(len(v[0]) for v in data.values())} bursts, {len(keys)} mode x path)")
    t = {}
    for k in keys:
        t.setdefault(k[0], []).append(best_t(g, k).x)
    out = {}
    for name in (n for n in modes.MODES if n in modes.SELECTABLE and n in t):
        v = round(max(t[name]), 2)
        out[name] = [v, v, round(v + g[2], 2), round(max(v, v + g[2]), 2)]
        print(name, out[name], "paths", [round(x, 2) for x in t[name]])
    path = Path(config.__file__).parent / "codes_data" / "fm" / "mode_thresholds.json"
    note = (f"FM (scripts/fm_threshold_table.py fit): per-burst calibration (scripts/fm_regime_study.py calib): "
            f"snr_est at P(codeword) = 0.9, steady noise (static, flutter columns) and fading (mobile column, "
            f"+{g[2]:.2f} dB); the worse of mic and flat. Slopes: steady {g[0]:.2f}, fading {g[1]:.2f} logits/dB "
            f"(predictor.FM_SLOPE_STEADY, FM_SLOPE_FADING).")
    path.write_text(json.dumps({"note": note, "channels": ["static", "flutter", "mobile", "worst"], "modes": out},
                               indent=1) + "\n")
    print("wrote", path, len(out), "modes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["measure", "table", "fit"])
    ap.add_argument("--source", choices=KNEES, default="shipped")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from data2g import config

    assert config.PROFILE == "fm", "run under DATA2G_PROFILE=fm"
    {"measure": lambda: measure(a), "table": table, "fit": fit}[a.what]()


if __name__ == "__main__":
    main()
