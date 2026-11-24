"""FM plan M6: the FM gear shifter judged by whole ARQ sessions (the real
modem, ARQ and shifter: scripts/phy_session.py's RealPhy on FmSessionChannel),
bulk transfer, paired seeds: each arm runs the same seeds, channels and data.

Arms are predictor settings (environment for the session's process, ARMS):
    prov      the provisional FM predictor and its table, as committed before the
              calibration (static column, 2 logits per dB, 6 dB for another band)
    fit       the calibrated predictor (FM's defaults)
    fitnoreg  fit, every burst read as steady noise (DATA2G_FM_REGIME=0)
    fit2      fit, steady slope 2.0 logits per dB (the fit's 1.50)
    fitr1     fit, fading from a noise spread of 1.0 dB (0.7)
    fitr1-    fitr1 without predictor.FM_STEADY_OFFSET_DB (fitr1 ran before it existed)
    off1      fitr1-, f108's band offset 1 dB lower
    off1s     the steady-only extra dB on f108 (FM's default). A first short burst on a new
              band (a probe, tried 2026-11-17) lost 0.6-1.5 kbps at every flat CNR above 10:
              runs/fm_session/probe_report.txt

Scenarios: each radio (TX path = radio) on static, flutter at 440 MHz and
mobile-rural at CNRs spanning its ladder, and a CNR ramp through threshold
(high to low and back over the horizon). Score: delivered bytes / horizon.

    DATA2G_PROFILE=fm python scripts/fm_session_study.py run [--seeds 6] [--workers 8] [--arms prov new]
    DATA2G_PROFILE=fm python scripts/fm_session_study.py report
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "fm_session"
HORIZON = 120.0
CNRS = {"mic": (8, 14, 20, 28), "flat": (10, 16, 24, 32)}
CONDS = ("static", "flutter@440", "mobile-rural")
ARMS = {"prov": {"DATA2G_FM_PREDICTOR": "prov",
                 "DATA2G_FM_THRESHOLDS": str(ROOT / "runs" / "fm_regime" / "thr_provisional.json")},
        "fit": {}, "fitnoreg": {"DATA2G_FM_REGIME": "0"},
        "fit2": {"DATA2G_FM_SLOPE_STEADY": "2.0"}, "fitr1": {"DATA2G_FM_FADING_SPREAD_DB": "1.0"},
        # predictor.FM_STEADY_OFFSET_DB: fitr1 is the shifter before it (none), off1s with it
        # (the default since); off1, f108's offset 1 dB lower on every regime
        "fitr1-": {"DATA2G_FM_STEADY_OFFSET": ""}, "off1": {"DATA2G_FM_STEADY_OFFSET": "",
                                                            "DATA2G_FM_BAND_OFFSET": "f108:-4.9"},
        "off1s": {}}


def ramp(t: float) -> float:
    """30 dB down to 6 and back over the horizon (a fade through threshold)."""
    return 6.0 + 24.0 * abs(1.0 - 2.0 * min(max(t, 0.0), HORIZON) / HORIZON)


def scenarios():
    out = [(r, c, cnr) for r in CNRS for c in CONDS for cnr in CNRS[r]]
    return out + [(r, "static", "ramp") for r in CNRS]


def one(args):
    radio, cond, cnr, seed = args
    sys.path.insert(0, str(ROOT / "scripts"))
    import linksim as L
    import phy_session as PS

    from data2g import modem

    modem.TX_PATH = radio
    c, mhz = (cond.split("@") + ["146"])[:2]
    ch = PS.FmSessionChannel(c, radio, ramp if cnr == "ramp" else float(cnr), seed=seed, horizon=HORIZON,
                             carrier_mhz=float(mhz))
    phy = PS.RealPhy(ch)
    res = L.run(L.make_policy("shift"), L.make_policy("shift"), None, [("a", 10_000_000, None, 0.0)], seed=seed,
                horizon=HORIZON, phy=phy, cap=1 if radio == "mic" else 2)
    return {"radio": radio, "condition": cond, "cnr": cnr, "seed": seed, "bps": round(8 * res["delivered"] / HORIZON),
            "bursts": res["bursts"], "lost_sync": res["lost_sync"], "timeouts": res["timeouts"]}


def run_arm(arm, seeds, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(r, c, cnr, s) for r, c, cnr in scenarios() for s in range(seeds)]
    cols = ["radio", "condition", "cnr", "seed", "bps", "bursts", "lost_sync", "timeouts"]
    with (OUT / f"{arm}.csv").open("w", newline="") as fh, ProcessPoolExecutor(workers) as ex:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for row in ex.map(one, jobs):
            w.writerow(row)
            fh.flush()
            print(arm, row, flush=True)


def report(arms):
    data = {}
    for a in arms:
        with (OUT / f"{a}.csv").open() as fh:
            for r in csv.DictReader(fh):
                data[(a, r["radio"], r["condition"], r["cnr"], int(r["seed"]))] = float(r["bps"])
    base, rest = arms[0], arms[1:]
    keys = sorted({k[1:4] for k in data if k[0] == base})
    rng = np.random.default_rng(0)
    print(f"bps, mean over seeds; paired difference against {base} with a 90% bootstrap interval")
    print("| radio | condition | CNR | " + base + " | " + " | ".join(f"{a} (diff)" for a in rest) + " |")
    print("|---|---|---|---|" + "---|" * len(rest))
    for k in keys:
        seeds = sorted(s for (a, *kk, s) in data if a == base and tuple(kk) == k)
        b = np.array([data[(base, *k, s)] for s in seeds])
        cells = []
        for a in rest:
            v = np.array([data.get((a, *k, s), np.nan) for s in seeds])
            d = v - b
            boot = [np.nanmean(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
            lo, hi = np.nanpercentile(boot, [5, 95])
            cells.append(f"{np.nanmean(v):.0f} ({np.nanmean(d):+.0f} [{lo:+.0f},{hi:+.0f}])")
        print(f"| {k[0]} | {k[1]} | {k[2]} | {b.mean():.0f} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report", "arm"])
    ap.add_argument("--arm")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.what == "arm":
        return run_arm(a.arm, a.seeds, a.workers)
    if a.what == "run":
        for arm in a.arms:  # one process each: the predictor and policy read their env at import
            env = {k: v for k, v in os.environ.items() if not k.startswith("DATA2G_FM_")}
            subprocess.run([sys.executable, __file__, "arm", "--arm", arm, "--seeds", str(a.seeds),
                            "--workers", str(a.workers)], env={**env, "DATA2G_PROFILE": "fm", **ARMS[arm]}, check=True)
    report(a.arms)


if __name__ == "__main__":
    main()
