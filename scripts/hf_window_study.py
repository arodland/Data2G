"""Issue #212: what windowed interleaving (codes.spread group,
config.SPREAD_WINDOW) and a per-window channel estimate (config.EST_WINDOW)
cost on HF, in dB.

Fixed modes, bursts of about SECONDS (the Pi's slow 256l x39 is 11.2 s), a
control codeword plus data codewords, through hfchannel (CFO +-50 Hz,
10 ppm, random lead) and modem.receive (offline: the window changes the data
path, not receiver state). Trial k has the same seed (payload, lead, CFO,
fading, noise) at every SNR and in every arm, so arms are paired. Metric:
data codewords failed (a miss loses them all), and its 10% point per mode
and channel, interpolated over the SNR grid. SNR is average power, the same
mode on both sides: the window does not change a burst's peak.

DD runs its DD_ITERS without the wall-clock budget (--dd-budget 1e9), so the
arms don't depend on machine load; latency is the Pi bench's business.

    python scripts/hf_window_study.py run [--only all w32 ...] [--seeds 20] [--jobs 8]
    python scripts/hf_window_study.py report

Pure Python: the native core has its windows compiled in (0), so a
tools/with_native.py arm would despread an env-windowed burst wrongly
(measured: 36 of 37 against Python's 37 on a clean 256l burst).
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "hf_window"
SECONDS = 11.5
VARIANTS = {  # frames; a frame is 144 ms
    "all": {},
    "w48": {"DATA2G_SPREAD_WINDOW": "48"},  # 6.9 s
    "w32": {"DATA2G_SPREAD_WINDOW": "32"},  # 4.6 s
    "w16": {"DATA2G_SPREAD_WINDOW": "16"},  # 2.3 s
    "e32": {"DATA2G_EST_WINDOW": "32"},     # the estimate per window alone (receiver-only)
    "we16": {"DATA2G_SPREAD_WINDOW": "16", "DATA2G_EST_WINDOW": "16"},  # what a window-by-window receiver sees
}
# 10% centre per mode (runs/ladder_10pct.csv, 2-codeword bursts: a rough centre only); mps from mpg
MODES = {"n4-qpsk-r1/3": (-9.75, -0.06, -2.0, 3.69), "n10-qpsk-r1/2": (-3.06, 5.94, 1.63, 3.94),
         "w48-qpsk-r1/2": (2.56, 8.56, 7.56, 7.38), "w48-16qam-r1/2": (8.31, 14.88, 14.13, 13.75),
         "w48-64l-r3/4": (17.5, 26.63, 25.81, None), "w48-256l-r5/8": (19.06, 26.38, 27.38, None)}
CHANNELS = ("awgn", "mpg", "mpp", "mpd", "mps")
FIELDS = ["mode", "channel", "snr", "seed", "n_data", "ok", "sync"]


def grid(mode, chan):
    awgn, mpg, mpp, mpd = MODES[mode]
    c = {"awgn": awgn, "mpg": mpg, "mpp": mpp, "mpd": mpd, "mps": mpg}[chan]
    if c is None:
        return []
    if chan == "awgn":
        return [round(c - 4 + 0.5 * i, 2) for i in range(15)]
    return [round(c - 8 + i, 2) for i in range(14)]  # long bursts' 10% points sit up to 5 dB under the centre


def trial(args):
    mode, chan, snr, seed = args
    import numpy as np

    import outcome_data as O
    from data2g import hfchannel, modem
    from data2g.arq import phy as PHY
    from data2g.config import FS

    rng = np.random.default_rng(seed)
    b = O.burst(mode, O.n_for(mode, SECONDS), rng)
    lead = int(rng.uniform(0.3, 1.0) * FS)
    x = np.concatenate([np.zeros(lead), PHY.tx_audio(b), np.zeros(FS // 2)])
    y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=float(rng.uniform(-50, 50)), ppm=10,
                                fading_preset=None if chan == "awgn" else chan, seed=seed)
    n_data = len(b.slots) - 1
    try:
        r = modem.receive(y)
    except Exception:  # noqa: BLE001 (no sync, or a header read past the audio)
        r = None
    if r is None or r["spec"].name != mode or r["n_cw"] != len(b.slots):
        return dict(mode=mode, channel=chan, snr=snr, seed=seed, n_data=n_data, ok=0, sync=0)
    rx = PHY.ModemRx(r, {})
    ok = sum(rx.decode(i, s.mask_id, 0, None) == s.payload for i, s in enumerate(b.slots) if i)
    return dict(mode=mode, channel=chan, snr=snr, seed=seed, n_data=n_data, ok=int(ok), sync=1)


def run_variant(name, seeds, jobs):
    from multiprocessing import Pool

    from data2g import codes, config
    from data2g.arq.modes import MODES as M

    assert config.SPREAD_WINDOW == int(VARIANTS[name].get("DATA2G_SPREAD_WINDOW", 0)), "env not applied"
    print(f"{name}: SPREAD_WINDOW {config.SPREAD_WINDOW} EST_WINDOW {config.EST_WINDOW} "
          f"groups {[codes.spread_group(M[m]) for m in MODES]} ({config.__file__})", flush=True)
    path = OUT / f"{name}.csv"
    done = set()
    if path.exists():  # resume: a trial is a pure function of its key
        done = {(r["mode"], r["channel"], float(r["snr"]), int(r["seed"])) for r in csv.DictReader(open(path))}
    work = [(m, c, s, k) for m in MODES for c in CHANNELS for s in grid(m, c) for k in range(seeds)
            if (m, c, s, k) not in done]
    print(f"{name}: {len(work)} trials to run, {len(done)} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(jobs) as pool, open(path, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if not done:
            w.writeheader()
        for row in pool.imap_unordered(trial, work, chunksize=4):
            w.writerow(row)
            f.flush()


def point10(rows):
    """SNR where the data codeword failure rate falls through 10%, from the
    top of the grid down (linear between grid points); None if never."""
    by = {}
    for r in rows:
        n, ok = by.get(r["snr"], (0, 0))
        by[r["snr"]] = (n + r["n_data"], ok + r["ok"])
    pts = sorted((s, 1 - ok / n) for s, (n, ok) in by.items())
    if not pts or pts[-1][1] > 0.1 or pts[0][1] <= 0.1:
        return None
    for (s0, f0), (s1, f1) in zip(reversed(pts[:-1]), reversed(pts[1:])):
        if f0 > 0.1:  # first crossing coming down from the top
            return s0 + (f0 - 0.1) / (f0 - f1) * (s1 - s0)
    return None


def report():
    data = {}
    for v in VARIANTS:
        p = OUT / f"{v}.csv"
        if p.exists():
            for r in csv.DictReader(open(p)):
                key = (r["mode"], r["channel"])
                data.setdefault(v, {}).setdefault(key, []).append(
                    dict(snr=float(r["snr"]), n_data=int(r["n_data"]), ok=int(r["ok"])))
    if "all" not in data:
        sys.exit("no baseline (all.csv)")
    arms = [v for v in VARIANTS if v in data and v != "all"]
    print("10% data-codeword failure point; arms in dB against `all` (positive: worse)")
    print(f"{'mode':16} {'chan':5} {'all':>7} " + " ".join(f"{v:>6}" for v in arms))
    for key, rows in sorted(data["all"].items(), key=lambda kv: (list(MODES).index(kv[0][0]), CHANNELS.index(kv[0][1]))):
        base = point10(rows)
        cells = []
        for v in arms:
            p = point10(data[v].get(key, []))
            cells.append("   n/a" if base is None or p is None else f"{p - base:+6.2f}")
        print(f"{key[0]:16} {key[1]:5} {'n/a' if base is None else f'{base:7.2f}':>7} " + " ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report", "variant"])
    ap.add_argument("--variant")
    ap.add_argument("--only", nargs="+", help="run: these variants only")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.what == "variant":
        return run_variant(a.variant, a.seeds, a.jobs)
    if a.what == "run":
        caps = {v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "DATA2G_THREADS")}
        for v in a.only or VARIANTS:  # one process each: config reads its overrides at import
            subprocess.run([sys.executable, str(ROOT / "tools" / "with_native.py"), "--python", "--dd-budget", "1e9",
                            __file__, "variant", "--variant", v, "--seeds", str(a.seeds), "--jobs", str(a.jobs)],
                           env={**os.environ, **caps, **VARIANTS[v]}, check=True)
    report()


if __name__ == "__main__":
    main()
