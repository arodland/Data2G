"""FM plan M3: a damaged burst start. The sending radio keys up later than
its TX delay allowed, or the receiver's squelch opens late: the receiver
hears hiss where the burst's start was. How late can the carrier arrive?

    python scripts/fm_start_study.py [--trials 40] [--workers 10] [--cnr 25]

Through data2g.fmchannel on the FM profile and the live streaming receiver
(fm_spacing_study.stream_receive, its 50 Hz w48 modes, FM's defaults: +-12.5
Hz, threshold 28.7, per-symbol noise). The carrier arrives LATE seconds after
the burst's first sample (-0.2: 0.2 s early, a normal TX delay). The burst
opens with 20 ms of silence, then ~164 ms of preamble (8 repeats), then the
header; w48 repeats its header after data frame 2. `transient` adds a
synthesiser settling from 1.5 kHz off in 20 ms after key-up, as cheap
handhelds do. Per point: bursts fully decoded, and acquired at all, out of
the trials (paired seeds across every LATE).
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_spacing_study as S  # noqa: E402

LATES = (-0.2, 0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.3, 0.45)
CONDS = ("static", "mobile-urban")


def job(args):
    from data2g import fmchannel as F
    from data2g.config import FS

    mode, radio, cond, transient, trial, cnr, drive = args
    spec = S.specs()[mode]
    sent, x = S.burst(spec, 1000 + trial)
    out = []
    for late in LATES:
        ch = F.FmChannel(cond, radio, cnr, seed=trial, duration_s=len(x) / FS + 4, drive_dev_hz=drive,
                         keyup_dev_hz=1500.0 if transient else 0.0, keyup_tau_s=0.02)
        lost, nosync, _ = S.stream_receive(spec, sent, x, ch, keyup_late_s=late)
        out.append((mode, radio, cond, transient, late, lost == 0, not nosync))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cnr", type=float, default=25.0)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    from data2g import config, fmchannel as F

    assert config.PROFILE == "fm", "run under DATA2G_PROFILE=fm"
    sp = S.specs()
    drives = {}
    for m in S.MODES:
        _, x = S.burst(sp[m], 0)
        for r in S.RADIOS:
            drives[(m, r)] = F.drive_for_bandwidth(x[abs(x) > 0], r, 15_000.0)[0]
    jobs = [(m, r, c, tr, t, a.cnr, drives[(m, r)]) for m in S.MODES for r in S.RADIOS for c in CONDS
            for tr in (False, True) for t in range(a.trials)]
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for rows in ex.map(job, jobs, chunksize=2):
            for mode, radio, cond, tr, late, ok, acq in rows:
                k = (mode, radio, cond, tr, late)
                n, o, q = res.get(k, (0, 0, 0))
                res[k] = (n + 1, o + ok, q + acq)
    print(f"CNR {a.cnr:g} dB: bursts fully decoded / acquired (of {a.trials}) by how late the carrier arrives (s)")
    print(f"{'mode':>10} {'radio':>4} {'condition':>12} {'transient':>9}  " + " ".join(f"{t:>7g}" for t in LATES))
    for mode in S.MODES:
        for radio in S.RADIOS:
            for cond in CONDS:
                for tr in (False, True):
                    cells = [res[(mode, radio, cond, tr, t)] for t in LATES]
                    print(f"{mode:>10} {radio:>4} {cond:>12} {str(tr):>9}  "
                          + " ".join(f"{o:>3}/{q:<3}" for _, o, q in cells))


if __name__ == "__main__":
    main()
