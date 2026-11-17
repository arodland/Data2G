"""FM ping verdict thresholds (data2g.arq.ping; FM plan M7): what the far
end measures of a ping burst for each station setup, on the FM simulator.

Cases (radio, tx_path, station level in Hz rms deviation for unit-RMS audio:
the width-limited drive, fork K's 1460 on mic and 2350 on flat):
    A mic radio, tx_path mic, normal level        -> level OK
    B mic radio, tx_path flat (overdriven)        -> limiter
    C flat radio, tx_path flat, normal            -> level OK
    D flat radio, tx_path mic                     -> flat-path tilt
    E mic radio, tx_path mic, 6 dB too hot        -> limiter
fmchannel normalises each burst to its own rms, so the ping's level (its
tx_level, the mode's gain) goes into the drive. Static and flutter at 440,
CNR 12..30, 4 seeds; both bands (f55 on both radios, f108 on flat).

    DATA2G_PROFILE=fm python scripts/fm_ping_study.py [--workers 4]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

LEVEL = {"mic": 1460.0, "flat": 2350.0}
CASES = {"A": ("mic", "mic", 1.0, "ok"), "B": ("mic", "flat", 1.0, "limiter"), "C": ("flat", "flat", 1.0, "ok"),
         "D": ("flat", "mic", 1.0, "tilt"), "E": ("mic", "mic", 2.0, "limiter")}
CNRS = (12, 16, 20, 25, 30)
CONDS = (("static", 146.0), ("flutter", 440.0))


def job(a):
    from data2g import fmchannel as F
    from data2g import modem
    from data2g.arq import burst as B
    from data2g.arq import frames as Fr
    from data2g.arq import ping as PG
    from data2g.arq.phy import tx_audio
    from data2g.arq.policy import GearShifter
    from data2g.config import FS

    case, cap, cond, mhz, cnr, seed = a
    radio, path, hot, want = CASES[case]
    mode = PG.PING_MODES[cap]
    pb = GearShifter().payload_bytes(mode)
    body = PG.ping_body("N0CALL", "W1AW", None, pb)
    payloads = Fr.Control(Fr.Core(ftype=Fr.SESSION), {Fr.T_PING: body}).pack(pb)
    b = B.TxBurst(mode, [B.Slot(B.ctl_mask(0, i, 0), 0, p) for i, p in enumerate(payloads)], 0, cap)
    x = tx_audio(b, tx_path=path, level=1.0)
    x = np.concatenate([np.zeros(FS // 2), x, np.zeros(FS // 2)])
    drive = LEVEL[radio] * hot * PG.tx_level(path)
    ch = F.FmChannel(cond, radio, cnr, seed=seed, duration_s=len(x) / FS + 2, carrier_mhz=mhz, drive_dev_hz=drive)
    y = ch.apply(x)
    try:
        r = modem.receive(y)
        if r["n_cw"] != len(payloads):
            raise ValueError("n_cw")
        m = PG.measure(r, payloads, PG.level(y * 0.1))
    except Exception:
        return (case, cap, cond, cnr, seed, want, None)
    return (case, cap, cond, cnr, seed, want, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(c, cap, cond, mhz, cnr, s) for c in CASES for cap in (1, 2) for cond, mhz in CONDS for cnr in CNRS
            for s in range(4) if not (cap == 2 and CASES[c][0] == "mic")]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, jobs, chunksize=2))
    from data2g.arq import ping as PG

    print("case cap cond cnr: snr dist tilt(bottom-top) -> verdict (mean over seeds; lost = not decoded)")
    conf = {}
    for c in CASES:
        for cap in (1, 2):
            for cond, _ in CONDS:
                for cnr in CNRS:
                    ms = [r[6] for r in rows if r[:4] == (c, cap, cond, cnr)]
                    if not ms:
                        continue
                    got = [m for m in ms if m is not None]
                    for m in got:
                        tx = CASES[c][1]
                        v = PG.verdict(dict(m, clip_permille=0, peak_dbfs=-20.0), tx, PG.PING_MODES[cap].split("-")[0])
                        k = ("limiter" if any("limiter" in s for s in v) else "tilt" if any("tx_path flat" in s for s in v)
                             else "low" if any("too low" in s for s in v) else "ok")
                        conf.setdefault((CASES[c][3], k), 0)
                        conf[(CASES[c][3], k)] += 1
                    if got:
                        print(f"{c} {cap} {cond:7} {cnr:2}: snr {np.mean([m['snr_db'] for m in got]):5.1f} "
                              f"dist {np.mean([m['dist_db'] for m in got]):5.2f} (max {max(m['dist_db'] for m in got):5.2f}) "
                              f"tilt {np.mean([m['thirds_db'][0] - m['thirds_db'][2] for m in got]):5.1f} "
                              f"noise_tilt {np.mean([m['noise_tilt_db'] for m in got]):5.1f} "
                              f"lost {len(ms) - len(got)}")
    print("confusion (want -> got: count):")
    for k in sorted(conf):
        print(f"  {k[0]:8} -> {k[1]:8} {conf[k]}")


if __name__ == "__main__":
    main()
