"""Fixed per-burst costs, paired: the same payload, channel seed, noise and
CFO through every arm, the preamble at the same absolute sample (the
lead-in difference goes into the pad ahead), one 1-codeword burst per
band through the full numpy receiver (modem.receive), as
scripts/sync_floor.py counts it: ok (right submode, count and start within
2 NCP), wrong (another header accepted) or fail.

Arms (modem module globals, patched per trial):
  lead800   LEADIN/LEADOUT 800 samples, the header copy on every w/w48 burst
  lead160   LEADIN/LEADOUT 160
  nocopy    lead160, and no header copy on these 1-codeword bursts

    python -m scripts.burst_trim --trials 2000 --out runs/burst_trim.csv

Found (2026-10-27, 2000 trials a cell, at each band's floor and 1.5 dB
under): lead160 matched lead800 in every one of the 48,000 paired trials
(0 differ). nocopy, failures at the floor / 1.5 dB under: w (ack-1f) AWGN
0.05% -> 7.2% / 7.0% -> 38.5%, MPG 1.3 -> 1.7 / 2.4 -> 3.9%, MPP 0.25 ->
1.35 / 0.4 -> 2.35%, MPD 0.1 -> 2.25 / 0.15 -> 3.45%; w48 (1 codeword)
AWGN 0 -> 0 / 0 -> 1.05%, MPG 0.4 -> 0.55 / 0.65 -> 1.35%, MPP 0.05 -> 1.0
/ 0.15 -> 1.7%, MPD 0.05 -> 1.0 / 0.1 -> 1.65%. No wrong headers in any arm.
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, hfchannel, modem  # noqa: E402
from data2g.config import FS, NCP, SUBMODES  # noqa: E402

BURST = {"w": "ack-1f", "w48": "w48-qpsk-r1/2", "n10": "n10-ack-4f"}
ARMS = {"lead800": (800, 0), "lead160": (160, 0), "nocopy": (160, 99)}
# runs/sync_floor.csv (2026-09-25, before the header copy): a cell at the floor and 1.5 dB under
FLOOR = {("w", "awgn"): -7.0, ("w", "mpg"): 1.75, ("w", "mpp"): 3.5, ("w", "mpd"): 5.25,
         ("n10", "awgn"): -9.5, ("n10", "mpg"): 4.25, ("n10", "mpp"): 0.0, ("n10", "mpd"): 2.25,
         ("w48", "awgn"): -3.75, ("w48", "mpg"): 5.25, ("w48", "mpp"): 5.5, ("w48", "mpd"): 7.75}


def trial(args):
    band, chan, snr, seed = args
    spec = SUBMODES[BURST[band]]
    rng = np.random.default_rng(seed)
    payload = rng.bytes(codes.payload_bytes(spec))
    lead = int(rng.uniform(0.3, 1.0) * FS)
    cfo = float(rng.uniform(-50, 50))
    out = {}
    keep = modem.LEADIN_SAMPLES, modem.LEADOUT_SAMPLES, modem.COPY_MIN_FRAMES
    for arm, (pad, cmin) in ARMS.items():
        if arm == "nocopy" and band not in modem.HEADER_COPY_BANDS:
            continue
        modem.LEADIN_SAMPLES = modem.LEADOUT_SAMPLES = pad
        modem.COPY_MIN_FRAMES = cmin
        try:
            x = modem.modulate([payload], spec)
            x = np.concatenate([np.zeros(lead + 800 - pad), x, np.zeros(FS // 2 + 800 - pad)])
            y = hfchannel.apply_channel(x, snr_db=snr, freq_offset_hz=cfo, ppm=10,
                                        fading_preset=None if chan == "awgn" else chan, seed=seed)
            try:
                r = modem.receive(y)
            except modem.SyncError:
                out[arm] = "fail"
                continue
            good = r["spec"] == spec and r["n_cw"] == 1 and abs(r["preamble_start"] - (lead + 800)) <= 2 * NCP
            out[arm] = "ok" if good else "wrong"
        finally:
            modem.LEADIN_SAMPLES, modem.LEADOUT_SAMPLES, modem.COPY_MIN_FRAMES = keep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--bands", nargs="+", default=list(BURST))
    ap.add_argument("--under", type=float, nargs="+", default=[1.5, 0.0], help="dB under each floor")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    with Pool(a.jobs) as pool:
        for (band, chan), floor in FLOOR.items():
            if band not in a.bands:
                continue
            for d in a.under:
                snr = floor - d
                res = pool.map(trial, [(band, chan, snr, 7919 * s + 3) for s in range(a.trials)], chunksize=8)
                row = dict(band=band, channel=chan, snr=snr, n=len(res))
                for arm in ARMS:
                    got = [r[arm] for r in res if arm in r]
                    if got:
                        row[f"{arm}_fail"] = sum(g != "ok" for g in got) / len(got)
                        row[f"{arm}_wrong"] = sum(g == "wrong" for g in got) / len(got)
                # paired: trials one arm lost and the other didn't
                for arm in ("lead160", "nocopy"):
                    if all(arm in r for r in res):
                        row[f"{arm}_only"] = sum(r[arm] != "ok" and r["lead800"] == "ok" for r in res)
                        row[f"{arm}_saved"] = sum(r[arm] == "ok" and r["lead800"] != "ok" for r in res)
                rows.append(row)
                print("  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
                      flush=True)
                with open(a.out, "w", newline="") as fh:
                    w = csv.DictWriter(fh, sorted({k for r in rows for k in r}, key=lambda k: list(row).index(k)
                                                  if k in row else 99))
                    w.writeheader()
                    w.writerows(rows)


if __name__ == "__main__":
    main()
