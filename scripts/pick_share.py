"""Which data modes the shipped shifter picks: loss_study sessions (real
modem, shift+cpm, PEP-referenced) over channel x SNR (-12..30 dB, 2 dB
steps) x bandwidth cap, every burst's submode logged; then per mode its
share of first-transmission data codewords at each cap that allows it.
ladder_page.COVERED (greyed) is every mode under RARE at every such cap.

    DATA2G_PEP_REF_DB=5 python tools/with_native.py scripts/pick_share.py --out runs/pick_share.csv
    python scripts/pick_share.py summarize runs/pick_share.csv
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
import csv
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import loss_study as LS  # noqa: E402

CAPS = (0, 1, 2)
RARE = 0.01


def job(args):
    cap, chan, snr, seed, horizon = args
    return [dict(r, cap=cap) for r in LS.one((chan, snr, seed, horizon, "shift+cpm", None, cap))]


def summarize(path):
    import ladder_page as LP

    from data2g.arq.modes import CAP_HZ, MODES

    cell = defaultdict(Counter)  # (cap, channel, snr) -> mode -> data codewords sent
    for r in csv.DictReader(open(path)):
        if r["kind"] == "data":
            cell[(int(r["cap"]), r["channel"], float(r["snr"]))][r["submode"]] += int(r["data_sent"])
    share, total, top = Counter(), Counter(), {}
    for k, c in cell.items():
        n = sum(c.values())
        total[k[0]] += n
        for m, v in c.items():
            share[(k[0], m)] += v
            if v / n > top.get(m, (0,))[0]:
                top[m] = (v / n, k)
    rare = []
    print(f"{'mode':16s} {'Hz':>5s}  " + " ".join(f"cap{c}%".rjust(6) for c in CAPS) + "  largest share of a cell")
    for m in sorted(MODES, key=lambda m: (LP.bps(MODES[m]), LP.width(MODES[m]))):
        w = LP.width(MODES[m])
        f = {c: share[(c, m)] / total[c] for c in CAPS if w <= CAP_HZ[c]}
        if all(v < RARE for v in f.values()):
            rare.append(m)
        t = top.get(m)
        print(f"{m:16s} {w:5d}  " + " ".join((f"{100 * f[c]:.2f}" if c in f else "").rjust(6) for c in CAPS)
              + (f"  {100 * t[0]:.0f}% at cap{t[1][0]} {t[1][1]} {t[1][2]:+.0f}" if t else "  never"))
    print(f"\nunder {RARE:.0%} at every cap: {sorted(rare)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--horizon", type=float, default=300.0)
    a = ap.parse_args()
    jobs = [(cap, ch, float(snr), seed, a.horizon) for cap in CAPS for ch in ("awgn", "mpg", "mpp", "mpd")
            for snr in range(-12, 31, 2) for seed in range(a.seeds)]
    cols = ["cap", "channel", "snr", "seed", "submode", "kind", "n_cw", "ctl_slots", "outcome", "data_sent", "data_ok"]
    with Pool(a.jobs) as pool, open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for i, rows in enumerate(pool.imap_unordered(job, jobs)):
            w.writerows(rows)
            f.flush()
            print(f"{i + 1}/{len(jobs)}", flush=True)
    summarize(a.out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(sys.argv[2])
    else:
        main()
