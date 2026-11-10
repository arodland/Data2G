"""What a held CPM lock's mid block and second header copy say (2026-11-10).

An early lock (cpm.find: front block + header copy 1) was never looked at
again: the receiver held BUSY to the end its word claimed. The mid block and
header copy 2 follow at rows the lock knows, as tones it knows. This
measures cpm.confirm (their tone_llr at the locked word, unfloored) and
cpm.reread (the best word over front, mid block and both copies) on:

  true   find's early locks on real bursts (start within 2 symbols), the
         SNR drawn from each grid's floor to 12 dB over it
  wrong  the same, where the early lock read a wrong word
  data   a lock forced on the same burst's data: the true word, a start a
         random number of symbols (and a fraction) into the burst
  cut    a true early lock whose burst stops after header copy 1 (noise on)

    python -m scripts.cpm_confirm_study [--trials 300] [--workers 8]

One row a trial to runs/cpm_confirm.csv; the summary prints quantiles, and
per floor the true locks under it and how many of those decoded whole.

Found (2026-11-10, 3000 trials a grid, seeds 70000+): no early lock read a
wrong word (0 of 7530), so reread changes nothing and is not used. confirm,
true locks under the floor (decoded whole) | cut under it | data under it:
    floor   c16r25 (2142)        c8r50 (2539)          c32r62 (2849)
      0     7 (0) | 97% | 99%    13 (3) | 99% | 99.9%   13 (0) | 100% | 100%
     -2     5 (0) | 89% | 98.5%   4 (0) | 97% | 99.5%    9 (0) | 99.9% | 100%
     -4     3 (0) | 75% | 96%     2 (0) | 93% | 99.0%    7 (0) | 99.3% | 99.9%
1734 / 1956 / 1858 true locks decoded whole. cpm.CONFIRM_LLR = -2.
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import csv  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from data2g.config import FS  # noqa: E402

FLOOR_DB = {"c16r25": -16.0, "c8r50": -13.0, "c32r62": -11.0}  # about each grid's AWGN floor
SPAN_DB = 12.0
CHANS = (None, "mpp", "mpd")


def trial(args):
    grid, seed = args
    from data2g import codes, cpm, hfchannel

    g = cpm.GRIDS[grid]
    rng = np.random.default_rng([seed, list(cpm.GRIDS).index(grid)])
    spec = cpm.grid_specs(grid)[int(rng.integers(2))]
    nd, dup = int(rng.integers(0, 4)), bool(rng.integers(2))
    ctl = codes.encode(cpm.CTL[grid], rng.bytes(codes.payload_bytes(cpm.CTL[grid])), 0, 0)
    coded = [ctl] * (1 + dup) + [codes.encode(spec, rng.bytes(codes.payload_bytes(spec)), 0, 0) for _ in range(nd)]
    x = cpm.modulate(spec, coded, dup=dup)
    chan = CHANS[int(rng.integers(len(CHANS)))]
    snr = FLOOR_DB[grid] + float(rng.uniform(0, SPAN_DB))
    lead = FS // 2 + int(rng.integers(g.T))
    p = hfchannel.active_power(x)
    faded = hfchannel.apply_channel(x, None, float(rng.uniform(-50, 50)), fading_preset=chan, seed=int(rng.integers(1 << 30)))
    lay = cpm.layout(grid, cpm.stream_symbols(grid, nd, dup))
    cut = faded.copy()
    cut[(lay.hdr_rows[0][-1] + 1) * g.T:] = 0.0  # the burst stops after its first header copy
    out = []
    for kind, sig in (("true", faded), ("cut", cut)):
        y = np.concatenate([np.zeros(lead), sig, np.zeros(FS)])
        y = hfchannel.awgn(y, snr, seed=seed + 1, s_power=p)
        lock = cpm.find(g, y[:lead + cpm.hdr_span(grid, 1) + 2 * g.T])
        if lock is None or abs(lock["start"] - lead) > 2 * g.T:
            continue  # no early lock on the burst (a late or mid lock reads both copies already)
        right = (lock["spec"], lock["n_data"], lock["dup"]) == (spec, nd, dup)
        w = cpm.reread(g, y, lock)
        row = dict(grid=grid, seed=seed, chan=chan or "awgn", snr=round(snr, 2), kind=kind if right else "wrong",
                   confirm=round(cpm.confirm(g, y, lock), 2), re_same=int(w[:3] == (lock["spec"], lock["n_data"], lock["dup"])),
                   re_right=int(w[:3] == (spec, nd, dup)), re_score=round(w[3], 2), re_margin=round(w[3] - w[4], 2), decoded="")
        if kind == "true" and right:
            # would the burst have been received? its control (and data, if any) through the real receive
            r = cpm.receive(y, dict(lock, end=min(lock["end"], len(y))))
            ok = codes.decode_many(cpm.CTL[grid], np.asarray(r["soft"][0])[None], 0, [0])[0][1]
            if nd:
                ok = ok and all(o for _, o in codes.decode_many(spec, np.asarray(r["soft"][1 + dup:]), 0, [0] * nd))  # encoded at index 0 above
            row["decoded"] = int(ok)
        out.append(row)
        if kind == "true" and right and lay.n > lay.hdr_rows[1][-1] + 8:
            # a lock with this word on the burst's data: any start past the front
            off = int(rng.integers(2, max(3, lay.n - lay.hdr_rows[1][-1] - 1))) * g.T + int(rng.integers(g.T))
            yy = np.concatenate([y, hfchannel.awgn(np.zeros(lay.n * g.T), 0.0, seed=seed + 2, s_power=p * 10 ** (-snr / 10))])
            fake = dict(lock, start=lock["start"] + off)
            w = cpm.reread(g, yy, fake)
            out.append(dict(row, kind="data", confirm=round(cpm.confirm(g, yy, fake), 2), re_same=int(w[:3] == (spec, nd, dup)),
                            re_right=0, re_score=round(w[3], 2), re_margin=round(w[3] - w[4], 2), decoded=""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/cpm_confirm.csv")
    a = ap.parse_args()
    from data2g import cpm

    jobs = [(g, 70000 + k) for g in cpm.GRIDS for k in range(a.trials)]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(trial, jobs, chunksize=4) for r in rs]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    q = (0.5, 1, 5, 25, 50, 75, 95, 99)
    print("confirm score percentiles", q)
    for grid in cpm.GRIDS:
        for kind in ("true", "wrong", "cut", "data"):
            v = np.array([r["confirm"] for r in rows if r["grid"] == grid and r["kind"] == kind])
            if len(v):
                same = np.mean([r["re_same"] for r in rows if r["grid"] == grid and r["kind"] == kind])
                print(f"{grid:7s} {kind:5s} n={len(v):4d}  " + " ".join(f"{x:7.1f}" for x in np.percentile(v, q))
                      + f"   reread keeps the word {same:.2f}")
        t = [r for r in rows if r["grid"] == grid and r["kind"] == "true"]
        for thr in (0.0, -2.0, -4.0, -6.0):
            low = [r for r in t if r["confirm"] < thr]
            other = {k: np.mean([r["confirm"] < thr for r in rows if r["grid"] == grid and r["kind"] == k]) for k in ("cut", "data")}
            print(f"{grid:7s} under {thr:4.0f}: true {len(low)} of {len(t)}, {sum(r['decoded'] for r in low)} of them decoded"
                  f" ({sum(r['decoded'] for r in t)} decoded in all); cut {other['cut']:.3f}, data {other['data']:.3f}")


if __name__ == "__main__":
    main()
