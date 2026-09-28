"""Gear-shifter phase C1: link abstraction. Per submode, P(codeword
decodes | its effective MI), where the effective MI is what the receiver
itself can compute: the mean, over the codeword's own channel uses, of
the AWGN BICM capacity (bits per coded bit) at |h|^2 / var from its own
channel estimate (MIESM). A good abstraction is one curve for every
channel; the spread of the per-channel 50% points says how good.

GPU, the ladder's harness (thresholds.Sim: 16-frame bursts, codewords
spread over the burst, the modem's own equalizer). SNRs span each
submode's ladder thresholds.

    PYTHONPATH=scripts uv run python scripts/link_abstraction.py --out runs/link_abstraction.csv
    PYTHONPATH=scripts uv run python scripts/link_abstraction.py --fit runs/link_abstraction.csv \\
        --write data2g/codes_data/link_abstraction.json
"""

import os

from data2g import threads  # noqa: E402

threads.limit(2)

import argparse
import csv
import json
from collections import defaultdict

import numpy as np
import torch

torch.set_num_threads(4)

from data2g import codes, constellation  # noqa: E402
from data2g.channel_torch import CHANNELS, llr  # noqa: E402
from data2g.config import DATA_SYMS_PER_FRAME, SUBMODES  # noqa: E402
from rx_audit import GRID, capacity_table  # noqa: E402
from thresholds import Sim, _payloads  # noqa: E402

BURST_FRAMES = 16  # as ladder.py

CHAN = ["awgn", "mpg", "mpp", "mpd"]


def ladder_thresholds(path="runs/ladder_final.csv") -> dict:
    out = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        out[r["name"]][r["channel"]] = float(r["threshold_db"])
    return out


def ladder_name(s) -> str:
    return f"{'' if s.band == 'w' else s.band + '-'}{s.code}-{s.constellation}-f{s.frames_per_cw}-k{s.k}@h{s.headroom:g}"


def run_point(sim: Sim, chan: str, snr: float, rng, g, table) -> tuple[np.ndarray, np.ndarray]:
    """-> (effective MI per codeword, decoded ok per codeword)."""
    s, b = sim.spec, sim.batch
    info = _payloads(s, b * sim.n_cw, rng)
    coded = codes.encode_info(s, info).reshape(b, sim.n_cw, -1)
    coded = codes.spread(coded, sim.m)
    cb = torch.tensor(coded.reshape(b, sim.n_f, DATA_SYMS_PER_FRAME, sim.ch.nc, sim.m), device=sim.device)
    c = CHANNELS[chan]
    with torch.no_grad():
        y, h, var = sim.ch.receive(sim.ch.channel(sim.ch.transmit(sim.points[(cb * sim.w).sum(-1)]), c, snr, g), c)
        l = llr(y, h, var, sim.points).reshape(b, -1)
        l = codes.despread(l, sim.n_cw, sim.m).reshape(b * sim.n_cw, -1)
        snr_cu = (10 * torch.log10((h.abs() ** 2 / var).clamp(min=1e-6))).cpu().numpy()
    mi_cu = np.interp(snr_cu, GRID, table)  # (b, n_f, 5, nc) bits per coded bit
    mi_bits = np.repeat(mi_cu.reshape(b, -1), sim.m, axis=1)  # the cu's value on each of its bits
    mi_cw = codes.despread(mi_bits, sim.n_cw, sim.m).reshape(b * sim.n_cw, -1).mean(axis=1)
    est, _ = codes.decode_llrs(s, l, device=sim.device)
    ok = codes.crc_ok(s, est) & ~(est != info).any(axis=1)
    return mi_cw, ok


def measure(out_path, device, per_point=512):
    thr = ladder_thresholds()
    done = set()
    new = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    if not new:
        done = {(r["submode"], r["channel"]) for r in csv.DictReader(open(out_path))}
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["submode", "channel", "snr", "mi_eff", "ok"])
        for s in SUBMODES.values():
            table = capacity_table(s.constellation)
            burst = max(1, BURST_FRAMES // s.frames_per_cw)
            sim = None
            t = thr[ladder_name(s)]
            for chan in CHAN:
                if (s.name, chan) in done:
                    continue
                base = t.get(chan, float("inf"))
                if not np.isfinite(base):
                    base = t["awgn"] + 10
                sim = sim or Sim(s, device, burst, batch=max(8, 256 // burst))
                rng, g = np.random.default_rng(0), torch.Generator(device=device).manual_seed(0)
                for snr in np.arange(base - 4, base + 6.01, 1.0):
                    n = 0
                    while n < per_point:
                        mi, ok = run_point(sim, chan, float(snr), rng, g, table)
                        w.writerows([s.name, chan, f"{snr:g}", f"{m:.4f}", int(o)] for m, o in zip(mi, ok))
                        n += len(ok)
                f.flush()
                print(f"{s.name:18s} {chan}", flush=True)


def fit(path, write=None):
    rows = defaultdict(list)
    for r in csv.DictReader(open(path)):
        rows[(r["submode"], r["channel"])].append((float(r["mi_eff"]), int(r["ok"])))
    res = {}
    for s in SUBMODES.values():
        per = {}
        for chan in CHAN:
            d = np.array(rows.get((s.name, chan), []))
            if len(d) < 50 or not 0.02 < d[:, 1].mean() < 0.98:
                continue  # all or nothing: no curve to fit
            per[chan] = logistic(d[:, 0], d[:, 1])
        allrows = np.array(sum((rows.get((s.name, c), []) for c in CHAN), []))
        if not len(allrows):
            continue
        a, mid = logistic(allrows[:, 0], allrows[:, 1])
        mids = [m for _, m in per.values()]
        res[s.name] = {"slope": a, "mi50": mid, "per_channel": {c: m for c, (_, m) in per.items()}}
        print(f"{s.name:18s} MI50 {mid:.3f} slope {a:6.1f}  per channel "
              + " ".join(f"{c} {m:.3f}" for c, (_, m) in per.items())
              + (f"  spread {max(mids) - min(mids):.3f}" if mids else ""))
    if write:
        with open(write, "w") as f:
            json.dump(res, f, indent=1)


def logistic(x, y, width=0.005) -> tuple[float, float]:
    """(slope a, MI50 m) of the logistic sigmoid(a (x - m)) matching the
    empirical curve: success rate per MI bin, made monotone (pool
    adjacent violators), interpolated at 10 / 50 / 90%; a = ln 81 /
    (MI90 - MI10). Regression fits diverged: success is close to a step
    in MI, so the data are nearly separable."""
    order = np.argsort(x)
    x, y = x[order], y[order].astype(float)
    edges = np.arange(x[0], x[-1] + width, width)
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 1)
    n = np.bincount(idx, minlength=len(edges)).astype(float)
    k = np.bincount(idx, weights=y, minlength=len(edges))
    keep = n > 0
    centres, n, k = edges[keep] + width / 2, n[keep], k[keep]
    # pool adjacent violators on the rates, weighted by counts
    blocks = [[k[i], n[i], centres[i] * n[i]] for i in range(len(n))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] / out[-2][1] > out[-1][0] / out[-1][1]:
            b2 = out.pop()
            out[-1] = [out[-1][0] + b2[0], out[-1][1] + b2[1], out[-1][2] + b2[2]]
    xs = np.array([c / m for _, m, c in out])
    ps = np.array([kk / m for kk, m, _ in out])

    def at(p):
        return float(np.interp(p, ps, xs)) if ps[0] < p < ps[-1] else float("nan")

    m10, m50, m90 = at(0.1), at(0.5), at(0.9)
    a = float(np.log(81) / max(m90 - m10, 1e-3)) if np.isfinite(m10 + m90) else float("nan")
    return a, m50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--fit")
    ap.add_argument("--write")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.out:
        measure(a.out, a.device)
    if a.fit:
        fit(a.fit, a.write)


if __name__ == "__main__":
    main()
