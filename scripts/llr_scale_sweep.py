"""Are the receiver's soft bits over- or under-confident, and would a global
LLR scale (alpha) or extra estimate-error variance (var + beta * mse) help?
Through channel_torch (clipper, fading, numpy estimator; genie delay
support, no decision-directed pass) at each mode's 1% threshold and 3 dB
below (average power).

Per setting: mean and worst-10% burst BMI (bits/cu), P(burst BMI < need)
with need = the mode's mean BMI at its AWGN threshold, unscaled, and the
same P for Chase-combined burst pairs (correctness-signed LLRs of two
independently faded bursts added), a proxy for IR combining. BMI at the
best alpha is the GMI. "oracle" picks alpha per burst: the most any
per-burst scaling could get. ratio = mean |y - h x|^2 / var (1 = the
variance is right; > 1 = overconfident), over all bursts and the worst 10%.

    PYTHONPATH=. python scripts/llr_scale_sweep.py --out runs/llr_scale_sweep.csv
"""

from data2g import threads  # noqa: E402

threads.limit(2)

import argparse
import csv
import time

import numpy as np
import torch

torch.set_num_threads(4)

from data2g import constellation, modem  # noqa: E402
from data2g.channel_torch import CHANNELS, BurstChannel, llr  # noqa: E402
from data2g.config import DATA_SYMS_PER_FRAME, SUBMODES  # noqa: E402

ALPHAS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25]
BETAS = [1.0, 3.0, 10.0, 30.0]
# 1% thresholds (dB, average power): runs/ladder_final.csv for QPSK/16-QAM,
# runs/constellation_cvar_0.01.csv (production sets) for the learned ones;
# where 1% is never reached, the 10% point (constellation_cvar_0.1.csv).
POINTS = {
    "w48-qpsk-r1/2": dict(awgn=2.75, mpg=14.75, mpp=8.5, mpd=8.0),
    "w48-16qam-r1/2": dict(awgn=8.5, mpg=21.0, mpp=14.75, mpd=14.25),
    "n10-qpsk-r1/2": dict(awgn=-4.25, mpg=10.75, mpp=3.0, mpd=2.25),
    "n10-16qam-r1/2": dict(awgn=2.0, mpg=17.25, mpp=11.75, mpd=9.0),
    "w48-64l-r1/2": dict(awgn=11.5, mpg=23.75, mpp=18.0, mpd=18.0),
    "w48-64l-r2/3": dict(awgn=14.25, mpg=27.0, mpp=21.5, mpd=22.75),
    "w48-64l-r3/4": dict(awgn=17.0, mpg=31.0, mpp=25.5, mpd=28.75),
    "w48-256l-r5/8": dict(awgn=18.25, mpg=31.0, mpp=26.25, mpd=36.25),
    "n10-64l-r3/4": dict(awgn=9.25, mpg=26.25, mpp=16.5, mpd=23.25),
    "n10-256l-r3/4": dict(awgn=14.0, mpg=32.5, mpp=23.25),
}


def simulate(ch, pts, chan, snr, n, batch, seed):
    """-> signed LLR inputs per burst: y, h, var, mse, x (complex/real, (N, cells)), bits (N, cells, m)."""
    m = constellation.bits_per_symbol(pts.cpu().numpy())
    w = torch.tensor(1 << np.arange(m - 1, -1, -1), device=ch.device)
    g = torch.Generator(device=ch.device).manual_seed(seed)
    frames = [f for f in range(ch.n_air) if f != ch.kc]
    out = {k: [] for k in ("y", "h", "var", "mse", "x", "bits")}
    stash, orig = [], modem.data_channel

    def grab(*a, **k):
        e = orig(*a, **k)
        stash.append(e)
        return e

    modem.data_channel = grab
    try:
        with torch.no_grad():
            for _ in range(n // batch):
                stash.clear()
                bits = torch.randint(0, 2, (batch, ch.n_f, DATA_SYMS_PER_FRAME, ch.nc, m), generator=g, device=ch.device)
                x = pts[(bits * w).sum(-1)]
                y, h, var = ch.receive(ch.channel(ch.transmit(x), CHANNELS[chan], snr, g), CHANNELS[chan])
                mse = torch.tensor(np.stack([e["mse"][frames] for e in stash]), dtype=var.dtype, device=ch.device)
                for k, v in zip(out, (y, h, var, mse, x, bits)):
                    out[k].append(v.reshape(batch, -1, v.shape[-1]) if k == "bits" else v.reshape(batch, -1))
    finally:
        modem.data_channel = orig
    return {k: torch.cat(v) for k, v in out.items()}


def signed_llrs(s, pts, beta=0.0, chunk=8):
    """(N, cells * m) correctness-signed LLRs, (1 - 2b) L: positive = right."""
    res = []
    for i in range(0, len(s["y"]), chunk):
        sl = slice(i, i + chunk)
        var = s["var"][sl] + beta * s["mse"][sl]
        L = llr(s["y"][sl], s["h"][sl], var, pts)
        res.append(((1.0 - 2.0 * s["bits"][sl].float()) * L).flatten(1))
    return torch.cat(res)


def burst_bmi(sl, m):
    """bits/cu per burst from signed LLRs (the decoders clamp at 16; BMI is taken unclamped)."""
    return m * (1.0 - torch.nn.functional.softplus(-sl).mean(1) / np.log(2))


def score(sl, m, need):
    b = burst_bmi(sl, m)
    half = len(sl) // 2
    pair = burst_bmi(sl[:half] + sl[half:2 * half], m)  # independent fades, same positions
    k = max(1, len(b) // 10)
    return dict(mean=b.mean().item(), worst10=b.topk(k, largest=False).values.mean().item(),
                p_out=(b < need).float().mean().item(), pair_p_out=(pair < need).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", nargs="+", default=list(POINTS))
    ap.add_argument("--bursts", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--below", type=float, default=3.0, help="second point, dB under the threshold")
    ap.add_argument("--frames", type=int, help="burst length in frames (default: 16, rounded to whole codewords)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.12)

    fields = ["mode", "channel", "snr", "method", "param", "mean", "worst10", "p_out", "pair_p_out",
              "ratio", "ratio_worst10", "need"]
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fields)
        wr.writeheader()
        for name in a.modes:
            spec = SUBMODES[name]
            fpc = spec.frames_per_cw
            ch = BurstChannel(spec, a.frames or max(fpc, 16 // fpc * fpc), device=a.device)
            pts = torch.tensor(constellation.load(spec.constellation), dtype=torch.complex64, device=a.device)
            m = constellation.bits_per_symbol(constellation.load(spec.constellation))
            need = None
            for chan, thr in POINTS[name].items():
                for snr in (thr, thr - a.below):
                    t0 = time.time()
                    s = simulate(ch, pts, chan, snr, a.bursts, a.batch, a.seed)
                    base = signed_llrs(s, pts)
                    b0 = burst_bmi(base, m)
                    if need is None:  # first point: AWGN at threshold
                        need = b0.mean().item()
                    r = ((s["y"] - s["h"] * s["x"]).abs() ** 2 / s["var"]).mean(1)
                    worst = b0.topk(max(1, len(b0) // 10), largest=False).indices
                    common = dict(mode=name, channel=chan, snr=snr, need=f"{need:.4f}",
                                  ratio=f"{r.mean().item():.3f}", ratio_worst10=f"{r[worst].mean().item():.3f}")
                    rows = [("alpha", al, score(al * base, m, need)) for al in ALPHAS]
                    rows += [("beta", be, score(signed_llrs(s, pts, be), m, need)) for be in BETAS]
                    per = torch.stack([burst_bmi(al * base, m) for al in ALPHAS]).max(0).values
                    rows.append(("oracle", 0, dict(mean=per.mean().item(),
                                                   worst10=per.topk(max(1, len(per) // 10), largest=False).values.mean().item(),
                                                   p_out=(per < need).float().mean().item(), pair_p_out=float("nan"))))
                    for method, p, sc in rows:
                        wr.writerow(dict(common, method=method, param=p, **{k: f"{v:.4f}" for k, v in sc.items()}))
                    f.flush()
                    best = max(rows[:len(ALPHAS)], key=lambda t: t[2]["mean"])
                    print(f"{name:15s} {chan:4s} {snr:6.2f} dB  BMI {rows[ALPHAS.index(1.0)][2]['mean']:.3f}"
                          f"  GMI {best[2]['mean']:.3f} @a={best[1]}  ratio {r.mean().item():.2f}"
                          f" (worst10 {r[worst].mean().item():.2f})  P_out {rows[ALPHAS.index(1.0)][2]['p_out']:.3f}"
                          f"  ({time.time() - t0:.0f}s)", flush=True)
                    del s, base
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
