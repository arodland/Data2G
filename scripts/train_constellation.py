"""Learn a 2^m-point constellation (positions, and with them the bit
labelling) by maximizing BMI through channel_torch: clipper, waveform-
domain fading, noise, and the real channel estimator.

    uv run python scripts/train_constellation.py --m 4 --snr 12 \
        --channels awgn mpp mpd --steps 400 --out runs/c16-snr12.npy
    # per submode: its band and clip headroom, each channel at its own
    # operating SNR (average power), warm-started from the current set
    uv run python scripts/train_constellation.py --m 6 --band w48 --headroom 5 \
        --at awgn:12.75 mpd:20.25 --init c64-snr18 --steps 1500 --out runs/c64-w48-r712.npy

Starts from Gray QAM (or --init) so a failure to improve is visible,
and prints both on the same held-out seeds at the end. Points are
renormalized to unit power each step (the modem's contract).
"""

import os

# Two BLAS/OpenMP threads, torch capped at 4, so a long GPU run leaves the CPU
# to the machine's owner (numpy and torch default to a thread per core).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import time

import numpy as np
import torch

torch.set_num_threads(4)

from data2g import constellation
from data2g.channel_torch import CHANNELS, BurstChannel, bmi, llr
from data2g.config import DATA_SYMS_PER_FRAME, SubmodeSpec


def evaluate(ch, points, chan, snr, m, seed=12345, batches=4, batch=32):
    g = torch.Generator(device=ch.device).manual_seed(seed)
    w = torch.tensor(1 << np.arange(m - 1, -1, -1), device=ch.device)
    tot = 0.0
    with torch.no_grad():
        for _ in range(batches):
            bits = torch.randint(0, 2, (batch, ch.n_f, DATA_SYMS_PER_FRAME, ch.nc, m), generator=g, device=ch.device)
            y, h, var = ch.receive(ch.channel(ch.transmit(points[(bits * w).sum(-1)]), chan, snr, g), chan)
            tot += m * bmi(llr(y, h, var, points), bits.float()).item()
    return tot / batches


def unit(p: torch.Tensor) -> torch.Tensor:
    return p / p.abs().pow(2).mean().sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--snr", type=float, help="one SNR for every --channels entry")
    ap.add_argument("--channels", nargs="+", default=["awgn", "mpp", "mpd"])
    ap.add_argument("--at", nargs="+", help="channel:snr pairs instead of --snr/--channels")
    ap.add_argument("--band", default="w")
    ap.add_argument("--headroom", type=float, help="clip headroom (default: the band's)")
    ap.add_argument("--init", help="constellation to start from (default: Gray QAM)")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.at:
        at = [(c, float(v)) for c, v in (p.split(":") for p in a.at)]
    elif a.snr is not None:
        at = [(c, a.snr) for c in a.channels]
    else:
        ap.error("--snr or --at")
    torch.manual_seed(a.seed)
    base = constellation.load(a.init) if a.init else constellation.gray_qam(a.m)
    assert len(base) == 2**a.m, "--init has the wrong number of points"
    spec = SubmodeSpec(0, "train", "ldpc", f"gray-qam{2**a.m}", 1, band=a.band, clip_headroom_db=a.headroom)
    ch = BurstChannel(spec, a.frames, device=a.device)
    init = torch.tensor(base, dtype=torch.complex64, device=a.device)
    param = torch.nn.Parameter(torch.view_as_real(init).clone())
    opt = torch.optim.Adam([param], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    w = torch.tensor(1 << np.arange(a.m - 1, -1, -1), device=a.device)
    g = torch.Generator(device=a.device).manual_seed(a.seed)

    t0, run = time.time(), []
    for step in range(a.steps):
        cname, snr = at[step % len(at)]
        chan = CHANNELS[cname]
        pts = unit(torch.view_as_complex(param))
        bits = torch.randint(0, 2, (a.batch, a.frames, DATA_SYMS_PER_FRAME, ch.nc, a.m), generator=g, device=a.device)
        y, h, var = ch.receive(ch.channel(ch.transmit(pts[(bits * w).sum(-1)]), chan, snr, g), chan)
        loss = -bmi(llr(y, h, var, pts), bits.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run.append(-loss.item() * a.m)
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}: BMI {np.mean(run[-50:]):.4f} bits/cu  ({time.time() - t0:.0f}s)", flush=True)

    learned = unit(torch.view_as_complex(param)).detach()
    np.save(a.out, learned.cpu().numpy().astype(np.complex128))
    print(f"\nheld-out BMI, bits/cu ({a.init or 'Gray QAM'} -> learned):")
    for name, snr in at:
        q = evaluate(ch, init, CHANNELS[name], snr, a.m)
        l = evaluate(ch, learned, CHANNELS[name], snr, a.m)
        print(f"  {name:5s} at {snr:5.2f} dB: {q:.4f} -> {l:.4f}  ({l - q:+.4f})", flush=True)


if __name__ == "__main__":
    main()
