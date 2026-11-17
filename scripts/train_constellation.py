"""Learn a 2^m-point constellation (positions, and with them the bit
labelling) by maximizing BMI through channel_torch: clipper, waveform-
domain fading, noise, and the real channel estimator.

    uv run python scripts/train_constellation.py --m 4 --snr 12 \
        --channels awgn mpp mpd --steps 400 --out runs/c16-snr12.npy
    # per submode: its band and clip headroom, each channel at its own
    # operating SNR (average power), warm-started from the current set
    uv run python scripts/train_constellation.py --m 6 --band w48 --headroom 5 \
        --at awgn:12.75 mpd:20.25 --init c64-snr18 --steps 1500 --out runs/c64-w48-r712.npy

FM (DATA2G_PROFILE=fm): `fm-<radio>[-<condition>]:<CNR>` entries train
through data2g.fmtorch (fmchannel's FM chain, radio and path by name, CNR
in fmchannel's convention), e.g.
    DATA2G_PROFILE=fm uv run python scripts/train_constellation.py --m 6 --band w48 \
        --at fm-mic:20 fm-flat:20 fm-mic-mobile-urban:25 --init c64-w48-r34 --out runs/c64-fm.npy
A condition may name its carrier, `fm-flat-flutter@440:12` (default 146 MHz).
FM_TX=1 trains under scripts/fm_tx.py's transmitter (tilt and no clipping on
flat for 16-QAM up, the mic drive backed off by family): the drive is picked
once, from the starting constellation, and held while the points move
(re-pick it after, as the ladder study does).

Starts from Gray QAM (or --init) so a failure to improve is visible,
and prints both on the same held-out seeds at the end. Points are
renormalized to unit power each step (the modem's contract).
"""

# Two BLAS/OpenMP threads, torch capped at 4, so a long GPU run leaves the CPU
# to the machine's owner (numpy and torch default to a thread per core).
from data2g import threads

threads.limit(2)

import argparse
import os
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))  # parallel trainings: fewer each

import fm_tx as T

from data2g import constellation, fmchannel
from data2g.channel_torch import CHANNELS, BurstChannel, bmi, llr
from data2g.config import DATA_SYMS_PER_FRAME, SubmodeSpec
from data2g.fmtorch import FmBurstChannel


def fm_channel(name: str, spec, frames: int, device, base=None) -> FmBurstChannel:
    """`fm-<radio>[-<condition>][@MHz]`: the longest radio name that fits, the rest the condition.
    Under FM_TX the settings' clipping and drive, the drive from a burst of `base` points."""
    rest, _, mhz = name[3:].partition("@")
    radio = max((r for r in fmchannel.RADIOS if rest == r or rest.startswith(r + "-")), key=len)
    c = FmBurstChannel(spec, frames, radio, rest[len(radio) + 1:] or "static", device=device,
                       carrier_mhz=float(mhz or 146.0), drive_dev_hz=None if not T.ON else 1000.0)
    if T.ON:
        m = int(np.log2(len(base)))
        if T.noclip(radio, m):
            c.headroom = T.NOCLIP_HEADROOM
        cells = torch.tensor(constellation.modulate(np.random.default_rng(0).integers(
            0, 2, frames * DATA_SYMS_PER_FRAME * c.nc * m), base).reshape(1, frames, DATA_SYMS_PER_FRAME, c.nc))
        with torch.no_grad():
            x = T.shape(c.transmit(cells.to(device, torch.complex64)).cpu().numpy()[0], spec.band, radio)
        c.drive = fmchannel.drive_for_bandwidth(x[np.abs(x) > 1e-9], radio, 15_000.0, "db26")[0] * T.drive_scale(radio, m)
        c.tilt = None
        if T.ON and radio == "flat":
            import fm_tilt_study

            from data2g.config import FS

            c.tilt = torch.tensor(fm_tilt_study.tilt_fir(spec.band, T.TILT_DB_OCT, FS), device=device)
    return c


def _tilt(c, x: torch.Tensor) -> torch.Tensor:
    """fm_tx.shape in torch (gradients flow): the FIR 'same', rms kept per burst."""
    if getattr(c, "tilt", None) is None:
        return x
    k = c.tilt.to(x.dtype)
    y = torch.nn.functional.conv1d(x[:, None], k.flip(-1)[None, None], padding=len(k) // 2)[:, 0]
    act = (x.abs() > 1e-9).to(x.dtype)
    return y * ((x.pow(2) * act).sum(-1, keepdim=True) / (y.pow(2) * act).sum(-1, keepdim=True)).sqrt()


def through(ch, fm, cname, points, idx, snr, g):
    """One batch through `cname`'s channel -> (y, h, var)."""
    if cname in fm:
        c = fm[cname]
        return c.receive(c.channel(_tilt(c, c.transmit(points[idx])), snr, g), points=points.detach().cpu().numpy())
    chan = CHANNELS[cname]
    return ch.receive(ch.channel(ch.transmit(points[idx]), chan, snr, g), chan)


def evaluate(ch, fm, points, cname, snr, m, seed=12345, batches=4, batch=32):
    g = torch.Generator(device=ch.device).manual_seed(seed)
    w = torch.tensor(1 << np.arange(m - 1, -1, -1), device=ch.device)
    tot = 0.0
    with torch.no_grad():
        for _ in range(batches):
            bits = torch.randint(0, 2, (batch, ch.n_f, DATA_SYMS_PER_FRAME, ch.nc, m), generator=g, device=ch.device)
            y, h, var = through(ch, fm, cname, points, (bits * w).sum(-1), snr, g)
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
    fm = {c: fm_channel(c, spec, a.frames, a.device, base) for c, _ in at if c.startswith("fm-")}
    init = torch.tensor(base, dtype=torch.complex64, device=a.device)
    param = torch.nn.Parameter(torch.view_as_real(init).clone())
    opt = torch.optim.Adam([param], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    w = torch.tensor(1 << np.arange(a.m - 1, -1, -1), device=a.device)
    g = torch.Generator(device=a.device).manual_seed(a.seed)

    t0, run = time.time(), []
    for step in range(a.steps):
        cname, snr = at[step % len(at)]
        pts = unit(torch.view_as_complex(param))
        bits = torch.randint(0, 2, (a.batch, a.frames, DATA_SYMS_PER_FRAME, ch.nc, a.m), generator=g, device=a.device)
        y, h, var = through(ch, fm, cname, pts, (bits * w).sum(-1), snr, g)
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
        q = evaluate(ch, fm, init, name, snr, a.m)
        l = evaluate(ch, fm, learned, name, snr, a.m)
        print(f"  {name:12s} at {snr:5.2f} dB: {q:.4f} -> {l:.4f}  ({l - q:+.4f})", flush=True)


if __name__ == "__main__":
    main()
