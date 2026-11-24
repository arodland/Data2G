"""FM plan M4/M5: the TX clipper on FM. dsp.tx_condition clips the analytic
envelope at a headroom over rms (HF SSB is peak-envelope limited). On FM
the limits differ: a flat (9600) port has no limiter and the binding limit
is 15 kHz occupied 26 dB down; a mic path pre-emphasises (750 us) and the
radio hard-limits at 5 kHz peak deviation. The tilt study found today's
clipper floors the best carriers near 21-24 dB SNR even at 6 dB headroom.

Through data2g.fmtorch.FmBurstChannel (BMI within 0.01 bit/cell of the
receiver), static (flutter at 440 MHz as a check), drive re-picked per
variant so the RF signal meets 15 kHz 26 dB down. Paths: f108 on flat and
f55 on flat (both with the +3 dB/octave TX tilt, fm_tilt_study), f55 on
mic (no tilt). Variants, each on the burst as transmitted unclipped (the
torch TX at 40 dB headroom, where the clipper never acts):

    envH      today's envelope clip-and-filter at H dB (3, 6, 9, 12)
    none      no TX clipping
    realC     clip the real audio (what sets the instantaneous deviation)
              at C dB over its rms, filter, three passes (bandwidth-aware:
              with the drive picked for 15 kHz, C trades clip noise against
              the rms the width limit allows)
    softC     the same with a soft limiter, C tanh(x / C)
    preC      mic: clip in the pre-emphasised domain the radio's limiter
              sees, at C dB over its rms, then undo the emphasis
    dsX       mic: no TX clipping, the drive at X% of what 15 kHz allows

Score: per-carrier BMI (true symbols) summed over the band, as dB of CNR
at equal throughput against the family's provisional headroom (16-QAM 3,
64- and 256-point 6 dB), and the clean per-carrier SNR (CNR 60).

    DATA2G_PROFILE=fm python scripts/fm_clip_study.py run [--workers 4]
    DATA2G_PROFILE=fm python scripts/fm_clip_study.py report
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm_tilt_study import _inv, tilt_fir  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "runs" / "fm_clip"
PATHS = (("f108", "flat", 3.0), ("f55", "flat", 3.0), ("f55", "mic", 0.0))
CONSTS = ("gray-qam16", "c64-w48-r23", "c256-w48-r58")
BASE = {"gray-qam16": "env3", "c64-w48-r23": "env6", "c256-w48-r58": "env6"}  # provisional f55/f108 headrooms
CNRS = (13, 16, 19, 22, 25, 28, 32)
CLEAN = 60
N_F, BURSTS = 6, 6
LEVELS = (6, 8, 10, 12)  # dB over rms for realC / softC / preC


def variants(radio: str, const: str) -> list[str]:
    v = [f"env{h}" for h in (3, 6, 9, 12)] + ["none"] + [f"real{c}" for c in LEVELS] + [f"soft{c}" for c in LEVELS]
    if radio == "mic":
        # the 15 kHz drive puts the pre-emphasised rms only ~4-5 dB under the radio's 5 kHz
        # limiter, so pre-clips go lower; and backing the drive off (no clipping) as its rival
        v += [f"pre{c}" for c in (3, 4, 5, 6, 8)] + [f"ds{x}" for x in (50, 70, 85)]
    return v


def _clip_filter(x, ch, level_db, soft=False, pre=None):
    """Real-sample clip at level_db over rms (or soft, c tanh(x / c)), then the band's
    TX filter, three passes. `pre`: (b, a, b_inv, a_inv) emphasis to clip in."""
    import torch
    from scipy import signal

    act = np.abs(x[0]) > 1e-9
    for _ in range(3):
        y = signal.lfilter(pre[0], pre[1], x, axis=-1) if pre else x
        c = np.sqrt(np.mean(y[:, act] ** 2)) * 10 ** (level_db / 20)
        y = c * np.tanh(y / c) if soft else np.clip(y, -c, c)
        y = signal.lfilter(pre[2], pre[3], y, axis=-1) if pre else y
        x = ch._filter(torch.tensor(y)).numpy() * act
    return x


def job(args):
    import torch

    torch.set_num_threads(1)
    from data2g import config, constellation, fmchannel
    from data2g.channel_torch import llr
    from data2g.config import DATA_SYMS_PER_FRAME, SubmodeSpec
    from data2g.fmtorch import FmBurstChannel

    band, radio, tilt, const, var, cond, cnrs = args
    head = int(var[3:]) if var.startswith("env") else 40
    # clip constants (the receiver's assumed clip-noise share) exist for 0-6 dB only: past
    # 6 the channel keeps 6 dB's (a slight over-estimate of clip noise) and clips at `head`
    spec = SubmodeSpec(0, "t", "ldpc", const, 1, band=band, clip_headroom_db=min(head, 6))
    ch = FmBurstChannel(spec, N_F, radio, cond, drive_dev_hz=1000.0, dtype=torch.float64,
                        carrier_mhz=440.0 if cond == "flutter" else 146.0)
    ch.headroom = head
    pts = constellation.load(const)
    m = int(np.log2(len(pts)))
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, (BURSTS, N_F * DATA_SYMS_PER_FRAME * ch.nc * m))
    cells = torch.tensor(constellation.modulate(bits.ravel(), pts).reshape(BURSTS, N_F, DATA_SYMS_PER_FRAME, ch.nc))
    with torch.no_grad():
        x = ch.transmit(cells).numpy()
    if tilt:  # the flat path's TX tilt; clipping that sets the deviation comes after it
        h_t = tilt_fir(band, tilt, config.FS)
        x = np.stack([np.convolve(r, h_t, "same") for r in x])
    if var.startswith(("real", "soft")):
        x = _clip_filter(x, ch, int(var[4:]), soft=var.startswith("soft"))
    elif var.startswith("pre"):
        r = fmchannel.RADIOS[radio]
        x = _clip_filter(x, ch, int(var[3:]), pre=(*fmchannel._emphasis(config.FS, r.emphasis_us, inverse=False),
                                                   *fmchannel._emphasis(config.FS, r.emphasis_us, inverse=True)))
    act = np.abs(x[0]) > 1e-9
    x = x / np.sqrt(np.mean(x[:, act] ** 2))
    ch.drive = fmchannel.drive_for_bandwidth(x[0][act], radio, 15_000.0, "db26")[0]
    if var.startswith("ds"):  # no TX clipping, the drive backed off to spare the radio's limiter
        ch.drive *= int(var[2:]) / 100
    # how often the radio's limiter acts at that drive (mic): the share of samples past 5 kHz
    lim = 0.0
    if fmchannel.RADIOS[radio].peak_dev_hz:
        r = fmchannel.RADIOS[radio]
        from scipy import signal

        pe = signal.lfilter(*fmchannel._emphasis(config.FS, r.emphasis_us, inverse=False), x[0][act] * ch.drive)
        lim = float(np.mean(np.abs(pe) > r.peak_dev_hz))
    pad = config.FS // 2
    xp = torch.nn.functional.pad(torch.tensor(x), (pad, pad))
    out = {"band": band, "radio": radio, "const": const, "var": var, "cond": cond, "drive": ch.drive,
           "limited": lim, "cnr": {}}
    for cnr in cnrs:
        with torch.no_grad():
            y = ch.channel(xp, cnr, seeds=list(range(BURSTS)))[:, pad:-pad]
            yd, h, var_ = ch.receive(y, points=pts)
            lam = llr(yd, h, var_, torch.tensor(pts))
        b = torch.tensor(bits.reshape(lam.shape), dtype=lam.real.dtype)
        loss = torch.nn.functional.softplus(-(1.0 - 2.0 * b) * lam) / np.log(2)
        bmi_k = m * (1.0 - loss.mean(dim=(0, 1, 2, 4)))
        err = (yd - h * cells).abs().pow(2).mean(dim=(0, 1, 2))
        sig = (h * cells).abs().pow(2).mean(dim=(0, 1, 2))
        out["cnr"][str(cnr)] = {"bmi": bmi_k.tolist(), "snr_db": (10 * torch.log10(sig / err)).tolist()}
    return out


def run(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(b, r, t, c, v, "static", CNRS + (CLEAN,)) for b, r, t in PATHS for c in CONSTS for v in variants(r, c)]
    # flutter at 440 MHz as a check: f108 flat, each family's baseline and the static winner's kind
    jobs += [("f108", "flat", 3.0, c, v, "flutter", (19, 25)) for c in CONSTS for v in (BASE[c], "none", "real10", "env12")]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(job, jobs))
    (OUT / "results.json").write_text(json.dumps(res))


def report():
    res = json.loads((OUT / "results.json").read_text())
    cn = [str(c) for c in CNRS]
    for band, radio, tilt in PATHS:
        print(f"\n## {band} on {radio}" + (f", TX tilt {tilt:+.0f} dB/oct" if tilt else "") + ", static")
        for const in CONSTS:
            rows = {r["var"]: r for r in res if (r["band"], r["radio"], r["const"], r["cond"]) == (band, radio, const, "static")}
            tot = {v: np.array([sum(r["cnr"][k]["bmi"]) for k in cn]) for v, r in rows.items()}
            base = tot[BASE[const]]
            print(f"{const} (baseline {BASE[const]}): total BMI (bits per OFDM symbol) at CNR 16 / 22 / 28; dB of "
                  "CNR saved at equal throughput (+ better; '>' past the baseline's grid top, '<' below its bottom); "
                  "drive Hz rms; clean SNR bottom/mean/top dB" + ("; radio limiter %" if radio == "mic" else ""))
            for v, r in rows.items():
                cells, gains = [], []
                for k in ("16", "22", "28"):
                    y = tot[v][cn.index(k)]
                    g = _inv(CNRS, base, y) - float(k)  # the CNR the baseline needs for this throughput, minus k
                    cells.append(f"{y:4.0f}")
                    gains.append(f"{g:+5.2f}" if not np.isnan(g) else (f">{CNRS[-1] - float(k):+.0f}" if y > base.max()
                                                                       else f"<{CNRS[0] - float(k):+.0f}"))
                s = np.array(r["cnr"][str(CLEAN)]["snr_db"])
                lim = f"  {100 * r['limited']:.2f}%" if radio == "mic" else ""
                print(f"  {v:8} {'/'.join(cells)}  {'/'.join(f'{g:>6}' for g in gains)}  drive {r['drive']:5.0f}  "
                      f"clean {s[0]:4.1f}/{s.mean():4.1f}/{s[-1]:4.1f}{lim}")
    fl = [r for r in res if r["cond"] == "flutter"]
    if fl:
        print("\n## f108 flat flutter 440: total BMI at CNR 19 / 25")
        for const in CONSTS:
            print(f"  {const}: " + "  ".join(f"{r['var']} {sum(r['cnr']['19']['bmi']):.0f}/{sum(r['cnr']['25']['bmi']):.0f}"
                                            for r in fl if r["const"] == const))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    if a.what == "run":
        run(a.workers)
    report()


if __name__ == "__main__":
    main()
