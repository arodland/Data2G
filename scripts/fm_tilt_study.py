"""FM plan Q9 / M5 step 3: the flat radio's f^2 discriminator noise tilt,
and which lever uses it: a TX tilt (frequency-domain gain on the whole
audio, pilots included, so the receiver's channel estimate absorbs it),
bit loading, or neither.

Through data2g.fmtorch.FmBurstChannel (BMI within 0.01 bit/cell of the
receiver), flat radio, drive re-picked per tilt so the RF signal meets
15 kHz occupied 26 dB down (fmchannel.drive_for_bandwidth, db26), the
modem's TX clipper at each family's headroom (HEADROOM) (the tilt filter runs after it, as a
radio-side filter would). Per band (f108 wide, f55 narrow on a flat radio),
tilt (dB/octave about the band's geometric centre, rms kept) and
constellation family: per-carrier BMI and SNR (true symbols, so the
estimator's error is in it) over a CNR grid, paired seeds across tilts.

    DATA2G_PROFILE=fm python scripts/fm_tilt_study.py run [--workers 4]
    DATA2G_PROFILE=fm python scripts/fm_tilt_study.py report
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "runs" / "fm_tilt"
BANDS = ("f108", "f55")
TILTS = (0.0, 1.5, 3.0, 4.5, 6.0)
CONSTS = ("gray-qam4", "gray-qam16", "c64-snr18", "c256-snr26")
CNRS = (10, 13, 16, 19, 22, 25, 28)
N_F, BURSTS = 6, 6
# the TX clipper per family, as the provisional f55/f108 modes (config.py): it caps SNR
# (2 dB headroom read 13 dB on every carrier), so the top families get the most
HEADROOM = {"gray-qam4": 0, "gray-qam16": 3, "c64-snr18": 6, "c256-snr26": 6}


def tilt_fir(band: str, db_oct: float, fs: int):
    """Linear-phase FIR, amplitude (f / fc)^(db_oct / 6.02) inside the band
    (fc its geometric centre), held flat outside it; applied with 'same'
    so the burst's timing is unchanged."""
    from scipy import signal

    from data2g import config

    b = config.BANDS[band]
    lo = config.CARRIER0 + config.RS * b.k0
    hi = lo + config.RS * (b.nc - 1)
    fc = np.sqrt(lo * hi)
    f = np.linspace(0, fs / 2, 513)
    g = (np.clip(f, lo, hi) / fc) ** (db_oct / (20 * np.log10(2)))  # db_oct dB per doubling
    return signal.firwin2(255, f, g, fs=fs)


def job(args):
    import torch

    torch.set_num_threads(1)
    from data2g import config, constellation, fmchannel
    from data2g.channel_torch import llr
    from data2g.config import DATA_SYMS_PER_FRAME, SubmodeSpec
    from data2g.fmtorch import FmBurstChannel

    band, tilt, const, cond, cnrs = args
    spec = SubmodeSpec(0, "t", "ldpc", const, 1, band=band, clip_headroom_db=HEADROOM[const])
    ch = FmBurstChannel(spec, N_F, "flat", cond, drive_dev_hz=1000.0, dtype=torch.float64,
                        carrier_mhz=440.0 if cond == "flutter" else 146.0)
    pts = constellation.load(const)
    m = int(np.log2(len(pts)))
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, (BURSTS, N_F * DATA_SYMS_PER_FRAME * ch.nc * m))
    cells = torch.tensor(constellation.modulate(bits.ravel(), pts).reshape(BURSTS, N_F, DATA_SYMS_PER_FRAME, ch.nc))
    pad = config.FS // 2
    with torch.no_grad():
        x = ch.transmit(cells).numpy()
    h_t = tilt_fir(band, tilt, config.FS)
    x = np.stack([np.convolve(r, h_t, "same") for r in x])
    act = np.abs(x[0]) > 1e-9
    x = x / np.sqrt(np.mean(x[:, act] ** 2))
    ch.drive = fmchannel.drive_for_bandwidth(x[0][act], "flat", 15_000.0, "db26")[0]
    xp = torch.nn.functional.pad(torch.tensor(x), (pad, pad))
    out = {"band": band, "tilt": tilt, "const": const, "cond": cond, "drive": ch.drive, "cnr": {}}
    for cnr in cnrs:
        with torch.no_grad():
            y = ch.channel(xp, cnr, seeds=list(range(BURSTS)))[:, pad:-pad]
            yd, h, var = ch.receive(y, points=pts)
            lam = llr(yd, h, var, torch.tensor(pts))
        b = torch.tensor(bits.reshape(lam.shape), dtype=lam.real.dtype)
        loss = torch.nn.functional.softplus(-(1.0 - 2.0 * b) * lam) / np.log(2)  # (B, f, s, nc, m)
        bmi_k = m * (1.0 - loss.mean(dim=(0, 1, 2, 4)))  # bits per cell, per carrier
        err = (yd - h * cells).abs().pow(2).mean(dim=(0, 1, 2))
        sig = (h * cells).abs().pow(2).mean(dim=(0, 1, 2))
        out["cnr"][str(cnr)] = {"bmi": bmi_k.tolist(), "snr_db": (10 * torch.log10(sig / err)).tolist()}
    return out


def run(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(b, t, c, "static", CNRS) for b in BANDS for t in TILTS for c in CONSTS]
    # flutter at 440 MHz as a check: f108, no tilt and +3, mid CNRs
    jobs += [("f108", t, c, "flutter", (16, 22)) for t in (0.0, 3.0) for c in CONSTS]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(job, jobs))
    (OUT / "results.json").write_text(json.dumps(res))


def _inv(cnrs, y, target):
    """The CNR at which the curve y(cnrs) (rising) reaches target, linear; nan past the grid."""
    return float(np.interp(target, y, cnrs, left=np.nan, right=np.nan))


def report():
    res = json.loads((OUT / "results.json").read_text())
    st = [r for r in res if r["cond"] == "static"]
    get = {(r["band"], r["tilt"], r["const"]): r for r in st}
    cn = [str(c) for c in CNRS]
    for band in BANDS:
        print(f"\n## {band} (flat), static")
        snr = np.array([get[(band, 0.0, "c64-snr18")]["cnr"][c]["snr_db"] for c in cn])
        print("per-carrier SNR, no tilt (dB): bottom carrier / mean / top carrier / slope top-bottom")
        for c, s in zip(cn, snr):
            print(f"  CNR {c}: {s[0]:.1f} / {s.mean():.1f} / {s[-1]:.1f} / {s[-1] - s[0]:+.1f}")
        print("drive (Hz rms) per tilt: " + ", ".join(f"{t:+.1f}: {get[(band, t, 'c64-snr18')]['drive']:.0f}" for t in TILTS))
        tot = {(t, c): np.array([sum(get[(band, t, c)]["cnr"][k]["bmi"]) for k in cn]) for t in TILTS for c in CONSTS}
        best = {t: np.max([tot[(t, c)] for c in CONSTS], axis=0) for t in TILTS}
        load = {t: np.array([np.sum(np.max([get[(band, t, c)]["cnr"][k]["bmi"] for c in CONSTS], axis=0)) for k in cn])
                for t in TILTS}

        def zones(t, n=3):
            out = []
            for k in cn:
                per = np.array([get[(band, t, c)]["cnr"][k]["bmi"] for c in CONSTS])  # (consts, nc)
                out.append(sum(per[:, z].sum(axis=1).max() for z in np.array_split(np.arange(per.shape[1]), n)))
            return np.array(out)

        print("total BMI, bits per OFDM symbol (sum over carriers): rows tilt dB/oct, columns CNR " + " ".join(cn))
        for t in TILTS:
            print(f"  tilt {t:+.1f}: " + "  ".join(f"{c.split('-')[-1]} " + " ".join(f"{v:5.0f}" for v in tot[(t, c)])
                                               for c in CONSTS) + "  best " + " ".join(f"{v:5.0f}" for v in best[t]))
        print("gain as dB of CNR at equal throughput, against no tilt (per family; best single) at CNR 13 / 19 / 25:")
        for t in TILTS[1:]:
            cells = []
            for c in CONSTS + ("best",):
                y0, y1 = (best[0.0], best[t]) if c == "best" else (tot[(0.0, c)], tot[(t, c)])
                cells.append(c.split("-")[-1] + " " + "/".join(f"{float(k) - _inv(CNRS, y0, y1[cn.index(k)]):+.2f}"
                                                               for k in ("13", "19", "25")))
            print(f"  tilt {t:+.1f}: " + "  ".join(cells))
        tb = max(TILTS, key=lambda t: best[t][cn.index("19")])
        print(f"best tilt at CNR 19: {tb:+.1f}")
        for name, curve in (("per-carrier loading, no tilt", load[0.0]), (f"per-carrier loading, tilt {tb:+.1f}", load[tb]),
                            ("3 zones, no tilt", zones(0.0)), (f"3 zones, tilt {tb:+.1f}", zones(tb)),
                            ("2 zones, no tilt", zones(0.0, 2))):
            print(f"  {name}: bits " + " ".join(f"{v:5.0f}" for v in curve) + "; dB vs best single at best tilt, CNR 13/19/25: "
                  + "/".join(f"{float(k) - _inv(CNRS, best[tb], curve[cn.index(k)]):+.2f}" for k in ("13", "19", "25")))
    fl = [r for r in res if r["cond"] == "flutter"]
    if fl:
        print("\n## f108 flutter 440 check: total BMI best single, CNR 16 / 22")
        for t in (0.0, 3.0):
            b = [max(sum(r["cnr"][k]["bmi"]) for r in fl if r["tilt"] == t) for k in ("16", "22")]
            print(f"  tilt {t:+.1f}: {b[0]:.0f} / {b[1]:.0f}")


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
