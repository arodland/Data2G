"""FM ladder screen (FM plan M4/M5): predict each candidate mode's 10% CNR
from BICM BMI through the FM training channel (data2g.fmtorch, within 0.01
bit/cell of the full receiver), then propose a ladder per band.

    DATA2G_PROFILE=fm python scripts/fm_ladder_screen.py bmi [--bursts 24] [--workers 8]
    DATA2G_PROFILE=fm python scripts/fm_ladder_screen.py propose --knees runs/fm_ladder/<calib>.csv

`bmi`: BMI per cell over a CNR grid for f55 on mic and flat radios and f108
on flat, on static (146 MHz), flutter (440) and mobile-rural (146), for
gray-qam4, gray-qam16 and the learned 64- and 256-point sets. Writes
runs/fm_ladder/screen.csv.

`propose`: candidates are every (constellation, code rate) whose LDPC code
the existing shift tables build (ldpc.layout's (base graph, Z) present),
codewords filling whole frames. A candidate's 10% CNR is where its BMI per
bit reaches rate + gap, the gap per constellation family calibrated from
streamed 10% knees (scripts/fm_ladder_study.py) of modes in that family:
gap = BMI(knee) / bits - rate, averaged. The ladder: the bps/CNR frontier
on the governing condition (static), thinned so rungs are at least 1 dB
apart, each the fastest within 1.5 dB of the last.

Candidates are named band:constellation:frames:k:headroom; register()
makes one a SubmodeSpec (header index 15, free on f55 and f108) so the
streamed harness can send and receive it.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_tx as T

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "fm_ladder"
CNRS = np.arange(0.0, 41.0, 1.0)
CONDS = (("static", 146.0), ("flutter", 440.0), ("mobile-rural", 146.0))
CASES = (("f55", "mic"), ("f55", "flat"), ("f108", "flat"))
CONSTS = ("gray-qam4", "gray-qam16", "c64-w48-r12", "c64-w48-r712", "c64-w48-r23", "c64-w48-r34",
          "c256-w48-r12", "c256-w48-r58")
# TX clip headroom per family, screened: on FM the drive is set by the occupied
# bandwidth, so BMI compares headrooms fairly (no PEP correction). The radio's
# limiter (mic) distorts data the pilots do not see; more TX clipping trades it
# for clip noise the receiver models (est clip_ratio).
HEADROOMS = {2: (0, 2), 4: (0, 1, 3), 6: (1, 3, 6), 8: (1, 3, 6)}
SUFFIX = "_tx" if T.ON else ""


def headrooms(radio: str, bits: int) -> tuple:
    """Under FM_TX the flat radio sends 16-QAM and up unclipped: one entry,
    6 (its clip constants, which the receiver keeps), sent with no clipping."""
    return (6,) if T.noclip(radio, bits) else HEADROOMS[bits]
RATES = (1 / 5, 1 / 4, 1 / 3, 2 / 5, 1 / 2, 3 / 5, 2 / 3, 3 / 4, 4 / 5, 5 / 6, 7 / 8)
BURST_S = 4.0  # bps at the most codewords a burst of at most this long carries
N_F = 8


def family(const: str) -> str:
    return const.split("-")[0] if const.startswith("c") else const


def bits_of(const: str) -> int:
    from data2g import constellation

    return int(np.log2(len(constellation.load(const))))


def register(name: str):
    """band:constellation:frames:k:headroom -> a registered SubmodeSpec (index 15)."""
    from data2g import config
    from data2g.config import SubmodeSpec

    band, const, frames, k, hr = name.split(":")
    s = SubmodeSpec(15, name, "ldpc", const, int(frames), k=int(k), band=band, clip_headroom_db=float(hr))
    config.SUBMODES[name] = s
    config.BY_INDEX[(s.sync_band, 15)] = s
    return s


def bmi_job(args):
    band, radio, cond, mhz, const, hr, bursts = args
    import torch

    torch.set_num_threads(1)
    import fm_bmi_check as B

    from data2g.config import SUBMODES
    from data2g.fmtorch import FmBurstChannel

    base = next(s for s in SUBMODES.values() if s.band == band)
    from dataclasses import replace

    spec = replace(base, constellation=const, clip_headroom_db=hr)
    B.N_F = N_F
    drive = None if T.ON else FmBurstChannel(spec, N_F, radio).drive
    rows = []
    for cnr in CNRS:
        ch_bmi = _bmi(spec, radio, cond, mhz, float(cnr), drive, bursts)
        ch_bmi, d = ch_bmi
        drive = d  # FM_TX: picked on the first call (the same burst at every CNR)
        rows.append((band, radio, cond, const, hr, float(cnr), round(ch_bmi, 4), round(d)))
    return rows


def _bmi(spec, radio, cond, mhz, cnr, drive, bursts):
    """fm_bmi_check.torch_bmi with the carrier frequency (flutter at 440)."""
    import torch

    from data2g import config, constellation, fmchannel
    from data2g.channel_torch import bmi, llr
    from data2g.config import DATA_SYMS_PER_FRAME
    from data2g.fmtorch import FmBurstChannel

    ch = FmBurstChannel(spec, N_F, radio, cond, drive_dev_hz=drive or 1000.0, carrier_mhz=mhz, dtype=torch.float64)
    pts = constellation.load(spec.constellation)
    m = int(np.log2(len(pts)))
    if T.noclip(radio, m):
        ch.headroom = T.NOCLIP_HEADROOM
    bits = np.stack([np.random.default_rng(100 + s).integers(0, 2, N_F * DATA_SYMS_PER_FRAME * ch.nc * m)
                     for s in range(bursts)])
    cells = torch.tensor(constellation.modulate(bits.ravel(), pts).reshape(bursts, N_F, DATA_SYMS_PER_FRAME, ch.nc))
    pad = config.FS // 2
    with torch.no_grad():
        x = ch.transmit(cells)
    if drive is None:  # FM_TX: the settings' TX shaping, then the drive from what is sent (fm_tilt/clip studies)
        xs = np.stack([T.shape(r, spec.band, radio) for r in x.numpy()])
        act = np.abs(xs[0]) > 1e-9
        xs = xs / np.sqrt(np.mean(xs[:, act] ** 2))
        ch.drive = fmchannel.drive_for_bandwidth(xs[0][act], radio, 15_000.0, "db26")[0] * T.drive_scale(radio, m)
        x = torch.tensor(xs, dtype=x.dtype)
    with torch.no_grad():
        x = torch.nn.functional.pad(x, (pad, pad))
        y = ch.channel(x, cnr, seeds=list(range(bursts)))[:, pad:-pad]
        yd, h, var = ch.receive(y, points=pts)
        lam = llr(yd, h, var, torch.tensor(pts))
    return m * bmi(lam, torch.tensor(bits.reshape(lam.shape), dtype=lam.real.dtype)).item(), ch.drive


def run_bmi(a):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(b, r, c, mhz, k, h, a.bursts) for b, r in CASES for c, mhz in CONDS for k in CONSTS
            for h in headrooms(r, bits_of(k))]
    with (OUT / f"screen{SUFFIX}.csv").open("w", newline="") as fh, ProcessPoolExecutor(a.workers) as ex:
        w = csv.writer(fh)
        w.writerow(["band", "radio", "condition", "const", "headroom", "cnr", "bmi", "drive"])
        for rows in ex.map(bmi_job, jobs):
            w.writerows(rows)
            fh.flush()


def candidates(band: str, radio: str):
    """(name, const, rate) for every buildable (constellation, rate) on band."""
    from data2g import ldpc
    from data2g.config import BANDS, DATA_SYMS_PER_FRAME

    cu = BANDS[band].nc * DATA_SYMS_PER_FRAME
    out = []
    for const in CONSTS:
        m = bits_of(const)
        # frames per codeword: the fewest giving at least 2000 coded bits (the
        # provisional ladder's lengths); longer codewords cost latency and windows
        f = next(f for f in range(1, 9) if cu * f * m >= 2000)
        hrs = headrooms(radio, m)
        n = cu * f * m
        for r in RATES:
            k0 = round(r * n / 8) * 8
            for dk in sorted(range(-int(0.04 * n) // 8, int(0.04 * n) // 8 + 1), key=abs):
                k = k0 + 8 * dk
                try:
                    bg, z = ldpc.layout(k, n)
                except StopIteration:
                    continue
                if f"bg{bg}_z{z}" in ldpc.tables():
                    out += [(f"{band}:{const}:{f}:{k}:{h}", const, k / n) for h in hrs]
                    break
    return out


def bps(name: str) -> tuple[float, int]:
    """Payload bps at the most codewords a burst of at most BURST_S carries."""
    from data2g import codes, modem
    from data2g.config import max_codewords

    s = register(name)
    n = max(1, max(c for c in range(1, max_codewords(s.sync_band) + 1) if modem.burst_seconds(s, c) <= BURST_S)
            if modem.burst_seconds(s, 1) <= BURST_S else 1)
    return 8 * codes.payload_bytes(s) * n / modem.burst_seconds(s, n), n


def propose(a):
    with (OUT / f"screen{SUFFIX}.csv").open() as fh:
        scr = list(csv.DictReader(fh))
    curve = {}
    for r in scr:
        curve.setdefault((r["band"], r["radio"], r["condition"], r["const"], float(r["headroom"])), []).append(
            (float(r["cnr"]), float(r["bmi"])))

    def cnr_at(key, need):
        """Lowest CNR from which BMI (bits/cell) stays at or above `need`,
        interpolated: a rung must keep working above its knee, and on the mic
        path BMI can fall again as CNR rises (the radio's limiter distortion,
        unseen by the pilots, makes the soft bits overconfident)."""
        pts = sorted(curve[key])
        c = np.array([p[0] for p in pts])
        b = np.minimum.accumulate(np.array([p[1] for p in pts])[::-1])[::-1]  # the worst from here up
        if b[-1] < need:
            return float("nan")
        return float(np.interp(need, b + 1e-9 * np.arange(len(b)), c))

    def bmi_at(key, cnr):
        pts = sorted(curve[key])
        return float(np.interp(cnr, [p[0] for p in pts], [p[1] for p in pts]))

    # the gap per family, from streamed knees of modes in it (static)
    from data2g.config import SUBMODES

    gaps = {}
    with open(a.knees or OUT / f"calib{SUFFIX}.csv") as fh:
        for r in csv.DictReader(fh):
            if r["condition"] != "static" or r["k0.1"] in ("", "nan"):
                continue
            s = SUBMODES[r["mode"]] if r["mode"] in SUBMODES else register(r["mode"])
            m = bits_of(s.constellation)
            rate = s.k / s.coded_bits
            hr = headrooms(r["radio"], m)[0] if T.noclip(r["radio"], m) else s.clip_headroom_db  # sent unclipped: screened as 6
            key = (s.band, r["radio"], "static", s.constellation, float(hr))
            b = [v for _, v in sorted(curve[key])]
            if max(b) - b[-1] > 0.1:  # BMI falls past its peak (mic limiter): the soft bits, not the code, set the knee
                print(f"calibration {r['mode']} {r['radio']}: skipped, BMI falls {max(b) - b[-1]:.2f} bit past its peak")
                continue
            got = bmi_at(key, float(r["k0.1"])) / m
            gaps.setdefault((family(s.constellation), r["radio"]), []).append(got - rate)
            print(f"calibration {r['mode']} {r['radio']}: knee {r['k0.1']} dB, BMI/bit {got:.3f}, rate {rate:.3f}, gap {got - rate:+.3f}")
    # per family and radio: the flat radio's gaps run larger under FM_TX (the screen credits its
    # tilt with more than the streamed receiver gets); a family missing on one radio takes the other's
    gap = {f: float(np.mean(v)) for f, v in gaps.items()}
    print("gap per (family, radio) (BMI per bit above rate at the 10% knee):",
          {f"{f}/{r}": round(v, 3) for (f, r), v in gap.items()})

    for band, radio in CASES:
        rows = []
        for name, const, rate in candidates(band, radio):
            g = gap.get((family(const), radio))
            if g is None:
                g = next((v for (f, _), v in gap.items() if f == family(const)), None)
            if g is None:
                continue
            m = bits_of(const)
            hr = float(name.split(":")[4])
            pred = {c: cnr_at((band, radio, c, const, hr), (rate + g) * m) for c, _ in CONDS}
            b, n = bps(name)
            rows.append((name, const, rate, b, n, pred))
        # frontier on static, then thin
        rows = [r for r in rows if np.isfinite(r[5]["static"])]
        rows.sort(key=lambda r: (r[5]["static"], -r[3]))
        front, best = [], -1.0
        for r in rows:
            if r[3] > best:
                front.append(r)
                best = r[3]
        ladder, last = [], -99.0
        for i, r in enumerate(front):
            if r[5]["static"] < last + 1.0:
                continue
            window = [q for q in front[i:] if q[5]["static"] <= max(r[5]["static"], last + 1.5)]
            pick = max(window, key=lambda q: q[3])
            if pick[5]["static"] < last + 1.0:
                continue
            ladder.append(pick)
            last = pick[5]["static"]
        print(f"\n## {band} on {radio}: proposed ladder ({len(ladder)} rungs of {len(front)} on the frontier)")
        print("| candidate | rate | bps | cw/burst | static | flutter@440 | mobile-rural |")
        print("|---|---|---|---|---|---|---|")
        for name, const, rate, b, n, p in ladder:
            print(f"| {name} | {rate:.3f} | {b:.0f} | {n} | {p['static']:.1f} | {p['flutter']:.1f} | {p['mobile-rural']:.1f} |")
        with (OUT / f"ladder_{band}_{radio}{SUFFIX}.txt").open("w") as fh:
            fh.write("\n".join(r[0] for r in ladder) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["bmi", "propose"])
    ap.add_argument("--bursts", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--knees")
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    run_bmi(a) if a.what == "bmi" else propose(a)


if __name__ == "__main__":
    main()
