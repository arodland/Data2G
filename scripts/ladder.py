"""Threshold every submode candidate on every channel; append to a CSV.

Resumable: (candidate, channel) pairs already in the CSV are skipped.
Data candidates are simulated in BURST_FRAMES-frame bursts with the
codewords spread over the burst (codes.spread), so fading thresholds
include the burst's time diversity. (The ACK was measured separately as
the 1-codeword burst it is: runs/ack_thresholds.txt.)

    PYTHONPATH=scripts uv run python scripts/ladder.py --out runs/ladder.csv
"""

import os

# Two BLAS/OpenMP threads, torch capped at 4, so a long GPU run leaves the CPU
# to the machine's owner (numpy and torch default to a thread per core).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import csv
import time

import torch

torch.set_num_threads(4)

from data2g.config import SUBMODES, SubmodeSpec, clip_peak_db
from thresholds import Sim, threshold

BURST_FRAMES = 16
CHANNELS = ["awgn", "mpg", "mpp", "mpd"]


def candidates() -> list[SubmodeSpec]:
    c = []

    def add(code, const, frames, k, band="w"):
        tag = "" if band == "w" else f"{band}-"
        c.append(SubmodeSpec(15, f"{tag}{code}-{const}-f{frames}-k{k}", code, const, frames, k=k, band=band))

    # robust: CA-polar on QPSK, below NR's 1/5 mother rate
    add("polar", "gray-qam4", 8, 192)   # 0.20 bits/cu
    add("polar", "gray-qam4", 4, 96)    # 0.20
    add("polar", "gray-qam4", 8, 96)    # 0.10
    add("polar", "gray-qam4", 4, 48)    # 0.10
    # NR LDPC, QPSK, 1920-bit codewords
    for k in (384, 640, 960, 1280, 1440):  # rates 1/5, 1/3, 1/2, 2/3, 3/4
        add("ldpc", "gray-qam4", 8, k)
    # Gray 16-QAM (learned gains nothing at 16 points), 1920 and 3840 bits
    for frames in (4, 8):
        n = 480 * frames
        for r in (1 / 3, 1 / 2, 2 / 3, 3 / 4):
            add("ldpc", "gray-qam16", frames, int(round(r * n / 8)) * 8)
    # learned 64 (trained at 18 dB), 2880 bits
    for k in (1440, 1680, 1920):  # rates 1/2, 7/12, 2/3
        add("ldpc", "c64-snr18", 4, k)
    # learned 256 (trained at 26 dB), 2880 bits
    for k in (1260, 1440):  # 3.5, 4.0 bits/cu
        add("ldpc", "c256-snr26", 3, k)
    return c


def narrow_candidates() -> list[SubmodeSpec]:
    """First pass at the 500 Hz (n10, 50 cu/frame) and 200 Hz (n4, 20
    cu/frame) bands: the same families, sized for their channel uses."""
    c = []

    def add(band, code, const, frames, k):
        c.append(SubmodeSpec(15, f"{band}-{code}-{const}-f{frames}-k{k}", code, const, frames, k=k, band=band))

    for band, cu in (("n4", 20), ("n10", 50)):
        for frames in (2, 4, 8):  # ACK-sized: 32-bit payload
            if 2 * cu * frames >= 80:
                add(band, "polar", "gray-qam4", frames, 48)
        fq = 24 if cu == 20 else 10  # QPSK codewords of ~1000 bits
        n = 2 * cu * fq
        for r in (1 / 5, 1 / 3, 1 / 2, 2 / 3, 3 / 4):
            add(band, "ldpc", "gray-qam4", fq, int(round(r * n / 8)) * 8)
        for r in (1 / 3, 1 / 2, 2 / 3, 3 / 4):
            add(band, "ldpc", "gray-qam16", fq, int(round(r * 2 * n / 8)) * 8)
    return c


def w48_candidates() -> list[SubmodeSpec]:
    """The 2400 Hz band (240 cu/frame): the wide families with half the
    frames per codeword, so codeword lengths match. No ACK: the narrower
    bands carry those."""
    c = []

    def add(code, const, frames, k):
        c.append(SubmodeSpec(15, f"w48-{code}-{const}-f{frames}-k{k}", code, const, frames, k=k, band="w48"))

    for k in (384, 640, 960, 1280, 1440):  # QPSK, 1920 bits: rates 1/5..3/4
        add("ldpc", "gray-qam4", 4, k)
    for r in (1 / 3, 1 / 2, 2 / 3, 3 / 4):  # 16-QAM, 3840 bits
        add("ldpc", "gray-qam16", 4, int(round(r * 3840 / 8)) * 8)
    for k in (1440, 1680):  # learned 64, 2880 bits: rates 1/2, 7/12
        add("ldpc", "c64-snr18", 2, k)
    return c


def top_candidates() -> list[SubmodeSpec]:
    """Past the stock clipper's ceiling (scripts/clip_study.py's -top panels)."""
    from clip_study import PANELS

    c = []
    for panel, band in (("w48-top", "w48"), ("w-top", "w")):
        tag = "" if band == "w" else f"{band}-"
        for _, code, const, frames, k, _, _ in PANELS[panel]:
            c.append(SubmodeSpec(15, f"{tag}{code}-{const}-f{frames}-k{k}", code, const, frames, k=k, band=band))
    return c


def measured_picks() -> list[tuple]:
    """(band, constellation, code rate, headroom) for every submode the
    clipper study measured, picked by scripts/pick_headroom.py's rule."""
    from clip_study import PANELS
    from pick_headroom import picks

    runs = {"w48": "runs/clip_w48.csv", "w48-top": "runs/clip_w48_top.csv", "w": "runs/clip_w.csv",
            "w-top": "runs/clip_w_top.csv", "n10": "runs/clip_n10.csv", "n4": "runs/clip_n4.csv"}
    out = []
    for panel, path in runs.items():
        band = panel.split("-")[0]
        p = picks(path)
        for name, code, const, frames, k, _, _ in PANELS[panel]:
            if p.get(name) is not None:
                spec = SubmodeSpec(15, name, code, const, frames, k=k, band=band)
                out.append((band, const, k / spec.coded_bits, p[name]))
    return out


def assign_headroom(spec: SubmodeSpec, measured: list[tuple]) -> float:
    """Its own pick if the study measured it, else the nearest measured
    submode's: same band and constellation at the closest code rate, then
    the same constellation on any band. Polar and QPSK always measured 0."""
    if spec.code == "polar" or spec.constellation == "gray-qam4":
        return 0.0
    rate = spec.k / spec.coded_bits
    for pool in ([m for m in measured if m[0] == spec.band and m[1] == spec.constellation],
                 [m for m in measured if m[1] == spec.constellation]):
        if pool:
            return min(pool, key=lambda m: (abs(m[2] - rate), m[3]))[3]
    return 0.0


def study_results() -> dict:
    """(band, constellation, frames, k, headroom, channel) -> threshold, for
    everything scripts/clip_study.py measured (same harness and bursts)."""
    from clip_study import PANELS

    runs = {"w48": "runs/clip_w48.csv", "w48-top": "runs/clip_w48_top.csv", "w": "runs/clip_w.csv",
            "w-top": "runs/clip_w_top.csv", "n10": "runs/clip_n10.csv", "n4": "runs/clip_n4.csv"}
    out = {}
    for panel, path in runs.items():
        band = panel.split("-")[0]
        spec_of = {name: (const, frames, k) for name, _, const, frames, k, _, _ in PANELS[panel]}
        for r in csv.DictReader(open(path)):
            const, frames, k = spec_of[r["submode"]]
            out[(band, const, frames, k, float(r["headroom"]), r["channel"])] = float(r["threshold_db"])
    return out


def done(path) -> dict:
    """(name, channel) -> threshold already in the CSV."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {(r["name"], r["channel"]): float(r["threshold_db"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ladder.csv")
    ap.add_argument("--submodes", action="store_true", help="run config.SUBMODES instead of the candidate list")
    ap.add_argument("--narrow", action="store_true", help="run the narrow-band candidates")
    ap.add_argument("--w48", action="store_true", help="run the 2400 Hz candidates")
    ap.add_argument("--channels", nargs="+", default=CHANNELS)
    ap.add_argument("--clip-picked", action="store_true",
                    help="every candidate (all bands + top end) at its picked clip headroom")
    ap.add_argument("--prior", default="runs/ladder_all.csv", help="earlier thresholds, as bisection starts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    seen = done(a.out)
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "code", "constellation", "frames", "k", "n", "bits_per_cu", "channel", "threshold_db",
                        "secs", "band", "headroom", "peak_db"])
        prior = done(a.prior) if a.clip_picked and os.path.exists(a.prior) else {}
        study = study_results() if a.clip_picked else {}
        if a.clip_picked:
            import dataclasses

            measured = measured_picks()
            todo = [dataclasses.replace(s, clip_headroom_db=assign_headroom(s, measured))
                    for s in candidates() + narrow_candidates() + w48_candidates() + top_candidates()]
            todo = list({s.name: s for s in todo}.values())  # the -top panels repeat a few
            todo = [dataclasses.replace(s, name=f"{s.name}@h{s.headroom:g}") for s in todo]
        else:
            todo = (SUBMODES.values() if a.submodes else narrow_candidates() if a.narrow
                    else w48_candidates() if a.w48 else candidates())
        for spec in todo:
            eta = spec.k / (spec.frames_per_cw * spec.cu_per_frame)
            burst = max(1, BURST_FRAMES // spec.frames_per_cw)
            sim = None
            awgn_thr = seen.get((spec.name, "awgn"))
            for chan in a.channels:
                if (spec.name, chan) in seen:
                    continue
                if awgn_thr == float("inf"):  # never decodes even without fading
                    w.writerow([spec.name, spec.code, spec.constellation, spec.frames_per_cw, spec.k,
                                spec.coded_bits, f"{eta:.3f}", chan, float("inf"), 0,
                                spec.band, spec.headroom, clip_peak_db(spec.band, spec.headroom)])
                    f.flush()
                    continue
                key = (spec.band, spec.constellation, spec.frames_per_cw, spec.k, spec.headroom, chan)
                if key in study:  # measured at exactly this setting already
                    thr = study[key]
                    if chan == "awgn":
                        awgn_thr = thr
                    w.writerow([spec.name, spec.code, spec.constellation, spec.frames_per_cw, spec.k,
                                spec.coded_bits, f"{eta:.3f}", chan, thr, "study",
                                spec.band, spec.headroom, clip_peak_db(spec.band, spec.headroom)])
                    f.flush()
                    print(f"== {spec.name} {chan}: {thr} dB (from the clip study)", flush=True)
                    continue
                sim = sim or Sim(spec, a.device, burst, batch=max(8, 256 // burst))  # ~256 codewords/step
                # Start from the old (stock-headroom) threshold, shifted by
                # however much this headroom moved the AWGN one; small first
                # step when there is such a prior.
                base = spec.name.split("@")[0]
                start, step = prior.get((base, chan)), 0.5
                old_awgn = prior.get((base, "awgn"))
                if (start is not None and chan != "awgn" and awgn_thr not in (None, float("inf"))
                        and old_awgn not in (None, float("inf")) and start != float("inf")):
                    start += awgn_thr - old_awgn
                if start is None or start == float("inf"):
                    start, step = (4 * eta - 3 if chan == "awgn" or awgn_thr is None else awgn_thr + 5), 2.0
                print(f"== {spec.name} ({eta:.2f} bits/cu) {chan}, start {start:.1f} dB", flush=True)
                t0 = time.time()
                thr = threshold(sim, chan, round(start * 4) / 4, step=step)
                if chan == "awgn":
                    awgn_thr = thr
                w.writerow([spec.name, spec.code, spec.constellation, spec.frames_per_cw, spec.k,
                            spec.coded_bits, f"{eta:.3f}", chan, thr, f"{time.time() - t0:.0f}",
                            spec.band, spec.headroom, clip_peak_db(spec.band, spec.headroom)])
                f.flush()
                print(f"   -> {thr} dB ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
