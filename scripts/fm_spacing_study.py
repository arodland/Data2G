"""FM plan Q19: carrier spacing, measured. 50 Hz or 100 Hz on the FM
profile's 16 kHz, through data2g.fmchannel.

    python scripts/fm_spacing_study.py run [--round r2] [--trials 20] [--workers 10]
    python scripts/fm_spacing_study.py report [--round r1]

Three numerologies, each its own process (config reads them at import):
    A   50 Hz, 48 carriers (w48: 350..2700 Hz), CP 32 samples (2 ms)
    A4  50 Hz, as A with CP 64 (4 ms): B2's rate exactly (the same CP fraction)
    B2 100 Hz, 24 carriers (w moved to 400..2700 Hz: CARRIER0 + FCENTER on the
        grid), CP 32 (2 ms): A's CP time, 9% slower than A; its 8-repeat
        preamble is half A's length
A 100 Hz grid with a 1 ms CP (A's rate) does not decode even in loopback:
acquisition timing is good to +-16 samples (modem.py), 1 ms at 16 kHz.
The same three modes on each, matched in code, constellation, coded bits
per codeword and headroom (no ACE on either): QPSK r1/2,
16-QAM r1/2, 64-point r3/4. Drive per (variant, mode, radio) at the 15 kHz
occupied limit. Conditions: every fmchannel condition, mobiles and flutter
at 146 and 440 MHz, plus mobile-rural at 130 km/h and a static path with
100 ppm sound-card clock error. Trials are seeded alike in every variant
(paired). Per point: codewords lost, and whether the burst was not
acquired at all. Writes runs/fm_spacing/<variant>.csv.
"""

import argparse
import csv
import functools
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "fm_spacing"


def out_dir(rnd: str) -> Path:
    return OUT if rnd == "r1" else OUT / rnd

_A = {"DATA2G_RS": "50", "DATA2G_NCP": "32", "DATA2G_CARRIER0": "950"}
_B2 = {"DATA2G_RS": "100", "DATA2G_NCP": "32", "DATA2G_CARRIER0": "400"}
# Round 1: HF's preamble threshold (24.1) everywhere. B2 lost 2-4x A's bursts
# to acquisition: its 8-repeat preamble is half A's length in time.
# Round 2: each variant at its own threshold (FM_SPACING_THRESHOLD: 1.1 x the
# largest noise peak over 1200 s, scripts/mf_detect_study.py noise under the
# variant), and B2 also with 16 repeats (A's preamble time). Measured
# (2026-11-15): 8 repeats, A (w48, 50 Hz) 22.7, B2 (w, 100 Hz) 24.3, with the
# same 20 s chunk statistics (median 17.0 / 16.9, 90% 19.9 / 18.9), so one
# threshold for both from the larger, 26.7 (config's precedent); 16 repeats
# 27.2 -> 29.9. HF's 24.1 is under FM's 8-repeat noise peaks.
ROUNDS = {
    "r1": {"A": _A, "A4": {**_A, "DATA2G_NCP": "64"}, "B2": _B2},
    "r2": {"A": {**_A, "FM_SPACING_THRESHOLD": "26.7"},
           "B2": {**_B2, "FM_SPACING_THRESHOLD": "26.7"},
           "B2p16": {**_B2, "DATA2G_PREAMBLE_REPEATS": "16", "FM_SPACING_THRESHOLD": "29.9"}},
}
CNRS = (4, 6, 8, 10, 13, 16, 20, 25, 30, 36)
MODES = ("qpsk-r1/2", "16qam-r1/2", "64l-r3/4")
RADIOS = tuple(os.environ.get("FM_RADIOS", "mic,flat").split(","))
# FM_MIC_RADIO: the fmchannel radio the "mic" rows use (scripts/fm_narrow_study.py: "mic-narrow",
# a 12.5 kHz channel radio), labelled "mic" so report_db pairs it; FM_OBW_HZ: the drive's bandwidth target
OBW_HZ = float(os.environ.get("FM_OBW_HZ", 15_000.0))


def radio_of(r):
    return os.environ.get("FM_MIC_RADIO", "mic") if r == "mic" else r
N_CW = 3


@functools.cache  # once per process: it registers (and, for DATA2G_SYMS_PER_FRAME, rescales) the modes in place
def specs():
    """The three modes in this process's numerology, registered so headers resolve."""
    from data2g import config
    from data2g.config import SUBMODES, SubmodeSpec

    if config.RS == 50:
        out = {"qpsk-r1/2": SUBMODES["w48-qpsk-r1/2"],
               "16qam-r1/2": replace(SUBMODES["w48-16qam-r1/2"], ace=(), clip_headroom_db=1),
               "64l-r3/4": SUBMODES["w48-64l-r3/4"]}
    else:
        out = {"qpsk-r1/2": SUBMODES["qpsk-r1/2"],
               "16qam-r1/2": replace(SUBMODES["16qam-r1/2"], clip_headroom_db=1),
               "64l-r3/4": SubmodeSpec(11, "w-64l-r3/4", "ldpc", "c64-w48-r34", 4, 2160, band="w", clip_headroom_db=5)}
    if config.DATA_SYMS_PER_FRAME != 5:  # DATA2G_SYMS_PER_FRAME (fm_pilot_study): the same codewords in fewer, longer frames
        for k, s in out.items():
            f = s.frames_per_cw * 5
            assert f % config.DATA_SYMS_PER_FRAME == 0, (s.name, config.DATA_SYMS_PER_FRAME)
            out[k] = replace(s, frames_per_cw=f // config.DATA_SYMS_PER_FRAME)
    for s in out.values():
        config.SUBMODES[s.name] = s
        config.BY_INDEX[(s.sync_band, s.index)] = s
    if "FM_SPACING_THRESHOLD" in os.environ:  # read by BandSpec.preamble_threshold at each detection
        config.PREAMBLE_THRESHOLDS[config.PREAMBLE_REPEATS] = float(os.environ["FM_SPACING_THRESHOLD"])
    return out


def conditions():
    from data2g import fmchannel as F

    out = []
    for name, c in F.CONDITIONS.items():
        moving = c.speed_kmh > 0 or c.flutter_db is not None
        for mhz in (F.CARRIERS_MHZ if moving else F.CARRIERS_MHZ[:1]):
            out.append((name, mhz, 0.0))
    out += [("mobile-rural@130", mhz, 0.0) for mhz in F.CARRIERS_MHZ]
    out.append(("static", 146.0, 100.0))
    return out


def burst(spec, seed):
    from data2g import codes, modem
    from data2g.config import FS

    rng = np.random.default_rng(seed)
    n = int(os.environ.get("FM_BURST_FRAMES", 0)) // spec.frames_per_cw or N_CW  # codewords: N_CW, or a burst length
    sent = [rng.bytes(codes.payload_bytes(spec)) for _ in range(n)]
    return sent, np.concatenate([np.zeros(FS // 2), modem.modulate(sent, spec), np.zeros(FS // 2)])


def stream_receive(spec, sent, x, ch, keyup_late_s: float = -0.2) -> tuple[int, int, int]:
    """The live receiver (FM_RX_STREAM): receiver.Receiver fed 0.25 s at a
    time, its Blanker (unless FM_RX_NO_BLANK) and StreamDetector at the
    band's threshold, accepting every mode of the burst's sync band. The
    audio before the burst is open-squelch hiss, the carrier keying up
    0.2 s (the sender's TX delay) before the burst. -> (lost, not acquired:
    no burst event within 0.1 s of the true start, burst events elsewhere:
    false alarms on the hiss)."""
    from data2g import modem, receiver
    from data2g.config import FS, SUBMODES

    lead = 2 * FS
    ch.keyup_s = lead / FS + keyup_late_s  # the carrier arrives this long after the burst starts (-0.2: TX delay)
    y = ch.apply(np.concatenate([np.zeros(lead), x[FS // 2:], np.zeros(FS // 2)]))
    names = [n for n, s in SUBMODES.items() if s.sync_band == spec.sync_band]
    rx = receiver.Receiver(modem.Accept.of(names), blank=not os.environ.get("FM_RX_NO_BLANK"))
    best, found, false = 0, False, 0
    for i in range(0, len(y), FS // 4):
        for kind, ev in rx.feed(y[i:i + FS // 4]):
            if kind != "burst":
                continue
            if abs(ev["header"]["start"] - lead) > FS // 10:
                false += 1
                continue
            found = ev["rx"] is not None or found
            try:
                b = modem.decode_received(ev["rx"])
            except Exception:
                continue
            best = max(best, sum(p == s and k for p, s, k in zip(b.payloads, sent, b.crc_ok)))
    return len(sent) - best, int(not found), false


def job(args):
    from data2g import fmchannel as F, modem
    from data2g.config import FS

    mode, radio, cond, mhz, ppm, trial, drive = args
    spec = specs()[mode]
    c = F.with_speed("mobile-rural", 130) if cond == "mobile-rural@130" else cond
    sent, x = burst(spec, 1000 + trial)
    rows = []
    for cnr in CNRS:
        stream = bool(os.environ.get("FM_RX_STREAM"))  # its audio is 2 s longer (the processes depend on the span)
        ch = F.FmChannel(c, radio_of(radio), cnr, seed=trial, duration_s=len(x) / FS + (4 if stream else 2), carrier_mhz=mhz,
                         drive_dev_hz=drive, ppm=ppm)
        if stream:
            lost, nosync, false = stream_receive(spec, sent, x, ch)
            rows.append((mode, radio, cond, mhz, ppm, trial, cnr, lost, nosync, false, len(sent)))
            continue
        try:
            y = ch.apply(x)
            if os.environ.get("FM_RX_BLANK"):  # as the live receiver (receiver.Receiver): impulse-blanked first
                from data2g import audio

                y = audio.Blanker()(y)
            b = modem.demodulate(y)
            lost, nosync = len(sent) - sum(p == s and k for p, s, k in zip(b.payloads, sent, b.crc_ok)), 0
        except Exception:  # SyncError and kin: the burst was not acquired
            lost, nosync = len(sent), 1
        rows.append((mode, radio, cond, mhz, ppm, trial, cnr, lost, nosync, 0, len(sent)))
    return rows


def run_variant(out: Path, variant, trials, workers):
    from data2g import config, fmchannel as F

    assert config.PROFILE == "fm", "run under DATA2G_PROFILE=fm"
    sp = specs()
    drives = {}
    for m in MODES:
        _, x = burst(sp[m], 0)
        for r in RADIOS:
            # FM_DRIVE_RADIO / FM_DRIVE_SCALE: the drive another radio's bandwidth target gives, scaled
            # (a 12.5 kHz radio's 2.5 kHz limiter holds it under 12.5 kHz at any drive)
            dr = os.environ.get("FM_DRIVE_RADIO", radio_of(r))
            drives[(m, r)] = F.drive_for_bandwidth(x[np.abs(x) > 0], dr, OBW_HZ)[0] * float(os.environ.get("FM_DRIVE_SCALE", 1))
    secs = {m: np.round(len(burst(sp[m], 0)[1]) / config.FS - 1.0, 3) for m in MODES}
    print(f"{variant}: RS {config.RS} NCP {config.NCP} M {config.M}; burst seconds {secs}; drives "
          + ", ".join(f"{k[0]}/{k[1]} {v:.0f}" for k, v in drives.items()), flush=True)
    jobs = [(m, r, c, mhz, ppm, t, drives[(m, r)]) for m in MODES for r in RADIOS
            for c, mhz, ppm in conditions() for t in range(trials)]
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{variant}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "radio", "condition", "mhz", "ppm", "trial", "cnr", "lost", "nosync", "false", "n"])
        with ProcessPoolExecutor(workers) as ex:
            for rows in ex.map(job, jobs, chunksize=2):
                w.writerows(rows)


def knee(fail, level=0.1):
    """Lowest CNR above which failure stays at or under `level`; nan: never within the grid."""
    fail = np.asarray(fail, float)
    bad = np.nonzero(fail > level)[0]
    if len(bad) == 0:
        return float(CNRS[0])
    i = bad[-1]
    if i == len(CNRS) - 1:
        return float("nan")
    return float(np.interp(level, [fail[i + 1], fail[i]], [CNRS[i + 1], CNRS[i]]))


FAMILY = {"static": "fixed", "fixed": "fixed", "fixed-fringe": "slow", "knife-edge": "slow", "flutter": "flutter",
          "pedestrian": "mobile", "mobile-rural": "mobile", "mobile-urban": "mobile", "mobile-bad-urban": "mobile",
          "mobile-hilly": "mobile", "mobile-rural@130": "mobile"}


def report_db(out: Path, variants):
    """Each variant against the round's first, in dB of CNR: codeword failure
    integrated over the CNR grid (two shifted copies differ by the shift),
    per condition family (flutter at 440 MHz apart), with a 90% bootstrap
    interval over trials (paired: every variant ran the same seeds), and the
    part from bursts not acquired (the rest is the data path).

    Round 2 (2026-11-15, 30 trials): 100 Hz no better anywhere outside 0.3
    dB; on mobile paths 1.1-2.1 dB worse, on slow fading 2.2-2.4 dB worse
    for QPSK and 16-QAM even with A's preamble time (B2p16), most of it
    acquisition; flutter at 440 MHz costs B2's 8-repeat preamble 0.6-2.6 dB."""
    from collections import defaultdict

    vs = [v for v in variants if (out / f"{v}.csv").exists()]
    col = {c: i for i, c in enumerate(CNRS)}
    fail = {v: defaultdict(dict) for v in vs}
    nos = {v: defaultdict(dict) for v in vs}
    false = {v: [0, 0] for v in vs}  # false alarms (streamed: burst events on the hiss before), trials
    for v in vs:
        with (out / f"{v}.csv").open() as f:
            for r in csv.DictReader(f):
                k = (r["mode"], r["radio"], r["condition"], r["mhz"], r["ppm"])
                fail[v][k][(int(r["trial"]), col[int(r["cnr"])])] = int(r["lost"]) / int(r.get("n") or N_CW)
                nos[v][k][(int(r["trial"]), col[int(r["cnr"])])] = int(r["nosync"])
                false[v][0] += int(r.get("false") or 0)
                false[v][1] += 1
    print("false alarms per trial (1.8 s of open-squelch hiss each, streamed only): "
          + ", ".join(f"{v} {a / max(n, 1):.4f}" for v, (a, n) in false.items()))

    def grid(d):  # {(trial, cnr index): x} -> (trials, cnrs)
        a = np.zeros((1 + max(t for t, _ in d), len(CNRS)))
        for (t, c), x in d.items():
            a[t, c] = x
        return a

    def area(a):
        return np.trapezoid(a, CNRS, axis=-1)

    groups = defaultdict(list)
    for k in fail[vs[0]]:
        groups[(k[0], FAMILY[k[2]] + (" 440" if k[2] == "flutter" and k[3].startswith("440") else ""))].append(k)
    rng = np.random.default_rng(0)
    print(f"dB of CNR against {vs[0]} (positive: worse): total [90% interval] / acquisition part")
    print(f"{'mode':>10} {'family':>11}  " + "  ".join(f"{v:>26}" for v in vs[1:]))
    for g in sorted(groups):
        cells = []
        for v in vs[1:]:
            tot = np.mean([area(grid(fail[v][k])) - area(grid(fail[vs[0]][k])) for k in groups[g]], axis=0)
            acq = np.mean([area(grid(nos[v][k])) - area(grid(nos[vs[0]][k])) for k in groups[g]], axis=0)
            boot = [tot[rng.integers(0, len(tot), len(tot))].mean() for _ in range(2000)]
            lo, hi = np.percentile(boot, [5, 95])
            cells.append(f"{tot.mean():+5.2f} [{lo:+5.2f},{hi:+5.2f}] / {acq.mean():+5.2f}")
        print(f"{g[0]:>10} {g[1]:>11}  " + "  ".join(f"{c:>26}" for c in cells))


def report(rnd):
    from collections import defaultdict

    data = {}
    for v in ROUNDS[rnd]:
        p = out_dir(rnd) / f"{v}.csv"
        if not p.exists():
            continue
        lost, sync, n = defaultdict(int), defaultdict(int), defaultdict(int)
        with p.open() as f:
            for r in csv.DictReader(f):
                k = (r["mode"], r["radio"], r["condition"], float(r["mhz"]), float(r["ppm"]), int(r["cnr"]))
                lost[k] += int(r["lost"])
                sync[k] += int(r["nosync"])
                n[k] += 1
        data[v] = (lost, sync, n)
    vs = list(data)
    keys = sorted({k[:5] for v in vs for k in data[v][0]}, key=lambda k: (MODES.index(k[0]), k[1], k[2], k[3], k[4]))
    print(f"CNR (dB, 15 kHz) for 10% codeword failure ('>36': not within the grid); "
          f"then failure at {CNRS[-1]} dB; columns {', '.join(vs)}")
    for mode, radio, cond, mhz, ppm in keys:
        ks, top = [], []
        for v in vs:
            lost, sync, n = data[v]
            fail = [lost[(mode, radio, cond, mhz, ppm, c)] / max(1, n[(mode, radio, cond, mhz, ppm, c)] * N_CW) for c in CNRS]
            k = knee(fail)
            ks.append(">36" if np.isnan(k) else f"{k:.1f}")
            top.append(f"{fail[-1]:.2f}")
        label = f"{cond}{'' if not ppm else f' {ppm:g}ppm'} {mhz:g}"
        print(f"{mode:>10} {radio:>4} {label:>28}  " + " ".join(f"{k:>5}" for k in ks) + "   " + " ".join(top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report", "variant"])
    ap.add_argument("--round", default="r2", choices=list(ROUNDS))
    ap.add_argument("--variant")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    if a.what == "report":
        report(a.round)
        return report_db(out_dir(a.round), list(ROUNDS[a.round]))
    if a.what == "variant":
        return run_variant(out_dir(a.round), a.variant, a.trials, a.workers)
    for v, env in ROUNDS[a.round].items():  # one process each: config reads the numerology at import
        subprocess.run([sys.executable, __file__, "variant", "--round", a.round, "--variant", v,
                        "--trials", str(a.trials), "--workers", str(a.workers)],
                       env={**os.environ, **env, "DATA2G_PROFILE": "fm"}, check=True)
    report(a.round)
    report_db(out_dir(a.round), list(ROUNDS[a.round]))


if __name__ == "__main__":
    main()
