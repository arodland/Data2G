"""Data2G-on-the-air detectors without a header (Receiver.data2g_on_air):
how often each reads present on bursts joined partway through, and on
noise, tones, CW, Hands, interference.PRESETS and narrow SSB passbands.
Thresholds are picked from these numbers.

OFDM (modem.pilot_presence + fold_verify, per sync band): the frame-pilot
fold peak and contrast over the newest `window`, verified against the band's
known pilot sequence (and n4 on n10). CPM (cpm.presence, per grid): per-bin
median-whitened active-noise-scale tone transitions with top tones trimmed.

    uv run --no-sync python scripts/presence_study.py --jobs 12 --n 40
    uv run --no-sync python scripts/presence_study.py --curve --jobs 12 --noise 400
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy import signal as sps  # noqa: E402

from data2g import audio, codes, cpm, cwid, hfchannel, interference, modem  # noqa: E402
from data2g.arq.policy import slots_for  # noqa: E402
from data2g.config import FS  # noqa: E402
from data2g.waveform import dsp, ofdm, sync  # noqa: E402

SPECS = {"w": "qpsk-r1/2", "w48": "w48-qpsk-r1/2", "n10": "n10-qpsk-r1/2", "n4": "n4-qpsk-r1/3"}
SYNC_BANDS = ("w", "w48", "n10")
GRIDS = ("c16r25", "c8r50", "c32r62")
BUF_S = 3.0  # buffer per trial (the receiver's keeps 1.9-4.8 s while listening)
OFDM_WINDOWS = (1.0, 1.5, 1.75)
CPM_WINDOWS = (1.5, 2.0, 3.0)
SNRS = (-15, -12, -9, -6, -3, 6)
CHANNELS = ("awgn", "mpp", "mpd", "mpg")


def ofdm_stats(x, band):
    d = sync.StreamDetector(ofdm.band(band))
    d.feed(dsp.to_baseband(x, 0))
    a, C = d.outputs_from(0)
    xs = x[a:]
    C = C[:, :max(0, len(xs) - modem.M + 1)]
    D = sync.pair_products(C, modem.FRAME_SAMPLES, modem.COPY_PAIRS)
    lvl = d.level()
    out = {}
    for w in OFDM_WINDOWS:
        win = int(w * FS)
        pk, ct = modem.pilot_presence(D, lvl, win)
        ok = modem.pilot_present(D, lvl, win, xs, band)
        out[w] = (pk, ct, int(ok))
    return out


def cpm_stats(x, grid):
    g = cpm.GRIDS[grid]
    return {w: cpm.presence(g, x[-int(w * FS):]) for w in CPM_WINDOWS}


def stats(x):
    return {"ofdm": {b: ofdm_stats(x, b) for b in SYNC_BANDS}, "cpm": {g: cpm_stats(x, g) for g in GRIDS}}


def ofdm_burst(band, rng):
    s = modem.SUBMODES[SPECS[band]]
    return modem.modulate([rng.bytes(codes.payload_bytes(s)) for _ in range(slots_for(s, 10.0))], s)


def cpm_burst(grid, rng):
    g = cpm.GRIDS[grid]
    return cpm.bandpass(g, cpm.tones(g, rng.integers(0, g.m, int(10 * g.rate))))


def job(args):
    kind, name, seed, n = args
    rng = np.random.default_rng(seed)
    out = defaultdict(list)
    L = int(BUF_S * FS)
    for t in range(n):
        if kind == "noise":
            out["noise"].append(stats(rng.normal(size=L)))
            continue
        if kind == "ssb":
            lo, hi = (300, 2400) if name == "ssb2400" else (400, 2200)
            w = rng.normal(size=L + FS)
            sos = sps.butter(6, [lo, hi], btype="bandpass", fs=FS, output="sos")
            out[name].append(stats(sps.sosfilt(sos, w)[FS:]))
            continue
        if kind == "preset":
            dur = 20.0
            nn = int(dur * FS)
            x = rng.normal(size=nn) + interference.Interference(
                interference.PRESETS[name], seed=seed * 100 + t, horizon=dur).render(0.0, nn, 1.0)
            x = audio.Blanker()(x)
            a = int(rng.integers(FS, nn - L))
            out[f"preset:{name}"].append(stats(x[a:a + L]))
            continue
        if kind == "narrow":
            f0 = float(rng.choice([1500.0, 1531.25]) if t % 2 == 0 else rng.uniform(1300, 1700))
            sig = {
                "tone": np.sin(2 * np.pi * f0 * np.arange(L) / FS),
                "cw": np.pad(cwid.audio(cwid.id_text("W1AW"), 20, freq=f0), (0, L))[:L],
                "hand": np.pad(cpm.tones(cpm.Grid(f"hand{f0:g}", 8, 6.25, center=f0), rng.integers(0, 8, 66)),
                               (0, L))[:L],
            }[name]
            for snr in (0, 20):
                r = hfchannel.awgn(sig, snr, seed=seed * 1000 + t, s_power=hfchannel.active_power(sig))
                out[f"{name} {snr:+d}"].append(stats(r))
            continue
        x = ofdm_burst(name, rng) if kind == "ofdm" else cpm_burst(name, rng)
        for ch in CHANNELS:
            for snr in SNRS:
                r = hfchannel.apply_channel(x, snr, freq_offset_hz=float(rng.uniform(-120, 120)),
                                            fading_preset=None if ch == "awgn" else ch, seed=seed * 1000 + t)
                a = int(rng.integers(int(1.5 * FS), len(r) - L))
                out[f"{kind} {name} {ch} {snr:+d}"].append(stats(r[a:a + L]))
    return dict(out)


CURVE_M = range(2, 10)  # OFDM fold frames (window (m + COPY_PAIRS) frames)
CURVE_S = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)  # CPM buffer lengths


def curve_job(args):
    """Noise only: OFDM contrast per fold length m, CPM presence per buffer length."""
    seed, n = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = rng.normal(size=int(3.5 * FS))
        row = {}
        for band in SYNC_BANDS:
            d = sync.StreamDetector(ofdm.band(band))
            d.feed(dsp.to_baseband(x, 0))
            a, C = d.outputs_from(0)
            C = C[:, :max(0, len(x) - a - modem.M + 1)]
            D = sync.pair_products(C, modem.FRAME_SAMPLES, modem.COPY_PAIRS)
            for m in CURVE_M:
                row[f"{band} {m}"] = modem.pilot_presence(D, d.level(), (m + modem.COPY_PAIRS) * modem.FRAME_SAMPLES)
        for g in GRIDS:
            for s in CURVE_S:
                row[f"{g} {s}"] = cpm.presence(cpm.GRIDS[g], x[-int(s * FS):])
        out.append(row)
    return out


def curve(jobs, n, fa):
    """Each detector's threshold at `fa` per window length, and a fit a + b / sqrt(length)."""
    with Pool(jobs) as p:
        rows = [r for part in p.map(curve_job, [(1300 + j, n) for j in range(jobs)]) for r in part]
    print(f"{len(rows)} noise buffers, thresholds at {fa:g}/hop")

    def fit(xs, ys):
        A = np.stack([np.ones(len(xs)), 1 / np.sqrt(xs)], axis=1)
        a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        if b < 0:
            return float(np.max(ys)), 0.0
        return a + max(0.0, float(np.max(ys - (a + b / np.sqrt(xs))))), b

    ms = np.array(list(CURVE_M), dtype=float)
    th = np.array([[np.quantile([r[f"{b} {m}"][1] for r in rows if r[f"{b} {m}"][0] >= 8.0] + [0.0] * sum(
        r[f"{b} {m}"][0] < 8.0 for r in rows), 1 - fa) for m in CURVE_M] for b in SYNC_BANDS])
    for b, t in zip(SYNC_BANDS, th):
        print(f"OFDM {b:3s} contrast (peak >= 8) by m {list(CURVE_M)}: " + " ".join(f"{v:.2f}" for v in t))
    a, b = fit(ms, th.max(axis=0))
    print(f"OFDM fit over bands: {a:.3f} + {b:.3f} / sqrt(m)")
    for g in GRIDS:
        n_sym = np.array([int(s * cpm.GRIDS[g].rate) for s in CURVE_S], dtype=float)
        t = np.array([np.quantile([r[f"{g} {s}"] for r in rows], 1 - fa) for s in CURVE_S])
        a, b = fit(n_sym, t)
        print(f"CPM {g} by s {CURVE_S}: " + " ".join(f"{v:.4f}" for v in t) + f"  fit {a:.4f} + {b:.4f} / sqrt(n)")


def report(path, fa, peak):
    """Per detector and window: the threshold noise passes `fa` of the time,
    what interference/narrowband signals do at it, and the bursts it detects."""
    with open(path) as f:
        agg = json.load(f)
    noise = agg["noise"]
    print(f"{len(noise)} noise buffers ({len(noise) * 0.25 / 3600:.2f} h); thresholds at {fa:g}/hop, OFDM peak >= {peak}")
    fa_cells = sorted(k for k in agg if k.split()[0] in ("tone", "cw", "hand") or k.startswith(("preset:", "ssb")))
    th = {}

    def ofdm_ok(s, band, w):
        return bool(s["ofdm"][band][str(w)][2])

    for band in SYNC_BANDS:
        for w in OFDM_WINDOWS:
            print(f"OFDM {band:3s} {w:4.2f}s (verified): "
                  + " ".join(f"{k} {np.mean([ofdm_ok(s, band, w) for s in agg[k]]):.2f}" for k in fa_cells))
    for g in GRIDS:
        for w in CPM_WINDOWS:
            v = np.array([s["cpm"][g][str(w)] for s in noise])
            th[g, w] = t = float(np.quantile(v, 1 - fa))
            print(f"CPM {g} {w:4.2f}s: th {t:.4f} (noise med {np.median(v):.4f}) | "
                  + " ".join(f"{k} {np.mean([s['cpm'][g][str(w)] >= t for s in agg[k]]):.2f}" for k in fa_cells))
    for kind, names, test in (
            ("ofdm", list(SPECS), lambda s, w: any(ofdm_ok(s, b, w) for b in SYNC_BANDS)),
            ("cpm", list(GRIDS), lambda s, w: any(s["cpm"][g][str(w)] >= th[g, w] for g in GRIDS))):
        ws = OFDM_WINDOWS if kind == "ofdm" else CPM_WINDOWS
        print(f"\n{kind} bursts detected; rows window, cols SNR {SNRS}")
        for n in names:
            for ch in CHANNELS:
                for w in ws:
                    row = [np.mean([test(s, w) for s in agg.get(f"{kind} {n} {ch} {snr:+d}", [])] or [np.nan])
                           for snr in SNRS]
                    print(f"  {n:7s} {ch:4s} {w:4.2f}s " + " ".join(f"{v:5.2f}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", metavar="JSON", help="analyse a finished run instead")
    ap.add_argument("--fa", type=float, default=2e-4, help="noise's pass rate per hop the thresholds are set at")
    ap.add_argument("--peak", type=float, default=8.0, help="OFDM: the fold peak floor under the contrast rule")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--n", type=int, default=20, help="trials per burst cell and job")
    ap.add_argument("--noise", type=int, default=400, help="noise buffers per job")
    ap.add_argument("--out", default="runs/presence_study.json")
    ap.add_argument("--curve", action="store_true", help="noise-only thresholds by window length, and their fit")
    a = ap.parse_args()
    if a.curve:
        return curve(a.jobs, a.noise, a.fa)
    if a.report:
        return report(a.report, a.fa, a.peak)
    cells = ([("noise", "", 900 + j, a.noise) for j in range(a.jobs)]
             + [("narrow", k, 700 + 13 * j + i, max(1, a.n // 2)) for i, k in enumerate(("tone", "cw", "hand"))
                for j in range(2)]
             + [("ssb", k, 800 + 13 * j + i, max(1, a.n // 2)) for i, k in enumerate(("ssb2400", "ssb2200"))
                for j in range(2)]
             + [("preset", k, 850 + 13 * j + i, max(1, a.n // 2))
                for i, k in enumerate(("carriers", "edge", "qrm_wide", "imp_long", "mid")) for j in range(2)]
             + [("ofdm", b, 100 + 17 * j + i, max(1, a.n // 4)) for i, b in enumerate(SPECS) for j in range(4)]
             + [("cpm", g, 500 + 17 * j + i, max(1, a.n // 4)) for i, g in enumerate(GRIDS) for j in range(4)])
    agg = defaultdict(list)
    with Pool(a.jobs) as p:
        for d in p.imap_unordered(job, cells):
            for k, v in d.items():
                agg[k] += v

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, tuple):
            return [conv(v) for v in o]
        return float(o) if isinstance(o, (np.floating, float)) else o

    with open(a.out, "w") as f:
        json.dump(conv(dict(agg)), f)
    print(f"{a.out}: {sum(len(v) for v in agg.values())} buffers, {len(agg)} cells")


if __name__ == "__main__":
    main()
