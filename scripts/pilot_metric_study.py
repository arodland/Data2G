"""Frame-pilot coherence metrics (modem.pilot_coherence): how well each
tells a real burst's pilots from noise and from a narrowband signal (a
steady tone, a CW key-down, a Hand's 8-FSK). The current metric,
|sum_f y_f conj(y'_f)| / sqrt(energies), reads ~1 for any one dominant
carrier, so a tone passes as pilots.

Per band, the mean over modem.PILOT_PAIRS consecutive pairs (the live
check's window), under three metrics:
  cur    the current one
  clipN  each carrier's |y| capped at N times the window's median first
  phase  each carrier's product normalized to unit length: |sum u_f| / N
  pat    the known pilot divided out, adjacent carriers' agreement (smooth)
  patp   pat with each adjacent pair at unit weight
  cur*patp  the current metric times patp
Thresholds: each metric's 99th percentile on noise (as PILOT_NOISE was
set). Reported: real bursts' windows over it (AWGN/MPP/MPD/MPG at
-6/-4/0/+10 dB), and narrowband windows over it.

    uv run python scripts/pilot_metric_study.py [--jobs 6] [--n 40]
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
from collections import defaultdict  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, cpm, cwid, hfchannel, modem  # noqa: E402
from data2g.arq.policy import slots_for  # noqa: E402
from data2g.config import FRAME_SAMPLES, FS, NCP, M  # noqa: E402
from data2g.waveform import dsp, ofdm  # noqa: E402

SPECS = {"w": "qpsk-r1/2", "w48": "w48-qpsk-r1/2", "n10": "n10-qpsk-r1/2", "n4": "n4-qpsk-r1/3"}
METRICS = ("cur", "patp")
GATES = (80, 90, 95)  # gateN: cur, vetoed (0) where patT is under noise's Nth percentile
ALL = METRICS + ("patT",) + tuple(f"gate{g}" for g in GATES)


def smooth(y, pilot, unit=False):
    """The pilot divided out, how coherently adjacent carriers agree: a real
    pilot leaves the channel, smooth over 50 Hz (a timing offset's linear
    phase slope included); a tone's leakage, or noise, leaves the pilot's
    pseudo-random pattern."""
    z = y * np.conj(pilot)
    a = z[1:] * np.conj(z[:-1])
    if unit:  # each adjacent pair counts once: one strong bin can't carry it
        return float(np.abs(np.sum(a / (np.abs(a) + 1e-30))) / len(a))
    return float(np.abs(np.sum(a)) / (np.sum(np.abs(a)) + 1e-30))


def pair_values(y0, y1, metric, pilot=None):
    if metric == "pat":
        return 0.5 * (smooth(y0, pilot) + smooth(y1, pilot))
    if metric == "patp":
        return 0.5 * (smooth(y0, pilot, True) + smooth(y1, pilot, True))
    if metric == "cur*patp":
        return pair_values(y0, y1, "cur") * pair_values(y0, y1, "patp", pilot)
    if metric.startswith("clip"):
        c = float(metric[4:])
        for y in (y0, y1):
            lim = c * np.median(np.abs(y))
            y *= np.minimum(1.0, lim / (np.abs(y) + 1e-30))
    p = y1 * np.conj(y0)
    if metric == "phase":
        return float(np.abs(np.sum(p / (np.abs(p) + 1e-30))) / len(p))
    return float(np.abs(np.sum(p)) / (np.sqrt(np.sum(np.abs(y0) ** 2) * np.sum(np.abs(y1) ** 2)) + 1e-30))


def windows(x, band, p0, cfo, n_frames):
    """Per metric, the means of PILOT_PAIRS consecutive pairs, each window
    of frame pilots from p0 (as pilot_coherence reads them)."""
    b = ofdm.band(band)
    z = dsp.freq_correct(dsp.to_baseband(np.asarray(x, dtype=np.float64)), cfo)
    ys = [b.demod_window(z, p0 + f * FRAME_SAMPLES + NCP, modem.DEMOD_BACKOFF) for f in range(n_frames)
          if p0 + f * FRAME_SAMPLES + NCP + M + modem.DEMOD_BACKOFF <= len(z)]
    out = {}
    k = modem.PILOT_PAIRS
    out["patT"], out["conc"] = [], []
    for i in range(0, len(ys) - k, k):
        win = ys[i:i + k + 1]
        a = sum((y * np.conj(b.pilot))[1:] * np.conj((y * np.conj(b.pilot))[:-1]) for y in win)  # per carrier pair, summed over frames
        out["patT"].append(float(np.abs(np.sum(a / (np.abs(a) + 1e-30))) / len(a)))
        e = sum(np.abs(y) ** 2 for y in win)
        out["conc"].append(float(e.max() / e.sum()))
    for m in METRICS:
        v = [pair_values(ys[f - 1].copy(), ys[f].copy(), m, b.pilot) for f in range(1, len(ys))]
        k = modem.PILOT_PAIRS
        out[m] = [float(np.mean(v[i:i + k])) for i in range(0, len(v) - k + 1, k)]
    return out


def lock_for(band, rng):
    """A real burst of the band (clean) and its lock: p0 and frame count."""
    s = modem.SUBMODES[SPECS[band]]
    x = modem.modulate([rng.bytes(codes.payload_bytes(s)) for _ in range(slots_for(s, 10.0))], s)
    y = np.concatenate([np.zeros(FS), x, np.zeros(FS)])
    p = modem.find_burst(y, [s.sync_band])
    return y, p["p0"], modem.frames_on_air(s, p["n_cw"]) + 1


def job(args):
    band, seed, n = args
    rng = np.random.default_rng(seed)
    out = defaultdict(lambda: defaultdict(list))  # kind -> metric -> windows
    for t in range(n):
        y, p0, nf = lock_for(band, rng)
        for ch, snr in [(c, s) for c in ("awgn", "mpp", "mpd", "mpg") for s in (-6, -4, 0, 10)]:
            if (t + snr) % 4:  # a quarter of the cells per trial: n/4 bursts a cell
                continue
            r = hfchannel.apply_channel(y, snr, fading_preset=None if ch == "awgn" else ch, seed=seed * 1000 + t)
            for m, v in windows(r, band, p0, 0.0, nf).items():
                out[f"real {ch} {snr:+d}"][m] += v
        noise = rng.normal(size=len(y))
        for m, v in windows(noise, band, p0, 0.0, nf).items():
            out["noise"][m] += v
        f0 = float(rng.uniform(1440, 1560))
        narrow = {
            "tone": np.sin(2 * np.pi * f0 * np.arange(len(y)) / FS),
            "cw": np.pad(cwid.audio(cwid.id_text("W1AW"), 20, freq=f0), (0, len(y)))[:len(y)],
            "hand": np.pad(cpm.tones(cpm.Grid(f"hand{f0:g}", 8, 6.25, center=f0), rng.integers(0, 8, 66)),
                           (0, len(y)))[:len(y)],
        }
        for kind, sig in narrow.items():
            for snr in (0, 20):
                r = hfchannel.awgn(sig, snr, seed=seed * 1000 + t + snr, s_power=hfchannel.active_power(sig))
                for m, v in windows(r, band, p0, 0.0, nf).items():
                    out[f"{kind} {snr:+d}"][m] += v
    return band, {k: dict(d) for k, d in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--n", type=int, default=40, help="bursts per band and job")
    a = ap.parse_args()
    cells = [(b, 100 + 17 * j + i, a.n) for i, b in enumerate(SPECS) for j in range(a.jobs)]
    with Pool(a.jobs) as p:
        res = p.map(job, cells)
    for band in SPECS:
        agg = defaultdict(lambda: defaultdict(list))
        for b, d in res:
            if b == band:
                for k, mv in d.items():
                    for m, v in mv.items():
                        agg[k][m] += v
        # pair windows: patp and cur of the same window, side by side
        for g in GATES:
            gp = float(np.percentile(agg["noise"]["patp"], g))
            gp = float(np.percentile(agg["noise"]["patT"], g))
            for k in agg:
                agg[k][f"gate{g}"] = list(np.where(np.array(agg[k]["patT"]) > gp, agg[k]["cur"], 0.0))
        thr = {m: float(np.percentile(agg["noise"][m], 99)) for m in ALL}
        print(f"  constants: patT gate " + "  ".join(f"{g}%: {np.percentile(agg['noise']['patT'], g):.3f}" for g in GATES))
        print(f"\n{band} ({len(agg['noise']['cur'])} noise windows): threshold (noise 99%) " +
              "  ".join(f"{m} {thr[m]:.2f}" for m in ALL))
        print(f"  {'windows over it':18s} " + " ".join(f"{m:>8s}" for m in ALL))
        for k in sorted(agg, key=lambda k: (k.split()[0] != "real", k)):
            if k == "noise":
                continue
            print(f"  {k:18s} " + " ".join(f"{np.mean(np.array(agg[k][m]) > thr[m]):8.2f}" for m in ALL))
        print("  patT / conc percentiles 10/50/90: " + "; ".join(
            f"{k} " + "/".join(f"{np.percentile(agg[k]['patT'], q):.2f}" for q in (10, 50, 90)) + " | "
            + "/".join(f"{np.percentile(agg[k]['conc'], q):.2f}" for q in (10, 50, 90))
            for k in sorted(agg) if k in ("noise", "tone +0", "tone +20", "hand +0", "hand +20")
            or k.startswith("real") and k.endswith("-6")))


if __name__ == "__main__":
    main()
