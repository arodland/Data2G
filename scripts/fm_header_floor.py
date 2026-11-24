"""FM header floors (modem.HEADER_MIN_SCORE) for f55 and f108, measured on
what an FM receiver hands the modem rather than white noise
(scripts/band_consts.py set 0.23 / 0.16 from white noise alone):

- false: the best header score at random starts in idle FM audio (open
  squelch hiss and an unmodulated carrier at 8 dB CNR, mic and flat radios,
  fmchannel via mf_detect_study.noise_audio), and off other bands' bursts
  through the FM channel at every start near their preambles;
- true: the score of a real header (the band's lowest rung) through the FM
  channel at low CNR, static and mobile-rural, each radio on its own TX
  path, read at the burst's true start (best of +-4 samples: acquisition's
  own error).
Everything through audio.Blanker first, as the live receiver.

    DATA2G_PROFILE=fm python scripts/fm_header_floor.py [--n 3000] [--trials 60] [--workers 8]
"""

from data2g import threads

threads.limit(1)

import argparse  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mf_detect_study import noise_audio  # noqa: E402

from data2g import audio, codes, config, fmchannel, modem  # noqa: E402
from data2g.config import FS, SUBMODES  # noqa: E402

BANDS = ("f55", "f108")
NOISE = ("fm-mic", "fm-flat", "fm-mic-idle8", "fm-flat-idle8")
PATHS = {"mic": "mic", "flat": "flat"}  # radio -> its TX path
CNRS = (-2, 0, 2, 4, 8, 12)  # the lowest rungs' static 10% knees sit at 0-2 dB, mobile at 7.5-11


def lowest(band):
    return min((s for s in SUBMODES.values() if s.band == band), key=lambda s: s.k / s.coded_bits)


def scores_on(x, band, starts):
    z = modem.dsp.to_baseband(audio.Blanker()(x))
    return [modem._read_header(z, int(s), band)["score"] for s in starts]


def false_job(args):
    source, seed, n = args
    x = noise_audio(source, seed, n * 0.02 + 4)
    rng = np.random.default_rng(seed)
    starts = rng.integers(FS, len(x) - 3 * FS, n)
    return source, {b: scores_on(x, b, starts) for b in BANDS}


def burst(spec, seed, tx_path):
    rng = np.random.default_rng(seed)
    sent = [rng.bytes(codes.payload_bytes(spec)) for _ in range(2)]
    return np.concatenate([np.zeros(FS), modem.modulate(sent, spec, tx_path=tx_path), np.zeros(FS)])


def cross_job(args):
    other, radio, seed = args
    spec = lowest(other)
    x = burst(spec, seed, PATHS[radio])
    ch = fmchannel.FmChannel("static", radio, 30.0, seed=seed, duration_s=len(x) / FS + 1,
                             drive_dev_hz=1460.0 if radio == "mic" else 2350.0)
    y = ch.apply(x)
    p = FS + modem.LEADIN_SAMPLES
    return other, {b: scores_on(y, b, range(p - 800, p + 4000, 7)) for b in BANDS if b != other}


def true_job(args):
    band, radio, cond, cnr, seed = args
    spec = lowest(band)
    x = burst(spec, seed, PATHS[radio])
    start = FS + modem.LEADIN_SAMPLES  # the preamble's first sample: burst_waveform puts it after the lead-in
    ch = fmchannel.FmChannel(cond, radio, cnr, seed=seed, duration_s=len(x) / FS + 1,
                             drive_dev_hz=1460.0 if radio == "mic" else 2350.0)
    y = ch.apply(x)
    return (band, radio, cond, cnr), max(scores_on(y, band, range(start - 4, start + 5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    assert config.PROFILE == "fm", "run under DATA2G_PROFILE=fm"
    # the genie start must be right: a clean burst's header reads near 1 there
    for b in BANDS:
        x = burst(lowest(b), 0, "flat")
        s = FS + modem.LEADIN_SAMPLES
        assert max(scores_on(x + 1e-4 * np.random.default_rng(0).standard_normal(len(x)), b, range(s - 4, s + 5))) > 0.8, b
    per = a.n // 10
    with ProcessPoolExecutor(a.workers) as ex:
        false = {}
        for src, d in ex.map(false_job, [(s, k, per) for s in NOISE for k in range(10)]):
            for b, v in d.items():
                false.setdefault((src, b), []).extend(v)
        cross = {}
        for other, d in ex.map(cross_job, [(o, r, k) for o in ("w", "n10", *BANDS) for r in PATHS for k in range(3)]):
            for b, v in d.items():
                cross.setdefault(b, []).extend(v)
        true = {}
        jobs = [(b, r, c, cnr, k) for b in BANDS for r in PATHS for c in ("static", "mobile-rural") for cnr in CNRS
                for k in range(a.trials) if not (b == "f108" and r == "mic")]  # f108 is flat-only
        for key, v in ex.map(true_job, jobs, chunksize=4):
            true.setdefault(key, []).append(v)
    print(f"false headers ({a.n} reads per source and band; blanked):")
    for (src, b), v in sorted(false.items()):
        v = np.array(v)
        print(f"  {b:5} {src:14} max {v.max():.3f}  99.9% {np.percentile(v, 99.9):.3f}  99% {np.percentile(v, 99):.3f}")
    for b, v in cross.items():
        print(f"  {b:5} {'other bursts':14} max {max(v):.3f}  ({len(v)} reads)")
    print(f"true headers (lowest rung, {a.trials} trials, genie start): fraction under each floor")
    floors = (0.16, 0.18, 0.20, 0.22, 0.23, 0.24, 0.26, 0.28, 0.30)
    print("  band  radio cond          cnr  median  " + "  ".join(f"<{f:.2f}" for f in floors))
    for key in sorted(true):
        v = np.array(true[key])
        print(f"  {key[0]:5} {key[1]:5} {key[2]:13} {key[3]:+3}  {np.median(v):.3f}   "
              + "  ".join(f"{np.mean(v < f):5.2f}" for f in floors))


if __name__ == "__main__":
    main()
