"""A band's receiver constants measured on noise, as the existing bands'
were (modem.HEADER_MIN_SCORE, PILOT_NOISE, PILOT_PATTERN):

- pilots (scripts/pilot_metric_study.py's rule): windows of modem.PILOT_PAIRS
  frame-pilot pairs at random places in white noise; PILOT_PATTERN is the
  pattern's 90th percentile, PILOT_NOISE the mean coherence's 99th among
  windows whose pattern passes it;
- the header floor (w48's rule, modem.HEADER_MIN_SCORE): the best score a
  header read gets at random starts in white noise, and off other bands'
  bursts at every offset near their preambles; the floor goes above both.

    DATA2G_PROFILE=fm python scripts/band_consts.py f55 f108 [--n 3000]
"""

from data2g import threads

threads.limit(1)

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, modem  # noqa: E402
from data2g.config import FRAME_SAMPLES, FS, SUBMODES  # noqa: E402


def pilots(band, n, rng):
    spec = next(s for s in SUBMODES.values() if s.band == band)
    lock = dict(spec=spec, n_cw=64 // spec.frames_per_cw, cfo=0.0)
    cs, pats = [], []
    for _ in range(n):
        x = rng.standard_normal((modem.PILOT_PAIRS + 3) * FRAME_SAMPLES)
        c, pat = modem.pilot_coherence(x, dict(lock, p0=int(rng.integers(0, FRAME_SAMPLES))), modem.PILOT_PAIRS,
                                       pattern=True)
        cs.append(np.mean(c[:modem.PILOT_PAIRS]))
        pats.append(pat)
    cs, pats = np.array(cs), np.array(pats)
    pattern = float(np.percentile(pats, 90))
    return pattern, float(np.percentile(cs[pats > pattern], 99)), float(np.percentile(cs, 99))


def headers(band, n, rng, others):
    z = modem.dsp.to_baseband(rng.standard_normal(n * 400 + 4 * FS))
    noise = [modem._read_header(z, int(s), band)["score"] for s in rng.integers(0, len(z) - 2 * FS, n)]
    cross = []
    for o in others:  # someone else's burst: read as this band's header at every start near its preamble
        spec = next(s for s in SUBMODES.values() if s.band == o and s.code == "ldpc")
        x = np.concatenate([np.zeros(FS // 2), modem.modulate([rng.bytes(codes.payload_bytes(spec))] * 4, spec),
                            np.zeros(FS)])
        zz = modem.dsp.to_baseband(x + 1e-3 * rng.standard_normal(len(x)))
        cross += [modem._read_header(zz, s, band)["score"] for s in range(FS // 2 - 800, FS // 2 + 4000, 7)]
    return max(noise), float(np.percentile(noise, 99.9)), max(cross)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bands", nargs="+")
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    sync = list(dict.fromkeys(s.sync_band for s in SUBMODES.values()))
    for band in a.bands:
        pat, coh, coh_all = pilots(band, a.n, rng)
        hmax, h999, cmax = headers(band, a.n, rng, [b for b in sync if b != band])
        print(f"{band}: PILOT_PATTERN {pat:.3f} PILOT_NOISE {coh:.3f} (coherence alone 99%: {coh_all:.3f}); "
              f"header on noise max {hmax:.3f} (99.9% {h999:.3f}), off other bands' bursts max {cmax:.3f}", flush=True)


if __name__ == "__main__":
    main()
