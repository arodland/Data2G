"""cpm.HEADER2_THRESHOLD: the two-copy header score (late_lock's read) on
audio holding no burst of the grid, at detect's best front placement
whatever its sync score (so stricter than a live candidate needs: one has
passed SYNC_THRESHOLD too). Sources: noise, OFDM bursts and other grids' CPM
bursts through AWGN/MPG at 0-30 dB, and the grid's own bursts with the front
cut off (a candidate inside a burst). Prints each source's largest score
(only starts passing PEAK_RATIO, as late_lock reads) and 1.1x the overall.

    uv run --no-sync python scripts/cpm_header2_fit.py --n 400
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse
from multiprocessing import Pool

import numpy as np

from data2g import codes, cpm, hfchannel, modem
from data2g.arq.modes import MODES, is_cpm
from data2g.config import FS


SOURCES = ("noise", "ofdm", "cpm-other", "cpm-cut")


def burst(spec, rng):
    if is_cpm(spec):
        ctl = codes.encode(cpm.CTL[spec.grid], rng.bytes(codes.payload_bytes(cpm.CTL[spec.grid])), 0, 0)
        data = [codes.encode(spec, rng.bytes(codes.payload_bytes(spec)), 0, 0) for _ in range(int(rng.integers(0, 4)))]
        return cpm.modulate(spec, [ctl] + data, dup=False)
    n_cw = int(rng.integers(1, 4))
    bits = np.stack([codes.encode(spec, rng.bytes(codes.payload_bytes(spec)), 0, 0) for _ in range(n_cw)])
    return modem.modulate_bits(codes.spread(bits, spec.bits_per_cu), spec)


def score(g, y):
    """late_lock's two-copy score at detect's best front placement: (over
    starts passing PEAK_RATIO, as late_lock reads, None if none; over all)."""
    _, s0, cfo = cpm.detect(g, y[:3 * cpm.hdr_span(g.name, 1)], front_only=True)
    reads = [(cpm.read_header(g, y, s, cfo)[3], cpm._peak_ratio(g, y, s, cfo) >= cpm.PEAK_RATIO)
             for s in cpm._tile_starts(g, s0)]
    return max((v for v, ok in reads if ok), default=None), max((v for v, _ in reads), default=0.0)


def one(args):
    grid, source, seed = args
    g = cpm.GRIDS[grid]
    rng = np.random.default_rng([seed, list(cpm.GRIDS).index(grid), SOURCES.index(source)])
    n = 3 * cpm.hdr_span(grid, 1) + cpm.hdr_span(grid, 2) + FS
    if source == "noise":
        y = rng.normal(0, 1, n)
    else:
        if source == "ofdm":
            spec = rng.choice([s for s in MODES.values() if not is_cpm(s)])
        elif source == "cpm-other":
            spec = rng.choice([s for s in cpm.SPECS.values() if s.grid != grid])
        else:  # cpm-cut: the grid's own burst, from a point past its front
            spec = rng.choice(cpm.grid_specs(grid))
        x = burst(spec, rng)
        if source == "cpm-cut":
            x = x[(len(cpm.preamble_pattern(g)) + int(rng.integers(1, 40))) * g.T:]
        x = np.concatenate([np.zeros(int(rng.integers(0, FS))), x, np.zeros(n)])[:n]
        y = hfchannel.apply_channel(x, snr_db=float(rng.choice([0, 10, 20, 30])),
                                    freq_offset_hz=float(rng.uniform(-50, 50)),
                                    fading_preset=[None, "mpg"][int(rng.integers(2))], seed=seed)
    return grid, source, score(g, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="buffers per grid and source")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    sources = SOURCES
    jobs = [(g, s, i) for g in cpm.GRIDS for s in sources for i in range(a.n)]
    out = {}
    with Pool(a.jobs) as pool:
        for g, s, v in pool.imap_unordered(one, jobs, chunksize=8):
            out.setdefault((g, s), []).append(v)
    for g in cpm.GRIDS:
        worst = worst_all = 0.0
        for s in sources:
            v = [x for x, _ in out[(g, s)] if x is not None]
            u = [x for _, x in out[(g, s)]]
            worst, worst_all = max(worst, max(v, default=0.0)), max(worst_all, max(u))
            print(f"{g} {s}: passing PEAK_RATIO {len(v)}/{len(u)}, max {max(v, default=0.0):.3f}; "
                  f"every placement max {max(u):.3f}, 99% {np.quantile(u, 0.99):.3f}")
        print(f"{g}: 1.1 x max = {1.1 * worst:.3f}, every placement {1.1 * worst_all:.3f} "
              f"(now {cpm.HEADER2_THRESHOLD[g]})")


if __name__ == "__main__":
    main()
