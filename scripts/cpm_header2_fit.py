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
    """At detect's best front placement: late_lock's two-copy score (None
    unless the placement passes PEAK_RATIO, as late_lock reads), the same
    over any placement, and the front's sync score and its one-copy header
    score (find's early lock: SYNC_THRESHOLD, HEADER_THRESHOLD)."""
    sync, s0, cfo = cpm.detect(g, y[:3 * cpm.hdr_span(g.name, 1)], front_only=True)
    two = cpm.read_header(g, y, s0, cfo)[3]
    ok = cpm._peak_ratio(g, y, s0, cfo) >= cpm.PEAK_RATIO
    return (two if ok else None), two, sync, cpm.read_header(g, y, s0, cfo, copies=1)[3]


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
            v = [x[0] for x in out[(g, s)] if x[0] is not None]
            u = [x[1] for x in out[(g, s)]]
            worst, worst_all = max(worst, max(v, default=0.0)), max(worst_all, max(u))
            print(f"{g} {s}: passing PEAK_RATIO {len(v)}/{len(u)}, max {max(v, default=0.0):.3f}; "
                  f"every placement max {max(u):.3f}, 99% {np.quantile(u, 0.99):.3f}")
        print(f"{g}: 1.1 x max = {1.1 * worst:.3f}, every placement {1.1 * worst_all:.3f} "
              f"(now {cpm.HEADER2_THRESHOLD[g]})")
        # the early lock's two scores on noise and OFDM (SYNC_THRESHOLD and
        # HEADER_THRESHOLD: 1.1x their largest, the provisional fit's rule)
        sync = max(x[2] for s in ("noise", "ofdm") for x in out[(g, s)])
        h1 = max(x[3] for s in ("noise", "ofdm") for x in out[(g, s)])
        print(f"{g}: noise/OFDM sync max {sync:.3f} -> {1.1 * sync:.3f} (now {cpm.SYNC_THRESHOLD[g]}); "
              f"one-copy header max {h1:.3f} -> {1.1 * h1:.3f} (now {cpm.HEADER_THRESHOLD[g]})")


if __name__ == "__main__":
    main()
