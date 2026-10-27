"""CPM acquisition statistics (2026-10-26, cpm-acq-gap): what each lock
gate reads on bursts the shipped receiver lost (genie decoded them) and on
audio holding no burst of the grid (noise, OFDM, other grids, the grid's
own burst with its front cut off), at detect's best front placement.

    python -m scripts.cpm_acq_fit --n 400
"""

from data2g import threads  # noqa: E402

threads.limit(1)

import argparse  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data2g import codes, cpm  # noqa: E402
from data2g.config import FS  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

SOURCES = ("noise", "ofdm", "cpm-other", "cpm-cut")


def stats(g, y, s0, cfo):
    """At a placement: the front's LLR (lf), front + best word over one (l1)
    or both (l2) header copies (cpm.read_header) and the words read."""
    h1, h2 = (cpm.read_header(g, y, s0, cfo, copies=c) for c in (1, 2))
    words = [w[:3] for w in cpm.valid_words(g.name)]
    return dict(lf=h1[5], l1=h1[3], l2=h2[3], w1=words.index(h1[:3]), w2=words.index(h2[:3]))


def burst(spec, rng):
    """A random burst of `spec`: CPM (control + 0-3 data codewords) or OFDM (1-3 codewords)."""
    from data2g import modem
    from data2g.arq.modes import is_cpm

    if is_cpm(spec):
        ctl = codes.encode(cpm.CTL[spec.grid], rng.bytes(codes.payload_bytes(cpm.CTL[spec.grid])), 0, 0)
        data = [codes.encode(spec, rng.bytes(codes.payload_bytes(spec)), 0, 0) for _ in range(int(rng.integers(0, 4)))]
        return cpm.modulate(spec, [ctl] + data, dup=False)
    n_cw = int(rng.integers(1, 4))
    bits = np.stack([codes.encode(spec, rng.bytes(codes.payload_bytes(spec)), 0, 0) for _ in range(n_cw)])
    return modem.modulate_bits(codes.spread(bits, spec.bits_per_cu), spec)


def false_one(args):
    from data2g import hfchannel
    from data2g.arq.modes import MODES, is_cpm

    grid, source, seed, mid = args
    g = cpm.GRIDS[grid]
    rng = np.random.default_rng([seed, list(cpm.GRIDS).index(grid), SOURCES.index(source), 7])
    M = max(cpm.mid_rows(grid)) * g.T if mid else 0  # a mid block's lead: room for every word's start
    n = M + 3 * cpm.hdr_span(grid, 1) + cpm.hdr_span(grid, 2) + FS
    if source == "noise":
        y = rng.normal(0, 1, n)
    else:
        if source == "ofdm":
            spec = rng.choice([s for s in MODES.values() if not is_cpm(s)])
        elif source == "cpm-other":
            spec = rng.choice([s for s in cpm.SPECS.values() if s.grid != grid])
        else:
            spec = rng.choice(cpm.grid_specs(grid))
        x = burst(spec, rng)
        if source == "cpm-cut" and mid:  # past its mid block and second header copy: data only
            ctl = codes.encode(cpm.CTL[grid], rng.bytes(codes.payload_bytes(cpm.CTL[grid])), 0, 0)
            x = cpm.modulate(spec, [ctl] + [codes.encode(spec, rng.bytes(codes.payload_bytes(spec)), 0, 0)
                                            for _ in range(3)], dup=False)
            lay = cpm.layout(grid, cpm.stream_symbols(grid, 3, False))
            x = x[(lay.hdr_rows[1][-1] + int(rng.integers(1, 40))) * g.T:]
        elif source == "cpm-cut":
            x = x[(len(cpm.preamble_pattern(g)) + int(rng.integers(1, 40))) * g.T:]
        x = np.concatenate([np.zeros(int(rng.integers(0, FS)) + M), x, np.zeros(n)])[:n]
        y = hfchannel.apply_channel(x, snr_db=float(rng.choice([-10, -5, 0, 10, 20, 30])),
                                    freq_offset_hz=float(rng.uniform(-50, 50)),
                                    fading_preset=[None, "mpg"][int(rng.integers(2))], seed=seed)
    if mid:
        sc, p, cfo = cpm.detect(g, y[M:M + 3 * cpm.hdr_span(grid, 1)], mid=True)
        p += M
        h = cpm.read_mid(g, y, p, cfo)
        return grid, source, dict(coarse=sc, lm=h[3] if h else -99.0, margin=h[3] - h[4] if h else 0.0)
    sc, s0, cfo = cpm.detect(g, y[:3 * cpm.hdr_span(grid, 1)], front_only=True)
    return grid, source, dict(stats(g, y, s0, cfo), coarse=sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--mid", action="store_true", help="find_mid's read at the best mid block placement")
    a = ap.parse_args()
    jobs = [(g, s, i, a.mid) for g in cpm.GRIDS for s in SOURCES for i in range(a.n)]
    out = {}
    with Pool(a.jobs) as pool:
        for g, s, v in pool.imap_unordered(false_one, jobs, chunksize=8):
            out.setdefault((g, s), []).append(v)
    for g in cpm.GRIDS:
        for s in SOURCES:
            v = out[(g, s)]
            print(f"{g} {s:9s} " + "  ".join(f"{k} q99 {np.quantile([x[k] for x in v], .99):7.2f} max {max(x[k] for x in v):7.2f}"
                                           for k in (("coarse", "lm", "margin") if a.mid
                                                     else ("coarse", "lf", "l1", "l2"))))


if __name__ == "__main__":
    main()
